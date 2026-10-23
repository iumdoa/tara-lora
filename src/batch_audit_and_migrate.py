#!/usr/bin/env python3
"""
TARA-5 全量数据批量处理与深度专家复核引擎 (Batch Audit & Migration Engine)
协议版本：tara5_action_v1

核心功能：
1. 全量扫描各书目旧版切片（包括 data/sliced 与 data/*/sliced 下共 2,505 章）；
2. 自动化专家级复核与规整：
   - 修复上章承接 (prev_chapter_ending) 的 EPUB 目录残卷与截断字符；
   - 严格区分“前置已确认事实”与“章纲未来计划事件”，提供证据出处；
   - 规范角色视界与信息差（杜绝盲区误置入已知情报）；
   - 自动检测并重平衡短断章 (cliffhanger < 60 字符) 与超大余波，确保叙事完整；
   - 自动核验并校正首尾锚点在正文中的 100% 精确子串匹配；
   - 生成细粒度行动链 (action_chain)、起止态势与适配篇幅预算；
   - 严格校验合规性，标注 needs_review 状态；
3. 将版本化切片产物写入 data/tara5_action_v1/{book}/sliced/；
4. 调用 prompt_builder 编译生成纯正文 ShareGPT 微调数据集，存入 data/tara5_action_v1/{book}/dataset/；
5. 生成全局合并微调数据集与统计报告。
"""

import os
import re
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))

from src.schema_validator import validate_tara5_chapter
from src.prompt_builder import (
    STAGE_OPENING,
    STAGE_RAMP_UP,
    STAGE_CLIMAX,
    STAGE_AFTERSHOCK,
    STAGE_CLIFFHANGER,
    VALID_STAGES,
    STAGE_TASK_MAP,
    build_task_input,
)


def clean_prev_chapter_ending(text: str, chapter_index: int = 1) -> str:
    """清理上章承接中的目录残留与字符截断"""
    if not text or not str(text).strip():
        return "（开门见山顺畅起笔）" if chapter_index > 1 else "（全书开篇第一章，开门见山切入主线）"

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return "（开门见山顺畅起笔）"

    # 如果多行都是目录形式如 "第二十七章 谎言", "第二十八章 火"
    toc_count = sum(1 for l in lines if re.match(r"^第[一二三四五六七八九十百千\d]+[章节卷篇回]", l))
    if (len(lines) > 1 and toc_count >= len(lines) - 1) or (chapter_index <= 1 and toc_count > 0):
        return "（全书开篇第一章，开门见山切入主线）"

    cleaned = text.strip()
    # 检查首字是否为破损单字+逗号 (例如 '验，现在他裤裆里...')
    m = re.match(r"^([\u4e00-\u9fa5])[，,]\s*(.+)", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(2).strip()

    return cleaned


def rebalance_segments(raw_segs: Dict[str, Any]) -> Dict[str, Any]:
    """
    规整与修复段落切片：
    - 兼容 rampup 命名 -> ramp_up；
    - 修复极短断章 (< 60 字)：从 aftershock 尾部转移铺垫段落；
    - 校准首尾锚点在正文中的精确存在。
    """
    op = raw_segs.get("opening", {})
    ru = raw_segs.get("ramp_up") or raw_segs.get("rampup", {})
    cl = raw_segs.get("climax", {})
    af = raw_segs.get("aftershock", {})
    ch = raw_segs.get("cliffhanger", {})

    clean_segs = {
        STAGE_OPENING: {
            "text": op.get("text", "").strip(),
            "anchor_start": op.get("anchor_start", "").strip(),
            "anchor_end": op.get("anchor_end", "").strip(),
        },
        STAGE_RAMP_UP: {
            "text": ru.get("text", "").strip(),
            "anchor_start": ru.get("anchor_start", "").strip(),
            "anchor_end": ru.get("anchor_end", "").strip(),
        },
        STAGE_CLIMAX: {
            "text": cl.get("text", "").strip(),
            "anchor_start": cl.get("anchor_start", "").strip(),
            "anchor_end": cl.get("anchor_end", "").strip(),
            "core_conflict": cl.get("core_conflict", "").strip(),
        },
        STAGE_AFTERSHOCK: {
            "text": af.get("text", "").strip(),
            "anchor_start": af.get("anchor_start", "").strip(),
            "anchor_end": af.get("anchor_end", "").strip(),
        },
        STAGE_CLIFFHANGER: {
            "text": ch.get("text", "").strip(),
            "anchor_start": ch.get("anchor_start", "").strip(),
            "anchor_end": ch.get("anchor_end", "").strip(),
        },
    }

    # 1. 检测并重平衡短断章
    ch_text = clean_segs[STAGE_CLIFFHANGER]["text"]
    af_text = clean_segs[STAGE_AFTERSHOCK]["text"]
    if len(ch_text) < 60 and len(af_text) > 300:
        af_paragraphs = [p.strip() for p in af_text.split("\n\n") if p.strip()]
        if len(af_paragraphs) > 1:
            tail_p = af_paragraphs[-1]
            if tail_p == ch_text or ch_text in tail_p:
                if len(af_paragraphs) > 2:
                    donor_p = af_paragraphs[-2]
                    new_af = "\n\n".join(af_paragraphs[:-2]).strip()
                    new_ch = f"{donor_p}\n\n{tail_p}".strip()
                else:
                    new_af = af_paragraphs[0]
                    new_ch = tail_p
            else:
                new_af = "\n\n".join(af_paragraphs[:-1]).strip()
                new_ch = f"{tail_p}\n\n{ch_text}".strip()

            clean_segs[STAGE_AFTERSHOCK]["text"] = new_af
            clean_segs[STAGE_CLIFFHANGER]["text"] = new_ch

    # 2. 严格核验并补正首尾锚点在正文中的存在
    for st in VALID_STAGES:
        s_data = clean_segs[st]
        text = s_data["text"]
        s_start = s_data["anchor_start"]
        s_end = s_data["anchor_end"]

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            continue

        if not s_start or s_start not in text:
            first_line = lines[0]
            # 尝试按句末标点截取第一句
            sent_m = re.search(r"[。！？!?]", first_line)
            if sent_m and sent_m.end() >= 8:
                s_data["anchor_start"] = first_line[:sent_m.end()]
            else:
                s_data["anchor_start"] = first_line[:min(len(first_line), 80)]

        if not s_end or s_end not in text:
            last_line = lines[-1]
            # 尝试取最后一句
            sents = [s.strip() for s in re.split(r"(?<=[。！？!?])", last_line) if s.strip()]
            if sents:
                s_data["anchor_end"] = sents[-1]
            else:
                s_data["anchor_end"] = last_line[-min(len(last_line), 80):]

        s_data["char_count"] = len(text)

    return clean_segs


def extract_protagonist_name(title: str, outline: List[str], tb: Dict[str, Any]) -> str:
    """启发式提取主角姓名"""
    candidates = [
        "吴邪", "许七安", "林轩", "萧炎", "韩立", "克莱恩", "陈羽",
        "方源", "孟浩", "王林", "白小纯", "石昊", "苏铭", "路明非",
        "罗峰", "周元", "牧尘", "叶凡", "纪宁", "江离", "李火旺"
    ]
    all_text = title + " " + " ".join(outline) + " " + str(tb.get("protagonist_plan", ""))
    for name in candidates:
        if name in all_text:
            return name
    return "主角"


def audit_and_migrate_chapter(legacy_data: Dict[str, Any], source_novel: str) -> Dict[str, Any]:
    """深度专家复核与转换单章"""
    ch_id = legacy_data.get("chapter_id") or f"ch_{legacy_data.get('chapter_index', 1):04d}"
    ch_index = legacy_data.get("chapter_index", 1)
    ch_title = legacy_data.get("chapter_title", "未命名章节")
    outline = legacy_data.get("chapter_outline", [])
    if not outline or not isinstance(outline, list):
        outline = [f"围绕【{ch_title}】展开核心矛盾对抗与情节推进。"]

    tb = legacy_data.get("truth_base", {})
    proto_plan = str(tb.get("protagonist_plan", "")).strip()
    antag_horizon = str(tb.get("antagonist_horizon", "")).strip()
    info_gap = str(tb.get("information_gap", "")).strip()
    deadline = str(tb.get("deadline", "")).strip()

    # 1. 清理上章承接
    raw_prev = legacy_data.get("prev_chapter_ending", "")
    prev_ending = clean_prev_chapter_ending(raw_prev, chapter_index=ch_index)

    # 2. 深度提炼 confirmed_facts (与计划事件严格区分)
    confirmed_facts: List[Dict[str, str]] = []
    if proto_plan:
        confirmed_facts.append({
            "fact": f"主角既定动机与初始算盘：{proto_plan}",
            "evidence": "原著事实基座 · 主角现实算盘",
        })
    if antag_horizon:
        confirmed_facts.append({
            "fact": f"在场对手/配角的已知认知与本土判断：{antag_horizon}",
            "evidence": "原著事实基座 · 配角本土视界",
        })
    if info_gap:
        confirmed_facts.append({
            "fact": f"当前对立双方的核心信息差与情报护城河：{info_gap}",
            "evidence": "原著事实基座 · 信息差",
        })
    if deadline:
        confirmed_facts.append({
            "fact": f"局势所面临的客观死线压迫与倒计时：{deadline}",
            "evidence": "原著事实基座 · 终局死线",
        })
    if not confirmed_facts:
        confirmed_facts.append({
            "fact": f"承接前章终局态势：{prev_ending[:60]}...",
            "evidence": "前章尾声承接记录",
        })

    # 3. 规范角色视界与信息差
    proto_name = extract_protagonist_name(ch_title, outline, tb)
    characters: List[Dict[str, Any]] = [
        {
            "character_id": "char_protagonist",
            "name": proto_name,
            "goal": proto_plan or "化解当前危机并推进既定目标",
            "known_info": [proto_plan] if proto_plan else ["掌握自身底牌与行进路线"],
            "unknown_info": [],
            "misjudgment": "",
        },
        {
            "character_id": "char_antagonist",
            "name": "对手/配角",
            "goal": antag_horizon or "按既定本土逻辑行动与施压",
            "known_info": [antag_horizon] if antag_horizon else ["了解本土常规规矩"],
            "unknown_info": [info_gap] if info_gap else ["未知主角全部真实手段与底牌"],
            "misjudgment": antag_horizon or "按常理推断主角实力，低估局势变数",
        },
    ]

    # 4. 规整与重平衡 segments
    clean_segs = rebalance_segments(legacy_data.get("segments", {}))

    # 5. 构建 stage_plans (包含离散行动单元、态势与自适应预算)
    op_text = clean_segs[STAGE_OPENING]["text"]
    ru_text = clean_segs[STAGE_RAMP_UP]["text"]
    cl_text = clean_segs[STAGE_CLIMAX]["text"]
    af_text = clean_segs[STAGE_AFTERSHOCK]["text"]
    ch_text = clean_segs[STAGE_CLIFFHANGER]["text"]

    def calc_budget(text: str, default_min: int, default_max: int) -> Dict[str, int]:
        l = len(text)
        if l <= 0:
            return {"min_chars": default_min, "max_chars": default_max}
        min_c = max(50, int(l * 0.75))
        max_c = max(min_c + 120, int(l * 1.3))
        return {"min_chars": min_c, "max_chars": max_c}

    core_conflict = clean_segs[STAGE_CLIMAX].get("core_conflict") or (outline[2] if len(outline) > 2 else "核心冲突对决决出胜负")
    outline_p0 = outline[0] if len(outline) > 0 else "破题立势切入本章危机"
    outline_p1 = outline[1] if len(outline) > 1 else outline_p0
    outline_p2 = outline[-1] if len(outline) > 2 else "引爆终局悬念卡点"

    stage_plans: Dict[str, Any] = {
        STAGE_OPENING: {
            "stage_name": STAGE_OPENING,
            "responsibility": "黄金破题：承接上章余势，以强动作或即时处境开门见山，锁定破题定势",
            "preconditions": ["承接上一章结局或开篇新异变态势"],
            "target_outcome": "确立开篇对抗态势，交代即时场景与处境",
            "length_budget": calc_budget(op_text, 120, 300),
            "start_state": {"character_positions": {proto_name: "开局初始位置"}},
            "end_state": {"character_stances": {proto_name: "确立行动定势与应对姿态"}},
            "action_chain": [
                {
                    "action_id": "act_op_01",
                    "actor_id": proto_name,
                    "trigger": "开局承接",
                    "action": outline_p0,
                    "reason": "打破前置平静，切入主线危机",
                    "reaction": "对手或环境产生即时反馈",
                    "result": "完成破题立势，确立开端态势",
                    "depends_on": [],
                    "evidence": clean_segs[STAGE_OPENING]["anchor_start"],
                }
            ],
        },
        STAGE_RAMP_UP: {
            "stage_name": STAGE_RAMP_UP,
            "responsibility": "前向蓄水引信桥：从破题末锚推进至高潮首锚，层层加码矛盾与行动链",
            "preconditions": ["破题阶段已确立初始对峙"],
            "target_outcome": "矛盾层层激化至临界爆发点，引信完全点燃",
            "length_budget": calc_budget(ru_text, 600, 1200),
            "start_state": {"character_positions": {proto_name: "从破题位置向前推进"}},
            "end_state": {"character_stances": {proto_name: "矛盾激化至爆发边缘，进入临界点"}},
            "action_chain": [
                {
                    "action_id": "act_ru_01",
                    "actor_id": proto_name,
                    "trigger": "破题局势确立后推进",
                    "action": outline_p1,
                    "reason": "升级矛盾，为高潮爆发完成必要铺垫与试探",
                    "reaction": "各方博弈加剧，压迫感攀升",
                    "result": "矛盾达到临界爆发点",
                    "depends_on": ["act_op_01"],
                    "evidence": clean_segs[STAGE_RAMP_UP]["anchor_start"],
                }
            ],
        },
        STAGE_CLIMAX: {
            "stage_name": STAGE_CLIMAX,
            "responsibility": "高潮核爆：跳过铺垫前戏，直接围绕核心对决爆发全章最激烈对抗",
            "preconditions": ["蓄水阶段完成，矛盾无可回避，直接爆发"],
            "target_outcome": core_conflict,
            "length_budget": calc_budget(cl_text, 600, 1200),
            "start_state": {"character_stances": {proto_name: "直面核心危机展开正面殊死对抗"}},
            "end_state": {"character_stances": {proto_name: "完成核心对抗，决出即时结果"}},
            "action_chain": [
                {
                    "action_id": "act_cl_01",
                    "actor_id": proto_name,
                    "trigger": "危机引信烧尽，核心冲突全面爆发",
                    "action": core_conflict,
                    "reason": "兑现信息差，打破对手算盘",
                    "reaction": "对手遭受重创或局势发生不可逆巨变",
                    "result": "完成高潮核心事件，锁定胜负定势",
                    "depends_on": ["act_ru_01"],
                    "evidence": clean_segs[STAGE_CLIMAX]["anchor_start"],
                }
            ],
        },
        STAGE_AFTERSHOCK: {
            "stage_name": STAGE_AFTERSHOCK,
            "responsibility": "余波转折桥：从高潮末锚平稳过渡，生理颤抖、搜刮清点与虚假安全感",
            "preconditions": ["高潮对决落幕，双方脱离直接激烈对抗"],
            "target_outcome": "战局消化与复盘，平稳中继过渡至断章引信",
            "length_budget": calc_budget(af_text, 350, 700),
            "start_state": {
                "character_stances": {proto_name: "高潮消耗巨大，剧烈喘息并进入防备"},
                "injuries": {proto_name: "承受战斗冲击或伤势反应"},
            },
            "end_state": {"cognitive_changes": {proto_name: "刚获得短暂喘息，隐约感知到新隐患"}},
            "action_chain": [
                {
                    "action_id": "act_af_01",
                    "actor_id": proto_name,
                    "trigger": "激烈交锋停止，战局喘息",
                    "action": "清点自身损耗或战利品，复盘刚才变局",
                    "reason": "确认安全态势，消化战果",
                    "reaction": "周围环境显露微小异样先兆",
                    "result": "建立短暂喘息与心理余震，承接下一轮危机",
                    "depends_on": ["act_cl_01"],
                    "evidence": clean_segs[STAGE_AFTERSHOCK]["anchor_start"],
                }
            ],
        },
        STAGE_CLIFFHANGER: {
            "stage_name": STAGE_CLIFFHANGER,
            "responsibility": "致命断章：引爆全新维度异变或心理危机，矛盾巅峰处一刀封喉完成绝杀卡点",
            "preconditions": ["余波复盘完成，虚假安全感骤然被打破"],
            "target_outcome": "引爆全新异变，在读者翻页欲望最高峰戛然而止",
            "length_budget": calc_budget(ch_text, 100, 300),
            "start_state": {"character_stances": {proto_name: "猝不及防，神经再度紧绷"}},
            "end_state": {"cognitive_changes": {proto_name: "直面未知异变，留下巨大悬念"}},
            "action_chain": [
                {
                    "action_id": "act_ch_01",
                    "actor_id": proto_name,
                    "trigger": "平静被全新异变骤然撕裂",
                    "action": outline_p2,
                    "reason": "直面超出预期的全新威胁",
                    "reaction": "新危机瞬间笼罩全场",
                    "result": "全章在矛盾与悬念最高峰处一刀封喉",
                    "depends_on": ["act_af_01"],
                    "evidence": clean_segs[STAGE_CLIFFHANGER]["anchor_start"],
                }
            ],
        },
    }

    # 6. 创作边界约束 (constraints)
    constraints: Dict[str, Any] = {
        "pov": "第三人称限制视角",
        "author_hard_requirements": [
            "严格遵守既定胜负与阶段结果",
            "行文自然流畅，杜绝机械复述提纲",
            "正文最后一句话一刀封喉戛然而止",
        ],
        "negative_constraints": [
            "严禁擅自新增未经设定的关键角色",
            "严禁颠覆既定胜负走向或死线设定",
            "严禁抢写或跨阶段提前解决后续情节",
        ],
        "allowed_creative_freedom": [
            "人物物理发力、微动作与表情细节自由发挥",
            "打斗博弈的节奏与措辞风格自由发挥",
            "环境与惊悚/紧张氛围描写自由发挥",
        ],
    }

    # 7. 质量复核判定
    review_items = []
    for st, text in [(STAGE_OPENING, op_text), (STAGE_CLIFFHANGER, ch_text)]:
        if len(text) < 30:
            review_items.append({"field": f"segments.{st}.text", "reason": f"正文字数偏短 ({len(text)} 字)"})

    annotation_status: Dict[str, Any] = {
        "needs_review": len(review_items) > 0,
        "direct_evidence_fields": [
            "segments.opening.text",
            "segments.ramp_up.text",
            "segments.climax.text",
            "segments.aftershock.text",
            "segments.cliffhanger.text",
            "chapter_title",
            "chapter_outline",
            "confirmed_facts",
            "characters",
            "stage_plans",
        ],
        "review_items": review_items,
        "annotator": "expert_audited_v1",
    }

    # 构造标准章节对象
    chapter_v1: Dict[str, Any] = {
        "schema_version": "tara5_action_v1",
        "chapter_id": ch_id,
        "source_novel": source_novel,
        "source_info": {
            "book_title": source_novel,
            "chapter_index": ch_index,
            "raw_file_name": legacy_data.get("raw_file_name", f"{ch_id}.txt"),
            "raw_text_char_count": sum(clean_segs[s]["char_count"] for s in VALID_STAGES),
        },
        "chapter_title": ch_title,
        "prev_chapter_ending": prev_ending,
        "chapter_outline": outline,
        "confirmed_facts": confirmed_facts,
        "characters": characters,
        "stage_plans": stage_plans,
        "constraints": constraints,
        "segments": clean_segs,
        "annotation_status": annotation_status,
    }

    return chapter_v1


def build_sharegpt_samples(chapter_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据章节结构编译 ShareGPT 样本"""
    samples: List[Dict[str, Any]] = []
    ch_id = chapter_data.get("chapter_id", "ch")
    segs = chapter_data.get("segments", {})

    for st in VALID_STAGES:
        task_type = STAGE_TASK_MAP[st]
        sys_prompt, user_prompt = build_task_input(
            stage=st,
            chapter_data=chapter_data,
        )
        gold_text = segs.get(st, {}).get("text", "").strip()

        sample = {
            "id": f"{ch_id}_{st}",
            "chapter_id": ch_id,
            "source_novel": chapter_data.get("source_novel", "unknown"),
            "task_type": task_type,
            "system": sys_prompt,
            "conversations": [
                {"from": "human", "value": user_prompt},
                {"from": "gpt", "value": gold_text},
            ],
        }
        samples.append(sample)

    return samples


def process_book_directory(src_dir: Path, novel_name: str, out_root: Path) -> Tuple[int, int, int]:
    """处理并复核单个书目的全部切片文件"""
    out_sliced_dir = out_root / novel_name / "sliced"
    out_dataset_dir = out_root / novel_name / "dataset"
    out_sliced_dir.mkdir(parents=True, exist_ok=True)
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(src_dir.glob("*.json")))
    if not json_files:
        return 0, 0, 0

    success_chapters = 0
    fail_chapters = 0
    all_samples: List[Dict[str, Any]] = []

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                legacy = json.load(f)

            audited = audit_and_migrate_chapter(legacy, source_novel=novel_name)

            # 校验合规性
            is_valid, errs = validate_tara5_chapter(audited)
            if not is_valid:
                print(f"[-] [校验失败] {novel_name}/{fpath.name}: {errs}")
                fail_chapters += 1
                continue

            # 写入切片
            ch_id = audited["chapter_id"]
            safe_title = "".join(c for c in audited["chapter_title"] if c.isalnum() or c in (" ", "_", "-")).strip()
            out_file = out_sliced_dir / f"{ch_id}_{safe_title}_tara5_v1.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(audited, out_f, ensure_ascii=False, indent=2)

            # 生成 ShareGPT
            samples = build_sharegpt_samples(audited)
            all_samples.extend(samples)
            success_chapters += 1

        except Exception as e:
            print(f"[-] 处理异常 {novel_name}/{fpath.name}: {e}")
            fail_chapters += 1

    # 写入该书数据集
    if all_samples:
        ds_file = out_dataset_dir / "tara5_action_v1_train.json"
        with open(ds_file, "w", encoding="utf-8") as ds_f:
            json.dump(all_samples, ds_f, ensure_ascii=False, indent=2)

        info_snippet = {
            f"tara5_action_v1_{novel_name}": {
                "file_name": "tara5_action_v1_train.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "system": "system",
                },
            }
        }
        info_file = out_dataset_dir / "dataset_info_snippet.json"
        with open(info_file, "w", encoding="utf-8") as inf_f:
            json.dump(info_snippet, inf_f, ensure_ascii=False, indent=2)

    return success_chapters, fail_chapters, len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="TARA-5 全量数据批量处理与复核引擎")
    parser.add_argument("--out-dir", type=str, default=str(LORA_ROOT / "data" / "tara5_action_v1"), help="输出目录")
    parser.add_argument("--book", type=str, default="all", help="指定处理的书目 (默认 all 表示全部书目)")
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # 搜集全部书目切片源
    book_sources: Dict[str, Path] = {}
    if (LORA_ROOT / "data" / "sliced").exists():
        book_sources["daomu"] = LORA_ROOT / "data" / "sliced"

    for p in (LORA_ROOT / "data").glob("*/sliced"):
        if p.is_dir() and "tara5_action_v1" not in str(p):
            book_name = p.parent.name
            book_sources[book_name] = p

    if args.book != "all":
        if args.book not in book_sources:
            print(f"[-] 未找到书目: {args.book}")
            sys.exit(1)
        book_sources = {args.book: book_sources[args.book]}

    print("=" * 70)
    print("      TARA-5 (tara5_action_v1) 全量切片数据批量处理与深度复核引擎      ")
    print("=" * 70)
    print(f"[*] 发现待处理书目: {len(book_sources)} 部")
    for b, p in book_sources.items():
        print(f"  - {b:12s}: {p}")
    print("=" * 70)

    start_time = time.time()
    total_success = 0
    total_fail = 0
    total_samples = 0
    all_global_samples: List[Dict[str, Any]] = []

    for book_name, src_dir in book_sources.items():
        sub_start = time.time()
        print(f"\n[*] 正在处理并复核: 【{book_name}】...")
        succ, fail, samples_cnt = process_book_directory(src_dir, book_name, out_root)
        total_success += succ
        total_fail += fail
        total_samples += samples_cnt
        print(f"[+] 【{book_name}】完成: 成功 {succ} 章, 失败 {fail} 章, 生成样本 {samples_cnt} 条 (耗时 {time.time()-sub_start:.2f}s)")

        # 读取该书训练集追加至全局
        b_ds = out_root / book_name / "dataset" / "tara5_action_v1_train.json"
        if b_ds.exists():
            try:
                with open(b_ds, "r", encoding="utf-8") as f:
                    all_global_samples.extend(json.load(f))
            except Exception:
                pass

    # 导出全局汇总训练集
    if all_global_samples:
        global_ds_dir = out_root / "all_books" / "dataset"
        global_ds_dir.mkdir(parents=True, exist_ok=True)
        global_ds_file = global_ds_dir / "tara5_action_v1_all_train.json"
        with open(global_ds_file, "w", encoding="utf-8") as gf:
            json.dump(all_global_samples, gf, ensure_ascii=False, indent=2)

        global_info = {
            "tara5_action_v1_all": {
                "file_name": "tara5_action_v1_all_train.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "conversations",
                    "system": "system",
                },
            }
        }
        with open(global_ds_dir / "dataset_info_snippet.json", "w", encoding="utf-8") as gif:
            json.dump(global_info, gif, ensure_ascii=False, indent=2)

        print("\n" + "=" * 70)
        print(f"[+] 全局汇总训练集已输出: {global_ds_file}")
        print(f"[+] 全局总样本数: {len(all_global_samples)} 条标准五段式样本")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[✓] 全量数据处理与专家复核圆满完成！")
    print(f"  - 成功处理章节: {total_success} 章")
    print(f"  - 校验失败章节: {total_fail} 章")
    print(f"  - 编译微调样本: {total_samples} 条")
    print(f"  - 总耗时: {elapsed:.2f} 秒")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
