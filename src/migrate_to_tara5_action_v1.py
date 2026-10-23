#!/usr/bin/env python3
"""
TARA-5 历史切片向 tara5_action_v1 协议迁移工具 (Data Migration Tool)
版本：tara5_action_v1

核心原则：
1. 严禁原地覆盖已有数据与切片！所有新格式数据输出至 `data/tara5_action_v1/{novel}/...`；
2. 完整填充 confirmed_facts, characters, stage_plans, constraints, segments 等字段；
3. 阶段名严格归一化为 opening, ramp_up, climax, aftershock, cliffhanger；
4. 明确标注 needs_review: true 以及 review_items，诚实记录规则迁移来源；
5. 调用 schema_validator 保证 100% 格式合规；
6. 支持同步导出基于 prompt_builder 的 ShareGPT 微调数据集。
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

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


def migrate_single_chapter(legacy_data: Dict[str, Any], source_novel: str = "default_novel") -> Dict[str, Any]:
    """
    将旧版切片数据转换为符合 tara5_action_v1 的标准化结构字典
    """
    ch_id = legacy_data.get("chapter_id") or f"ch_{legacy_data.get('chapter_index', 1):04d}"
    ch_title = legacy_data.get("chapter_title", "未命名章节")
    prev_ending = legacy_data.get("prev_chapter_ending", "（本篇开门见山）").strip()
    outline = legacy_data.get("chapter_outline", [])
    if not outline or not isinstance(outline, list):
        outline = [f"围绕【{ch_title}】展开核心矛盾对抗与情节推进。"]

    tb = legacy_data.get("truth_base", {})
    proto_plan = tb.get("protagonist_plan", "").strip()
    antag_horizon = tb.get("antagonist_horizon", "").strip()
    info_gap = tb.get("information_gap", "").strip()
    deadline = tb.get("deadline", "").strip()

    # 1. 提炼 confirmed_facts (与计划事件严格区分)
    confirmed_facts: List[Dict[str, str]] = []
    if proto_plan:
        confirmed_facts.append({
            "fact": f"主角现实算盘与初始动机：{proto_plan}",
            "evidence": "历史切片事实基座 · 主角算盘",
        })
    if antag_horizon:
        confirmed_facts.append({
            "fact": f"配角/对手本土视界与判断基准：{antag_horizon}",
            "evidence": "历史切片事实基座 · 配角视界",
        })
    if info_gap:
        confirmed_facts.append({
            "fact": f"冲突双方的核心信息差护城河：{info_gap}",
            "evidence": "历史切片事实基座 · 信息差",
        })
    if deadline:
        confirmed_facts.append({
            "fact": f"局势所面临的终局死线与倒计时：{deadline}",
            "evidence": "历史切片事实基座 · 终局死线",
        })
    if not confirmed_facts:
        confirmed_facts.append({
            "fact": f"承接前情局势：{prev_ending[:60]}...",
            "evidence": "前章尾声承接记录",
        })

    # 2. 提炼 characters
    characters: List[Dict[str, Any]] = [
        {
            "character_id": "protagonist",
            "name": "主角",
            "goal": proto_plan or "化解当前危机并推进既定目标",
            "known_info": [info_gap] if info_gap else ["了解自身手段与真实意图"],
            "unknown_info": [],
            "misjudgment": "",
        },
        {
            "character_id": "antagonist",
            "name": "配角/对手",
            "goal": antag_horizon or "按既定本土逻辑行动与施压",
            "known_info": [],
            "unknown_info": [info_gap] if info_gap else ["未知主角全部底牌"],
            "misjudgment": antag_horizon or "低估变局复杂性",
        },
    ]

    # 3. 规整 segments (兼容 rampup 命名，并计算 char_count)
    raw_segs = legacy_data.get("segments", {})
    op_seg = raw_segs.get("opening", {})
    ru_seg = raw_segs.get("ramp_up") or raw_segs.get("rampup", {})
    cl_seg = raw_segs.get("climax", {})
    af_seg = raw_segs.get("aftershock", {})
    ch_seg = raw_segs.get("cliffhanger", {})

    segments: Dict[str, Any] = {}
    for st_name, s_data in [
        (STAGE_OPENING, op_seg),
        (STAGE_RAMP_UP, ru_seg),
        (STAGE_CLIMAX, cl_seg),
        (STAGE_AFTERSHOCK, af_seg),
        (STAGE_CLIFFHANGER, ch_seg),
    ]:
        s_text = s_data.get("text", "").strip()
        s_start = s_data.get("anchor_start", "").strip()
        s_end = s_data.get("anchor_end", "").strip()

        # 兜底：若旧数据无锚点，安全取首尾第一句
        if not s_start and s_text:
            first_line = s_text.split("\n")[0].strip()
            s_start = first_line[:min(len(first_line), 50)]
        if not s_end and s_text:
            last_line = [l.strip() for l in s_text.split("\n") if l.strip()][-1]
            s_end = last_line[-min(len(last_line), 50):]

        segments[st_name] = {
            "anchor_start": s_start,
            "anchor_end": s_end,
            "text": s_text,
            "char_count": len(s_text),
        }

    # 4. 构建 stage_plans (含 action_chain、起止 state、预算)
    core_conflict = cl_seg.get("core_conflict", "").strip()
    outline_p0 = outline[0] if len(outline) > 0 else "开门见山打破平静"
    outline_p1 = outline[1] if len(outline) > 1 else (outline[0] if outline else "升级矛盾")
    outline_p2 = outline[2] if len(outline) > 2 else (core_conflict or "高潮爆发")

    stage_plans: Dict[str, Any] = {
        STAGE_OPENING: {
            "stage_name": STAGE_OPENING,
            "responsibility": "黄金破题：承接上章余势，以强动作或即时处境开门见山，锁定破题定势",
            "preconditions": ["承接上一章断章态势或开篇新异变"],
            "target_outcome": "完成破题立势，展现主角即时处境",
            "length_budget": {"min_chars": 150, "max_chars": 350},
            "start_state": {"character_positions": {"protagonist": "初始场景位置"}},
            "end_state": {"character_stances": {"protagonist": "确立行动定势与应对姿态"}},
            "action_chain": [
                {
                    "action_id": "act_op_01",
                    "actor_id": "protagonist",
                    "trigger": "开局承接",
                    "action": outline_p0,
                    "reason": "打破前置平静，切入主线危机",
                    "reaction": "对手或环境出现即时异变反馈",
                    "result": "确立本章开局对抗态势",
                    "depends_on": [],
                    "evidence": segments[STAGE_OPENING]["anchor_start"],
                }
            ],
        },
        STAGE_RAMP_UP: {
            "stage_name": STAGE_RAMP_UP,
            "responsibility": "蓄水引信桥：从破题末锚推进至高潮首锚，层层加码矛盾与行动链",
            "preconditions": ["破题阶段已确立初始对峙"],
            "target_outcome": "矛盾层层激化至临界爆发点，引信完全点燃",
            "length_budget": {"min_chars": 600, "max_chars": 1200},
            "start_state": {"character_positions": {"protagonist": "从破题位置向前推进"}},
            "end_state": {"character_stances": {"protagonist": "危机逼近临界点，做好决战爆发准备"}},
            "action_chain": [
                {
                    "action_id": "act_ru_01",
                    "actor_id": "protagonist",
                    "trigger": "破题局势确立后推进",
                    "action": outline_p1,
                    "reason": "升级矛盾，为高潮爆发完成必要铺垫与试探",
                    "reaction": "各方博弈加剧，压迫感攀升",
                    "result": "矛盾达到临界爆发点",
                    "depends_on": ["act_op_01"],
                    "evidence": segments[STAGE_RAMP_UP]["anchor_start"],
                }
            ],
        },
        STAGE_CLIMAX: {
            "stage_name": STAGE_CLIMAX,
            "responsibility": "高潮核爆：跳过铺垫前戏，直接围绕核心对决爆发全章最激烈对抗",
            "preconditions": ["蓄水阶段完成，矛盾无可回避，直接爆发"],
            "target_outcome": core_conflict or "高潮对决决出胜负或即时战果",
            "length_budget": {"min_chars": 600, "max_chars": 1200},
            "start_state": {"character_stances": {"protagonist": "直面核心危机展开肉搏/交锋"}},
            "end_state": {"character_stances": {"protagonist": "完成核心对抗，决出即时结果"}},
            "action_chain": [
                {
                    "action_id": "act_cl_01",
                    "actor_id": "protagonist",
                    "trigger": "危机引信烧尽，核心冲突全面爆发",
                    "action": core_conflict or "实施绝地反击或正面殊死对抗",
                    "reason": "兑现信息差，打破对手算盘",
                    "reaction": "对手遭受重创或局势发生不可逆巨变",
                    "result": "完成高潮核心事件，锁定胜负定势",
                    "depends_on": ["act_ru_01"],
                    "evidence": segments[STAGE_CLIMAX]["anchor_start"],
                }
            ],
        },
        STAGE_AFTERSHOCK: {
            "stage_name": STAGE_AFTERSHOCK,
            "responsibility": "余波转折桥：从高潮末锚平稳过渡，生理颤抖、搜刮清点与虚假安全感",
            "preconditions": ["高潮对决落幕，双方脱离直接激烈对抗"],
            "target_outcome": "战局消化与复盘，平稳中继过渡至断章引信",
            "length_budget": {"min_chars": 350, "max_chars": 700},
            "start_state": {
                "character_stances": {"protagonist": "高潮消耗巨大，剧烈喘息并进入防备"},
                "injuries": {"protagonist": "承受战斗冲击或伤势反应"},
            },
            "end_state": {"cognitive_changes": {"protagonist": "刚获得短暂喘息，隐约感知到新隐患"}},
            "action_chain": [
                {
                    "action_id": "act_af_01",
                    "actor_id": "protagonist",
                    "trigger": "激烈交锋停止，战局喘息",
                    "action": "清点自身损耗或战利品，复盘刚才变局",
                    "reason": "确认安全态势，消化战果",
                    "reaction": "周围环境显露微小异样先兆",
                    "result": "建立短暂喘息与心理余震，承接下一轮危机",
                    "depends_on": ["act_cl_01"],
                    "evidence": segments[STAGE_AFTERSHOCK]["anchor_start"],
                }
            ],
        },
        STAGE_CLIFFHANGER: {
            "stage_name": STAGE_CLIFFHANGER,
            "responsibility": "致命断章：引爆全新维度异变或心理危机，矛盾巅峰处一刀封喉完成绝杀卡点",
            "preconditions": ["余波复盘完成，虚假安全感骤然被打破"],
            "target_outcome": "引爆全新异变，在读者翻页欲望最高峰戛然而止",
            "length_budget": {"min_chars": 150, "max_chars": 350},
            "start_state": {"character_stances": {"protagonist": "猝不及防，神经再度紧绷"}},
            "end_state": {"cognitive_changes": {"protagonist": "直面未知异变，留下巨大悬念"}},
            "action_chain": [
                {
                    "action_id": "act_ch_01",
                    "actor_id": "protagonist",
                    "trigger": "平静被全新异变骤然撕裂",
                    "action": outline_p2 or "目睹或遭遇突发未知新危机",
                    "reason": "直面超出预期的全新威胁",
                    "reaction": "新危机瞬间笼罩全场",
                    "result": "全章在矛盾与悬念最高峰处一刀封喉",
                    "depends_on": ["act_af_01"],
                    "evidence": segments[STAGE_CLIFFHANGER]["anchor_start"],
                }
            ],
        },
    }

    # 5. 创作边界约束 (constraints)
    constraints: Dict[str, Any] = {
        "pov": "第三人称限制视角",
        "author_hard_requirements": [
            "严格遵守既定胜负与阶段结果",
            "行文自然流畅，杜绝机械复述提纲",
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

    # 6. 标注审核状态 (诚实标注待复核项)
    annotation_status: Dict[str, Any] = {
        "needs_review": True,
        "direct_evidence_fields": [
            "segments.opening.text",
            "segments.ramp_up.text",
            "segments.climax.text",
            "segments.aftershock.text",
            "segments.cliffhanger.text",
            "chapter_title",
            "chapter_outline",
        ],
        "review_items": [
            {
                "field": "stage_plans.*.action_chain",
                "reason": "细粒度行动单元由旧版章纲及事实基座规则迁移生成，需人工核对细粒度动作依赖与出处",
            },
            {
                "field": "confirmed_facts",
                "reason": "已确认事实由旧版 truth_base 转换，需复核证据出处",
            },
        ],
        "annotator": "rule_migrated_v1",
    }

    chapter_v1: Dict[str, Any] = {
        "schema_version": "tara5_action_v1",
        "chapter_id": ch_id,
        "source_novel": source_novel,
        "source_info": {
            "book_title": source_novel,
            "chapter_index": legacy_data.get("chapter_index", 1),
            "raw_file_name": legacy_data.get("raw_file_name", f"{ch_id}.txt"),
        },
        "chapter_title": ch_title,
        "prev_chapter_ending": prev_ending,
        "chapter_outline": outline,
        "confirmed_facts": confirmed_facts,
        "characters": characters,
        "stage_plans": stage_plans,
        "constraints": constraints,
        "segments": segments,
        "annotation_status": annotation_status,
    }

    return chapter_v1


def build_sharegpt_samples_from_v1(chapter_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    根据 tara5_action_v1 章节结构，通过 prompt_builder 编译纯净的 ShareGPT 微调样本
    GPT 回复仅为纯文学正文，彻底消除任何合成 <think> 标签污染！
    """
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


def main():
    parser = argparse.ArgumentParser(description="TARA-5 历史切片向 tara5_action_v1 迁移工具")
    parser.add_argument("--src-dir", type=str, default=str(LORA_ROOT / "data" / "sliced"), help="旧版切片输入目录")
    parser.add_argument("--novel", type=str, default="daomu", help="所属书目名称或别名")
    parser.add_argument("--out-dir", type=str, default="", help="新协议输出根目录 (默认 data/tara5_action_v1/{novel})")
    parser.add_argument("--limit", type=int, default=0, help="处理最大章节数 (0 表示全部)")
    parser.add_argument("--export-dataset", action="store_true", default=True, help="是否同步导出 ShareGPT 微调训练集")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    novel = args.novel
    if args.out_dir:
        out_root = Path(args.out_dir)
    else:
        out_root = LORA_ROOT / "data" / "tara5_action_v1" / novel

    out_sliced_dir = out_root / "sliced"
    out_dataset_dir = out_root / "dataset"
    out_sliced_dir.mkdir(parents=True, exist_ok=True)
    out_dataset_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(src_dir.glob("*.json")))
    if not json_files:
        print(f"[-] 在 {src_dir} 未找到 JSON 文件！")
        sys.exit(1)

    if args.limit > 0:
        json_files = json_files[:args.limit]

    print("=" * 65)
    print(f"[*] 开始执行 TARA-5 数据迁移 (-> tara5_action_v1)")
    print(f"[*] 输入目录: {src_dir}")
    print(f"[*] 书目标识: {novel}")
    print(f"[*] 目标切片目录: {out_sliced_dir}")
    print(f"[*] 目标数据集目录: {out_dataset_dir}")
    print(f"[*] 待处理文件数: {len(json_files)}")
    print("=" * 65)

    all_sharegpt_samples: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    for fpath in json_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                legacy_data = json.load(f)
            migrated_ch = migrate_single_chapter(legacy_data, source_novel=novel)

            # 严格 Schema 校验
            is_valid, errors = validate_tara5_chapter(migrated_ch)
            if not is_valid:
                print(f"[-] [Schema校验失败] {fpath.name}: {errors}")
                fail_count += 1
                continue

            # 写入版本化切片文件
            ch_id = migrated_ch["chapter_id"]
            safe_title = "".join(c for c in migrated_ch["chapter_title"] if c.isalnum() or c in (" ", "_", "-")).strip()
            out_file = out_sliced_dir / f"{ch_id}_{safe_title}_tara5_v1.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(migrated_ch, out_f, ensure_ascii=False, indent=2)

            # 编译 ShareGPT 微调样本
            if args.export_dataset:
                samples = build_sharegpt_samples_from_v1(migrated_ch)
                all_sharegpt_samples.extend(samples)

            success_count += 1
            if success_count % 10 == 0:
                print(f"[+] 已安全迁移 {success_count}/{len(json_files)} 章节...")

        except Exception as e:
            print(f"[-] 迁移出错 {fpath.name}: {e}")
            fail_count += 1

    if args.export_dataset and all_sharegpt_samples:
        ds_file = out_dataset_dir / "tara5_action_v1_train.json"
        with open(ds_file, "w", encoding="utf-8") as ds_f:
            json.dump(all_sharegpt_samples, ds_f, ensure_ascii=False, indent=2)

        info_snippet = {
            f"tara5_action_v1_{novel}": {
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

        print(f"[+] ShareGPT 数据集已保存至: {ds_file} (共 {len(all_sharegpt_samples)} 条五段样本)")

    print("=" * 65)
    print(f"[✓] 迁移执行完成！成功: {success_count}, 失败: {fail_count}")
    print(f"[*] 注意：所有迁移生成样本已自动标记 'needs_review: true'，需按需人工复核。")
    print("=" * 65)


if __name__ == "__main__":
    main()
