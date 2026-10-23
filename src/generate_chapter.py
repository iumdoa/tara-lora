#!/usr/bin/env python3
"""
TARA-5 (Five-Stage Anchor & Style Contrast) 全章反向装配生成器
协议版本：tara5_action_v1

执行五段式反向装配全闭环流程：
1. 【高潮切片 (tara5_climax)】推演爆发点，锁定 Anchor_3 与 Anchor_4
2. 【断章切片 (tara5_cliffhanger)】推演全章绝杀卡点，锁定 Anchor_5 与 Anchor_6
3. 【黄金破题 (tara5_opening)】承接上一章结尾，开门见山，锁定 Anchor_1 与 Anchor_2
4. 【蓄水引信桥 (tara5_ramp_up)】从 Anchor_2 推进，强制咬入 Anchor_3
5. 【余波转折桥 (tara5_aftershock)】从 Anchor_4 复盘喘息，强制咬入 Anchor_5
6. 【无缝装配 (Safe Assembly)】按顺序装配 Opening + Ramp_up + Climax + Aftershock + Cliffhanger

统一复用 prompt_builder.py 与 assembly.py，杜绝训练与推理模板漂移！
"""

import os
import re
import sys
import json
import argparse
from pathlib import Path

if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))

from src.prompt_builder import (
    STAGE_OPENING,
    STAGE_RAMP_UP,
    STAGE_CLIMAX,
    STAGE_AFTERSHOCK,
    STAGE_CLIFFHANGER,
    build_task_input,
)
from src.assembly import safe_assemble_chapter, strip_exact_anchor_prefix


def run_generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_think_and_text(output_str: str) -> tuple[str, str]:
    """解析并分离历史遗留 <think> 思考链与正文（新版本以纯正文为主）"""
    think_match = re.search(r"<think>(.*?)</think>", output_str, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        body_text = output_str[think_match.end():].strip()
        return think_content, body_text
    return "", output_str.strip()


def extract_anchors_from_output(think_str: str, body_text: str) -> tuple[str, str]:
    """
    自适应提取首尾锚点：
    1. 优先从正文首末有效句子提取；
    2. 若正文存在明确段落，取第一行/句作为起点，最后一行/句作为终点；
    3. 兼容检查历史 think 中的显式锚点标注。
    """
    start_anchor = ""
    end_anchor = ""

    if think_str:
        m_start = re.search(r"(?:起点铁锚|破题首句|引爆句)[^\n:：]*[:：]\s*[“\"「](.*?)[”\"」]", think_str)
        m_end = re.search(r"(?:终点铁锚|破题定势铁锚|卡点末句|高潮末句)[^\n:：]*[:：]\s*[“\"「](.*?)[”\"」]", think_str)
        if m_start:
            start_anchor = m_start.group(1).strip()
        if m_end:
            end_anchor = m_end.group(1).strip()

    lines = [l.strip() for l in body_text.split("\n") if l.strip() and not l.strip().startswith("【")]
    if not start_anchor and lines:
        first = lines[0]
        # 若第一句过长，截取第一句结束标点处
        sent_end = re.search(r"[。！？!?]", first)
        if sent_end and sent_end.end() >= 8:
            start_anchor = first[:sent_end.end()]
        else:
            start_anchor = first[:min(len(first), 80)]

    if not end_anchor and lines:
        last = lines[-1]
        sent_parts = re.split(r"(?<=[。！？!?])", last)
        valid_sents = [s.strip() for s in sent_parts if s.strip()]
        if valid_sents:
            end_anchor = valid_sents[-1]
        else:
            end_anchor = last[-min(len(last), 80):]

    return start_anchor, end_anchor


def build_default_test_chapter() -> dict:
    """构建符合 tara5_action_v1 规范的默认测试章节定义"""
    return {
        "schema_version": "tara5_action_v1",
        "chapter_id": "ch_test_market",
        "source_novel": "webnovel_universe",
        "chapter_title": "第七十八章 灵气黑市的规矩",
        "prev_chapter_ending": "卷帘门哗啦一声被狂暴的气劲撕碎，刺耳的警报响彻长巷。陈羽一把拽住吓呆的胖子闪进阴影，身后七八名黑衣修士手持附魔符文弩，杀气腾腾地封死了两头巷口！",
        "chapter_outline": [
            "陈羽凭借敏锐感知识破黑市伏击圈，借着巷道杂物与反光镜制造假身引诱弩箭齐射脱困",
            "黑市暗堂执事带队步步紧逼，以势压人逼交灵石配方，陈羽假意周旋实则暗中调动地脉灵气",
            "谈判破裂瞬间，陈羽果断引爆杂货铺电表箱的灵气逆流，借狂暴冲击波近身反杀前排修士撕开生门",
            "两人夺路冲进古玩暗街，身后突然传来一声令人头皮发麻的幽沉古钟鸣响，整条街的黑影齐齐转头睁开了幽绿的竖瞳，全新杀机引爆全章卡点"
        ],
        "confirmed_facts": [
            {"fact": "陈羽深谙黑市阵法灵气节点在杂货铺后墙电表箱内", "evidence": "前文第72章伏笔探索"},
            {"fact": "黑市符文弩二次装填具有约15秒死线间隙", "evidence": "修士法器性能已知设定"},
            {"fact": "在场黑市执事误判陈羽只是毫无根基的年轻散修肥羊", "evidence": "执事已知本土视界"}
        ],
        "characters": [
            {
                "character_id": "protagonist",
                "name": "陈羽",
                "goal": "在治安巡逻队赶到前全身而退，绝不暴露真实底牌",
                "known_info": ["已知后墙电表箱节点弱点"],
                "unknown_info": [],
                "misjudgment": ""
            },
            {
                "character_id": "antagonist",
                "name": "黑市执事",
                "goal": "强行扣押并搜魂夺取灵石配方",
                "known_info": ["手下持有附魔弩箭且巷道已被封锁"],
                "unknown_info": ["不知陈羽拥有引爆地脉灵气的隐秘秘法"],
                "misjudgment": "认定翻手即可镇压"
            }
        ],
        "stage_plans": {
            STAGE_OPENING: {
                "stage_name": STAGE_OPENING,
                "responsibility": "承接上章弩箭封路险境，开篇强动作突围破局，确立开局态势",
                "preconditions": ["承接被黑衣修士堵塞在巷道死角的危局"],
                "target_outcome": "识破伏击并诱使第一波弩箭射偏，建立破题定势",
                "length_budget": {"min_chars": 180, "max_chars": 300},
                "start_state": {"character_positions": {"protagonist": "巷道垃圾箱阴影后"}},
                "end_state": {"character_stances": {"protagonist": "紧靠砖墙，冷静等待反击时机"}},
                "action_chain": [
                    {
                        "action_id": "act_op_01",
                        "actor_id": "protagonist",
                        "trigger": "弩箭上膛瞄准",
                        "action": "陈羽脚踢反光镜与废弃铁桶制造声东击西假身",
                        "reason": "诱骗第一轮符文齐射",
                        "reaction": "对方弩箭齐发落空，打碎砖墙",
                        "result": "争取到宝贵的装填间隙并闪入中段死角",
                        "depends_on": [],
                        "evidence": "破题动作推演"
                    }
                ]
            },
            STAGE_RAMP_UP: {
                "stage_name": STAGE_RAMP_UP,
                "responsibility": "前向蓄水引信桥：执事压上，假意周旋与暗中布局，将矛盾蓄满至临界点",
                "preconditions": ["开局假身争取到片刻对峙时间"],
                "target_outcome": "层层加码压迫，引导执事走到电表箱杀伤半径内",
                "length_budget": {"min_chars": 600, "max_chars": 900},
                "start_state": {"character_positions": {"protagonist": "杂货铺侧门掩体"}},
                "end_state": {"character_stances": {"protagonist": "指尖凝聚微型引爆灵丝，杀意已决"}},
                "action_chain": [
                    {
                        "action_id": "act_ru_01",
                        "actor_id": "protagonist",
                        "trigger": "黑市执事步步紧逼逼交配方",
                        "action": "陈羽故作惊慌讨价还价，暗地以鞋底导引地脉灵气渗入电表箱",
                        "reason": "将敌方主力诱入灵气逆流覆盖范围",
                        "reaction": "执事冷笑逼近，自以为稳操胜券",
                        "result": "灵气逆流蓄能达到爆炸临界点",
                        "depends_on": ["act_op_01"],
                        "evidence": "蓄水布局推演"
                    }
                ]
            },
            STAGE_CLIMAX: {
                "stage_name": STAGE_CLIMAX,
                "responsibility": "高潮核爆：跳过铺垫前戏，果断引爆灵气逆流，借冲击波近身反杀破开生门",
                "preconditions": ["执事进入杀伤核心区，灵气逆流完成导引"],
                "target_outcome": "引爆节点，近身重创前排撕开突围缺口",
                "length_budget": {"min_chars": 600, "max_chars": 900},
                "start_state": {"character_stances": {"protagonist": "瞬间暴起发难"}},
                "end_state": {"character_stances": {"protagonist": "斩落敌首，夺路狂奔冲出生门"}},
                "action_chain": [
                    {
                        "action_id": "act_cl_01",
                        "actor_id": "protagonist",
                        "trigger": "谈判破裂执事挥手示意格杀",
                        "action": "陈羽果断引爆杂货铺电表箱，暴烈气浪瞬间掀翻中路，陈羽借烟尘飞身突刺",
                        "reason": "信息差绝杀，打乱合围",
                        "reaction": "黑衣修士阵型瞬间四分五裂，执事被冲击波掀翻咳血",
                        "result": "撕开包围网生门，确立绝对突围战果",
                        "depends_on": ["act_ru_01"],
                        "evidence": "高潮对决推演"
                    }
                ]
            },
            STAGE_AFTERSHOCK: {
                "stage_name": STAGE_AFTERSHOCK,
                "responsibility": "余波转折桥：高潮平息，战后脱力喘息，清点损耗，过渡至暗街",
                "preconditions": ["成功突破封锁，脱离直接近身格斗"],
                "target_outcome": "消化战况，制造片刻安全感假象，顺畅咬入暗街危机引信",
                "length_budget": {"min_chars": 400, "max_chars": 600},
                "start_state": {"character_stances": {"protagonist": "肋下微痛，拉着胖子狂奔喘息"}},
                "end_state": {"cognitive_changes": {"protagonist": "察觉街道死寂过于反常"}},
                "action_chain": [
                    {
                        "action_id": "act_af_01",
                        "actor_id": "protagonist",
                        "trigger": "冲入古玩暗街脱离身后追兵视线",
                        "action": "隐入古玩店檐下大口喘息，检查随身灵石与伤势",
                        "reason": "确认脱险并调整内息",
                        "reaction": "胖子双腿发软瘫倒，长街陷入诡异死寂",
                        "result": "获得虚假喘息，周围暗生异变",
                        "depends_on": ["act_cl_01"],
                        "evidence": "余波复盘推演"
                    }
                ]
            },
            STAGE_CLIFFHANGER: {
                "stage_name": STAGE_CLIFFHANGER,
                "responsibility": "致命断章：引爆全新未知恐怖异变，矛盾巅峰处一刀封喉",
                "preconditions": ["余波平静中暗流涌动"],
                "target_outcome": "古钟鸣响与绿瞳惊现，在读者翻页欲望最高潮戛然而止",
                "length_budget": {"min_chars": 150, "max_chars": 300},
                "start_state": {"character_stances": {"protagonist": "突然停步，寒毛倒竖"}},
                "end_state": {"cognitive_changes": {"protagonist": "意识到陷入更恐怖的未知领域"}},
                "action_chain": [
                    {
                        "action_id": "act_ch_01",
                        "actor_id": "protagonist",
                        "trigger": "古钟幽鸣猝然响起",
                        "action": "陈羽猛然抬头，目睹暗街两侧阴影活化睁开绿色竖瞳",
                        "reason": "直面未知异变",
                        "reaction": "整条街所有黑影齐齐盯视二人",
                        "result": "全章在最惊悚悬念处瞬间切断",
                        "depends_on": ["act_af_01"],
                        "evidence": "断章卡点推演"
                    }
                ]
            }
        },
        "constraints": {
            "pov": "第三人称主角限制视角",
            "author_hard_requirements": ["动作凌厉，杜绝废话", "最后一句一刀封喉"],
            "negative_constraints": ["严禁擅自新增高阶外援", "严禁更改突围成功的既定结果"],
            "allowed_creative_freedom": ["动作物理碰撞细节", "光影法阵与氛围描写"]
        },
        "segments": {
            STAGE_OPENING: {"anchor_start": "", "anchor_end": "", "text": ""},
            STAGE_RAMP_UP: {"anchor_start": "", "anchor_end": "", "text": ""},
            STAGE_CLIMAX: {"anchor_start": "", "anchor_end": "", "text": ""},
            STAGE_AFTERSHOCK: {"anchor_start": "", "anchor_end": "", "text": ""},
            STAGE_CLIFFHANGER: {"anchor_start": "", "anchor_end": "", "text": ""},
        },
        "annotation_status": {"needs_review": False}
    }


def main():
    parser = argparse.ArgumentParser(description="TARA-5 全章自动化五段式装配流水线 (tara5_action_v1)")
    default_base = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_base).exists():
        default_base = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--base_model", type=str, default=default_base, help="基座模型路径")
    parser.add_argument("--adapter_path", type=str, default=str(LORA_ROOT / "output" / "tara_webnovel_7books"), help="LoRA 适配器路径")
    parser.add_argument("--chapter_file", type=str, default="", help="可选: 符合 tara5_action_v1 规范的章节 JSON 文件")
    parser.add_argument("--output_file", type=str, default=str(LORA_ROOT / "output" / "generated_chapter_v1.md"), help="生成保存路径")
    args = parser.parse_args()

    print("=" * 70)
    print("      TARA-5 五段式反向装配全闭环流水线 (tara5_action_v1)      ")
    print("=" * 70)

    # 1. 加载章节规划数据
    if args.chapter_file and Path(args.chapter_file).exists():
        print(f"[*] 从文件读取章节规划: {args.chapter_file}")
        with open(args.chapter_file, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
    else:
        print("[*] 使用默认网文宇宙实战测试题 (灵气黑市的规矩)")
        chapter_data = build_default_test_chapter()

    title = chapter_data.get("chapter_title", "未命名章节")

    print(f"[*] 正在加载 Tokenizer 与基座模型: {args.base_model}...")
    tokenizer_path = args.adapter_path if (Path(args.adapter_path) / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if Path(args.adapter_path).exists():
        print(f"[*] 正在挂载 LoRA 适配器: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"[!] 警告：未找到适配器 {args.adapter_path}，使用基座模型进行推演。")

    model.eval()

    # =============================================================
    # TARA-5 核心反向生成流 (Pure Reverse Order: 3 -> 5 -> 1 -> 2 -> 4)
    #
    # 步骤 1：【高潮切片 (climax)】先写！由章纲、事实与状态驱动，锁定 Anchor_3 与 Anchor_4
    # 步骤 2：【断章切片 (cliffhanger)】次写！锁定 Anchor_5 与 Anchor_6
    # 步骤 3：【破题切片 (opening)】承接上章结尾，锁定 Anchor_1 与 Anchor_2
    # 步骤 4：【蓄水引信桥 (ramp_up)】逆向倒推：从 Anchor_2 咬入 Anchor_3
    # 步骤 5：【余波转折中继桥 (aftershock)】逆向倒推：从 Anchor_4 咬入 Anchor_5
    # 步骤 6：【顺序装配与去重 (Safe Assembly)】按阅读顺序装配
    # =============================================================

    # 步骤 1：高潮核爆切片 (climax)
    print("\n>>> [反向生成 1/5] 启动高潮核爆切片推演 (STAGE_CLIMAX)...")
    sys_climax, user_climax = build_task_input(STAGE_CLIMAX, chapter_data)
    climax_raw = run_generate(model, tokenizer, sys_climax, user_climax, max_new_tokens=850, temperature=0.7)
    climax_think, climax_body = parse_think_and_text(climax_raw)
    anchor_3, anchor_4 = extract_anchors_from_output(climax_think, climax_body)
    print(f"  [+] 锁定 Anchor_3 (高潮爆发首句/蓄水终点目标): {anchor_3[:40]}...")
    print(f"  [+] 锁定 Anchor_4 (高潮定势末句/余波起点): {anchor_4[:40]}...")

    # 步骤 2：致命断章切片 (cliffhanger)
    print("\n>>> [反向生成 2/5] 启动致命断章切片推演 (STAGE_CLIFFHANGER)...")
    sys_cliff, user_cliff = build_task_input(STAGE_CLIFFHANGER, chapter_data)
    cliff_raw = run_generate(model, tokenizer, sys_cliff, user_cliff, max_new_tokens=350, temperature=0.7)
    cliff_think, cliff_body = parse_think_and_text(cliff_raw)
    anchor_5, anchor_6 = extract_anchors_from_output(cliff_think, cliff_body)
    print(f"  [+] 锁定 Anchor_5 (断章引爆首句/余波终点目标): {anchor_5[:40]}...")
    print(f"  [+] 锁定 Anchor_6 (全章绝杀卡点): {anchor_6[:40]}...")

    # 步骤 3：黄金破题切片 (opening)
    print("\n>>> [反向生成 3/5] 启动黄金破题切片推演 (STAGE_OPENING)...")
    sys_opening, user_opening = build_task_input(STAGE_OPENING, chapter_data)
    op_raw = run_generate(model, tokenizer, sys_opening, user_opening, max_new_tokens=350, temperature=0.7)
    op_think, op_body = parse_think_and_text(op_raw)
    anchor_1, anchor_2 = extract_anchors_from_output(op_think, op_body)
    print(f"  [+] 锁定 Anchor_1 (破题首句): {anchor_1[:40]}...")
    print(f"  [+] 锁定 Anchor_2 (破题定势末句/蓄水起点): {anchor_2[:40]}...")

    # 步骤 4：前向蓄水引信桥 (ramp_up)
    print("\n>>> [反向生成 4/5] 启动前向蓄水引信桥推演 (STAGE_RAMP_UP · 有界咬合)...")
    sys_rampup, user_rampup = build_task_input(
        STAGE_RAMP_UP,
        chapter_data,
        context=op_body,
        start_anchor=anchor_2,
        target_anchor=anchor_3,
    )
    ru_raw = run_generate(model, tokenizer, sys_rampup, user_rampup, max_new_tokens=900, temperature=0.7)
    ru_think, ru_body = parse_think_and_text(ru_raw)

    # 步骤 5：余波转折中继桥 (aftershock)
    print("\n>>> [反向生成 5/5] 启动余波转折中继桥推演 (STAGE_AFTERSHOCK · 有界咬合)...")
    sys_aftershock, user_aftershock = build_task_input(
        STAGE_AFTERSHOCK,
        chapter_data,
        context=climax_body,
        start_anchor=anchor_4,
        target_anchor=anchor_5,
    )
    af_raw = run_generate(model, tokenizer, sys_aftershock, user_aftershock, max_new_tokens=650, temperature=0.7)
    af_think, af_body = parse_think_and_text(af_raw)

    # =============================================================
    # 步骤 6：阅读顺序安全装配与熔断自愈 (Safe Assembly)
    # 装配顺序：Opening -> Ramp_up -> Climax -> Aftershock -> Cliffhanger
    # 锚点归属：Prev-Retained, Next-Stripped
    # =============================================================
    print("\n>>> [步骤 6] 执行阅读顺序安全装配 (Opening -> Ramp_up -> Climax -> Aftershock -> Cliffhanger)...")

    segments = {
        STAGE_OPENING: op_body,
        STAGE_RAMP_UP: ru_body,
        STAGE_CLIMAX: climax_body,
        STAGE_AFTERSHOCK: af_body,
        STAGE_CLIFFHANGER: cliff_body,
    }

    anchors_dict = {
        "anchor_1": anchor_1,
        "anchor_2": anchor_2,
        "anchor_3": anchor_3,
        "anchor_4": anchor_4,
        "anchor_5": anchor_5,
        "anchor_6": anchor_6,
    }

    out_file = Path(args.output_file)
    drafts_dir = out_file.parent / "drafts"

    full_text, success, diag_logs = safe_assemble_chapter(
        title=title,
        segments=segments,
        anchors=anchors_dict,
        drafts_dir=drafts_dir,
        allow_soft_fail=False,
    )

    print("\n" + "=" * 70)
    print("                 【装配接缝双侧检验与诊断报告】                 ")
    print("=" * 70)
    for log in diag_logs:
        print(f"  {log}")
    print("=" * 70)

    if success:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        final_output = f"# {title}\n\n{full_text}\n"
        out_file.write_text(final_output, encoding="utf-8")

        print("\n" + "=" * 70)
        print("                【TARA-5 最终顺序装配正文预览】                ")
        print("=" * 70)
        print(final_output[:600] + "\n...\n" + final_output[-400:])
        print("\n" + "=" * 70)
        print(f"[+] 全章总字数: {len(full_text)} 字")
        print(f"[+] 完整单章已成功写入: {out_file}")
        print("=" * 70 + "\n")
    else:
        print(f"\n[-] [装配熔断] 由于检测到关键接缝缺失或正文严重截断，已停止破损装配！")
        print(f"[*] 完整诊断与各段落草稿快照已存至: {drafts_dir}")


if __name__ == "__main__":
    main()
