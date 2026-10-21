#!/usr/bin/env python3
"""
TARA-5 (Five-Stage Anchor & Style Contrast) 全章反向装配生成器

执行五段式反向装配全闭环流程：
1. 【高潮切片 (tara_climax)】推演爆发点，锁定 Anchor_3 与 Anchor_4
2. 【断章切片 (tara_cliffhanger)】推演全章绝杀卡点，锁定 Anchor_5 与 Anchor_6
3. 【破题切片 (tara_opening)】承接上一章结尾，开门见山，锁定 Anchor_1 与 Anchor_2
4. 【蓄水桥 (tara_rampup)】从 Anchor_2 推进，强制咬入 Anchor_3
5. 【余波桥 (tara_aftershock)】从 Anchor_4 复盘喘息，强制咬入 Anchor_5
6. 【无缝装配 (Zero-Token Assembly)】按顺序装配 Opening + Ramp-up + Climax + Aftershock + Cliffhanger
"""

import os
import re
import sys
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
from prompts.extraction_prompts import (
    TRAIN_SYSTEM_PROMPT_OPENING,
    TRAIN_SYSTEM_PROMPT_RAMPUP,
    TRAIN_SYSTEM_PROMPT_CLIMAX,
    TRAIN_SYSTEM_PROMPT_AFTERSHOCK,
    TRAIN_SYSTEM_PROMPT_CLIFFHANGER,
)


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


def parse_think_and_text(output_str: str):
    """解析并分离 <think> 思考链与正文"""
    think_match = re.search(r"<think>(.*?)</think>", output_str, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        body_text = output_str[think_match.end():].strip()
        return think_content, body_text
    return "", output_str.strip()


def extract_anchors(think_str: str, body_text: str):
    """从 think 或正文中提取起点铁锚与终点铁锚"""
    start_anchor = ""
    end_anchor = ""

    m_start = re.search(r"(?:起点铁锚|破题首句|引爆句)[^\n:：]*[:：]\s*[“\"「](.*?)[”\"」]", think_str)
    m_end = re.search(r"(?:终点铁锚|破题定势铁锚|卡点末句|高潮末句)[^\n:：]*[:：]\s*[“\"「](.*?)[”\"」]", think_str)

    if m_start:
        start_anchor = m_start.group(1).strip()
    if m_end:
        end_anchor = m_end.group(1).strip()

    lines = [l.strip() for l in body_text.split("\n") if l.strip() and not l.strip().startswith("【")]
    if not start_anchor and lines:
        start_anchor = lines[0]
    if not end_anchor and lines:
        if len(lines[-1]) < 12 and len(lines) > 1:
            end_anchor = f"{lines[-2]} {lines[-1]}"
        else:
            end_anchor = lines[-1]

    return start_anchor, end_anchor


def main():
    parser = argparse.ArgumentParser(description="TARA-5 全章自动化五段式装配流水线")
    default_base = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_base).exists():
        default_base = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--base_model", type=str, default=default_base, help="基座模型路径")
    parser.add_argument("--adapter_path", type=str, default=str(LORA_ROOT / "output" / "tara_webnovel_7books"), help="LoRA 适配器路径")
    parser.add_argument("--output_file", type=str, default=str(LORA_ROOT / "output" / "generated_chapter_7books.md"), help="生成保存路径")
    args = parser.parse_args()

    print("=" * 70)
    print("      TARA 网文体系多流派融合通用生成流水线 (7-Book Webnovel System)      ")
    print("=" * 70)
    print(f"[*] 正在加载 Tokenizer 与基座模型: {args.base_model}...")
    tokenizer_path = args.adapter_path if (Path(args.adapter_path) / "tokenizer_config.json").exists() else args.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    print(f"[*] 正在挂载通用网文体系 LoRA 适配器: {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    # 设定网文体系综合测试命题（融合都市异能、强冲突博弈与惊险断章）
    title = "第七十八章 灵气黑市的规矩"
    prev_chapter_ending = "卷帘门哗啦一声被狂暴的气劲撕碎，刺耳的警报响彻长巷。陈羽一把拽住吓呆的胖子闪进阴影，身后七八名黑衣修士手持附魔符文弩，杀气腾腾地封死了两头巷口！"
    outline = [
        "陈羽凭借敏锐感知识破黑市伏击圈，借着巷道杂物与反光镜制造假身引诱弩箭齐射脱困",
        "黑市暗堂执事带队步步紧逼，以势压人逼交灵石配方，陈羽假意周旋实则暗中调动地脉灵气",
        "谈判破裂瞬间，陈羽果断引爆杂货铺电表箱的灵气逆流，借狂暴冲击波近身反杀前排修士撕开生门",
        "两人夺路冲进古玩暗街，身后突然传来一声令人头皮发麻的幽沉古钟鸣响，整条街的黑影齐齐转头睁开了幽绿的竖瞳，全新杀机引爆全章卡点"
    ]
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline)])
    truth_base_str = (
        "- 主角现实算盘：绝不在此刻暴露真正底牌，必须在治安巡逻队赶到前把水彻底搅浑全身而退\n"
        "- 配角本土视界：黑市执事认定两个年轻散修不过是误打误撞的肥羊，翻手即可强行镇压搜魂\n"
        "- 信息差护城河：陈羽早已看穿黑市阵法的灵气循环节点位于杂货铺后墙电表箱内\n"
        "- 终局死线：巷口符文弩箭二次上膛只需 15 秒，一旦合围将被万箭穿心"
    )

    print(f"\n【章节测试命题】{title}")
    print(f"【上一章结尾承接】\n{prev_chapter_ending}")
    print(f"【章纲规划】\n{outline_str}")
    print(f"【事实基座 (Truth Base)】\n{truth_base_str}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 步骤 1：生成高潮切片 (tara_climax) -> 锁定 Anchor_3 与 Anchor_4
    # -------------------------------------------------------------
    print("\n>>> [步骤 1/5] 启动高潮核爆切片生成 (tara_climax)...")
    climax_prompt = (
        f"【章节标题】{title}\n"
        f"【章纲要点】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【任务指示】\n"
        f"请跳过铺垫前戏，直接为本章撰写最核心的高潮爆发切片（约 600~800 字）。\n"
        f"必须先在 <think> 中推演高潮反杀与物理动作，并在正文中锁定明确的【高潮首句】与【高潮末句】。"
    )
    climax_raw = run_generate(model, tokenizer, TRAIN_SYSTEM_PROMPT_CLIMAX, climax_prompt, max_new_tokens=800, temperature=0.7)
    climax_think, climax_body = parse_think_and_text(climax_raw)
    anchor_3, anchor_4 = extract_anchors(climax_think, climax_body)
    print(f"[*] 锁定 Anchor_3 (高潮爆发首句): {anchor_3}")
    print(f"[*] 锁定 Anchor_4 (高潮定势末句): {anchor_4}")

    # -------------------------------------------------------------
    # 步骤 2：生成断章切片 (tara_cliffhanger) -> 锁定 Anchor_5 与 Anchor_6
    # -------------------------------------------------------------
    print("\n>>> [步骤 2/5] 启动致命断章切片生成 (tara_cliffhanger)...")
    cliff_start_hint = "身后突然传来一声令人头皮发麻的幽沉古钟鸣响，整条暗街的诡异黑影齐刷刷转过头来，直勾勾睁开了惨绿色的竖瞳！"
    cliffhanger_prompt = (
        f"【章节标题】{title}\n"
        f"【起始铁锚 (断章危机引爆句)】{cliff_start_hint}\n\n"
        f"【任务指示】\n"
        f"请从断章危机引爆句出发，在 150~250 字内引爆全新维度危机，在矛盾最高峰处戛然而止完成致命卡点。\n"
        f"最后一句话必须一刀封喉，制造让读者不可抗拒的翻页渴望。"
    )
    cliff_raw = run_generate(model, tokenizer, TRAIN_SYSTEM_PROMPT_CLIFFHANGER, cliffhanger_prompt, max_new_tokens=350, temperature=0.7)
    cliff_think, cliff_body = parse_think_and_text(cliff_raw)
    anchor_5, anchor_6 = extract_anchors(cliff_think, cliff_body)
    print(f"[*] 锁定 Anchor_5 (断章引爆首句): {anchor_5}")
    print(f"[*] 锁定 Anchor_6 (全章绝杀卡点): {anchor_6}")

    # -------------------------------------------------------------
    # 步骤 3：生成破题切片 (tara_opening) -> 承接上章结尾，锁定 Anchor_1 与 Anchor_2
    # -------------------------------------------------------------
    print("\n>>> [步骤 3/5] 启动破题切片生成 (tara_opening)...")
    op_prompt = (
        f"【章节标题】{title}\n"
        f"【上一章结尾承接】\n{prev_chapter_ending}\n\n"
        f"【章纲要点】\n{outline_str}\n\n"
        f"【任务指示】\n"
        f"请根据上一章结尾的余势与本章开局，撰写本章黄金破题切片（约 150~250 字）。\n"
        f"必须紧扣上章尾声或以强动作反差开门见山，杜绝空洞环境闲笔，并在正文末尾锁定明确的【破题定势铁锚】。"
    )
    op_raw = run_generate(model, tokenizer, TRAIN_SYSTEM_PROMPT_OPENING, op_prompt, max_new_tokens=350, temperature=0.7)
    op_think, op_body = parse_think_and_text(op_raw)
    anchor_1, anchor_2 = extract_anchors(op_think, op_body)
    print(f"[*] 锁定 Anchor_1 (破题首句): {anchor_1}")
    print(f"[*] 锁定 Anchor_2 (破题定势末句): {anchor_2}")

    # -------------------------------------------------------------
    # 步骤 4：生成前向蓄水引信桥 (tara_rampup) -> Anchor_2 咬入 Anchor_3 (带章纲、基座与破题穿透)
    # -------------------------------------------------------------
    print("\n>>> [步骤 4/5] 启动前向蓄水引信桥 (tara_rampup)...")
    op_context = op_body[:280].strip() if len(op_body) > 280 else op_body
    rampup_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 开篇破题局势】\n{op_context}\n\n"
        f"【起始铁锚 (破题定势)】{anchor_2}\n"
        f"【终点铁锚 (Target Anchor · 高潮爆发首句)】{anchor_3}\n\n"
        f"【任务指示】\n"
        f"请从起始铁锚出发，严格围绕总章纲与前文破题局势展开情节推进（约 600~800 字），层层施压蓄积矛盾。\n"
        f"【强制约束】文章正文最后一句必须严丝合缝、完全咬入指定的【终点铁锚】！"
    )
    ru_raw = run_generate(model, tokenizer, TRAIN_SYSTEM_PROMPT_RAMPUP, rampup_prompt, max_new_tokens=850, temperature=0.7)
    _, ru_body = parse_think_and_text(ru_raw)

    # -------------------------------------------------------------
    # 步骤 5：生成余波转折中继桥 (tara_aftershock) -> Anchor_4 咬入 Anchor_5 (带章纲、基座与高潮穿透)
    # -------------------------------------------------------------
    print("\n>>> [步骤 5/5] 启动余波转折中继桥 (tara_aftershock)...")
    cl_context = climax_body[-280:].strip() if len(climax_body) > 280 else climax_body
    aftershock_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 高潮战果局势】\n...{cl_context}\n\n"
        f"【起始铁锚 (高潮落定末句)】{anchor_4}\n"
        f"【终点铁锚 (Target Anchor · 断章引爆首句)】{anchor_5}\n\n"
        f"【任务指示】\n"
        f"请从高潮落定末句出发，严格紧扣前文战况，撰写战后余波、生理颤抖、搜刮复盘与虚假安全感（约 400~500 字）。\n"
        f"【强制约束】正文最后一句必须顺理成章、完全咬入指定的【终点铁锚】！"
    )
    af_raw = run_generate(model, tokenizer, TRAIN_SYSTEM_PROMPT_AFTERSHOCK, aftershock_prompt, max_new_tokens=600, temperature=0.7)
    _, af_body = parse_think_and_text(af_raw)

    # -------------------------------------------------------------
    # 装配：五段零 Token 本地无缝咬合装配 (Zero-Token Assembly)
    # -------------------------------------------------------------
    print("\n>>> [装配完成] 执行五段无缝咬合装配...")

    def clean_text(text):
        lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith("【")]
        return "\n\n".join(lines)

    p_opening = clean_text(op_body)
    p_rampup = clean_text(ru_body)
    p_climax = clean_text(climax_body)
    p_aftershock = clean_text(af_body)
    p_cliff = clean_text(cliff_body)

    # 智能接缝去重：若后一段开头重复了前一段的终点锚点，则优雅去重
    def strip_leading_anchor(text, anchor):
        text = text.strip()
        if anchor and text.startswith(anchor):
            text = text[len(anchor):].strip()
        return text

    p_rampup = strip_leading_anchor(p_rampup, anchor_2)
    p_climax = strip_leading_anchor(p_climax, anchor_3)
    p_aftershock = strip_leading_anchor(p_aftershock, anchor_4)
    p_cliff = strip_leading_anchor(p_cliff, anchor_5)

    full_parts = [p_opening, p_rampup, p_climax, p_aftershock, p_cliff]
    full_chapter = f"{title}\n\n" + "\n\n".join([p for p in full_parts if p.strip()])

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(full_chapter, encoding="utf-8")

    print("\n" + "=" * 70)
    print("                【TARA-5 五段式反向装配完整成章效果展示】                ")
    print("=" * 70)
    print(full_chapter)
    print("\n" + "=" * 70)
    print(f"[+] 全章字数: {len(full_chapter)} 字")
    print(f"[+] 完整单章已保存至: {out_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
