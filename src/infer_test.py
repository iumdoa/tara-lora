#!/usr/bin/env python3
"""
TARA-5 微调效果测试脚本 (Inference & Verification)
加载微调后的 LoRA 适配器，测试五段式三核双桥装配与咬合效果：
- opening: 破题黄金切片（承接上章结尾）
- rampup: 前向蓄水引信桥（咬合高潮起点）
- climax: 高潮核爆切片（信息差反杀）
- aftershock: 余波转折中继桥（咬合断章起点）
- cliffhanger: 致命断章切片（新危机绝杀卡点）
"""

import os
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
from prompts.extraction_prompts import (
    TRAIN_SYSTEM_PROMPT_OPENING,
    TRAIN_SYSTEM_PROMPT_RAMPUP,
    TRAIN_SYSTEM_PROMPT_CLIMAX,
    TRAIN_SYSTEM_PROMPT_AFTERSHOCK,
    TRAIN_SYSTEM_PROMPT_CLIFFHANGER,
)


def run_inference(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 1024):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_str, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = outputs[0][len(inputs["input_ids"][0]):]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="TARA-5 LoRA 推理验证")
    default_base = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_base).exists():
        default_base = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--base_model", type=str, default=default_base, help="基座模型 ID 或路径")
    parser.add_argument("--adapter_path", type=str, default=str(LORA_ROOT / "output" / "tara_lora_5"), help="LoRA 适配器路径")
    parser.add_argument("--task", type=str, default="cliffhanger", choices=["opening", "rampup", "climax", "aftershock", "cliffhanger"], help="测试任务类型")
    args = parser.parse_args()

    print("=" * 60)
    print("       TARA-5 反向装配 LoRA 模型效果实测       ")
    print("=" * 60)
    print(f"[*] 基座模型: {args.base_model}")
    print(f"[*] LoRA 适配器: {args.adapter_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path if Path(args.adapter_path).exists() else args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    if Path(args.adapter_path).exists():
        print(f"[*] 正在挂载 LoRA 权重: {args.adapter_path}...")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"[!] 警告：未找到适配器 {args.adapter_path}，将直接使用基座模型输出对比。")

    model.eval()

    if args.task == "opening":
        system_prompt = TRAIN_SYSTEM_PROMPT_OPENING
        user_prompt = (
            "【章节标题】第十八章 枯井下的活死人\n"
            "【上一章结尾承接】\n"
            "荒山破庙的瓦片被暴雨砸得噼啪作响，忽然从后院枯井深处传来一声指甲抓挠石壁的刺耳锐响。胖子手电猛地一晃，照出一只惨白浮肿的手腕刚刚缩回井口！\n\n"
            "【章纲要点】\n"
            "1. 吴邪与胖子确认枯井异变，冒险绑绳下井探路\n"
            "2. 井底透出诡异荧绿磷光，发现水下暗道\n\n"
            "【任务指示】\n"
            "请根据上一章结尾的余势与本章开局，撰写本章黄金破题切片（约 150~250 字）。\n"
            "必须紧扣上章尾声或以强动作反差开门见山，杜绝空洞环境闲笔，并在正文末尾锁定明确的【破题定势铁锚】。"
        )
    elif args.task == "rampup":
        system_prompt = TRAIN_SYSTEM_PROMPT_RAMPUP
        target_anchor = "我刚想把刀拔出来，忽然摸到那尸体的肚皮下面，竟然有一张微弱呼吸着的人脸！"
        user_prompt = (
            "【章节标题】第十八章 枯井下的活死人\n"
            "【起始铁锚 (破题定势)】胖子抄起折叠铲，反手挽紧探阴爪的麻绳，二人一前一后翻身跃入深井。\n"
            f"【终点铁锚 (Target Anchor · 高潮爆发首句)】{target_anchor}\n\n"
            "【任务指示】\n"
            "请从起始铁锚出发，撰写前半段情节推进（约 600~800 字），层层施压蓄积矛盾。\n"
            "【强制约束】文章正文最后一句必须严丝合缝、完全咬入指定的【终点铁锚】！"
        )
    elif args.task == "climax":
        system_prompt = TRAIN_SYSTEM_PROMPT_CLIMAX
        user_prompt = (
            "【章节标题】第十九章 铜甲血尸\n"
            "【章纲要点】\n"
            "1. 青铜棺盖炸裂，血尸破棺而出直扑二人\n"
            "2. 胖子工兵铲侧身掩护，吴邪以黑金古刀直刺血尸命门封魂镇钉\n"
            "3. 两人配合用雷管将血尸撕裂引爆\n\n"
            "【事实基座卡 (Truth Base)】\n"
            "- 主角现实算盘：绝不硬拼尸毒，必须用最后半包雷管引爆血尸命门方有一线生机\n"
            "- 配角本土视界：血尸铜皮铁骨刀枪不入，胖子以为工兵铲能拍碎其颈椎，实则险被断骨反刺\n"
            "- 信息差护城河：吴邪在古籍中见过此类铜甲血尸命门在后心第三椎骨的封魂镇钉\n"
            "- 终局死线：井底石室正在急速渗入剧毒地下水，水位已漫过脚踝，雷管引线受潮即废\n\n"
            "【任务指示】\n"
            "请跳过铺垫前戏，直接为本章撰写最核心的高潮爆发切片（约 600~800 字）。\n"
            "必须先在 <think> 中推演高潮反杀与物理动作，并在正文中锁定明确的【高潮首句】与【高潮末句】。"
        )
    elif args.task == "aftershock":
        system_prompt = TRAIN_SYSTEM_PROMPT_AFTERSHOCK
        user_prompt = (
            "【章节标题】第十九章 铜甲血尸\n"
            "【起始铁锚 (高潮落定末句)】轰的一声闷响，血尸焦黑的残躯被气浪狠狠掀飞，撞碎在青铜棺椁上散成一地腥臭的碎骨。\n"
            "【终点铁锚 (Target Anchor · 断章引爆首句)】胖子刚抹了一把脸上的冷汗，忽然整个人直愣愣地盯住坍塌的石壁后方，手里的工兵铲啪嗒一声掉在地上。\n\n"
            "【任务指示】\n"
            "请从高潮落定末句出发，撰写战后余波、生理颤抖、搜刮复盘与虚假安全感（约 400~500 字）。\n"
            "【强制约束】正文最后一句必须顺理成章、完全咬入指定的【终点铁锚】！"
        )
    elif args.task == "cliffhanger":
        system_prompt = TRAIN_SYSTEM_PROMPT_CLIFFHANGER
        user_prompt = (
            "【章节标题】第十九章 铜甲血尸\n"
            "【起始铁锚 (断章危机引爆句)】胖子刚抹了一把脸上的冷汗，忽然整个人直愣愣地盯住坍塌的石壁后方，手里的工兵铲啪嗒一声掉在地上。\n\n"
            "【任务指示】\n"
            "请从断章危机引爆句出发，在 150~250 字内引爆全新维度危机，在矛盾最高峰处戛然而止完成致命卡点。\n"
            "最后一句话必须一刀封喉，制造让读者不可抗拒的翻页渴望。"
        )

    print(f"\n【当前测试任务】tara_{args.task}")
    print("\n[*] 正在生成输出...\n")
    output = run_inference(model, tokenizer, system_prompt, user_prompt)
    print(output)
    print("\n" + "=" * 60)
    print("[+] 推理测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
