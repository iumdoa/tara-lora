#!/usr/bin/env python3
"""
TARA-5 微调效果测试脚本 (Inference & Verification)
协议版本：tara5_action_v1

加载微调后的 LoRA 适配器，实测五段式反向装配各项任务生成效果：
- opening: 黄金破题切片（承接上章结尾）
- ramp_up: 前向蓄水引信桥（咬合高潮首句）
- climax: 高潮核爆切片（信息差反杀与核心肉戏）
- aftershock: 余波转折中继桥（咬合断章首句）
- cliffhanger: 致命断章切片（新危机绝杀卡点）

100% 复用 prompt_builder.py，确保推理输入与微调训练集完全同构！
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

from src.prompt_builder import (
    STAGE_OPENING,
    STAGE_RAMP_UP,
    STAGE_CLIMAX,
    STAGE_AFTERSHOCK,
    STAGE_CLIFFHANGER,
    VALID_STAGES,
    normalize_stage_name,
    build_task_input,
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


def build_default_test_data() -> dict:
    """构建用于独立任务测试的 tara5_action_v1 结构"""
    return {
        "schema_version": "tara5_action_v1",
        "chapter_id": "ch_infer_test",
        "source_novel": "daomu",
        "chapter_title": "第十九章 铜甲血尸",
        "prev_chapter_ending": "荒山破庙的瓦片被暴雨砸得噼啪作响，忽然从后院枯井深处传来一声指甲抓挠石壁的刺耳锐响。胖子手电猛地一晃，照出一只惨白浮肿的手腕刚刚缩回井口！",
        "chapter_outline": [
            "吴邪与胖子确认枯井异变，冒险绑绳下井探路，发现水下暗道与青铜古棺",
            "青铜棺盖炸裂，铜甲血尸破棺而出直扑二人，胖子工兵铲被反震脱手",
            "吴邪抓住血尸扑空的刹那，以黑金古刀刺入其后心封魂镇钉，二人配合引爆雷管将血尸炸碎",
            "坍塌石壁后传出惊悚异响，手电余光照出无数双猩红眼睛，全新万尸巢异变引爆卡点"
        ],
        "confirmed_facts": [
            {"fact": "铜甲血尸铜皮铁骨，寻常刀枪难伤，唯有后心第三椎骨的封魂镇钉是命门", "evidence": "古籍七星鲁王残卷记载"},
            {"fact": "吴邪与胖子随身仅剩最后半包雷管，且井底渗水严重，引线一旦浸湿即失效", "evidence": "前置背包清点记录"},
            {"fact": "胖子以为只是寻常尸变，低估了铜甲反震力道", "evidence": "胖子已知本土视界"}
        ],
        "characters": [
            {
                "character_id": "protagonist",
                "name": "吴邪",
                "goal": "绝不与血尸硬拼，寻找命门以最少代价脱困",
                "known_info": ["血尸命门在后心封魂镇钉"],
                "unknown_info": [],
                "misjudgment": ""
            },
            {
                "character_id": "antagonist",
                "name": "铜甲血尸",
                "goal": "扑杀侵入墓室的一切活物",
                "known_info": [],
                "unknown_info": [],
                "misjudgment": ""
            }
        ],
        "stage_plans": {
            STAGE_OPENING: {
                "stage_name": STAGE_OPENING,
                "responsibility": "承接上章井口浮肿白手悬念，开篇下井探路，确立水下暗道态势",
                "preconditions": ["承接荒庙井口发现异响与怪手"],
                "target_outcome": "二人翻入枯井，建立破题定势",
                "length_budget": {"min_chars": 150, "max_chars": 280},
                "start_state": {"character_positions": {"protagonist": "井口湿冷石台上"}},
                "end_state": {"character_stances": {"protagonist": "反手挽紧绳索潜入水下暗道"}},
                "action_chain": [
                    {
                        "action_id": "act_op_01",
                        "actor_id": "protagonist",
                        "trigger": "手电照出怪手缩回",
                        "action": "吴邪咬紧手电筒，示意胖子绑紧麻绳一前一后滑入深井",
                        "reason": "探明井底异变真相",
                        "reaction": "井底透出幽绿水光与扑面阴气",
                        "result": "二人成功下井，破题确立探墓局势",
                        "depends_on": [],
                        "evidence": "破题行动推演"
                    }
                ]
            },
            STAGE_RAMP_UP: {
                "stage_name": STAGE_RAMP_UP,
                "responsibility": "前向蓄水引信桥：探索水下暗道，青铜棺异动，将危险弓弦拉满",
                "preconditions": ["二人已抵达水下暗室"],
                "target_outcome": "棺椁异响加剧，逼至开棺爆发临界点",
                "length_budget": {"min_chars": 600, "max_chars": 850},
                "start_state": {"character_positions": {"protagonist": "摸索至中央青铜棺椁旁"}},
                "end_state": {"character_stances": {"protagonist": "雷管捏在掌心，冷汗浸透后背"}},
                "action_chain": [
                    {
                        "action_id": "act_ru_01",
                        "actor_id": "protagonist",
                        "trigger": "棺内传来沉闷撞击声",
                        "action": "吴邪示意胖子架铲警戒，自己绕到侧后方寻找撬缝",
                        "reason": "抢先占据观察与撤退路线",
                        "reaction": "棺椁四周墨斗线寸寸绷断，尸气喷涌",
                        "result": "冲突矛盾达到爆炸临界点",
                        "depends_on": ["act_op_01"],
                        "evidence": "蓄水推演"
                    }
                ]
            },
            STAGE_CLIMAX: {
                "stage_name": STAGE_CLIMAX,
                "responsibility": "高潮核爆：跳过铺垫前戏，血尸破棺扑杀，工兵铲被震飞，吴邪飞身刺命门引爆雷管",
                "preconditions": ["青铜棺盖炸裂，双方已展开面对面生死肉搏"],
                "target_outcome": "刺穿封魂镇钉并引爆雷管，彻底瓦解血尸",
                "length_budget": {"min_chars": 600, "max_chars": 850},
                "start_state": {"character_stances": {"protagonist": "被扑倒在积水中，黑金古刀横胸招架"}},
                "end_state": {"character_stances": {"protagonist": "引爆雷管翻滚跃入侧坑，血尸粉碎"}},
                "action_chain": [
                    {
                        "action_id": "act_cl_01",
                        "actor_id": "protagonist",
                        "trigger": "胖子工兵铲被血尸铜甲崩断震退",
                        "action": "吴邪滑铲贴地突进，古刀反撩精准切入血尸后心第三椎骨封魂镇钉，并将雷管塞入其甲缝后飞扑卧倒",
                        "reason": "兑现弱点信息差完成绝地反杀",
                        "reaction": "火光与狂暴冲击波将铜甲血尸炸裂成漫天腥臭碎骨",
                        "result": "彻底消灭铜甲血尸，锁定高潮胜负",
                        "depends_on": ["act_ru_01"],
                        "evidence": "高潮搏杀推演"
                    }
                ]
            },
            STAGE_AFTERSHOCK: {
                "stage_name": STAGE_AFTERSHOCK,
                "responsibility": "余波转折桥：硝烟弥漫，战后脱力喘息，清点伤势，获得片刻喘息假象",
                "preconditions": ["血尸被炸碎，激烈搏杀结束"],
                "target_outcome": "战况复盘，平稳顺畅咬入断章引信",
                "length_budget": {"min_chars": 350, "max_chars": 550},
                "start_state": {"character_stances": {"protagonist": "耳朵嗡嗡作响，从烂泥积水中撑起半身"}},
                "end_state": {"cognitive_changes": {"protagonist": "刚想松一口气，忽然察觉坍塌石壁后的死寂过于反常"}},
                "action_chain": [
                    {
                        "action_id": "act_af_01",
                        "actor_id": "protagonist",
                        "trigger": "爆炸烟尘缓缓散开",
                        "action": "扯出急救绷带帮胖子止住臂上鲜血，确认雷管耗尽",
                        "reason": "确认安全并评估剩余生存资源",
                        "reaction": "胖子大口喘着粗气抹脸上的黑灰",
                        "result": "建立虚假喘息，为新异变做铺垫",
                        "depends_on": ["act_cl_01"],
                        "evidence": "余波推演"
                    }
                ]
            },
            STAGE_CLIFFHANGER: {
                "stage_name": STAGE_CLIFFHANGER,
                "responsibility": "致命断章：引爆全新殉葬坑万尸巢危机，矛盾巅峰处一刀封喉",
                "preconditions": ["余波喘息刚刚建立"],
                "target_outcome": "无数红眼怪影现身，在最绝望悬念处戛然而止",
                "length_budget": {"min_chars": 150, "max_chars": 280},
                "start_state": {"character_stances": {"protagonist": "手电光斑猛地凝固在破裂的石壁缝隙中"}},
                "end_state": {"cognitive_changes": {"protagonist": "意识到刚杀死的不过是一只守门哨兵"}},
                "action_chain": [
                    {
                        "action_id": "act_ch_01",
                        "actor_id": "protagonist",
                        "trigger": "石壁深处传来连绵不绝的抓挠声",
                        "action": "吴邪死死捂住胖子的嘴，手电微光照出缝隙内密密麻麻睁开的上百只猩红血瞳",
                        "reason": "直面超出数量级的未知死局",
                        "reaction": "黑暗中的低吼此起彼伏，如潮水般涌来",
                        "result": "故事在恐怖顶峰瞬间断章卡死",
                        "depends_on": ["act_af_01"],
                        "evidence": "断章卡点推演"
                    }
                ]
            }
        },
        "constraints": {
            "pov": "第一人称（吴邪）限制视角",
            "negative_constraints": ["严禁擅自复活已死角色", "严禁破坏雷管耗尽的既定事实"],
            "allowed_creative_freedom": ["动作微细节描写", "惊悚悬疑氛围渲染"]
        },
        "segments": {
            STAGE_OPENING: {
                "anchor_start": "胖子抄起折叠铲，反手挽紧探阴爪的麻绳，二人一前一后翻身跃入深井。",
                "anchor_end": "胖子抄起折叠铲，反手挽紧探阴爪的麻绳，二人一前一后翻身跃入深井。",
                "text": "胖子抄起折叠铲，反手挽紧探阴爪的麻绳，二人一前一后翻身跃入深井。四周弥漫着尸臭与湿冷的阴气，石壁上隐隐渗出绿色黏液。"
            },
            STAGE_CLIMAX: {
                "anchor_start": "我刚想把刀拔出来，忽然摸到那尸体的肚皮下面，竟然有一张微弱呼吸着的人脸！",
                "anchor_end": "轰的一声闷响，血尸焦黑的残躯被气浪狠狠掀飞，撞碎在青铜棺椁上散成一地腥臭的碎骨。",
                "text": "我刚想把刀拔出来，忽然摸到那尸体的肚皮下面，竟然有一张微弱呼吸着的人脸！...轰的一声闷响，血尸焦黑的残躯被气浪狠狠掀飞，撞碎在青铜棺椁上散成一地腥臭的碎骨。"
            },
            STAGE_CLIFFHANGER: {
                "anchor_start": "胖子刚抹了一把脸上的冷汗，忽然整个人直愣愣地盯住坍塌的石壁后方，手里的工兵铲啪嗒一声掉在地上。",
                "anchor_end": "手电光芒在此时骤然闪烁了两下，终于彻底熄灭，黑暗中只剩下那密密麻麻的红光如萤火般悄然逼近。",
                "text": "胖子刚抹了一把脸上的冷汗，忽然整个人直愣愣地盯住坍塌的石壁后方，手里的工兵铲啪嗒一声掉在地上。...手电光芒在此时骤然闪烁了两下，终于彻底熄灭，黑暗中只剩下那密密麻麻的红光如萤火般悄然逼近。"
            }
        },
        "annotation_status": {"needs_review": False}
    }


def main():
    parser = argparse.ArgumentParser(description="TARA-5 LoRA 推理验证 (tara5_action_v1)")
    default_base = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_base).exists():
        default_base = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--base_model", type=str, default=default_base, help="基座模型 ID 或路径")
    parser.add_argument("--adapter_path", type=str, default=str(LORA_ROOT / "output" / "tara_lora_5"), help="LoRA 适配器路径")
    parser.add_argument("--task", type=str, default="cliffhanger", help=f"测试任务类型 (支持: {list(VALID_STAGES)})")
    parser.add_argument("--chapter_file", type=str, default="", help="可选: 指定符合 tara5_action_v1 规范的章节 JSON 文件")
    args = parser.parse_args()

    norm_task = normalize_stage_name(args.task)

    print("=" * 60)
    print("       TARA-5 反向装配 LoRA 模型效果实测 (tara5_action_v1)       ")
    print("=" * 60)
    print(f"[*] 基座模型: {args.base_model}")
    print(f"[*] LoRA 适配器: {args.adapter_path}")
    print(f"[*] 测试阶段: {norm_task}")

    # 读取或构建章节数据
    if args.chapter_file and Path(args.chapter_file).exists():
        print(f"[*] 加载指定章节文件: {args.chapter_file}")
        with open(args.chapter_file, "r", encoding="utf-8") as f:
            chapter_data = json.load(f)
    else:
        chapter_data = build_default_test_data()

    # 统一使用 prompt_builder 构造任务提示词 (保证与训练集 100% 同构)
    system_prompt, user_prompt = build_task_input(norm_task, chapter_data)

    print("\n" + "-" * 50)
    print("【统一构造的 System Prompt】")
    print(system_prompt)
    print("\n【统一构造的 User Prompt】")
    print(user_prompt)
    print("-" * 50 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path if (Path(args.adapter_path) / "tokenizer_config.json").exists() else args.base_model, trust_remote_code=True)
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
        print(f"[!] 提示：未找到适配器 {args.adapter_path}，使用基座模型推理。")

    model.eval()

    print(f"\n[*] 正在针对【{norm_task}】进行推演生成...\n")
    output = run_inference(model, tokenizer, system_prompt, user_prompt)

    print("=" * 60)
    print("                     【模型输出正文】                     ")
    print("=" * 60)
    print(output)
    print("=" * 60)
    print(f"[+] 输出字数: {len(output)} 字")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
