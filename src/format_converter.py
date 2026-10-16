#!/usr/bin/env python3
"""
TARA 数据集转换器 (Format Converter)

将切分后的 `data/sliced/*.json` 转换为大模型微调标准格式：
- ShareGPT 格式 (LLaMA-Factory 默认推荐格式)
- Alpaca 格式
"""

import sys
import json
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))
from prompts.extraction_prompts import (
    TRAIN_SYSTEM_PROMPT_CLIMAX,
    TRAIN_SYSTEM_PROMPT_RAMPUP,
    TRAIN_SYSTEM_PROMPT_AFTERMATH,
)


def build_samples_from_sliced_chapter(data: dict) -> list:
    """将单个切分好的章节转化为 3 个标准的训练样本"""
    samples = []
    title = data.get("chapter_title", "未命名章节")
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(data.get("chapter_outline", []))])
    tb = data.get("truth_base", {})
    truth_base_str = (
        f"- 主角现实算盘：{tb.get('protagonist_plan', '')}\n"
        f"- 配角本土视界：{tb.get('antagonist_horizon', '')}\n"
        f"- 信息差护城河：{tb.get('information_gap', '')}\n"
        f"- 终局死线：{tb.get('deadline', '')}"
    )

    climax = data.get("climax", {})
    ramp_up = data.get("ramp_up", {})
    aftermath = data.get("aftermath", {})

    # ==========================================
    # 任务 1：高潮核爆切片定点生成 (tara_climax)
    # ==========================================
    climax_prompt = (
        f"【章节标题】{title}\n"
        f"【章纲要点】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【任务指示】\n"
        f"请跳过铺垫前戏，直接为本章撰写最核心的高潮爆发切片（约600~800字）。\n"
        f"必须先在 <think> 中推演高潮反杀与物理动作，并在正文中锁定明确的【高潮首句】与【高潮末句】。"
    )

    climax_think = (
        f"<think>\n"
        f"1. 高潮冲突定位：{climax.get('core_conflict', '核心对决')}\n"
        f"2. 信息差兑现：依托事实基座中的信息差，主角反客为主，瞬间撕开对手的傲慢伪装。\n"
        f"3. 物理与动作细节：避免空洞修辞，着重描写肌肉发力、穴位打击与实物碰撞。\n"
        f"4. 锚点锁定：\n"
        f"   - 起点铁锚 (Climax_Start)：{climax.get('climax_start', '')}\n"
        f"   - 终点铁锚 (Climax_End)：{climax.get('climax_end', '')}\n"
        f"</think>\n"
    )
    climax_output = f"{climax_think}{climax.get('text', '')}"

    samples.append({
        "id": f"{data.get('chapter_id', 'ch')}_climax",
        "task_type": "tara_climax",
        "system": TRAIN_SYSTEM_PROMPT_CLIMAX,
        "conversations": [
            {"from": "human", "value": climax_prompt},
            {"from": "gpt", "value": climax_output}
        ]
    })

    # ==========================================
    # 任务 2：前向倒推插值咬合 (tara_rampup)
    # ==========================================
    target_anchor = ramp_up.get("target_anchor_sentence", climax.get("climax_start", ""))
    rampup_prompt = (
        f"【章节标题】{title}\n"
        f"【开局初始情境】{ramp_up.get('opening_state', '')}\n"
        f"【终点铁锚 (Target Anchor)】{target_anchor}\n\n"
        f"【任务指示】\n"
        f"请从开局情境出发，撰写前半段情节推进（约600~900字），层层蓄积矛盾。\n"
        f"【强制约束】正文最后一句必须严丝合缝过渡并咬入【终点铁锚】！"
    )

    rampup_think = (
        f"<think>\n"
        f"1. 铺垫推进链条：从初始情境出发，让反派步步紧逼，主角顺势示弱布设陷阱。\n"
        f"2. 终点咬合自省：正文最后一句话必须是通向高潮的临门一脚，使读者顺畅滑入：“{target_anchor}”。\n"
        f"3. 去水文检查：删除一切不服务于激化矛盾的环境闲笔。\n"
        f"</think>\n"
    )
    rampup_output = f"{rampup_think}{ramp_up.get('text', '')}"

    samples.append({
        "id": f"{data.get('chapter_id', 'ch')}_rampup",
        "task_type": "tara_rampup",
        "system": TRAIN_SYSTEM_PROMPT_RAMPUP,
        "conversations": [
            {"from": "human", "value": rampup_prompt},
            {"from": "gpt", "value": rampup_output}
        ]
    })

    # ==========================================
    # 任务 3：尾声余波与断章留钩 (tara_aftermath)
    # ==========================================
    start_anchor = aftermath.get("from_anchor_sentence", climax.get("climax_end", ""))
    aftermath_prompt = (
        f"【章节标题】{title}\n"
        f"【承接高潮终点】{start_anchor}\n"
        f"【断章目标】{aftermath.get('hook', '制造全新悬念')}\n\n"
        f"【任务指示】\n"
        f"请承接高潮终点，撰写本章尾声余波（约400~600字）。\n"
        f"重点烘托现场各方震动反响，并于末尾引爆新危机，完成致命卡点断章。"
    )

    aftermath_think = (
        f"<think>\n"
        f"1. 承接点：紧贴“{start_anchor}”的震撼余势起笔。\n"
        f"2. 侧面烘托：描写在场配角与路人的生理性震颤，借他者之口坐实主角手段之狠辣。\n"
        f"3. 断章钩子：将危机推向更高维度的外部势力，并在最后一刻戛然而止。\n"
        f"</think>\n"
    )
    aftermath_output = f"{aftermath_think}{aftermath.get('text', '')}"

    samples.append({
        "id": f"{data.get('chapter_id', 'ch')}_aftermath",
        "task_type": "tara_aftermath",
        "system": TRAIN_SYSTEM_PROMPT_AFTERMATH,
        "conversations": [
            {"from": "human", "value": aftermath_prompt},
            {"from": "gpt", "value": aftermath_output}
        ]
    })

    return samples


def main():
    sliced_dir = LORA_ROOT / "data" / "sliced"
    dataset_dir = LORA_ROOT / "data" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(sliced_dir.glob("*.json"))
    if not json_files:
        print(f"[-] 在 {sliced_dir} 中未找到切分文件！请先运行 segmenter.py")
        sys.exit(1)

    all_train_samples = []
    for fpath in json_files:
        print(f"[*] 正在处理: {fpath.name}")
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            samples = build_samples_from_sliced_chapter(data)
            all_train_samples.extend(samples)

    out_file = dataset_dir / "tara_train_dataset.json"
    with open(out_file, "w", encoding="utf-8") as out_f:
        json.dump(all_train_samples, out_f, ensure_ascii=False, indent=2)

    # 附带生成 LLaMA-Factory 所需的 dataset_info.json 配置片段
    dataset_info_snippet = {
        "tara_webnovel_sft": {
            "file_name": "tara_train_dataset.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system"
            }
        }
    }
    info_file = dataset_dir / "dataset_info_snippet.json"
    with open(info_file, "w", encoding="utf-8") as inf_f:
        json.dump(dataset_info_snippet, inf_f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"[+] 转换完成！共生成 {len(all_train_samples)} 条标准训练样本")
    print(f"- 训练数据集: {out_file}")
    print(f"- LLaMA-Factory 注册片段: {info_file}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
