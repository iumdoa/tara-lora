#!/usr/bin/env python3
"""
TARA-5 数据集转换器 (Format Converter)

将五段式切分后的 `data/sliced/*.json` 转换为大模型微调标准格式：
- ShareGPT 格式 (LLaMA-Factory 默认推荐格式)
- 包含 5 个核心装配任务:
  1. tara_opening (破题黄金切片)
  2. tara_rampup (前向蓄水引信桥)
  3. tara_climax (高潮核爆切片)
  4. tara_aftershock (余波转折中继桥)
  5. tara_cliffhanger (致命断章切片)
"""

import sys
import json
import argparse
from pathlib import Path

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


def build_samples_from_sliced_chapter(data: dict) -> list:
    """将单个 TARA-5 切分好的章节转化为 5 个标准的微调训练样本"""
    samples = []
    ch_id = data.get("chapter_id", "ch")
    title = data.get("chapter_title", "未命名章节")
    prev_ending = data.get("prev_chapter_ending", "（本篇开门见山）").strip()
    outline_list = data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])
    
    tb = data.get("truth_base", {})
    truth_base_str = (
        f"- 主角现实算盘：{tb.get('protagonist_plan', '')}\n"
        f"- 配角本土视界：{tb.get('antagonist_horizon', '')}\n"
        f"- 信息差护城河：{tb.get('information_gap', '')}\n"
        f"- 终局死线：{tb.get('deadline', '')}"
    )

    segments = data.get("segments", {})
    op = segments.get("opening", {})
    ru = segments.get("ramp_up", {})
    cl = segments.get("climax", {})
    af = segments.get("aftershock", {})
    ch = segments.get("cliffhanger", {})

    # ==========================================
    # 任务 1：破题黄金切片 (tara_opening)
    # ==========================================
    op_start = op.get("anchor_start", "")
    op_end = op.get("anchor_end", "")
    op_prompt = (
        f"【章节标题】{title}\n"
        f"【上一章结尾承接】\n{prev_ending}\n\n"
        f"【章纲要点】\n{outline_str}\n\n"
        f"【任务指示】\n"
        f"请根据上一章结尾的余势与本章开局，撰写本章黄金破题切片（约 150~250 字）。\n"
        f"必须紧扣上章尾声或以强动作反差开门见山，杜绝空洞环境闲笔，并在正文末尾锁定明确的【破题定势铁锚】。"
    )
    op_think = (
        f"<think>\n"
        f"1. 上章余势承接：承接上章结尾的悬念或动作冲击，绝不作无关的环境闲笔。\n"
        f"2. 动态破题设计：用具体动作或对抗打破平衡，直接切入核心主角的即时处境。\n"
        f"3. 锚点锁定：\n"
        f"   - 破题首句：{op_start}\n"
        f"   - 破题定势铁锚：{op_end}\n"
        f"</think>\n"
    )
    op_output = f"{op_think}{op.get('text', '').strip()}"
    samples.append({
        "id": f"{ch_id}_opening",
        "task_type": "tara_opening",
        "system": TRAIN_SYSTEM_PROMPT_OPENING,
        "conversations": [
            {"from": "human", "value": op_prompt},
            {"from": "gpt", "value": op_output}
        ]
    })

    # ==========================================
    # 任务 2：前向蓄水引信桥 (tara_rampup) - 注入章纲、事实基座与破题上下文穿透
    # ==========================================
    ru_start = ru.get("anchor_start", op_end)
    ru_end = ru.get("anchor_end", cl.get("anchor_start", ""))
    op_context = op.get("text", "").strip()
    if len(op_context) > 280:
        op_context = f"{op_context[:280]}..."

    rampup_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 开篇破题局势】\n{op_context}\n\n"
        f"【起始铁锚 (破题定势)】{ru_start}\n"
        f"【终点铁锚 (Target Anchor · 高潮爆发首句)】{ru_end}\n\n"
        f"【任务指示】\n"
        f"请从起始铁锚出发，严格围绕总章纲与前文破题局势展开情节推进（约 600~800 字），层层施压蓄积矛盾。\n"
        f"【强制约束】文章正文最后一句必须严丝合缝、完全咬入指定的【终点铁锚】！"
    )
    rampup_think = (
        f"<think>\n"
        f"1. 蓄水推进链条：紧密承接前文破题态势，让矛盾步步升级，主角展开谋划与行动，将危机弓弦拉满。\n"
        f"2. 终点咬合自省：正文最后一句话必须是通向高潮的临门一脚，使读者顺畅滑入：“{ru_end}”。\n"
        f"3. 剧情因果自洽：严格恪守事实基座中的信息差与主角算盘，杜绝跳脱主线的无端幻觉。\n"
        f"</think>\n"
    )
    ru_text = ru.get("text", "").strip()
    if ru_end and not ru_text.endswith(ru_end):
        ru_text = f"{ru_text}\n\n{ru_end}"
    rampup_output = f"{rampup_think}{ru_text}"
    samples.append({
        "id": f"{ch_id}_rampup",
        "task_type": "tara_rampup",
        "system": TRAIN_SYSTEM_PROMPT_RAMPUP,
        "conversations": [
            {"from": "human", "value": rampup_prompt},
            {"from": "gpt", "value": rampup_output}
        ]
    })

    # ==========================================
    # 任务 3：高潮核爆切片 (tara_climax) - 注入蓄水临界态势穿透
    # ==========================================
    cl_start = cl.get("anchor_start", "")
    cl_end = cl.get("anchor_end", "")
    ru_context = ru_text[-280:].strip() if len(ru_text) > 280 else ru_text

    climax_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 蓄水临界态势】\n...{ru_context}\n\n"
        f"【任务指示】\n"
        f"请跳过铺垫前戏，直接围绕核心对决撰写最核心的高潮爆发切片（约 600~800 字）。\n"
        f"必须先在 <think> 中推演高潮反杀与物理动作，并在正文中锁定明确的【高潮首句】与【高潮末句】。"
    )
    climax_think = (
        f"<think>\n"
        f"1. 高潮冲突定位：{cl.get('core_conflict', '核心对决')}\n"
        f"2. 信息差兑现：依托事实基座中的信息差，主角反客为主，瞬间撕开对手的傲慢伪装。\n"
        f"3. 物理与动作细节：避免空洞修辞，着重描写肌肉发力、器械交锋与实物碰撞。\n"
        f"4. 锚点锁定：\n"
        f"   - 起点铁锚 (Climax_Start)：{cl_start}\n"
        f"   - 终点铁锚 (Climax_End)：{cl_end}\n"
        f"</think>\n"
    )
    climax_output = f"{climax_think}{cl.get('text', '').strip()}"
    samples.append({
        "id": f"{ch_id}_climax",
        "task_type": "tara_climax",
        "system": TRAIN_SYSTEM_PROMPT_CLIMAX,
        "conversations": [
            {"from": "human", "value": climax_prompt},
            {"from": "gpt", "value": climax_output}
        ]
    })

    # ==========================================
    # 任务 4：余波转折中继桥 (tara_aftershock) - 注入章纲、事实基座与高潮战况穿透
    # ==========================================
    af_start = af.get("anchor_start", cl_end)
    af_end = af.get("anchor_end", ch.get("anchor_start", ""))
    cl_context = cl.get("text", "").strip()
    if len(cl_context) > 280:
        cl_context = f"...{cl_context[-280:]}"

    aftershock_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 高潮战果局势】\n{cl_context}\n\n"
        f"【起始铁锚 (高潮落定末句)】{af_start}\n"
        f"【终点铁锚 (Target Anchor · 断章引爆首句)】{af_end}\n\n"
        f"【任务指示】\n"
        f"请从高潮落定末句出发，严格紧扣前文战况，撰写战后余波、生理颤抖、搜刮复盘与虚假安全感（约 400~500 字）。\n"
        f"【强制约束】正文最后一句必须顺理成章、完全咬入指定的【终点铁锚】！"
    )
    aftershock_think = (
        f"<think>\n"
        f"1. 承接点：紧贴高潮落定末句的震撼余势起笔，严丝合缝继承高潮战果。\n"
        f"2. 侧面烘托与复盘：描写在场各方的生理性震颤，清点战利品或消化变局，制造短暂喘息与虚假安全感。\n"
        f"3. 终点咬合自省：末尾顺畅滑入下一轮危机的引信：“{af_end}”。\n"
        f"</think>\n"
    )
    af_text = af.get("text", "").strip()
    if af_end and not af_text.endswith(af_end):
        af_text = f"{af_text}\n\n{af_end}"
    aftershock_output = f"{aftershock_think}{af_text}"
    samples.append({
        "id": f"{ch_id}_aftershock",
        "task_type": "tara_aftershock",
        "system": TRAIN_SYSTEM_PROMPT_AFTERSHOCK,
        "conversations": [
            {"from": "human", "value": aftershock_prompt},
            {"from": "gpt", "value": aftershock_output}
        ]
    })

    # ==========================================
    # 任务 5：致命断章切片 (tara_cliffhanger) - 注入章纲与战后余波态势穿透
    # ==========================================
    ch_start = ch.get("anchor_start", af_end)
    ch_end = ch.get("anchor_end", "")
    af_context = af_text[-250:].strip() if len(af_text) > 250 else af_text

    cliffhanger_prompt = (
        f"【章节标题】{title}\n"
        f"【本章总章纲】\n{outline_str}\n\n"
        f"【事实基座卡 (Truth Base)】\n{truth_base_str}\n\n"
        f"【前置剧情穿透 · 战后余波态势】\n...{af_context}\n\n"
        f"【起始铁锚 (断章危机引爆句)】{ch_start}\n\n"
        f"【任务指示】\n"
        f"请从断章危机引爆句出发，严格紧扣全章剧情脉络，在 150~250 字内引爆全新维度危机，在矛盾最高峰处戛然而止完成致命卡点。\n"
        f"最后一句话必须一刀封喉，制造让读者不可抗拒的翻页渴望。"
    )
    cliffhanger_think = (
        f"<think>\n"
        f"1. 危机引爆：承接断章首句，瞬时拉升异变与压迫感。\n"
        f"2. 绝杀卡点：在动作尚未完成或惊悚真相刚露一角时瞬间切断，拒绝任何交代式总结。\n"
        f"3. 锚点锁定：\n"
        f"   - 起点铁锚：{ch_start}\n"
        f"   - 绝杀卡点末句：{ch_end}\n"
        f"</think>\n"
    )
    cliffhanger_output = f"{cliffhanger_think}{ch.get('text', '').strip()}"
    samples.append({
        "id": f"{ch_id}_cliffhanger",
        "task_type": "tara_cliffhanger",
        "system": TRAIN_SYSTEM_PROMPT_CLIFFHANGER,
        "conversations": [
            {"from": "human", "value": cliffhanger_prompt},
            {"from": "gpt", "value": cliffhanger_output}
        ]
    })

    return samples


def main():
    parser = argparse.ArgumentParser(description="TARA-5 五段式微调数据集转换器")
    parser.add_argument("--sliced-dir", type=str, default=str(LORA_ROOT / "data" / "sliced"), help="切分文件目录")
    parser.add_argument("--dataset-dir", type=str, default=str(LORA_ROOT / "data" / "dataset"), help="数据集输出目录")
    args = parser.parse_args()

    sliced_dir = Path(args.sliced_dir)
    dataset_dir = Path(args.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(sliced_dir.glob("*.json")))
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
        "tara_5_sft": {
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
    print(f"[+] 转换完成！共生成 {len(all_train_samples)} 条标准五段式训练样本")
    print(f"- 训练数据集: {out_file}")
    print(f"- LLaMA-Factory 注册片段: {info_file}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
