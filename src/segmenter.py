#!/usr/bin/env python3
"""
TARA (Three-Anchor Reverse Assembly) 章节逆向切分与结构化引擎

功能：
1. 演示模式 (--demo): 对内置高品质网文章节执行标准切分，展示切片、首尾锚点与事实基座
2. API 生产模式 (--file <path>): 调用兼容 OpenAI/DeepSeek 的 LLM API，对海量小说章节执行逆向抽取
"""

import os
import sys
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path

# 引入 Prompt 模板
CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))
from prompts.extraction_prompts import (
    REVERSE_EXTRACTION_SYSTEM_PROMPT,
    REVERSE_EXTRACTION_USER_TEMPLATE,
)


def run_demo_segmentation():
    """使用内置样例演示标准切分逻辑"""
    raw_file = LORA_ROOT / "data" / "raw" / "sample_chapter.txt"
    if not raw_file.exists():
        print(f"[-] 错误：找不到样例文件 {raw_file}")
        sys.exit(1)

    print(f"[*] 正在读取样例章节: {raw_file}")
    with open(raw_file, "r", encoding="utf-8") as f:
        full_text = f.read()

    title = "第47章 生死契与倒栽葱"

    # 精准定位高潮爆发铁锚与收尾铁锚
    climax_start_anchor = "那一掌拍下的速度极快，但林舟抬手更快！"
    climax_end_anchor = "双腿抽搐了两下，便直挺挺地耷拉在了地面上，气绝身亡！"

    if climax_start_anchor not in full_text or climax_end_anchor not in full_text:
        print("[-] 错误：未在文章中匹配到锚点标识句子！")
        sys.exit(1)

    # 1. 切分 Ramp-up 前半段铺垫
    rampup_part = full_text.split(climax_start_anchor)[0].strip()

    # 2. 切分 Climax 高潮切片
    after_start = full_text[full_text.index(climax_start_anchor):]
    climax_part = after_start[: after_start.index(climax_end_anchor) + len(climax_end_anchor)].strip()

    # 3. 切分 Aftermath 尾声余波
    aftermath_part = full_text[full_text.index(climax_end_anchor) + len(climax_end_anchor):].strip()

    # 构建标准结构化数据
    sliced_data = {
        "chapter_id": "ch_0047",
        "chapter_title": title,
        "chapter_outline": [
            "林舟身中阴煞假装重伤，等待贪图筑基丹的敌人上门",
            "灵药阁赵管事深夜带人踹门强索丹药，以此借题发挥",
            "林舟反客为主诱使赵横签下生死契，绝境反杀一招秒杀",
            "搜身获得幕后黑手周少钦信函，内门金丹神识惊动，断章留钩"
        ],
        "truth_base": {
            "protagonist_plan": "利用阴煞之气掩饰实力并在体内炼出先天死气，诱敌立下生死契，合法将其格杀立威",
            "antagonist_horizon": "认定林舟不过是经脉断裂的残废散修，即便强行逼迫或击杀，也不过翻掌之间，贪图筑基丹以邀功",
            "information_gap": "赵横不知林舟非但没废，反而悟出先天死气；且以为林舟签生死契是自暴自弃，实为林舟的杀人执照",
            "deadline": "今夜不除赵横，丹药必被内门强行夺走；同时必须在内门金丹巡山巡查前解决战斗"
        },
        "ramp_up": {
            "opening_state": "林舟在柴房调息，身负暗伤，面对手持法旨强行踹门查房的赵管事",
            "target_anchor_sentence": climax_start_anchor,
            "text": rampup_part,
            "word_count": len(rampup_part)
        },
        "climax": {
            "core_conflict": "生死契成立后的雷霆反杀：避开刚猛掌风，点破天泉穴，倒栽葱砸碎青石地板",
            "climax_start": climax_start_anchor,
            "climax_end": climax_end_anchor,
            "text": climax_part,
            "word_count": len(climax_part)
        },
        "aftermath": {
            "from_anchor_sentence": climax_end_anchor,
            "hook": "搜出首座真传周少钦的谋害密函，同时内门首峰暴怒的金丹期神识正凌空横扫而来",
            "text": aftermath_part,
            "word_count": len(aftermath_part)
        }
    }

    # 保存切分成果
    output_dir = LORA_ROOT / "data" / "sliced"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "sample_chapter_sliced.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sliced_data, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print("【TARA 切分原型验证成功】")
    print(f"章节标题: {title}")
    print(f"总字数: {len(full_text)} 字")
    print(f"- Ramp-up (前半铺垫): {sliced_data['ramp_up']['word_count']} 字")
    print(f"- Climax (核心高潮): {sliced_data['climax']['word_count']} 字")
    print(f"- Aftermath (尾声余波): {sliced_data['aftermath']['word_count']} 字")
    print(f"铁锚 1 (Climax_Start): {climax_start_anchor}")
    print(f"铁锚 2 (Climax_End): {climax_end_anchor}")
    print(f"结果已写入: {out_file}")
    print("="*50 + "\n")
    return out_file


def run_api_segmentation(file_path: str, api_key: str, api_base: str, model: str):
    """通过 LLM API 逆向拆解任意真实章节"""
    path = Path(file_path)
    if not path.exists():
        print(f"[-] 文件不存在: {file_path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    title = path.stem
    print(f"[*] 正在调用 API ({model}) 逆向拆解章节: {title} ({len(content)} 字)...")

    prompt = REVERSE_EXTRACTION_USER_TEMPLATE.format(title=title, content=content)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": REVERSE_EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"}
    }

    url = api_base.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            result_json = res_data["choices"][0]["message"]["content"]
            parsed = json.loads(result_json)
            
            output_dir = LORA_ROOT / "data" / "sliced"
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{title}_sliced.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(parsed, out_f, ensure_ascii=False, indent=2)
            print(f"[+] 逆向拆解成功！已保存至: {out_file}")
    except Exception as e:
        print(f"[-] API 调用或解析失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="TARA 章节逆向拆解切分工具")
    parser.add_argument("--demo", action="store_true", help="运行内置高质量样例切分演示")
    parser.add_argument("--file", type=str, help="待切分的章节文本文件路径")
    parser.add_argument("--api-key", type=str, default=os.getenv("LLM_API_KEY", ""), help="LLM API Key")
    parser.add_argument("--api-base", type=str, default=os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1"), help="API Base URL")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="大模型名称")

    args = parser.parse_args()

    if args.demo:
        run_demo_segmentation()
    elif args.file:
        if not args.api_key:
            print("[-] 请提供 --api-key 或设置 LLM_API_KEY 环境变量！")
            sys.exit(1)
        run_api_segmentation(args.file, args.api_key, args.api_base, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
