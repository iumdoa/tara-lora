#!/usr/bin/env python3
"""
TARA-5 文风偏好对比数据集构建器 (AI Draft vs. Human Original Contrast Builder)

功能：
1. 读取 data/sliced/*.json 中的【章纲】与【上一章结尾】
2. 调用 DeepSeek API 盲写一版完整的 AI 初稿 (约 2000~2500 字，作为负样本 Rejected)
3. 提取原著真实人类作者正文 (作为正样本 Chosen)
4. 生成标准的 DPO 偏好对齐数据集: data/dataset/tara_contrast_dpo.json
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))

# 自动加载 .env 环境变量
env_path = LORA_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'").strip('"'))

from prompts.extraction_prompts import AI_DRAFT_USER_TEMPLATE


def call_deepseek_api(user_prompt: str, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-flash") -> str:
    """调用 DeepSeek API 生成 AI 初稿"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一位网络小说作家，请根据给定的章节大纲与上一章结尾承接，撰写本章核心正文（篇幅约 1500~2000 字）。"},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.8,
        "max_tokens": 2500,
        "thinking": {"type": "disabled"}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"DeepSeek 生成初稿失败: {e}")


def reconstruct_human_original(data: dict) -> str:
    """从五段切片中重构出人类原著的完整正文"""
    segments = data.get("segments", {})
    parts = []
    for k in ["opening", "ramp_up", "climax", "aftershock", "cliffhanger"]:
        seg = segments.get(k, {})
        text = seg.get("text", "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def process_contrast_for_chapter(json_path: Path, raw_dir: Path, api_key: str, base_url: str, model: str) -> dict:
    """为单章生成 AI 初稿并构建 DPO 对比对"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 如果已经生成过对比数据，直接读取
    contrast = data.get("contrast", {})
    if contrast.get("ai_draft") and contrast.get("human_original"):
        print(f"[+] 已存在对比草稿，跳过生成: {json_path.name}")
        return {
            "id": f"{data.get('chapter_id', 'ch')}_dpo",
            "prompt": contrast.get("prompt", ""),
            "chosen": contrast.get("human_original", ""),
            "rejected": contrast.get("ai_draft", "")
        }

    title = data.get("chapter_title", "")
    prev_ending = data.get("prev_chapter_ending", "（前情提要）")
    outline_list = data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    user_prompt = AI_DRAFT_USER_TEMPLATE.format(
        title=title,
        prev_ending=prev_ending,
        outline=outline_str
    )

    print(f"[*] 正在调用 API 盲写 AI 对比草稿: {title}...")
    ai_draft = ""
    for attempt in range(1, 4):
        try:
            ai_draft = call_deepseek_api(user_prompt, api_key, base_url, model)
            break
        except Exception as e:
            print(f"[-] 尝试 {attempt}/3 出错: {e}")
            time.sleep(3)

    if not ai_draft:
        print(f"[!] 无法生成 AI 草稿: {title}")
        return None

    # 获取人类原著正文
    raw_file = raw_dir / f"{json_path.stem.replace('_sliced', '')}.txt"
    if raw_file.exists():
        human_original = raw_file.read_text(encoding="utf-8").strip()
    else:
        human_original = reconstruct_human_original(data)

    # 回写到 sliced.json 留档持久化
    data["contrast"] = {
        "prompt": user_prompt,
        "ai_draft": ai_draft,
        "human_original": human_original,
        "ai_word_count": len(ai_draft),
        "human_word_count": len(human_original)
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] 成功构建对比对: {title} (AI草稿: {len(ai_draft)}字 vs 人类原著: {len(human_original)}字)")

    return {
        "id": f"{data.get('chapter_id', 'ch')}_dpo",
        "prompt": user_prompt,
        "chosen": human_original,
        "rejected": ai_draft
    }


def main():
    parser = argparse.ArgumentParser(description="TARA-5 AI与原著文风对比数据集构建器")
    parser.add_argument("--sliced-dir", type=str, default="data/sliced", help="切分文件目录")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="原始章节目录")
    parser.add_argument("--output", type=str, default="data/dataset/tara_contrast_dpo.json", help="输出 DPO 数据集路径")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""), help="DeepSeek API 密钥")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com/v1", help="API 地址")
    parser.add_argument("--model", type=str, default="deepseek-flash", help="生成草稿所用的模型 (默认 deepseek-flash 即 DeepSeek-V4.1-Flash)")
    parser.add_argument("--threads", type=int, default=5, help="并发线程数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理的章节数量 (0为全部)")
    args = parser.parse_args()

    if not args.api_key:
        print("[-] 错误：未指定 DeepSeek API Key！")
        print("    请设置环境变量 export DEEPSEEK_API_KEY='sk-...' 或通过 --api-key 传入。")
        sys.exit(1)

    sliced_dir = Path(args.sliced_dir)
    raw_dir = Path(args.raw_dir)
    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(sliced_dir.glob("*.json")))
    if args.limit > 0:
        json_files = json_files[:args.limit]
    if not json_files:
        print(f"[-] 在 {sliced_dir} 下未发现切片 JSON！请先运行 segmenter.py")
        sys.exit(1)

    print(f"[*] 开始为 {len(json_files)} 个章节批量生成 AI 初稿并构建 DPO 对比集 (线程数: {args.threads})...")

    dpo_samples = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(process_contrast_for_chapter, f, raw_dir, args.api_key, args.base_url, args.model): f
            for f in json_files
        }
        for future in as_completed(futures):
            res = future.result()
            if res:
                dpo_samples.append({
                    "conversations": [
                        {"from": "human", "value": res["prompt"]}
                    ],
                    "chosen": {"from": "gpt", "value": res["chosen"]},
                    "rejected": {"from": "gpt", "value": res["rejected"]}
                })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dpo_samples, f, ensure_ascii=False, indent=2)

    # 导出 DPO 的 dataset_info 配置片段
    snippet = {
        "tara_style_dpo": {
            "file_name": "tara_contrast_dpo.json",
            "ranking": True,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    }
    with open(out_file.parent / "dataset_info_dpo_snippet.json", "w", encoding="utf-8") as f:
        json.dump(snippet, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"[+] DPO 文风对比数据集构建完成！共生成 {len(dpo_samples)} 条样本")
    print(f"[+] DPO 数据文件: {out_file}")
    print(f"[+] 注册配置片段: {out_file.parent / 'dataset_info_dpo_snippet.json'}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
