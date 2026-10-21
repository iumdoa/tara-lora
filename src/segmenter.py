#!/usr/bin/env python3
"""
TARA-5 (Five-Stage Anchor & Cross-Chapter Linkage) 章节逆向切分引擎

核心功能：
1. 自动关联上一章结尾 (prev_chapter_ending)，构建连贯破题切片
2. 调用 DeepSeek API 执行五段式 (三核双桥) 逆向拆解
3. 严格执行六铁锚原文子串校验 (Verbatim Substring Assert) 与自闭合校验
4. 多线程高并发批量处理 data/raw/ 下全部章节
"""

import os
import re
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

from prompts.extraction_prompts import (
    REVERSE_EXTRACTION_SYSTEM_PROMPT,
    REVERSE_EXTRACTION_USER_TEMPLATE,
)


def call_deepseek_api(system_prompt: str, user_prompt: str, api_key: str, base_url: str = "https://api.deepseek.com/v1", model: str = "deepseek-flash") -> str:
    """调用兼容 OpenAI 规范的 DeepSeek API"""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 8192,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"}
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        raise RuntimeError(f"API HTTP Error {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"API Request Failed: {e}")


def clean_json_response(raw_resp: str) -> dict:
    """清理 API 返回并解析 JSON"""
    text = raw_resp.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def extract_chapter_number(filename: str) -> int:
    """从文件名如 ch_0003_xxx.txt 中提取序号"""
    m = re.search(r"ch_(\d+)", filename)
    if m:
        return int(m.group(1))
    return 999999


def get_prev_chapter_ending(raw_dir: Path, current_idx: int) -> str:
    """自动获取前置正文章节末尾的 200 字断章作为本章破题衔接的依据"""
    for prev_idx in range(current_idx - 1, 0, -1):
        prev_pattern = f"ch_{prev_idx:04d}_*.txt"
        matches = list(raw_dir.glob(prev_pattern))
        if not matches:
            matches = [f for f in raw_dir.glob("*.txt") if extract_chapter_number(f.name) == prev_idx]
        
        if matches:
            f = matches[0]
            # 排除非正文设定集或说明资料
            if any(k in f.name for k in ["简介", "制作说明", "目录", "资料", "地形篇", "明器", "怪物"]):
                continue
            text = f.read_text(encoding="utf-8").strip()
            # 必须是具有正文字数的真实章节
            if len(text) >= 800:
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                clean_text = "\n".join(lines)
                return clean_text[-250:].strip()

    return "（本书/本卷开门见山第一章，直接切入核心动态）"


def get_volume_sampled_files(raw_dir: Path, ratio: float = 0.1) -> list:
    """按各大分卷等距均匀抽取指定比例的代表性章节"""
    files = sorted(list(raw_dir.glob("*.txt")), key=lambda p: extract_chapter_number(p.name))
    volumes = {}
    current_vol = "第一卷"
    for f in files:
        m = re.search(r"ch_(\d+)_(.*)\.txt", f.name)
        if not m:
            continue
        title = m.group(2)
        if any(k in title for k in ["简介", "资料", "目录", "地形篇", "明器", "怪物", "制作说明"]):
            continue

        # 模式 A: 文件名直接带有分卷前缀，如 "第一卷 京察风云_第一章 牢狱之灾"
        if "_" in title and any(k in title.split("_")[0] for k in ["卷", "部", "篇"]):
            file_vol = title.split("_")[0]
        else:
            # 模式 B: 独立的卷首纯标题文件
            if any(k in title for k in ["卷", "部", "篇"]) and len(f.read_text(encoding="utf-8").strip()) < 800:
                current_vol = title
                continue
            file_vol = current_vol

        if len(f.read_text(encoding="utf-8").strip()) < 800:
            continue

        if file_vol not in volumes:
            volumes[file_vol] = []
        volumes[file_vol].append(f)

    sampled = []
    print("\n" + "=" * 60)
    print("【各大分卷抽样明细】")
    for vol, ch_list in volumes.items():
        k = max(1, round(len(ch_list) * ratio))
        step = len(ch_list) / k
        indices = [min(int(i * step + step / 2), len(ch_list) - 1) for i in range(k)]
        selected = [ch_list[idx] for idx in sorted(list(set(indices)))]
        sampled.extend(selected)
        print(f"  * {vol:35s}: 共 {len(ch_list):3d} 章 -> 抽取 {len(selected):2d} 章")
    print("=" * 60 + "\n")

    return sorted(sampled, key=lambda p: extract_chapter_number(p.name))


def validate_and_refine_segments(data: dict, full_text: str) -> bool:
    """严格校验所有 6 个锚点在原文中的精确匹配，并执行自闭合咬合校验"""
    segments = data.get("segments", {})
    required_keys = ["opening", "ramp_up", "climax", "aftershock", "cliffhanger"]
    for k in required_keys:
        if k not in segments:
            print(f"[-] 缺少段落: {k}")
            return False

    op = segments["opening"]
    ru = segments["ramp_up"]
    cl = segments["climax"]
    af = segments["aftershock"]
    ch = segments["cliffhanger"]

    # 1. 校验原文精确存在
    anchors = [
        ("opening.start", op.get("anchor_start", "")),
        ("opening.end", op.get("anchor_end", "")),
        ("climax.start", cl.get("anchor_start", "")),
        ("climax.end", cl.get("anchor_end", "")),
        ("cliffhanger.start", ch.get("anchor_start", "")),
        ("cliffhanger.end", ch.get("anchor_end", "")),
    ]

    for name, anchor in anchors:
        if not anchor or anchor not in full_text:
            print(f"[-] 锚点校验失败: {name} 未在原文中完全匹配！内容: {anchor[:40]}...")
            return False

    # 2. 确保咬合一致性
    ru["anchor_end"] = cl["anchor_start"]
    af["anchor_end"] = ch["anchor_start"]

    # 3. 确保正文末尾自闭合
    if not ru.get("text", "").strip().endswith(cl["anchor_start"]):
        ru["text"] = f"{ru.get('text', '').strip()}\n\n{cl['anchor_start']}"
    if not af.get("text", "").strip().endswith(ch["anchor_start"]):
        af["text"] = f"{af.get('text', '').strip()}\n\n{ch['anchor_start']}"
    if not ch.get("text", "").strip().endswith(ch["anchor_end"]):
        ch["text"] = f"{ch.get('text', '').strip()}\n\n{ch['anchor_end']}"

    # 4. 统计字数
    for k in required_keys:
        segments[k]["word_count"] = len(segments[k].get("text", ""))

    return True


def process_single_chapter(raw_file: Path, raw_dir: Path, output_dir: Path, api_key: str, base_url: str, model: str) -> bool:
    """处理单个章节的五段式逆向抽取"""
    full_text = raw_file.read_text(encoding="utf-8").strip()
    current_idx = extract_chapter_number(raw_file.name)
    title = raw_file.stem.split("_", 2)[-1] if "_" in raw_file.stem else raw_file.stem
    prev_ending = get_prev_chapter_ending(raw_dir, current_idx)

    # 过滤设定集、资料与卷首导读
    if any(k in raw_file.name for k in ["简介", "资料", "目录", "地形篇", "明器", "怪物"]):
        print(f"[!] 跳过设定集/非剧情章节: {raw_file.name}")
        return False

    # 过于简短的章节跳过（如纯卷头标语等）
    if len(full_text) < 800:
        print(f"[!] 跳过超短文件: {raw_file.name} (字数: {len(full_text)})")
        return False

    out_file = output_dir / f"{raw_file.stem}_sliced.json"
    if out_file.exists():
        print(f"[+] 已经存在切片文件，跳过: {out_file.name}")
        return True

    print(f"[*] 正在调用 API 拆解五段式: {raw_file.name} (承接上章尾: {prev_ending[:30]}...)")
    user_prompt = REVERSE_EXTRACTION_USER_TEMPLATE.format(
        title=title,
        prev_ending=prev_ending,
        content=full_text
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            raw_resp = call_deepseek_api(REVERSE_EXTRACTION_SYSTEM_PROMPT, user_prompt, api_key, base_url, model)
            data = clean_json_response(raw_resp)
            data["chapter_id"] = f"ch_{current_idx:04d}"
            data["chapter_index"] = current_idx
            data["prev_chapter_ending"] = prev_ending

            if validate_and_refine_segments(data, full_text):
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"[OK] 成功生成五段闭环切片: {out_file.name}")
                return True
            else:
                print(f"[!] 尝试 {attempt}/{max_retries} 锚点比对未通过，正在重试...")
        except Exception as e:
            print(f"[-] 尝试 {attempt}/{max_retries} 发生异常: {e}")
            time.sleep(2)

    print(f"[FAILED] 章节拆解失败: {raw_file.name}")
    return False


def main():
    parser = argparse.ArgumentParser(description="TARA-5 五段式跨章闭环抽取引擎")
    parser.add_argument("--dir", type=str, default="data/raw", help="原始章节文本所在目录")
    parser.add_argument("--output", type=str, default="data/sliced", help="切分结果保存目录")
    parser.add_argument("--api-key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", ""), help="DeepSeek API 密钥")
    parser.add_argument("--base-url", type=str, default="https://api.deepseek.com/v1", help="API 基础地址")
    parser.add_argument("--model", type=str, default="deepseek-flash", help="模型名称 (默认 deepseek-flash 即 DeepSeek-V4.1-Flash)")
    parser.add_argument("--threads", type=int, default=5, help="并发线程数")
    parser.add_argument("--limit", type=int, default=0, help="限制处理的章节数量 (0为全部)")
    parser.add_argument("--sample-by-volume", type=float, default=0.0, help="按大卷抽样比例 (如 0.1 代表每卷抽取 10 百分比)")
    args = parser.parse_args()

    raw_dir = Path(args.dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.api_key:
        print("[-] 错误：未指定 DeepSeek API Key！")
        print("    请设置环境变量 export DEEPSEEK_API_KEY='sk-...' 或通过 --api-key 传入。")
        sys.exit(1)

    if args.sample_by_volume > 0.0:
        all_files = get_volume_sampled_files(raw_dir, args.sample_by_volume)
        print(f"[*] 启用大卷等距抽样 (比例 {args.sample_by_volume*100:.0f}%)，共精选出 {len(all_files)} 个代表性章节")
    else:
        all_files = sorted(list(raw_dir.glob("*.txt")), key=lambda p: extract_chapter_number(p.name))
        if args.limit > 0:
            all_files = all_files[:args.limit]
        print(f"[*] 共扫描到 {len(all_files)} 个章节文本，启动并发处理 (线程数: {args.threads})...")

    success_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(process_single_chapter, f, raw_dir, output_dir, args.api_key, args.base_url, args.model): f
            for f in all_files
        }
        for future in as_completed(futures):
            f = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as e:
                print(f"[-] 处理文件 {f.name} 时产生致命错误: {e}")

    print("\n" + "=" * 60)
    print(f"[+] 批量处理完成！成功切分 {success_count} / {len(all_files)} 个章节")
    print(f"[+] 结构化切片已保存至: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
