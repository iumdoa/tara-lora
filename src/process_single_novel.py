#!/usr/bin/env python3
"""
TARA 单书全自动端到端处理引擎 (Single Novel Full Pipeline Runner)

功能：
1. 规范识别并分章（自动抓取 NCX 卷名与章节名）
2. 按大卷等距抽取 10% 代表性章节
3. 并发调用 DeepSeek-V4.1-Flash (无思考极速模式，max_tokens=8192) 提炼五段切片与大纲
4. 盲写 AI 初稿并构建 DPO 对齐样本
5. 编译 ShareGPT SFT 数据集
6. 自动聚合入全局网文多流派通用大库 (merge_datasets.py)
7. 将原书文件移入 processed_books/ 归档
"""

import os
import re
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
PROCESSED_DIR = LORA_ROOT / "processed_books"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_book_alias(filename: str) -> str:
    """根据中文书名猜测简洁的英文/拼音别名"""
    alias_map = {
        "斗破苍穹": "doupo",
        "诡秘之主": "guimi",
        "凡人修仙传": "fanren",
        "大王饶命": "dawang",
        "神秘复苏": "shenmi",
        "我师兄实在太稳健了": "wenjian",
        "我有一座冒险屋": "maoxianwu",
        "我真没想重生啊": "chongsheng",
        "绍宋": "shaosong",
        "大奉打更人": "dafeng",
        "盗墓笔记": "daomu",
        "苟在初圣魔门当人材": "momen",
    }
    for k, v in alias_map.items():
        if k in filename:
            return v
    # 兜底：取前两个汉字字符
    clean = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", "", filename)
    return clean[:8].lower() if clean else "book"


def run_cmd(cmd: list, desc: str):
    print("\n" + "=" * 70, flush=True)
    print(f"[*] 正在执行: {desc}", flush=True)
    print(f"[*] 命令: {' '.join(cmd)}", flush=True)
    print("=" * 70, flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"\n[-] 阶段执行失败: {desc} (code {res.returncode})", flush=True)
        sys.exit(res.returncode)


def update_tracks_registry(alias: str, book_filename: str, dataset_dir: Path):
    """自动将处理完成的书目与实际 SFT 产出同步到 config/tracks.json"""
    tracks_file = LORA_ROOT / "config" / "tracks.json"
    if not tracks_file.exists():
        return
    try:
        import json
        with open(tracks_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        sft_file = dataset_dir / "tara_train_dataset.json"
        sft_count = 0
        if sft_file.exists():
            with open(sft_file, "r", encoding="utf-8") as f:
                sft_count = len(json.load(f))

        clean_title = re.sub(r"\(.*?\)|\[.*?\]|（.*?）", "", book_filename).strip()
        clean_title = clean_title.replace(".epub", "").replace(".txt", "").strip()

        matched = False
        for track in cfg.get("tracks", []):
            all_books = track.get("all_15_books", [])
            for b in all_books:
                if clean_title in b or any(k in b for k in ["苟在初圣魔门当人材"] if k in clean_title):
                    assigned = track.setdefault("assigned_books", [])
                    found_existing = False
                    for ab in assigned:
                        if ab.get("alias") == alias or ab.get("title") == clean_title:
                            ab["sft_count"] = sft_count
                            found_existing = True
                            break
                    if not found_existing:
                        assigned.append({
                            "alias": alias,
                            "title": clean_title,
                            "sft_count": sft_count
                        })
                    track["current_sft"] = sum(ab.get("sft_count", 0) for ab in assigned)
                    assigned_titles = {ab["title"] for ab in assigned}
                    missing = [item for item in all_books if not any(at in item for at in assigned_titles)]
                    track["recommended_missing"] = missing[:4]
                    matched = True
                    break
            if matched:
                break

        if matched:
            with open(tracks_file, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
            print(f"[+] 成功同步更新赛道注册表: {tracks_file} (当前赛道有效 SFT 增至: {track['current_sft']})")
    except Exception as e:
        print(f"[!] 同步更新 tracks.json 提示: {e}")


def main():
    parser = argparse.ArgumentParser(description="TARA 单书全自动端到端处理引擎")
    parser.add_argument("book_file", type=str, help="待处理的电子书文件路径 (.epub)")
    parser.add_argument("--alias", type=str, default="", help="别名/存储目录名 (默认自动识别)")
    parser.add_argument("--ratio", type=float, default=0.1, help="分卷抽样比例 (默认 0.1)")
    parser.add_argument("--threads", type=int, default=5, help="并发线程数 (默认 5)")
    args = parser.parse_args()

    book_path = Path(args.book_file)
    if not book_path.exists():
        print(f"[-] 错误：找不到文件 {book_path}")
        sys.exit(1)

    alias = args.alias if args.alias else get_book_alias(book_path.name)
    book_raw_dir = LORA_ROOT / "data" / alias / "raw"
    book_sliced_dir = LORA_ROOT / "data" / alias / "sliced"
    book_dataset_dir = LORA_ROOT / "data" / alias / "dataset"

    print("\n" + "#" * 70)
    print(f"#  开始自动化处理小说: {book_path.name}")
    print(f"#  标识别名: {alias}")
    print(f"#  抽样比例: {args.ratio * 100:.0f}%")
    print(f"#  数据目录: data/{alias}/")
    print("#" * 70 + "\n")

    # 1. 拆解章节 (novel_splitter.py)
    cmd_split = [
        sys.executable, "-u", str(CURRENT_DIR / "novel_splitter.py"),
        str(book_path),
        "--output-dir", str(book_raw_dir)
    ]
    run_cmd(cmd_split, f"1/4 电子书智能分章 (EPUB TOC 解析)")

    # 2. 跑数据流水线 (切片 + DPO 盲写 + SFT 编译)
    cmd_pipeline = [
        sys.executable, "-u", str(CURRENT_DIR / "run_pipeline.py"),
        "--raw-dir", str(book_raw_dir),
        "--sliced-dir", str(book_sliced_dir),
        "--dataset-dir", str(book_dataset_dir),
        "--sample-by-volume", str(args.ratio),
        "--threads", str(args.threads)
    ]
    run_cmd(cmd_pipeline, f"2/4 10% 大卷抽样切片与 DPO/SFT 编译")

    # 3. 聚合进全局网文体系总库 (merge_datasets.py)
    cmd_merge = [sys.executable, "-u", str(CURRENT_DIR / "merge_datasets.py")]
    run_cmd(cmd_merge, f"3/4 自动聚合进全局网文多流派通用总库")

    # 同步更新赛道配置
    update_tracks_registry(alias, book_path.name, book_dataset_dir)

    # 4. 移动原书文件到 processed_books/
    dest_path = PROCESSED_DIR / book_path.name
    print(f"\n[*] 4/4 正在将已处理图书移动至归档目录: {dest_path}...")
    shutil.move(str(book_path), str(dest_path))
    print(f"[+] 归档完成！")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] 小说 《{book_path.name}》 处理并入库全流程圆满成功！")
    print(f"- 章节原始文本: {book_raw_dir}")
    print(f"- 五段切片结果: {book_sliced_dir}")
    print(f"- 该书独立数据集: {book_dataset_dir}")
    print(f"- 已归档文件: {dest_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
