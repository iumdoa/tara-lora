#!/usr/bin/env python3
"""
TARA-5 全量端到端自动化数据流水线 (End-to-End Pipeline Runner)

一键完成：
1. 五段式逆向切片与六铁锚原文校验 (segmenter.py)
2. AI 初稿盲写与 DPO 偏好对齐集构建 (contrast_builder.py)
3. 五段式 ShareGPT SFT 数据集编译 (format_converter.py)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))


def run_command_stream(cmd: list, desc: str):
    print("\n" + "=" * 70, flush=True)
    print(f"[*] 启动阶段: {desc}", flush=True)
    print(f"[*] 命令: {' '.join(cmd)}", flush=True)
    print("=" * 70, flush=True)
    start_time = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    cost = time.time() - start_time
    if proc.returncode != 0:
        print(f"\n[-] 阶段失败，退出码: {proc.returncode} (耗时: {cost:.1f}s)", flush=True)
        sys.exit(proc.returncode)
    print(f"\n[+] 阶段完成！耗时: {cost:.1f}s\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="TARA-5 全量自动化流水线")
    parser.add_argument("--raw-dir", type=str, default=str(LORA_ROOT / "data" / "raw"), help="原始章节目录")
    parser.add_argument("--sliced-dir", type=str, default=str(LORA_ROOT / "data" / "sliced"), help="切片输出目录")
    parser.add_argument("--dataset-dir", type=str, default=str(LORA_ROOT / "data" / "dataset"), help="数据集输出目录")
    parser.add_argument("--threads", type=int, default=5, help="API 并发调用线程数 (默认 5)")
    parser.add_argument("--limit", type=int, default=0, help="处理章节数量 (0为全量)")
    parser.add_argument("--sample-by-volume", type=float, default=0.0, help="按大卷抽样比例 (如 0.1 代表每卷抽取 10 百分比)")
    parser.add_argument("--skip-dpo", action="store_true", help="跳过 DPO 对比构建")
    parser.add_argument("--use-opencode", action="store_true", help="使用 OpenCode Go DeepSeek-V4.1-Flash (消耗 OpenCode 额度)")
    args = parser.parse_args()

    if args.use_opencode:
        os.environ["OPENCODE_USE_CLI"] = "1"

    raw_dir = Path(args.raw_dir)
    sliced_dir = Path(args.sliced_dir)
    dataset_dir = Path(args.dataset_dir)
    dpo_file = dataset_dir / "tara_contrast_dpo.json"

    # 1. 执行五段式逆向拆解
    cmd_seg = [
        sys.executable, "-u", str(CURRENT_DIR / "segmenter.py"),
        "--dir", str(raw_dir),
        "--output", str(sliced_dir),
        "--threads", str(args.threads)
    ]
    if args.sample_by_volume > 0.0:
        cmd_seg.extend(["--sample-by-volume", str(args.sample_by_volume)])
    elif args.limit > 0:
        cmd_seg.extend(["--limit", str(args.limit)])
    run_command_stream(cmd_seg, "1/3 五段式跨章闭环抽取与六铁锚原文校验")

    # 2. 执行 AI 盲写与 DPO 对比集构建
    if not args.skip_dpo:
        cmd_dpo = [
            sys.executable, "-u", str(CURRENT_DIR / "contrast_builder.py"),
            "--raw-dir", str(raw_dir),
            "--sliced-dir", str(sliced_dir),
            "--output", str(dpo_file),
            "--threads", str(args.threads)
        ]
        if args.limit > 0:
            cmd_dpo.extend(["--limit", str(args.limit)])
        run_command_stream(cmd_dpo, "2/3 DeepSeek-V4.1-Flash AI初稿盲写与 DPO 偏好对齐集构建")
    else:
        print("[*] 跳过 DPO 对比构建阶段", flush=True)

    # 3. 执行 SFT 数据集格式转换
    cmd_conv = [
        sys.executable, "-u", str(CURRENT_DIR / "format_converter.py"),
        "--sliced-dir", str(sliced_dir),
        "--dataset-dir", str(dataset_dir)
    ]
    run_command_stream(cmd_conv, "3/3 五段式多任务 SFT 训练集转换 (ShareGPT)")

    print("=" * 70)
    print("[+] TARA-5 数据准备全流程大获成功！")
    print(f"- 原始目录: {raw_dir}")
    print(f"- 切片目录: {sliced_dir}")
    print(f"- SFT 训练集: {dataset_dir / 'tara_train_dataset.json'}")
    if not args.skip_dpo:
        print(f"- DPO 对比集: {dpo_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
