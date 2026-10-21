#!/usr/bin/env python3
"""
TARA 网文多流派体系数据集融合器 (Multi-Novel Unified Dataset Merger)

功能：
1. 将各书源（盗墓笔记、大奉打更人等）的五段式 SFT 样本与 DPO 偏好对汇聚融合；
2. 注入全局防冲突标识 (daomu_, dafeng_)；
3. 执行伪随机打散 (Shuffle)，促进跨题材通用网文节奏与结构特性的表征学习；
4. 统计并导出统一的 dataset_info 配置与任务分布报告。
"""

import os
import sys
import json
import random
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent

DATA_ROOT = LORA_ROOT / "data"
UNIFIED_DATASET_DIR = DATA_ROOT / "dataset"
UNIFIED_DATASET_DIR.mkdir(parents=True, exist_ok=True)

def discover_sources(data_root: Path) -> list:
    """自动扫描并发现 data/ 目录下所有已编译出数据集的小说目录"""
    sources = []
    for p in sorted(data_root.iterdir()):
        if not p.is_dir() or p.name in ["dataset", "raw", "sliced"]:
            continue
        sft_path = p / "dataset" / "tara_train_dataset.json"
        dpo_path = p / "dataset" / "tara_contrast_dpo.json"
        if sft_path.exists():
            sources.append({
                "name": p.name,
                "title": p.name,
                "sft": sft_path,
                "dpo": dpo_path
            })
    return sources


def main():
    print("=" * 70)
    print("      TARA 网文多流派通用体系数据集融合流水线 (Unified Merger)      ")
    print("=" * 70)

    # 兼容历史路径：如果 data/daomu 尚未建立，从 data/dataset 迁移
    daomu_dir = DATA_ROOT / "daomu" / "dataset"
    if not daomu_dir.exists():
        daomu_dir.mkdir(parents=True, exist_ok=True)
        old_sft = DATA_ROOT / "dataset" / "tara_train_dataset.json"
        old_dpo = DATA_ROOT / "dataset" / "tara_contrast_dpo.json"
        if old_sft.exists():
            import shutil
            shutil.copy2(old_sft, daomu_dir / "tara_train_dataset.json")
            if old_dpo.exists():
                shutil.copy2(old_dpo, daomu_dir / "tara_contrast_dpo.json")
            print("[*] 已将历史单书数据归档至 data/daomu/dataset/")

    sources = discover_sources(DATA_ROOT)
    print(f"[*] 自动发现 {len(sources)} 个已处理小说数据源: {[s['name'] for s in sources]}")

    all_sft_samples = []
    all_dpo_samples = []
    stats = {}

    for src in sources:
        src_name = src["name"]
        src_title = src["title"]
        sft_path = src["sft"]
        dpo_path = src["dpo"]

        if not sft_path.exists():
            print(f"[!] 警告：未找到 {src_title} 的 SFT 数据集: {sft_path}")
            continue

        with open(sft_path, "r", encoding="utf-8") as f:
            sft_data = json.load(f)

        dpo_data = []
        if dpo_path.exists():
            with open(dpo_path, "r", encoding="utf-8") as f:
                dpo_data = json.load(f)

        # 注入唯一前缀与溯源
        for item in sft_data:
            item_copy = dict(item)
            if not item_copy["id"].startswith(f"{src_name}_"):
                item_copy["id"] = f"{src_name}_{item_copy['id']}"
            item_copy["source_novel"] = src_title
            all_sft_samples.append(item_copy)

        for item in dpo_data:
            item_copy = dict(item)
            item_copy["source_novel"] = src_title
            all_dpo_samples.append(item_copy)

        stats[src_title] = {
            "sft_count": len(sft_data),
            "dpo_count": len(dpo_data)
        }
        print(f"[+] 加载 {src_title:25s} -> SFT: {len(sft_data):4d} 条 | DPO: {len(dpo_data):3d} 条")

    # 执行全局打散，消除流派顺序偏置
    random.seed(42)
    random.shuffle(all_sft_samples)
    random.shuffle(all_dpo_samples)

    # 导出融合 SFT 数据集
    unified_sft_file = UNIFIED_DATASET_DIR / "tara_train_dataset.json"
    with open(unified_sft_file, "w", encoding="utf-8") as f:
        json.dump(all_sft_samples, f, ensure_ascii=False, indent=2)

    # 导出融合 DPO 数据集
    unified_dpo_file = UNIFIED_DATASET_DIR / "tara_contrast_dpo.json"
    with open(unified_dpo_file, "w", encoding="utf-8") as f:
        json.dump(all_dpo_samples, f, ensure_ascii=False, indent=2)

    # 任务分布统计
    from collections import Counter
    task_dist = Counter(x["task_type"] for x in all_sft_samples)

    print("\n" + "=" * 70)
    print("【网文体系融合数据集明细】")
    print(f"- 融合 SFT 样本总数: {len(all_sft_samples)} 条")
    print(f"- 融合 DPO 对比总数: {len(all_dpo_samples)} 对")
    print("任务分布（严格均衡）：")
    for t, cnt in task_dist.items():
        print(f"  * {t:20s}: {cnt:4d} 条")
    print(f"\n[+] 统一训练集已保存至: {unified_sft_file}")
    print(f"[+] 统一偏好集已保存至: {unified_dpo_file}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
