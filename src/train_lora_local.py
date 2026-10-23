#!/usr/bin/env python3
"""
TARA (Three-Anchor Reverse Assembly) 本地极速微调脚本
协议版本：tara5_action_v1

基于 HuggingFace Transformers + PEFT (QLoRA / LoRA)
专为本地 8GB 显存显卡优化，默认支持 Qwen2.5-1.5B-Instruct。

安全保护规范：
- 不裁剪提示中间内容；
- 不截断回答尾部；
- 超长默认报错；
- --overlength skip 只在本次训练内存中过滤，不修改源数据文件；
- --check-only 不加载模型、不启动训练；
- 严格校验回答边界与可监督 token；
- 支持验证集评估：按章节组或书源严格划分，同章五段绝不跨界，避免上下文泄漏，报告 eval_loss。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

# 默认设置 HuggingFace 镜像端点，确保国内快速稳定下载
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 避免显存碎片化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent


class TaraShareGPTDataset(Dataset):
    """
    处理 TARA ShareGPT 数据集：
    - 对 System 和 User Prompt 部分做 Label Masking (-100)
    - 仅对 Assistant (纯正文) 计算 Loss，确保模型学到精准反向装配与锚点咬合
    - 绝不在训练中裁剪或截断
    """
    def __init__(
        self,
        data_or_path: Union[str, Path, List[Dict[str, Any]]],
        tokenizer,
        max_length: int = 2560,
        overlength: str = "error",
        split_name: str = "train",
    ):
        if max_length <= 0 or overlength not in ("error", "skip"):
            raise ValueError("max_length 必须为正数，overlength 必须是 error 或 skip")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split_name = split_name

        if isinstance(data_or_path, (str, Path)):
            with open(data_or_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        elif isinstance(data_or_path, list):
            raw_data = data_or_path
        else:
            raise ValueError("data_or_path 必须是文件路径或样本字典列表")

        if not isinstance(raw_data, list) or not raw_data:
            raise ValueError(f"[{split_name}] 数据集必须是非空样本数组")

        self.samples = []
        self.report = {
            "split": split_name,
            "total": len(raw_data),
            "kept": 0,
            "overlength": 0,
            "max_length": max_length,
            "longest": 0,
            "by_task": {},
            "overlength_examples": [],
        }

        # 一次性预检并冻结编码结果；不改原始数据，也不在训练中裁剪
        for index, item in enumerate(raw_data):
            if not isinstance(item, dict):
                raise ValueError(f"[{split_name}] 样本索引 {index} 不是对象")
            ident = item.get("id", str(index))
            conversations = item.get("conversations")
            if (
                not isinstance(conversations, list)
                or len(conversations) != 2
                or not all(isinstance(m, dict) for m in conversations)
                or [m.get("from") for m in conversations] != ["human", "gpt"]
                or not all(isinstance(m.get("value"), str) and m["value"].strip() for m in conversations)
            ):
                raise ValueError(f"[{split_name}] 样本 {ident}: 需要一组非空 human/gpt 对话，不能静默丢弃多轮内容")

            system_text = item.get("system", "")
            if not isinstance(system_text, str):
                raise ValueError(f"[{split_name}] 样本 {ident}: system 必须是字符串")

            messages = [{"role": "system", "content": system_text}] if system_text else []
            messages.append({"role": "user", "content": conversations[0]["value"]})
            prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_str = tokenizer.apply_chat_template(
                messages + [{"role": "assistant", "content": conversations[1]["value"]}],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full_str, add_special_tokens=False)["input_ids"]

            if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
                raise ValueError(f"[{split_name}] 样本 {ident}: 聊天模板的回答边界不一致或没有可监督 token")

            task = str(item.get("task_type", "unknown"))
            counts = self.report["by_task"].setdefault(task, {"total": 0, "kept": 0, "overlength": 0})
            counts["total"] += 1
            self.report["longest"] = max(self.report["longest"], len(full_ids))

            if len(full_ids) > max_length:
                counts["overlength"] += 1
                self.report["overlength"] += 1
                if len(self.report["overlength_examples"]) < 10:
                    self.report["overlength_examples"].append(
                        {"index": index, "id": ident, "tokens": len(full_ids)}
                    )
                continue

            self.samples.append({
                "input_ids": full_ids,
                "labels": [-100] * len(prompt_ids) + full_ids[len(prompt_ids):],
                "attention_mask": [1] * len(full_ids),
            })
            counts["kept"] += 1

        self.report["kept"] = len(self.samples)
        print(f"[*] 数据预检 [{split_name}]（原文件不修改；提示与回答均不截断）：")
        print(json.dumps(self.report, ensure_ascii=False, indent=2))

        if self.report["overlength"] and overlength == "error":
            raise ValueError(
                f"[{split_name}] 有 {self.report['overlength']} 条样本超过 max_length={max_length}；"
                "请按资源条件调整长度，或显式使用 --overlength skip 仅在本次训练内存中排除它们"
            )
        if not self.samples:
            raise ValueError(f"[{split_name}] 没有可用于训练的完整样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {key: list(value) for key, value in self.samples[idx].items()}


def split_dataset_by_chapter(
    raw_data: List[Dict[str, Any]],
    val_ratio: float = 0.1,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    按章节组或书源划分训练集与验证集：
    - 同一章节的五段样本绝对不跨界！保证全留在 train 或全留在 val；
    - 采用连续章节块划分，防止相邻章节上下文泄漏。
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        return raw_data, []

    # 聚合章节映射: chapter_key -> list of samples
    chapter_groups: Dict[str, List[Dict[str, Any]]] = {}
    for item in raw_data:
        ch_key = item.get("chapter_id")
        if not ch_key:
            # 兼容从 id 解析 (如 ch_0003_climax)
            s_id = str(item.get("id", ""))
            ch_key = s_id.rsplit("_", 1)[0] if "_" in s_id else s_id
        chapter_groups.setdefault(ch_key, []).append(item)

    unique_chapters = list(chapter_groups.keys())
    val_chapter_count = max(1, round(len(unique_chapters) * val_ratio))
    train_chapter_count = len(unique_chapters) - val_chapter_count

    if train_chapter_count <= 0:
        # 章节过少无法切分时，全部留给训练
        print("[!] 警告：章节总数过少，无法执行有效的独立章节验证集切分，保持纯训练模式。")
        return raw_data, []

    # 连续块划分（取末尾章节块作为验证集，避免随机打乱导致的相邻上下文穿透泄漏）
    train_keys = unique_chapters[:train_chapter_count]
    val_keys = unique_chapters[train_chapter_count:]

    train_data = []
    for k in train_keys:
        train_data.extend(chapter_groups[k])

    val_data = []
    for k in val_keys:
        val_data.extend(chapter_groups[k])

    print(
        f"[*] 章节组隔离切分完成: 训练集包含 {len(train_keys)} 章 ({len(train_data)} 样本), "
        f"验证集包含 {len(val_keys)} 章 ({len(val_data)} 样本)。同章五段绝对无跨界泄漏。"
    )
    return train_data, val_data


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    ratio = 100 * trainable_params / all_param
    print(f"[i] 可训练参数: {trainable_params:,} / {all_param:,} ({ratio:.2f}%)")


def train():
    parser = argparse.ArgumentParser(description="TARA 本地 LoRA 微调 (tara5_action_v1)")
    default_model = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_model).exists():
        default_model = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--model_name_or_path", type=str, default=default_model,
                        help="预训练基座模型 ID 或本地路径")
    parser.add_argument("--dataset_path", type=str,
                        default=str(LORA_ROOT / "data" / "dataset" / "tara_train_dataset.json"),
                        help="TARA 训练集 JSON 路径")
    parser.add_argument("--val_dataset_path", type=str, default="",
                        help="可选独立验证集路径 (若未指定，可通过 --val_ratio 按章节划分)")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="按章节组切分验证集的比例 (默认 0.0 即不切分；推荐 0.1)")
    parser.add_argument("--output_dir", type=str,
                        default=str(LORA_ROOT / "output" / "tara_lora_5"),
                        help="LoRA 适配器权重保存路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="单卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=2560,
                        help="完整样本的 token 上限；超长默认停止，不裁剪提示或回答")
    parser.add_argument("--overlength", choices=["error", "skip"], default="error",
                        help="超长样本策略：error 停止；skip 仅从本次训练内存集合中排除，不修改数据文件")
    parser.add_argument("--check-only", action="store_true",
                        help="只加载 tokenizer 并预检数据，不加载模型或启动训练")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="是否启用 4-bit 量化节省显存")
    args = parser.parse_args()

    print("=" * 60)
    print("       TARA-5 网文反向装配法 —— 本地微调与校验引擎       ")
    print("=" * 60)
    print(f"[*] 基座模型: {args.model_name_or_path}")
    print(f"[*] 数据集: {args.dataset_path}")
    if args.val_dataset_path:
        print(f"[*] 独立验证集: {args.val_dataset_path}")
    elif args.val_ratio > 0:
        print(f"[*] 章节组自动验证切分比例: {args.val_ratio * 100:.1f}%")
    print(f"[*] 输出目录: {args.output_dir}")
    print(f"[*] 训练轮次: {args.epochs} Epochs, 批大小: {args.batch_size} x {args.grad_accum}")
    print(f"[*] 显卡设备: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    # 1. 加载 Tokenizer
    print("[*] 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 构建训练集与验证集
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        raw_train_data = json.load(f)

    if args.val_dataset_path:
        with open(args.val_dataset_path, "r", encoding="utf-8") as vf:
            raw_val_data = json.load(vf)
        train_items = raw_train_data
        val_items = raw_val_data
    elif args.val_ratio > 0.0:
        train_items, val_items = split_dataset_by_chapter(raw_train_data, val_ratio=args.val_ratio)
    else:
        train_items = raw_train_data
        val_items = []

    # 在模型加载、显存分配及训练开始前检查全部样本
    train_dataset = TaraShareGPTDataset(
        train_items,
        tokenizer,
        max_length=args.max_length,
        overlength=args.overlength,
        split_name="train",
    )

    eval_dataset = None
    if val_items:
        eval_dataset = TaraShareGPTDataset(
            val_items,
            tokenizer,
            max_length=args.max_length,
            overlength=args.overlength,
            split_name="eval",
        )

    if args.check_only:
        print("[+] 只读预检完成，未加载模型或启动训练。")
        return

    # 3. 模型量化配置 (适配 8G 显存)
    device_map = "auto" if torch.cuda.is_available() else None
    bnb_config = None
    if args.use_4bit and torch.cuda.is_available():
        print("[*] 启用 4-bit NF4 量化模式以适配 8GB 显存...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    # 4. 加载基座模型
    print(f"[*] 正在加载基座模型 (来源: {args.model_name_or_path})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    if args.use_4bit and torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)

    # 5. 配置 LoRA
    print(f"[*] 注入 LoRA 适配器: rank={args.lora_rank}, alpha={args.lora_alpha}...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 6. 使用已预检的完整样本
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # 7. 配置训练参数
    has_eval = eval_dataset is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch" if has_eval else "no",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit" if (args.use_4bit and torch.cuda.is_available()) else "adamw_torch",
        report_to="none",
        dataloader_num_workers=0
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # 8. 启动训练
    print("\n[+] 开始执行训练...")
    train_result = trainer.train()

    eval_metrics = {}
    if has_eval:
        print("[*] 正在评估验证集损失 (Eval Loss)...")
        eval_metrics = trainer.evaluate()

    print("\n" + "=" * 60)
    print(f"[+] 训练顺利完成！")
    print(f"  - 最终训练损失 (Train Loss): {train_result.training_loss:.4f}")
    if has_eval and "eval_loss" in eval_metrics:
        print(f"  - 最终验证损失 (Eval Loss):  {eval_metrics['eval_loss']:.4f}")
    print(f"[*] 正在保存 LoRA 适配器权重至: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[+] 保存完成！可使用 src/infer_test.py 检验模型效果。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()
