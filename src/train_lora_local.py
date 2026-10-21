#!/usr/bin/env python3
"""
TARA (Three-Anchor Reverse Assembly) 本地极速微调脚本
基于 HuggingFace Transformers + PEFT (QLoRA / LoRA)
专为本地 8GB 显存显卡优化，默认支持 Qwen2.5-1.5B-Instruct。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

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
    - 仅对 Assistant (思考过程与正文) 计算 Loss，确保模型学到精准反向装配与锚点咬合
    """
    def __init__(self, data_path: str, tokenizer, max_length: int = 1536):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)
        print(f"[+] 成功加载训练集: {data_path}，总样本数: {len(self.raw_data)}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.raw_data[idx]
        system_text = item.get("system", "")
        conversations = item.get("conversations", [])
        
        user_text = ""
        assistant_text = ""
        for msg in conversations:
            if msg.get("from") == "human":
                user_text = msg.get("value", "")
            elif msg.get("from") == "gpt":
                assistant_text = msg.get("value", "")

        # 构造 Qwen ChatML 标准格式
        # Prompt 部分 (System + User)
        messages_prompt = []
        if system_text:
            messages_prompt.append({"role": "system", "content": system_text})
        messages_prompt.append({"role": "user", "content": user_text})

        # 包含 Assistant 的完整多轮对话
        messages_full = list(messages_prompt)
        messages_full.append({"role": "assistant", "content": assistant_text})

        prompt_str = self.tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_str = self.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_str, add_special_tokens=False)["input_ids"]

        # 截断超长样本
        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            if len(prompt_ids) >= self.max_length:
                prompt_ids = prompt_ids[:self.max_length - 64]

        # 构造 labels：Prompt 部分全部置为 -100 (不计梯度)
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        # 确保 labels 与 input_ids 等长
        if len(labels) < len(full_ids):
            labels.extend([-100] * (len(full_ids) - len(labels)))
        elif len(labels) > len(full_ids):
            labels = labels[:len(full_ids)]

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids)
        }


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
    parser = argparse.ArgumentParser(description="TARA 本地 LoRA 微调")
    default_model = str(LORA_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
    if not Path(default_model).exists():
        default_model = "Qwen/Qwen2.5-1.5B-Instruct"

    parser.add_argument("--model_name_or_path", type=str, default=default_model,
                        help="预训练基座模型 ID 或本地路径")
    parser.add_argument("--dataset_path", type=str,
                        default=str(LORA_ROOT / "data" / "dataset" / "tara_train_dataset.json"),
                        help="TARA 训练集 JSON 路径")
    parser.add_argument("--output_dir", type=str,
                        default=str(LORA_ROOT / "output" / "tara_lora_5"),
                        help="LoRA 适配器权重保存路径")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=1, help="单卡 batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=1024, help="单样本最大上下文长度")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA Rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA Alpha")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="是否启用 4-bit 量化节省显存")
    args = parser.parse_args()

    print("=" * 60)
    print("       TARA 网文三锚点反向装配法 —— 本地 Pilot 试训引擎       ")
    print("=" * 60)
    print(f"[*] 基座模型: {args.model_name_or_path}")
    print(f"[*] 数据集: {args.dataset_path}")
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

    # 2. 模型量化配置 (适配 8G 显存)
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

    # 3. 加载基座模型
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

    # 4. 配置 LoRA
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

    # 5. 加载数据集
    dataset = TaraShareGPTDataset(args.dataset_path, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )

    # 6. 配置训练参数
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
        train_dataset=dataset,
        data_collator=data_collator
    )

    # 7. 启动训练
    print("\n[+] 开始执行训练...")
    train_result = trainer.train()

    print("\n" + "=" * 60)
    print(f"[+] 训练顺利完成！最终 Loss: {train_result.training_loss:.4f}")
    print(f"[*] 正在保存 LoRA 适配器权重至: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[+] 保存完成！可使用 src/infer_test.py 检验模型咬合效果。")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train()
