# TARA (Three-Anchor Reverse Assembly) 网文微调训练体系

本模块专为 **Mytha / Bitou Writing** 研发，旨在通过 LoRA 微调，让开源大语言模型（如 Qwen2.5-7B/14B）深度掌握 **“三锚点反向装配写作法”（TARA）**。

---

## 一、 核心痛点与解决哲学

### 传统大模型的自回归生成弊端
* **误差线性累积**：传统“从头写到尾”的生成方式是一个初值发散过程，模型在写到 1500 字时，注意力已严重稀释，导致“开头注水冗长、高潮草草收尾”。
* **无目标漂移**：因为终点模糊，模型在中途极易自由发挥、偏离主线，生成无法收回的垃圾情节。

### TARA 锚点反向装配哲学
将一个 2000~2500 字的章节拆解为 **双向有界定点问题（Boundary-Value Infilling）**：

1. **步骤 1：高潮核爆切片（Climax Anchor，先写约 600~800 字）**  
   - 输入：章纲 + 事实基座（Truth Base）。  
   - 产出：矛盾爆发的最巅峰，锁定两个关键铁锚：
     - `Climax_Start`：高潮爆发的第一句话。
     - `Climax_End`：高潮冲突结束的最后一句话。
2. **步骤 2：前向倒推咬合（Ramp-up Interpolation，写约 650~900 字）**  
   - 起点：章节初始状态。  
   - **终点强制约束**：文章末句必须严丝合缝咬入 `Climax_Start`！  
   - 效果：模型所有字句都只能为高潮做引信和蓄势，**从数学上杜绝放飞与灌水**。
3. **步骤 3：尾声余波推演（Aftermath & Cliffhanger，写约 400~600 字）**  
   - 起点：承接 `Climax_End`。  
   - 产出：路人与各方震动反响、战利品/秘密行动、引爆外部倒计时、章末留钩断章。
4. **步骤 4：零 Token 本地无缝拼装（Assembly）**  
   - 程序层直接无缝咬合拼装成完整单章。

---

## 二、 目录结构

```
lora/
├── README.md               # 方案与操作指引
├── schema/
│   └── tara_dataset_spec.json  # 结构化数据模式定义
├── prompts/
│   └── extraction_prompts.py   # 逆向章节拆解 Prompt 模板
├── src/
│   ├── novel_splitter.py       # 小说分章工具（支持 .epub 电子书与 .txt 全本）
│   ├── segmenter.py            # 章节逆向切分核心脚本（支持单章/批量 API 与本地演示）
│   ├── format_converter.py     # 转化为 LLaMA-Factory / ShareGPT 标准训练集
│   ├── train_lora_local.py     # 本地极速 LoRA/QLoRA 试训脚本 (适配 8GB 显存)
│   └── infer_test.py           # 微调效果推理验证与锚点咬合测试
└── data/
    ├── raw/                    # 原始章节文本输入
    ├── sliced/                 # 切分后的结构化 JSON 数据
    └── dataset/                # 最终用于 LoRA 微调的训练集
```

---

## 三、 微调任务分类（模型需掌握的三大技能）

每个训练章节将被拆解为以下三种 SFT/CoT 训练样本：

| 任务类型 | 指令目标 | 输入内容 | 期望输出 |
| :--- | :--- | :--- | :--- |
| **`tara_climax`** | 锻造高潮切片 | 章纲要求 + 事实基座卡 (主角算盘/信息差/限制) | `<think>`思考推演`</think>` + 高潮正文（含明确首尾锚句） |
| **`tara_rampup`** | 前向倒推咬合 | 章节开局起点 + **目标咬合首句 (`Climax_Start`)** + 预算字数 | `<think>`铺垫与情绪蓄水推演`</think>` + 推进正文（末句直达锚点） |
| **`tara_aftermath`** | 承接余波断章 | 高潮末句 (`Climax_End`) + 本章留钩目标 + 预算字数 | `<think>`反应烘托与悬念推演`</think>` + 尾声正文（末句卡章） |

---

## 四、 快速上手与全流程命令

不需要第三方依赖，纯 Python 3 标准库即可运行：

```bash
cd /home/iumdoa/Project/lora

# 1. 如果有整本小说 (.epub 或 .txt)，自动分章提取纯文本到 data/raw/
python3 src/novel_splitter.py /path/to/novel.epub
# 或者：python3 src/novel_splitter.py /path/to/novel.txt

# 2. 逆向结构化拆解
# 方式 A：体验内置高质量玄幻单章示例
python3 src/segmenter.py --demo

# 方式 B：指定 API 批量拆解 data/raw/ 下提取出的所有章节
export LLM_API_KEY="your-api-key"
python3 src/segmenter.py --dir data/raw/ --api-key $LLM_API_KEY --model deepseek-chat

# 3. 将切分好的结构化数据转换为 LLaMA-Factory 标准训练集
python3 src/format_converter.py

# 4. 查看生成的训练样本
cat data/dataset/tara_train_dataset.json | head -n 40
```

---

## 五、 本地极速微调与效果验证 (Pilot Run)

针对本地 8GB 显存显卡（如 Quadro RTX 4000 / RTX 3070 / RTX 4060），使用 4-bit QLoRA 仅需 4GB 显存，10~15 分钟即可跑完 210 条样本：

```bash
# 1. 启动本地 LoRA 微调 (默认基座为 Qwen2.5-1.5B-Instruct)
.venv/bin/python src/train_lora_local.py \
  --epochs 3 \
  --batch_size 1 \
  --grad_accum 8 \
  --learning_rate 2e-4

# 2. 检验模型锚点咬合效果 (测试 tara_rampup 终点咬合)
.venv/bin/python src/infer_test.py --task rampup

# 3. 检验模型高潮爆发切片 (测试 tara_climax)
.venv/bin/python src/infer_test.py --task climax
```

---

## 六、 云端算力训练方案 (LLaMA-Factory 7B/14B)

当本地验证数据闭环与格式收敛无误后，可将生成的训练集上传至云端（如 AutoDL / 阿里云 PAI 单卡 4090），执行全量工业级训练：

1. 注册数据集：将 `data/dataset/dataset_info_snippet.json` 内容追加至 LLaMA-Factory 的 `dataset_info.json`。
2. 启动训练：
```bash
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset tara_webnovel_sft \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir saves/qwen7b_tara_lora \
    --overwrite_output_dir \
    --cutoff_len 1536 \
    --learning_rate 2e-4 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 50 \
    --warmup_ratio 0.05 \
    --fp16 True
```

