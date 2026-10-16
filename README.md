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
│   ├── segmenter.py            # 章节逆向切分核心脚本（支持 API 与本地离线演示）
│   └── format_converter.py     # 转化为 LLaMA-Factory / ShareGPT 标准训练集
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

## 四、 快速上手与演示

不需要额外依赖，纯 Python 3 标准库即可运行：

```bash
cd /home/iumdoa/Projects/lora

# 1. 运行切分原型验证（内置高品质玄幻示例单章）
python3 src/segmenter.py --demo

# 2. 将切分好的结构化数据转换为 LLaMA-Factory 标准训练集
python3 src/format_converter.py

# 3. 查看生成的训练样本
cat data/dataset/tara_train_dataset.json | head -n 40
```
