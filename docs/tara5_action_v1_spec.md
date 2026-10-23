# TARA-5 行动与状态驱动反向装配数据协议规范 (tara5_action_v1)

> **版本**：`tara5_action_v1`  
> **状态**：正式发布 (Ready for Pilot & Mytha Integration)  
> **适用范围**：`lora` 微调训练引擎、Mytha 文学工程规划引擎、章节反向装配生成器

---

## 一、架构不可逾越之核心红线

1. **反向推演生成，阅读顺序装配**：
   - **正文推理推演顺序**：
     $$\text{高潮 (climax)} \longrightarrow \text{结尾 (cliffhanger)} \longrightarrow \text{开篇 (opening)} \longrightarrow \text{蓄水桥 (ramp\_up)} \longrightarrow \text{余波桥 (aftershock)}$$
   - **最终读者阅读装配顺序**：
     $$\text{开篇 (opening)} \longrightarrow \text{蓄水桥 (ramp\_up)} \longrightarrow \text{高潮 (climax)} \longrightarrow \text{余波桥 (aftershock)} \longrightarrow \text{结尾 (cliffhanger)}$$

2. **引擎职责严格正交划分**：
   - **Mytha 文学工程引擎**：负责已确认事实 (`confirmed_facts`)、稳定角色视界与信息差 (`characters`)、完整总章纲 (`chapter_outline`)、阶段行动链与起止态势 (`stage_plans`)、创作约束边界 (`constraints`) 与篇幅预算 (`length_budget`)。
   - **微调 Writer 大模型**：负责将指定事件与动作表现写成生动自然的小说正文（措辞表达、对白潜台词、物理微动作、视听感官与节奏张力）。
   - **红线禁令**：Writer **严禁重新规划故事走向**，**严禁擅自新增未经设定的关键角色**，**严禁颠覆胜负或改变既定因果**。文学允许自由描写，但绝不能把正文训练成机械复述提纲。
   - **输出纯净性**：Writer 的输出**仅为纯小说自然正文**，新版本彻底移除合成 `<think>` 标签及程序控制参数。

3. **反向生成信息严格单向隔离**：
   - **高潮 (climax)**：仅依赖总章纲相关行动、已确认事实、人物认知、高潮起止态势；**严禁依赖尚未生成的蓄水正文**。
   - **结尾 (cliffhanger)**：仅依赖总章纲结局态势、结尾行动链、悬念卡点目标；**严禁依赖尚未生成的余波正文**。
   - **开篇 (opening)**：依赖上章结尾承接、本章开局初始态势及开篇行动链。
   - **蓄水桥 (ramp_up)**：提供已生成开篇的必要上下文、开篇末锚、高潮首锚、蓄水行动链与起止态势。
   - **余波桥 (aftershock)**：提供已生成高潮的必要上下文、高潮末锚、断章首锚、余波行动链与起止态势。

---

## 二、标准阶段命名与锚点归属规范

### 1. 阶段名称规范
全工程统一使用下划线命名，严禁混用驼峰或连写：
- `opening`（黄金破题切片）
- `ramp_up`（前向蓄水引信桥，彻底废弃旧版 `rampup`）
- `climax`（高潮核爆切片）
- `aftershock`（余波转折中继桥）
- `cliffhanger`（致命断章切片）

### 2. 接缝锚点归属原则 (Prev-Retained, Next-Stripped)
| 接缝名称 | 前置阶段 (Prev) | 后置阶段 (Next) | 共享锚点 Key | 归属与去重决策 |
| :--- | :--- | :--- | :--- | :--- |
| **Junction 1** | `opening` | `ramp_up` | `anchor_2` | `opening` 尾部完整保留；`ramp_up` 头部精准剥除重复字符 |
| **Junction 2** | `ramp_up` | `climax` | `anchor_3` | `ramp_up` 尾部完整保留；`climax` 头部精准剥除重复字符 |
| **Junction 3** | `climax` | `aftershock` | `anchor_4` | `climax` 尾部完整保留；`aftershock` 头部精准剥除重复字符 |
| **Junction 4** | `aftershock` | `cliffhanger` | `anchor_5` | `aftershock` 尾部完整保留；`cliffhanger` 头部精准剥除重复字符 |

> [!IMPORTANT]
> **精准剥离禁令**：去重算法仅允许对首部精准匹配的锚点字符序列进行切片截断，**严禁模糊匹配后直接整行丢弃整段正文**（防止同一行中后续非锚点正文被误删）；若前段末尾遗漏锚句而后段具备，系统强制触发【防丢句保护】，保留后段头部，杜绝剧情丢失。

---

## 三、章节级 JSON 数据协议定义

协议定义于 `schema/tara5_action_v1_schema.json`。

### 核心字段清单
```json
{
  "schema_version": "tara5_action_v1",
  "chapter_id": "ch_0001",
  "source_novel": "daomu",
  "source_info": {
    "book_title": "盗墓笔记",
    "author": "南派三叔",
    "volume_title": "七星鲁王宫",
    "chapter_index": 1,
    "raw_file_name": "ch_0003_第一章 血尸_sliced.json",
    "raw_text_char_count": 3117
  },
  "chapter_title": "第一章 血尸",
  "prev_chapter_ending": "上一章末尾文本...",
  "chapter_outline": [
    "1. 阅读顺序事件一...",
    "2. 阅读顺序事件二..."
  ],
  "confirmed_facts": [
    {
      "fact": "本章开始前已确凿发生/已知的事实",
      "evidence": "出处依据片段或前章伏笔"
    }
  ],
  "characters": [
    {
      "character_id": "char_sanya",
      "name": "三伢子",
      "goal": "即时行动算盘",
      "known_info": ["已知关键情报"],
      "unknown_info": ["认知盲区"],
      "misjudgment": "主观误判"
    }
  ],
  "stage_plans": {
    "opening": { "$ref": "#/definitions/stage_plan" },
    "ramp_up": { "$ref": "#/definitions/stage_plan" },
    "climax": { "$ref": "#/definitions/stage_plan" },
    "aftershock": { "$ref": "#/definitions/stage_plan" },
    "cliffhanger": { "$ref": "#/definitions/stage_plan" }
  },
  "constraints": {
    "pov": "第三人称限制视角",
    "author_hard_requirements": ["作者硬性风格规范"],
    "negative_constraints": ["严禁擅自新增未经设定的关键角色", "严禁篡改胜负结局"],
    "allowed_creative_freedom": ["动作微细节", "生理反应白描", "气氛渲染"]
  },
  "segments": {
    "opening": {
      "anchor_start": "该阶段首句精确原文",
      "anchor_end": "该阶段末句精确原文",
      "text": "阶段自然正文",
      "char_count": 103
    },
    "ramp_up": { "...": "..." },
    "climax": { "...": "..." },
    "aftershock": { "...": "..." },
    "cliffhanger": { "...": "..." }
  },
  "annotation_status": {
    "needs_review": true,
    "direct_evidence_fields": ["segments.*.text", "chapter_title", "chapter_outline"],
    "review_items": [
      {
        "field": "stage_plans.*.action_chain",
        "reason": "规则迁移推断生成，需人工核对细粒度动作依赖"
      }
    ],
    "annotator": "rule_migrated_v1"
  }
}
```

### 行动单元 (`action_unit`) 规范
每个阶段的 `action_chain` 细化为可执行行动单元：
- `action_id`: 单元唯一标识 (如 `act_cl_001`)
- `actor_id`: 行动主体角色 ID
- `trigger`: 触发条件或时机
- `action`: 具体物理行动描述
- `reason`: 行动背后的动机或认知依据
- `reaction`: 其他角色或环境的即时反馈
- `result`: 该行动造成的直接结果
- `depends_on`: 依赖的前序 `action_id` 列表
- `evidence`: 原文依据出处

---

## 四、五大装配任务输入构造规则与示例

训练集 ShareGPT 编译与在线推理 100% 调用 `src/prompt_builder.py` 中的 `build_task_input(stage, chapter_data, ...)`。

### 1. 高潮核爆 (`tara5_climax`)
- **System Prompt**：
  > 你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 高潮核爆专家）。你的职责是跳过铺垫前戏，直接围绕指定的高潮核心行动链撰写爆发力最强、冲突最集中、情绪最饱满的小说正文（约 600~900 字）。着重刻画高频动作、生理对抗、心理博弈与局势瞬间逆转。【严格边界】严格遵守既定胜负与态势结果，不得新增非计划关键角色或无端篡改设定；正文首句与末句需清晰锁定高潮爆发与定局点。
- **User Prompt 包含字段**：
  - `【章节标题】`
  - `【故事总章纲】`
  - `【已确认事实基座】`
  - `【核心角色认知与动机】`（含角色目标、盲区与误判）
  - `【本阶段文学职责】`
  - `【起始前置态势】`
  - `【核心行动执行链】`
  - `【终点定局态势】`
  - `【目标达成结果】`
  - `【篇幅预算】`
  - `【创作约束与边界】`
  - `【任务指示】`

### 2. 致命断章 (`tara5_cliffhanger`)
- **System Prompt**：
  > 你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 致命断章专家）。你的职责是紧扣本章结局态势与终局悬念目标，在 150~300 字内引爆全新维度异变或心理危机，在矛盾最高峰处戛然而止完成绝杀卡点。最后一句话必须一刀封喉，制造令读者不可抗拒的翻页渴望。【严格边界】直接切入终局危机，杜绝说明性交代或大段背景复述；禁止改变本章既定走向。
- **User Prompt 包含字段**：
  - 聚焦终局死线与全新异变目标，完全隔离未生成的余波正文。

### 3. 黄金破题 (`tara5_opening`)
- **System Prompt**：
  > 你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 开篇破题专家）。你的职责是将指定的开篇行动计划与开局状态转化为极具吸引力的小说正文（约 150~300 字）。紧扣上章余势顺畅起笔或通过强动作反差开门见山，杜绝空洞环境闲笔。【严格边界】专注于词句表现、人物微动作与即时氛围；禁止擅自改动已确认事实或提前展开后续高潮事件。
- **User Prompt 包含字段**：
  - `【上一章结尾承接】`、开篇态势与破题行动链。

### 4. 前向蓄水引信桥 (`tara5_ramp_up`)
- **System Prompt**：
  > 你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 蓄水引信桥专家）。你的职责是从【破题定势铁锚】出发，把承上启下的蓄水行动链推演为层层施压、暗流涌动的小说正文（约 600~900 字）。【严格边界】正文必须严丝合缝由指定的【终点铁锚 (高潮爆发句)】落笔定调收尾；专注于矛盾升级与人物博弈，禁止提前抢写高潮爆发核心事件。
- **User Prompt 包含字段**：
  - 前置破题上下文、`【起始铁锚】`（开篇末句）、`【终点铁锚】`（高潮爆发首句）、蓄水行动链与咬合约束。

### 5. 余波转折中继桥 (`tara5_aftershock`)
- **System Prompt**：
  > 你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 余波转折桥专家）。你的职责是从【高潮定势末锚】出发，将战后余波、生理颤抖、战利品复盘与虚假安全感转化为张弛有度的小说正文（约 400~600 字）。【严格边界】正文末尾必须顺理成章、完全咬入指定的【终点铁锚 (断章引爆句)】；专注于战局消化与中继过渡，禁止重复回放高潮对决。
- **User Prompt 包含字段**：
  - 高潮战果上下文、`【起始铁锚】`（高潮落定末句）、`【终点铁锚】`（断章引爆首句）、余波行动链与咬合约束。

---

## 五、旧数据无损迁移与批量命令

### 1. 迁移工具
`src/migrate_to_tara5_action_v1.py`

### 2. 保护原则
- **绝不原地覆盖**任何旧版切片或数据集；
- 所有新协议产物输出至 `data/tara5_action_v1/{novel}/...`；
- 自动标记 `needs_review: true`，记录迁移原因与待复核字段。

### 3. 操作命令
```bash
# 小样本迁移验证 (处理前 5 章并同步导出 ShareGPT 数据集)
.venv/bin/python src/migrate_to_tara5_action_v1.py \
    --src-dir data/sliced \
    --novel daomu \
    --limit 5 \
    --export-dataset

# 校验生成的切片文件合规性
.venv/bin/python -c "from src.schema_validator import validate_file; from pathlib import Path; print(validate_file(Path('data/tara5_action_v1/daomu/sliced/ch_0003_第一章 血尸_tara5_v1.json')))"

# 全量安全批量迁移 (示例: 将某书目全量迁移至独立版本目录)
.venv/bin/python src/migrate_to_tara5_action_v1.py \
    --src-dir data/daomu/sliced \
    --novel daomu \
    --export-dataset
```

---

## 六、验证与回归测试总结

### 1. 单元测试套件覆盖
测试文件：`tests/test_tara5_action_v1.py`
1. `test_01_schema_validation`: 验证合规样本通过，缺失必填项、枚举非法、预算颠倒等违规项精准报错拦截。
2. `test_02_train_vs_infer_prompt_consistency`: 严格证明 5 个阶段在训练集与推理装配调用时生成的 System Prompt 与 User Prompt 100% 相同。
3. `test_03_assembly_anchor_ownership_and_deduplication`: 验证 Prev-Retained, Next-Stripped 原则生效，整章锚点出现且仅出现一次，同行后续正文完好保留无截断。
4. `test_04_dataset_split_no_leakage`: 验证按章节组隔离划分验证集时，同章五段 100% 同属一侧，无相邻上下文交叉泄漏。

### 2. 训练安全性预检 (`--check-only`)
```bash
.venv/bin/python src/train_lora_local.py \
    --dataset_path data/tara5_action_v1/daomu/dataset/tara5_action_v1_train.json \
    --check-only
```
- **结果**：5/5 样本无截断通过、Token 边界精确对齐、零显存消耗、未加载基座模型或启动训练。

---

## 七、Mytha 系统对接备忘录

1. **接口输出格式**：
   Mytha 向 Writer 交付的章节规划 JSON 必须符合 `schema/tara5_action_v1_schema.json`。
2. **阶段名称**：
   请确保调用端使用 `opening`, `ramp_up`, `climax`, `aftershock`, `cliffhanger`。
3. **因果与质量边界声明**：
   > [!WARNING]
   > “格式校验通过 (`validate_tara5_chapter == True`)”**仅代表数据协议结构与字段完备合规**，绝不等于“情节因果或正文文学质量已经通过”。文学因果自洽度、动机充实度与戏剧张力必须依赖 Mytha 上游规划的精确输入及人工文学复核。
