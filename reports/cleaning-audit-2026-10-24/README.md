# 昨晚数据清洗复核（2026-10-24）

结论：结构与导出一致性通过，内容质量未通过；不建议将本批次作为“已清洗合格”数据直接训练。

## 范围与方法

复核对象是 2026-10-23 18:32:59 +0800 的提交 `dfab044` 所对应当前 TARA-5 数据：18 本、2,505 章、12,525 条训练样本。未发现单独运行日志，因此时间归属依据提交记录，并非据此断言昨夜任务运行时段。

全量执行现有 Schema 校验、单书与合并集比对、从切片重建 prompt/回答比对、原文空白归一后的连续匹配、段落包含关系、角色引用、证据占位、锚点、字数和预算检查。所有 2,505 章均匹配到本地原文。语义核对仅针对下述具体案例，不代表全量人工审读。

原始切片与训练集未修改。所有问题在旁路文件中设置 `needs_review=true`；原数据的 `annotation_status` 仍需后续处理。`input_manifest.json` 保存复核时切片 SHA-256。

## 优先处理的问题

| 优先级 | 问题 | 结果 / 影响 |
| --- | --- | --- |
| P0 | 串书污染 | 《仙逆》ch_0062（第60章 即墨老人）的 aftershock、cliffhanger 及相应规划被替换为《我的模拟长生路》的玄京城毁灭、李凡“还真”剧情，已传播到训练集。 |
| P1 | 整段嵌套/重复 | 363 章存在 579 对整段包含关系，五阶段切分仍有明显重复。 |
| P1 | 证据占位符 | 2,505 章共 5,010 个 confirmed_facts.evidence 仍是“本阶段既定设定与局势信息差记录”或“本章开局死线与时间窗口要求”，不是原文证据。 |
| P1 | 行动主体引用失效 | 12,525 个 actor_id 均不能在本章 characters.character_id 中解析；大量为“主角”，角色表使用 char_protagonist。 |
| P1 | 原文不一致候选 | 896 章、1,240 段空白归一后无法在原文连续匹配。可能包括清理、标点更改、删改或伪造，需要逐条复核，不能全部认定伪造。 |
| P1 | 作者作话残留 | momen ch_0866 cliffhanger 留有活动称号、盟主打赏名单；xiuxian ch_0425 aftershock 是感谢打赏和加更安排。已人工确认两处。 |
| P1 | 开局事实时间穿越 | changsheng ch_0062 的 confirmed_facts 仍含“至死不知”“导致捕仙失败”“道玄子祭出本命道基残剑”；characters 的认知/目标也保留终局信息。 |
| P1 | 样本 ID 不唯一 | 合并集有 2,770 个重复 ID 值，涉及 7,900 条记录；是跨书 ID 冲突，不代表这些记录正文相同。详见 duplicate_sample_ids.json。 |
| P1 | 默认训练长度预检失败 | 513/12,525 条超过 2,560 tokens（4.10%），最长 6,408。蓄水 265、高潮 174、余波 65、结尾 9；训练脚本正常拦截，未启动训练。 |
| P2 | 字数与预算未同步 | 357 个 char_count 与实际 Python 字符长度不符；659 段不在自己的长度预算内，预算问题标为待复核。 |
| P2 | 边界句重复 | 4,155 对相邻阶段保留相同接缝锚句，标记供核查训练目标与装配约定；不等同于整段重复，装配时可能另外去重。 |

## 关键证据与原因

- 串书：`data/tara5_action_v1/xianni/sliced/ch_0062_第二卷 修真血影_第60章 即墨老人_tara5_v1.json`。原文为王林、张虎、即墨老人、飞剑斗法，清洗后结尾却为李凡“还真”。`src/deep_clean_tara5_v1.py` 的 `fix_changsheng_ch0062` 仅检查 chapter_id 含 ch_0062；调用处对所有书目无条件执行，缺少 novel_key 限定。
- 重复例：bailian ch_0086 aftershock/cliffhanger；changsheng ch_0043 ramp_up 包含 climax 和 aftershock。修复函数仅覆盖部分阶段组合，并受 idx > 150 / idx > 50 门槛限制，不能保证全局去重。
- 占位证据：清洗代码给 confirmed_facts 第三、第四项写入通用说明，却将其描述为“真实证据依据”。前两项虽有引文，是否支撑对应 fact 仍需语义复核。
- 清洗后的 char_count、length_budget 未随正文修改更新。
- 其中 2,295 章虽发现问题，原标注仍为 needs_review=false；旁路 chapter_review_flags.json 用于覆盖复核状态，不会自动被训练加载器读取。

## 通过的检查与限制

- 2,505/2,505 章通过项目现有 Schema 校验（其未校验上述语义质量及 actor_id 引用完整性）。
- 18 个单书训练集合并内容与全局训练集完全一致；每章五任务齐全；导出的提示和回答与当前切片及 prompt_builder 一致。因此内容问题也已同步进入训练集。
- 本次使用本地 Qwen2.5-1.5B-Instruct tokenizer，离线执行 `--check-only`；未加载模型或启动训练。预检日志仅保留脚本默认的部分超长样本示例，并非 513 条完整明细。
- 原文匹配只忽略空白，不忽略标点或做模糊匹配；未全量判定因果、角色认知、证据是否真正支持事实，也未穷尽所有作者作话表达。

## 标记文件与复现

- `issues.csv`：可用表格软件筛选优先级、问题类型、文件、字段、证据、状态。
- `issues.jsonl`：同一份逐字段机器可读清单，含唯一 issue_id。
- `chapter_review_flags.json`：逐章 needs_review=true 与问题 ID 关联。
- `duplicate_sample_ids.json`：跨书重复 ID 的全部相关记录。
- `summary.json`：全量统计；`training-precheck.log`：训练长度预检日志。

复现内容检查：`.venv/bin/python reports/cleaning-audit-2026-10-24/audit.py`

复现长度检查：`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/train_lora_local.py --dataset_path data/tara5_action_v1/all_books/dataset/tara5_action_v1_all_train.json --check-only`

处理顺序建议：先隔离串书样本并修复书目限定，再处理重复切片和作话残留；重建可追溯事实与行动主体，更新字数/预算、书目命名空间和复核状态，最后重新编译并做训练预检。
