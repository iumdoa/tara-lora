# TARA-5 数据修复与发布

训练默认读取 `data/tara5_action_v1/all_books/dataset/tara5_action_v1_ready_2560.json`。
这份数据仅包含通过原文位置校验、独立模型语义复核且本地 tokenizer 编码不超过 2,560 tokens 的完整样本。

- `tara5_action_v1_all_train.json`：所有通过内容复核的完整五任务数据；可能存在超长样本。
- `tara5_action_v1_long_context.json`：超出默认长度的完整样本，不截断、不删除。采用更长上下文训练前需按实际资源另做预检。
- `data/tara5_action_v1/quarantine/`：未通过内容或来源复核的原始章节，保留原因并设为 `needs_review=true`；不进入训练导出。

复核流程为模型重新标注、独立模型复核、程序校验，不应称为人工专家全审。模型复核仍有误判可能。
原来的角色认知、事实、行动模板不会被当成真实证据延续。已确认事实只引用前章原句；不明确的认知信息留空，不生成占位事实。

正文全部以原始章节字符偏移提取，五阶段严格按原文顺序且互不重叠。相邻阶段之间只允许空白间隔。原文 SHA-256、首尾范围、事实来源和行动证据位置保存在章节 JSON 中。
桥接输入两端的锚句属于相邻阶段，只作为上下文；训练目标不重复锚句。兼容装配器仍可处理历史共享锚点。

```bash
# 通过现有 .env 中的服务重建标注，先暂存，不覆盖生产数据
.venv/bin/python src/repair_tara5_data.py --workers 12
# 可按单书重试隔离章节；保留成功检查点，不重跑已通过章节
.venv/bin/python src/repair_tara5_data.py --book bailian --retry-quarantine --workers 8
# 原文验证、token 分流和暂存导出；有未完成章节时拒绝发布
.venv/bin/python src/publish_tara5_repair.py
# 有修复前快照时才允许发布：隔离文件保留到 quarantine
.venv/bin/python src/publish_tara5_repair.py --apply
# 独立检查产物一致性和来源
.venv/bin/python src/verify_tara5_release.py
# 本地 tokenizer 检查，绝不加载模型/启动训练
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/train_lora_local.py --check-only
.venv/bin/python -m unittest discover -s tests -v
```

本次工作目录为 `output/cleaning-repair-2026-10-24/`，包含修复前 `before.tar.gz`、逐章模型响应、独立复核响应、处理记录、候选切片和暂存发布文件。报告与隔离清单位于 `reports/cleaning-repair-2026-10-24/`。

`deep_clean_tara5_v1.py` 现在仅转入上述暂存重建入口；不再运行硬编码剧情补丁或原地模板替换。

当服务余额不足时，处理器会立即熔断后续请求，保存 `pending_service` 状态；这不同于内容质检失败。恢复后运行相同重建命令即可接续，包括已暂存到隔离区的章节。

`publish_tara5_repair.py --allow-pending --apply` 是显式的部分发布选项：只发布已验证样本，其余全部留档隔离，并在报告中记录 `partial_service_blocked`，不代表全量清洗完成。默认发布仍拒绝任何未完成的服务处理记录。
