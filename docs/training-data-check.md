# 训练数据只读预检

训练脚本不会修改采集数据或转换后的数据集。完整聊天序列超过 `--max_length` 时，默认在加载模型前停止，报告总数、各任务超长数、最长序列及部分样本 ID。不会删除提示中间的 token，也不会截掉回答尾部。

仅检查数据（只加载 tokenizer，不加载模型或启动训练）：

```bash
.venv/bin/python src/train_lora_local.py --check-only
```

默认长度仍为 2560；若有超长样本，上述命令以错误结束。可按显存和模型能力调整 `--max_length` 后重检，不能只为通过检查而盲目增大。

若明确接受本次训练跳过超长样本，可先检查过滤情况：

```bash
.venv/bin/python src/train_lora_local.py --check-only --overlength skip
```

正式训练时使用同样的 `--max_length` 和 `--overlength skip`，去掉 `--check-only` 即可。跳过只发生于本次内存集合，源文件保留不动；报告中的分任务计数可用于判断过滤是否造成任务分布偏移。没有可用样本时仍停止。

当前训练格式为一组非空 human/gpt 对话。多轮、空回答或聊天模板回答边界不匹配会被拒绝，避免错误标签。所有入选样本完整保留，system/user 与 assistant 起始前缀全部屏蔽为 -100，只有回答及其模板结束标记参与监督。

回归测试（使用临时合成数据，不启动训练）：

```bash
.venv/bin/python -m unittest discover -s tests -v
```
