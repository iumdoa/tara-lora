import collections
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
summary=json.loads((REPORT/'summary.json').read_text())
quarantine=json.loads((REPORT/'quarantine_manifest.json').read_text())
blocked={r['file']:r for r in quarantine}
old=ROOT/'reports/cleaning-audit-2026-10-24/issues.jsonl'
chapters={}
for line in old.read_text().splitlines():
    issue=json.loads(line);path=issue['file']
    row=chapters.setdefault(path,{'original_file':path,'issue_ids':[], 'issue_types':collections.Counter()})
    row['issue_ids'].append(issue['issue_id']);row['issue_types'][issue['code']]+=1
for path,row in chapters.items():
    record=blocked.get(path)
    row['disposition']='reannotated_and_verified' if record is None else record['status']
    if record:
        original=Path(path);row['current_file']=str(Path('data/tara5_action_v1/quarantine')/original.parent.parent.name/'sliced'/original.name)
        row['reasons']=record.get('reasons',[])
    else: row['current_file']=path
(REPORT/'issue_dispositions.json').write_text(json.dumps(list(chapters.values()),ensure_ascii=False,indent=2))
pending=summary['pending_service_chapters']; rejected=summary['quarantined_chapters']-pending
status='部分完成，服务余额不足，尚未完成全量重标注。' if pending else '全量处理已完成；通过项导出，无法验证项隔离保留。'
text=f'''# 数据清洗修复结果（2026-10-24）

**状态：{status}**

| 项目 | 数量 |
| --- | ---: |
| 原始章节 | {summary['original_chapters']} |
| 已重新标注且通过原文/结构/语义检查 | {summary['approved_chapters']} |
| 已验证完整训练样本 | {summary['all_samples']} |
| 默认 2,560 token 内可训练样本 | {summary['ready_samples']} |
| 单独保留的完整超长样本 | {summary['long_context_samples']} |
| 内容或切分复核未通过、隔离保留 | {rejected} |
| 标注服务余额不足而尚未完成、隔离保留 | {pending} |

## 已落地的修复

- 《仙逆》ch_0062 的串书污染已对照完整原文重建，恢复王林、丹宝、飞剑与剑鞘情节。
- 通过项的所有正文均由原文件精确字符位置提取；五段不重叠，之间不得遗漏非空白文字。没有通过生成新正文“修补”原文。
- confirmed_facts 仅引用同书前章原句；删除未经证实的全知认知和机械证据占位。没有依据的信息不填充。
- 行动计划重新提炼，actor_id 必须能解析到角色；行动证据有阶段内原文位置。更新真实字数、预算和锚点。
- 修复昨晚导出代码的 system/user 对调；训练和推理使用同一正确序列化函数。
- 样本 ID 加入书目命名空间；训练/验证分组采用“书目+章节”，按各书顺序切分。
- 未通过或未完成的样本不进入训练导出，原数据保留在 quarantine 与完整快照中；训练器拒绝 needs_review=true 样本。
- 全文超长样本单独保留，不裁剪回答、不提高默认长度绕过检查。默认训练路径已指向 ready_2560 数据集。
- 服务返回 HTTP 402 后停止后续请求；余额问题记录为 pending_service，不能当作内容不合格。恢复后按记录接续。

## 验证与边界

`verification.json` 检查来源 SHA-256、原文偏移、阶段间完整性、前章事实证据、行动证据、角色引用、逐书/合并导出一致性、唯一 ID 和分流守恒。
`training-precheck.log` 使用本地 Qwen2.5-1.5B-Instruct tokenizer 对 ready 数据执行真实训练预检；未加载模型、未启动训练。
`tests.log` 保存回归测试结果。通过检查不等于保证所有文学判断绝无误差。

语义复核为模型标注加独立模型审核；《仙逆》ch_0062 由本助手逐句对照原文处理，均不是“人类专家全审”。认知字段没有来源支持时留空。
隔离项保留具体原因，包含目录/设定资料、不自然的五段划分、计划不受原文支持和服务中断；不以“全部已修复”掩盖未完成项。

此前检查报告的提示一致性检查沿用了错误的 tuple 解包顺序，因此漏报了 system/user 对调。本次已用系统角色内容断言和导出回归测试修正，而不是沿用旧报告的通过结论。

## 位置

- 默认训练集：`data/tara5_action_v1/all_books/dataset/tara5_action_v1_ready_2560.json`
- 已验证完整集与长上下文集：同目录 `tara5_action_v1_all_train.json`、`tara5_action_v1_long_context.json`
- 隔离原始章节：`data/tara5_action_v1/quarantine/`
- 逐章处理结果与原问题 ID 对应：`issue_dispositions.json`
- 未通过/未完成的具体原因：`quarantine_manifest.json`
- 每条样本的真实 token 数与分流：`sample_lengths.json`
- 修复前完整快照：`output/cleaning-repair-2026-10-24/before.tar.gz`
- 原文标注、独立审核响应与断点：`output/cleaning-repair-2026-10-24/`

## 恢复与复现

服务端恢复余额或在本地 `.env` 配置可用服务后：

```bash
.venv/bin/python src/repair_tara5_data.py --workers 12
.venv/bin/python src/repair_tara5_data.py --retry-quarantine --workers 8
.venv/bin/python src/publish_tara5_repair.py --apply
.venv/bin/python src/verify_tara5_release.py
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python src/train_lora_local.py --check-only
.venv/bin/python -m unittest discover -s tests -v
```

默认发布拒绝未完成的处理记录；`--allow-pending` 只用于明确标注为部分完成的安全发布。已通过检查点不重跑。
'''
(REPORT/'README.md').write_text(text)
print(status)
