#!/usr/bin/env python3
"""
TARA-5 章节安全装配与接缝咬合去重引擎 (Assembly Engine)
版本：tara5_action_v1

核心规范：
1. 装配阅读顺序：
   opening (开篇) -> ramp_up (蓄水桥) -> climax (高潮) -> aftershock (余波桥) -> cliffhanger (结尾断章)
2. 锚点归属原则：
   【前段保留，后段剥离 (Prev-Retained, Next-Stripped)】
   - Junction 1 (破题 -> 蓄水): opening 保留 anchor_2，ramp_up 剥离开头重复的 anchor_2；
   - Junction 2 (蓄水 -> 高潮): ramp_up 保留 anchor_3，climax 剥离开头重复的 anchor_3；
   - Junction 3 (高潮 -> 余波): climax 保留 anchor_4，aftershock 剥离开头重复的 anchor_4；
   - Junction 4 (余波 -> 断章): aftershock 保留 anchor_5，cliffhanger 剥离开头重复的 anchor_5。
3. 精准边界去重：
   - 严禁模糊匹配后直接整行丢弃 (防止误删同一行中后续非锚句正文)；
   - 仅精准剔除前缀中确认重复的锚点字符序列；
   - 若前段末尾未咬入锚句，而后段开头包含，则坚决保留后段头部，杜绝剧情丢句。
4. 熔断与自愈机制：
   - 检测生成截断（过短、非标点收尾）；
   - 检测关键接缝缺锚；
   - 诊断失败时输出详细诊断并持久化草稿。
"""

import re
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# 标准段落顺序
ASSEMBLY_ORDER = ["opening", "ramp_up", "climax", "aftershock", "cliffhanger"]

# 接缝定义: (junction_name, prev_stage, next_stage, anchor_key)
JUNCTION_DEFS = [
    ("Junction 1 [破题 -> 蓄水]", "opening", "ramp_up", "anchor_2"),
    ("Junction 2 [蓄水 -> 高潮]", "ramp_up", "climax", "anchor_3"),
    ("Junction 3 [高潮 -> 余波]", "climax", "aftershock", "anchor_4"),
    ("Junction 4 [余波 -> 断章]", "aftershock", "cliffhanger", "anchor_5"),
]

TERMINAL_PUNCTUATION = ("。", "！", "？", "”", "’", "…", '"', "'", "!", "?")


def normalize_anchor_text(s: str) -> str:
    """去除所有空白与标点符号，进行鲁棒性锚点比对"""
    if not s:
        return ""
    return re.sub(r"[\s\u3000，。！？、“”\"'：；·…—\-~`\.,!?:;]", "", s)


def strip_exact_anchor_prefix(text: str, anchor: str) -> Tuple[str, bool]:
    """
    精准从 text 开头剥除 anchor，绝不丢弃整行。
    
    返回: (stripped_text, did_strip)
    - 优先精确前缀匹配: text.lstrip().startswith(anchor)
    - 次之基于归一化比对字符序列，逐字符贪婪匹配对应长度，截取原字符串切片
    """
    if not text or not anchor:
        return text, False

    leading_ws_len = len(text) - len(text.lstrip())
    body = text[leading_ws_len:]

    # 1. 严格全匹配前缀
    if body.startswith(anchor):
        stripped = body[len(anchor):].lstrip("\r\n")
        return stripped, True

    # 2. 标点微调匹配 (寻找最短切片满足 normalize_anchor_text(body[:k]) == normalize_anchor_text(anchor))
    norm_anc = normalize_anchor_text(anchor)
    if not norm_anc:
        return text, False

    # 仅在前 max(len(anchor) * 2, 60) 字符范围内寻找匹配终点，严禁大范围跨行匹配
    search_limit = min(len(body), max(len(anchor) + 30, int(len(anchor) * 1.5)))
    for k in range(1, search_limit + 1):
        candidate_norm = normalize_anchor_text(body[:k])
        if candidate_norm == norm_anc:
            # 找到了对应锚点的字符切片，剥除并保留原行后续内容
            stripped = body[k:].lstrip("\r\n")
            return stripped, True

    return text, False


def safe_assemble_chapter(
    title: str,
    segments: Dict[str, str],
    anchors: Optional[Dict[str, str]] = None,
    drafts_dir: Optional[Path] = None,
    allow_soft_fail: bool = False,
) -> Tuple[str, bool, List[str]]:
    """
    安全装配全章并进行双侧咬合核验。

    :param title: 章节标题
    :param segments: 五段正文字典（支持 ramp_up 或 rampup 命名）
    :param anchors: 锚点字典（anchor_2, anchor_3, anchor_4, anchor_5 等）
    :param drafts_dir: 若指定，在失败或完成时写入诊断快照草稿
    :param allow_soft_fail: 若为 True，在非致命缺锚时仍返回组装文本
    :return: (assembled_text, success, diagnostic_logs)
    """
    anchors = anchors or {}
    report_lines: List[str] = []
    has_fatal_error = False

    # 标准化输入段落 key
    clean_segs: Dict[str, str] = {}
    for st in ASSEMBLY_ORDER:
        val = segments.get(st, "")
        if not val and st == "ramp_up":
            val = segments.get("rampup", "")
        clean_segs[st] = val.strip()

    # 1. 检查是否存在段落缺失或极度截断
    for name in ASSEMBLY_ORDER:
        seg_text = clean_segs[name]
        if not seg_text:
            report_lines.append(f"[-] [缺失错误] 阶段 {name} 正文为空，无法装配！")
            has_fatal_error = True
        elif len(seg_text) < 20:
            report_lines.append(f"[-] [严重截断] 阶段 {name} 长度仅 {len(seg_text)} 字，疑似模型异常中断！")
            has_fatal_error = True
        elif len(seg_text) < 60:
            report_lines.append(f"[!] [长度偏短] 阶段 {name} 长度仅 {len(seg_text)} 字，请注意内容丰满度。")
        elif not seg_text.endswith(TERMINAL_PUNCTUATION):
            report_lines.append(f"[!] [截断警告] 阶段 {name} 末尾可能未完整结束 (末尾: '{seg_text[-1]}')")

    # 2. 接缝双侧校验与去重决策 (Prev-Retained, Next-Stripped)
    for j_name, prev_name, next_name, anchor_key in JUNCTION_DEFS:
        anchor = anchors.get(anchor_key, "").strip()
        prev_t = clean_segs[prev_name]
        next_t = clean_segs[next_name]

        if not anchor:
            report_lines.append(f"[i] {j_name}: 未指定锚点 {anchor_key}，按原样顺接。")
            continue

        norm_anc = normalize_anchor_text(anchor)
        if not norm_anc:
            continue

        # 检查前段尾部与后段头部候选区
        prev_tail_norm = normalize_anchor_text(prev_t[-max(len(anchor) * 3, 120):])
        next_head_norm = normalize_anchor_text(next_t[:max(len(anchor) * 3, 120)])

        prev_has = norm_anc in prev_tail_norm
        next_has = norm_anc in next_head_norm

        if prev_has and next_has:
            # 双侧均包含锚点 -> 前段保留，后段精准剥离前缀锚点
            stripped_text, did_strip = strip_exact_anchor_prefix(next_t, anchor)
            if did_strip:
                clean_segs[next_name] = stripped_text
                report_lines.append(f"[OK] {j_name}: 双侧均确认包含锚句，已精准剥离后段首部重复锚点 (前段保留)。")
            else:
                report_lines.append(f"[!] {j_name}: 双侧存在相似文本，但无法精准对齐前缀字符，保留后段原文以防误伤。")
        elif not prev_has and next_has:
            # 前段遗漏，后段具备 -> 坚决保留后段头部，杜绝漏句
            report_lines.append(f"[!] [防丢保护] {j_name}: 前段末尾未咬入锚句，已保留后段开头锚句，防止关键情节丢失！")
        elif prev_has and not next_has:
            # 前段具备，后段未重复 -> 正常连接
            report_lines.append(f"[OK] {j_name}: 前段已包含锚句，后段开头未重复，保持顺接。")
        else:
            # 两侧均未检测到完整锚点
            report_lines.append(f"[?] [缺锚提示] {j_name}: 双侧均未检测到完整锚点: “{anchor[:25]}...”")
            # 蓄水->高潮 属于核心转折，若两者均无，标记关注
            if "高潮" in j_name:
                report_lines.append(f"    -> 警告：高潮起始锚点缺失，因果咬合可能存在轻微断层。")

    # 3. 诊断报告持久化
    if drafts_dir:
        drafts_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        draft_info = {
            "timestamp": timestamp,
            "title": title,
            "status": "FAILED" if has_fatal_error else "SUCCESS",
            "anchors": anchors,
            "diagnostics": report_lines,
            "segments_before": segments,
            "segments_assembled": clean_segs,
        }
        draft_file = drafts_dir / f"chapter_assembly_{timestamp}.json"
        try:
            draft_file.write_text(json.dumps(draft_info, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # 4. 组装结果
    if has_fatal_error and not allow_soft_fail:
        return "", False, report_lines

    # 最终装配顺序：Opening -> Ramp_up -> Climax -> Aftershock -> Cliffhanger
    assembled_parts = [
        clean_segs["opening"],
        clean_segs["ramp_up"],
        clean_segs["climax"],
        clean_segs["aftershock"],
        clean_segs["cliffhanger"],
    ]
    # 段与段之间使用双换行标准空行分隔
    assembled_text = "\n\n".join([p for p in assembled_parts if p]).strip()

    return assembled_text, not has_fatal_error, report_lines
