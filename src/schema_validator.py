#!/usr/bin/env python3
"""
TARA-5 数据协议校验器 (Schema Validator)
版本：tara5_action_v1

针对 `schema/tara5_action_v1_schema.json` 进行纯标准库自包含校验，
无需第三方外部依赖，严格保证数据结构、字段、枚举与嵌套约束的合规性。
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

VALID_STAGES = ("opening", "ramp_up", "climax", "aftershock", "cliffhanger")
SCHEMA_VERSION = "tara5_action_v1"


def validate_tara5_chapter(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    全面校验章节数据字典是否严格遵循 tara5_action_v1 规范。
    返回: (is_valid, error_list)
    """
    errors: List[str] = []

    # 1. 根节点必填项
    required_root_keys = [
        "schema_version",
        "chapter_id",
        "source_novel",
        "chapter_title",
        "chapter_outline",
        "confirmed_facts",
        "characters",
        "stage_plans",
        "constraints",
        "segments",
        "annotation_status",
    ]
    for k in required_root_keys:
        if k not in data:
            errors.append(f"根节点缺少必填字段: '{k}'")

    if errors:
        return False, errors

    # 2. 版本号校验
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 '{SCHEMA_VERSION}', 实际为: '{data.get('schema_version')}'")

    # 3. 总章纲校验
    outline = data.get("chapter_outline")
    if not isinstance(outline, list) or len(outline) == 0:
        errors.append("chapter_outline 必须为非空字符串列表")
    else:
        for idx, item in enumerate(outline):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"chapter_outline[{idx}] 必须为有效非空字符串")

    # 4. 已确认事实 (confirmed_facts)
    facts = data.get("confirmed_facts")
    if not isinstance(facts, list):
        errors.append("confirmed_facts 必须为列表")
    else:
        for idx, f in enumerate(facts):
            if not isinstance(f, dict):
                errors.append(f"confirmed_facts[{idx}] 必须为对象")
                continue
            if "fact" not in f or not str(f["fact"]).strip():
                errors.append(f"confirmed_facts[{idx}] 缺少非空 'fact' 字段")
            if "evidence" not in f or not str(f["evidence"]).strip():
                errors.append(f"confirmed_facts[{idx}] 缺少非空 'evidence' 证据出处字段")

    # 5. 角色设定 (characters)
    chars = data.get("characters")
    if not isinstance(chars, list):
        errors.append("characters 必须为列表")
    else:
        for idx, c in enumerate(chars):
            if not isinstance(c, dict):
                errors.append(f"characters[{idx}] 必须为对象")
                continue
            for field in ("character_id", "name", "goal"):
                if field not in c or not str(c[field]).strip():
                    errors.append(f"characters[{idx}] 缺少非空 '{field}' 字段")

    # 6. 创作约束 (constraints)
    constraints = data.get("constraints")
    if not isinstance(constraints, dict):
        errors.append("constraints 必须为对象")
    else:
        for field in ("pov", "negative_constraints", "allowed_creative_freedom"):
            if field not in constraints:
                errors.append(f"constraints 缺少必填字段: '{field}'")
        if "negative_constraints" in constraints and not isinstance(constraints["negative_constraints"], list):
            errors.append("constraints.negative_constraints 必须为列表")
        if "allowed_creative_freedom" in constraints and not isinstance(constraints["allowed_creative_freedom"], list):
            errors.append("constraints.allowed_creative_freedom 必须为列表")

    # 7. 五阶段计划 (stage_plans)
    plans = data.get("stage_plans")
    if not isinstance(plans, dict):
        errors.append("stage_plans 必须为对象")
    else:
        for stage in VALID_STAGES:
            if stage not in plans:
                errors.append(f"stage_plans 缺少阶段定义: '{stage}'")
                continue
            p = plans[stage]
            if not isinstance(p, dict):
                errors.append(f"stage_plans['{stage}'] 必须为对象")
                continue
            if p.get("stage_name") != stage:
                errors.append(f"stage_plans['{stage}'].stage_name 必须为 '{stage}'，实际为: '{p.get('stage_name')}'")
            for req in ("responsibility", "preconditions", "target_outcome", "length_budget", "start_state", "end_state", "action_chain"):
                if req not in p:
                    errors.append(f"stage_plans['{stage}'] 缺少必填字段: '{req}'")

            # 校验 length_budget
            budget = p.get("length_budget")
            if not isinstance(budget, dict) or "min_chars" not in budget or "max_chars" not in budget:
                errors.append(f"stage_plans['{stage}'].length_budget 必须包含 min_chars 和 max_chars")
            elif budget["min_chars"] > budget["max_chars"]:
                errors.append(f"stage_plans['{stage}'].length_budget min_chars ({budget['min_chars']}) 不能大于 max_chars ({budget['max_chars']})")

            # 校验 action_chain
            actions = p.get("action_chain")
            if not isinstance(actions, list):
                errors.append(f"stage_plans['{stage}'].action_chain 必须为列表")
            else:
                for a_idx, act in enumerate(actions):
                    if not isinstance(act, dict):
                        errors.append(f"stage_plans['{stage}'].action_chain[{a_idx}] 必须为对象")
                        continue
                    for af in ("action_id", "actor_id", "action", "result"):
                        if af not in act or not str(act[af]).strip():
                            errors.append(f"stage_plans['{stage}'].action_chain[{a_idx}] 缺少必填字段: '{af}'")

    # 8. 五阶段正文切片 (segments)
    segments = data.get("segments")
    if not isinstance(segments, dict):
        errors.append("segments 必须为对象")
    else:
        for stage in VALID_STAGES:
            if stage not in segments:
                errors.append(f"segments 缺少阶段切片: '{stage}'")
                continue
            seg = segments[stage]
            if not isinstance(seg, dict):
                errors.append(f"segments['{stage}'] 必须为对象")
                continue
            for req in ("anchor_start", "anchor_end", "text"):
                if req not in seg or not str(seg[req]).strip():
                    errors.append(f"segments['{stage}'] 缺少非空必填字段: '{req}'")
            text = seg.get("text", "")
            anc_start = seg.get("anchor_start", "")
            anc_end = seg.get("anchor_end", "")
            if anc_start and anc_start not in text:
                errors.append(f"segments['{stage}'] 的 anchor_start 未在正文精确出现")
            if anc_end and anc_end not in text:
                errors.append(f"segments['{stage}'] 的 anchor_end 未在正文精确出现")

    # 9. 标注审核状态 (annotation_status)
    status = data.get("annotation_status")
    if not isinstance(status, dict):
        errors.append("annotation_status 必须为对象")
    else:
        if "needs_review" not in status or not isinstance(status["needs_review"], bool):
            errors.append("annotation_status.needs_review 必须为布尔值")

    return len(errors) == 0, errors


def validate_file(file_path: Path) -> Tuple[bool, List[str]]:
    """读取指定 JSON 文件并进行 Schema 校验"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return validate_tara5_chapter(data)
    except Exception as e:
        return False, [f"文件解析或读取失败: {e}"]
