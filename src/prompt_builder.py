#!/usr/bin/env python3
"""
TARA-5 统一任务输入与提示词构造引擎 (Prompt & Input Builder)
版本：tara5_action_v1

核心原则：
1. 单一真实来源 (Single Source of Truth)：训练数据转换 (format_converter) 与
   推理装配 (generate_chapter / infer_test) 100% 复用本模块，消除模板漂移。
2. 阶段名称严格规范：opening, ramp_up, climax, aftershock, cliffhanger。
3. 反向生成信息隔离：
   - 高潮 (climax)：仅依赖总章纲相关行动、已确认事实、人物认知、高潮起止状态；严禁依赖尚未生成的蓄水正文。
   - 结尾 (cliffhanger)：仅依赖章纲结局状态、结尾行动、悬念目标；严禁依赖尚未生成的余波正文。
   - 开篇 (opening)：承接上章结尾，明确开篇行动与起局状态。
   - 蓄水桥 (ramp_up)：依赖开篇必要上下文、开篇末锚、高潮首锚、蓄水行动链与起止状态。
   - 余波桥 (aftershock)：依赖高潮必要上下文、高潮末锚、断章首锚、余波行动链与起止状态。
4. Writer 职责边界：输出纯小说自然正文，杜绝合成模板标签 (<think>) 与程序控制字段；文学表达允许自由发挥，严禁机械背诵提纲。
"""

from typing import Dict, Any, Tuple, Optional, List

# 标准五阶段名称常量
STAGE_OPENING = "opening"
STAGE_RAMP_UP = "ramp_up"
STAGE_CLIMAX = "climax"
STAGE_AFTERSHOCK = "aftershock"
STAGE_CLIFFHANGER = "cliffhanger"

VALID_STAGES = (
    STAGE_OPENING,
    STAGE_RAMP_UP,
    STAGE_CLIMAX,
    STAGE_AFTERSHOCK,
    STAGE_CLIFFHANGER,
)

# 统一阶段任务名称 (用于 ShareGPT 数据集 task_type 标识)
STAGE_TASK_MAP = {
    STAGE_OPENING: "tara5_opening",
    STAGE_RAMP_UP: "tara5_ramp_up",
    STAGE_CLIMAX: "tara5_climax",
    STAGE_AFTERSHOCK: "tara5_aftershock",
    STAGE_CLIFFHANGER: "tara5_cliffhanger",
}

# 兼容旧版命名
LEGACY_STAGE_MAP = {
    "rampup": STAGE_RAMP_UP,
    "tara_opening": STAGE_OPENING,
    "tara_rampup": STAGE_RAMP_UP,
    "tara_climax": STAGE_CLIMAX,
    "tara_aftershock": STAGE_AFTERSHOCK,
    "tara_cliffhanger": STAGE_CLIFFHANGER,
    "tara5_rampup": STAGE_RAMP_UP,
}


def normalize_stage_name(name: str) -> str:
    """标准化阶段名称为统一的 opening, ramp_up, climax, aftershock, cliffhanger"""
    clean = name.strip().lower()
    if clean in VALID_STAGES:
        return clean
    if clean in LEGACY_STAGE_MAP:
        return LEGACY_STAGE_MAP[clean]
    raise ValueError(f"未知的 TARA-5 阶段名称: {name} (允许值: {VALID_STAGES})")


# ==============================================================================
# 微调 Writer 专用 System Prompts (纯文学创作执行者设定，无机械模板)
# ==============================================================================
SYSTEM_PROMPTS = {
    STAGE_OPENING: (
        "你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 开篇破题专家）。\n"
        "你的职责是将指定的开篇行动计划与开局状态转化为极具吸引力的小说正文（约 150~300 字）。\n"
        "紧扣上章余势顺畅起笔或通过强动作反差开门见山，杜绝空洞环境闲笔。\n"
        "【严格边界】专注于词句表现、人物微动作与即时氛围；禁止擅自改动已确认事实或提前展开后续高潮事件。"
    ),
    STAGE_RAMP_UP: (
        "你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 蓄水引信桥专家）。\n"
        "你的职责是从【破题定势铁锚】出发，把承上启下的蓄水行动链推演为层层施压、暗流涌动的小说正文（约 600~900 字）。\n"
        "【严格边界】正文必须严丝合缝由指定的【终点铁锚 (高潮爆发句)】落笔定调收尾；专注于矛盾升级与人物博弈，禁止提前抢写高潮爆发核心事件。"
    ),
    STAGE_CLIMAX: (
        "你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 高潮核爆专家）。\n"
        "你的职责是跳过铺垫前戏，直接围绕指定的高潮核心行动链撰写爆发力最强、冲突最集中、情绪最饱满的小说正文（约 600~900 字）。\n"
        "着重刻画高频动作、生理对抗、心理博弈与局势瞬间逆转。\n"
        "【严格边界】严格遵守既定胜负与态势结果，不得新增非计划关键角色或无端篡改设定；正文首句与末句需清晰锁定高潮爆发与定局点。"
    ),
    STAGE_AFTERSHOCK: (
        "你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 余波转折桥专家）。\n"
        "你的职责是从【高潮定势末锚】出发，将战后余波、生理颤抖、战利品复盘与虚假安全感转化为张弛有度的小说正文（约 400~600 字）。\n"
        "【严格边界】正文末尾必须顺理成章、完全咬入指定的【终点铁锚 (断章引爆句)】；专注于战局消化与中继过渡，禁止重复回放高潮对决。"
    ),
    STAGE_CLIFFHANGER: (
        "你是由 Mytha 文学工程引擎驱动的网文作家（TARA-5 致命断章专家）。\n"
        "你的职责是紧扣本章结局态势与终局悬念目标，在 150~300 字内引爆全新维度异变或心理危机，在矛盾最高峰处戛然而止完成绝杀卡点。\n"
        "最后一句话必须一刀封喉，制造令读者不可抗拒的翻页渴望。\n"
        "【严格边界】直接切入终局危机，杜绝说明性交代或大段背景复述；禁止改变本章既定走向。"
    ),
}


def get_system_prompt(stage: str) -> str:
    """获取指定阶段的标准 System Prompt"""
    st = normalize_stage_name(stage)
    return SYSTEM_PROMPTS[st]


# ==============================================================================
# 结构化辅助格式化函数
# ==============================================================================
def format_action_chain(actions: List[Dict[str, Any]]) -> str:
    """将行动单元列表格式化为简洁直观的行动流"""
    if not actions:
        return "（无指定显式行动链）"
    lines = []
    for idx, act in enumerate(actions, 1):
        actor = act.get("actor_id", "角色")
        action_text = act.get("action", "")
        trigger = act.get("trigger", "")
        reason = act.get("reason", "")
        result = act.get("result", "")

        parts = [f"{idx}. [{actor}] 执行：{action_text}"]
        if trigger:
            parts.append(f"   - 触发时机：{trigger}")
        if reason:
            parts.append(f"   - 动机认知：{reason}")
        if result:
            parts.append(f"   - 达成结果：{result}")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def format_narrative_state(state: Dict[str, Any]) -> str:
    """格式化相关人物的状态（位置、姿态、持有物、伤势、认知）"""
    if not state or not isinstance(state, dict):
        return "（维持前置自然态势）"
    lines = []
    if state.get("character_positions"):
        items = [f"{k}: {v}" for k, v in state["character_positions"].items() if v]
        if items:
            lines.append(f"  * 空间位置：{'; '.join(items)}")
    if state.get("character_stances"):
        items = [f"{k}: {v}" for k, v in state["character_stances"].items() if v]
        if items:
            lines.append(f"  * 姿态动作：{'; '.join(items)}")
    if state.get("held_items"):
        items = [f"{k}: {v}" for k, v in state["held_items"].items() if v]
        if items:
            lines.append(f"  * 持有器物：{'; '.join(items)}")
    if state.get("injuries"):
        items = [f"{k}: {v}" for k, v in state["injuries"].items() if v]
        if items:
            lines.append(f"  * 伤势状态：{'; '.join(items)}")
    if state.get("cognitive_changes"):
        items = [f"{k}: {v}" for k, v in state["cognitive_changes"].items() if v]
        if items:
            lines.append(f"  * 认知变化：{'; '.join(items)}")
    return "\n".join(lines) if lines else "（维持前置自然态势）"


def format_confirmed_facts(facts: List[Dict[str, Any]]) -> str:
    """格式化已确认事实清单"""
    if not facts:
        return "（暂无特殊前置锁死事实）"
    lines = []
    for idx, f in enumerate(facts, 1):
        if isinstance(f, dict):
            fact_desc = f.get("fact", "")
            evidence = f.get("evidence", "")
            if evidence:
                lines.append(f"{idx}. {fact_desc} (出处：{evidence})")
            else:
                lines.append(f"{idx}. {fact_desc}")
        elif isinstance(f, str):
            lines.append(f"{idx}. {f}")
    return "\n".join(lines)


def format_characters_info(characters: List[Dict[str, Any]]) -> str:
    """格式化角色视界与信息差"""
    if not characters:
        return "（按常规角色设定行动）"
    lines = []
    for c in characters:
        name = c.get("name", c.get("character_id", "未知"))
        goal = c.get("goal", "")
        mis = c.get("misjudgment", "")
        unknown = c.get("unknown_info", [])

        chunk = [f"• 【{name}】目标：{goal or '推进既定计划'}"]
        if unknown:
            chunk.append(f"  - 认知盲区：{', '.join(unknown)}")
        if mis:
            chunk.append(f"  - 关键误判：{mis}")
        lines.append("\n".join(chunk))
    return "\n".join(lines)


def format_constraints(constraints: Dict[str, Any]) -> str:
    """格式化创作约束边界"""
    if not constraints or not isinstance(constraints, dict):
        return (
            "• 视角：第三人称限制视角\n"
            "• 负面约束：严禁擅自新增关键角色，严禁违背既定战局结果\n"
            "• 自由度：用词、动作细节、神态描写与行文节奏自由发挥"
        )
    pov = constraints.get("pov", "第三人称限制视角")
    neg = constraints.get("negative_constraints", [])
    free = constraints.get("allowed_creative_freedom", [])

    lines = [f"• 叙事视角：{pov}"]
    if neg:
        lines.append(f"• 负面禁令（严禁越界）：{'; '.join(neg)}")
    if free:
        lines.append(f"• 创作自由发挥：{'; '.join(free)}")
    return "\n".join(lines)


# ==============================================================================
# 各阶段核心 Prompt 构造器 (训练与推理共用同一套规则)
# ==============================================================================
def build_climax_prompt(chapter_data: Dict[str, Any]) -> str:
    """
    高潮核爆切片任务输入构造 (tara5_climax)
    【信息可用性】：仅依赖总章纲相关计划、事实、人物认知、高潮起止状态、篇幅预算。
    【绝对隔离】：绝不依赖尚未生成的蓄水正文。
    """
    title = chapter_data.get("chapter_title", "未命名章节")
    outline_list = chapter_data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    plans = chapter_data.get("stage_plans", {})
    cl_plan = plans.get(STAGE_CLIMAX, {})

    facts_str = format_confirmed_facts(chapter_data.get("confirmed_facts", []))
    chars_str = format_characters_info(chapter_data.get("characters", []))
    constraints_str = format_constraints(chapter_data.get("constraints", {}))

    actions_str = format_action_chain(cl_plan.get("action_chain", []))
    start_state_str = format_narrative_state(cl_plan.get("start_state", {}))
    end_state_str = format_narrative_state(cl_plan.get("end_state", {}))
    target_outcome = cl_plan.get("target_outcome", "完成核心冲突对抗并决出即时结果")
    budget = cl_plan.get("length_budget", {"min_chars": 600, "max_chars": 900})

    prompt = (
        f"【章节标题】{title}\n\n"
        f"【故事总章纲】\n{outline_str}\n\n"
        f"【已确认事实基座】\n{facts_str}\n\n"
        f"【核心角色认知与动机】\n{chars_str}\n\n"
        f"【本阶段文学职责】高潮核爆 (Climax) —— 跳过铺垫前戏，直接进入全章最激烈矛盾爆发\n"
        f"【起始前置态势】\n{start_state_str}\n"
        f"【核心行动执行链】\n{actions_str}\n"
        f"【终点定局态势】\n{end_state_str}\n"
        f"【目标达成结果】{target_outcome}\n"
        f"【篇幅预算】{budget.get('min_chars', 600)} ~ {budget.get('max_chars', 900)} 字\n\n"
        f"【创作约束与边界】\n{constraints_str}\n\n"
        f"【任务指示】\n"
        f"请将上述行动与态势转化为自然生动的小说正文，着重呈现对决动作的张力与节奏。\n"
        f"禁止机械复述提纲，禁止擅自修改对决胜负走向，直接输出正文。"
    )
    return prompt


def build_cliffhanger_prompt(chapter_data: Dict[str, Any]) -> str:
    """
    致命断章切片任务输入构造 (tara5_cliffhanger)
    【信息可用性】：仅依赖总章纲终局状态、结尾行动计划、收束或悬念目标。
    【绝对隔离】：绝不依赖尚未生成的余波正文。
    """
    title = chapter_data.get("chapter_title", "未命名章节")
    outline_list = chapter_data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    plans = chapter_data.get("stage_plans", {})
    ch_plan = plans.get(STAGE_CLIFFHANGER, {})

    actions_str = format_action_chain(ch_plan.get("action_chain", []))
    start_state_str = format_narrative_state(ch_plan.get("start_state", {}))
    end_state_str = format_narrative_state(ch_plan.get("end_state", {}))
    target_outcome = ch_plan.get("target_outcome", "引爆全新异变并在高潮点戛然而止完成绝杀卡点")
    constraints_str = format_constraints(chapter_data.get("constraints", {}))
    budget = ch_plan.get("length_budget", {"min_chars": 150, "max_chars": 300})

    prompt = (
        f"【章节标题】{title}\n\n"
        f"【故事总章纲】\n{outline_str}\n\n"
        f"【本阶段文学职责】致命断章 (Cliffhanger) —— 围绕本章终局危机引爆全新维度杀机或惊悚真相\n"
        f"【起局前置态势】\n{start_state_str}\n"
        f"【结尾行动执行链】\n{actions_str}\n"
        f"【绝杀卡点终态】\n{end_state_str}\n"
        f"【核心卡点目标】{target_outcome}\n"
        f"【篇幅预算】{budget.get('min_chars', 150)} ~ {budget.get('max_chars', 300)} 字\n\n"
        f"【创作约束与边界】\n{constraints_str}\n\n"
        f"【任务指示】\n"
        f"请将上述结尾行动与终局悬念转化为极具压迫感的小说正文。\n"
        f"最后一句话必须一刀封喉、戛然而止，杜绝多余交代的闲笔，直接输出正文。"
    )
    return prompt


def build_opening_prompt(chapter_data: Dict[str, Any]) -> str:
    """
    破题黄金切片任务输入构造 (tara5_opening)
    【信息可用性】：提供上一章结尾承接、本章开局状态与开篇行动计划。
    """
    title = chapter_data.get("chapter_title", "未命名章节")
    prev_ending = chapter_data.get("prev_chapter_ending", "（本篇开门见山）").strip()
    outline_list = chapter_data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    plans = chapter_data.get("stage_plans", {})
    op_plan = plans.get(STAGE_OPENING, {})

    facts_str = format_confirmed_facts(chapter_data.get("confirmed_facts", []))
    actions_str = format_action_chain(op_plan.get("action_chain", []))
    start_state_str = format_narrative_state(op_plan.get("start_state", {}))
    end_state_str = format_narrative_state(op_plan.get("end_state", {}))
    target_outcome = op_plan.get("target_outcome", "顺利承接上章余势，完成破题并确立开篇态势")
    constraints_str = format_constraints(chapter_data.get("constraints", {}))
    budget = op_plan.get("length_budget", {"min_chars": 150, "max_chars": 300})

    prompt = (
        f"【章节标题】{title}\n\n"
        f"【上一章结尾承接】\n{prev_ending}\n\n"
        f"【故事总章纲】\n{outline_str}\n\n"
        f"【已确认事实基座】\n{facts_str}\n\n"
        f"【本阶段文学职责】黄金破题 (Opening) —— 承接上章悬念余势，以强动作或即时处境开门见山\n"
        f"【开局初始态势】\n{start_state_str}\n"
        f"【开篇行动执行链】\n{actions_str}\n"
        f"【破题定势终态】\n{end_state_str}\n"
        f"【目标达成结果】{target_outcome}\n"
        f"【篇幅预算】{budget.get('min_chars', 150)} ~ {budget.get('max_chars', 300)} 字\n\n"
        f"【创作约束与边界】\n{constraints_str}\n\n"
        f"【任务指示】\n"
        f"请将上述开篇态势转化为自然流畅的小说正文。\n"
        f"杜绝大段死寂的环境空话，在正文结尾确立明确的行动定势，直接输出正文。"
    )
    return prompt


def build_ramp_up_prompt(
    chapter_data: Dict[str, Any],
    opening_context: Optional[str] = None,
    opening_anchor_end: Optional[str] = None,
    climax_anchor_start: Optional[str] = None,
) -> str:
    """
    前向蓄水引信桥任务输入构造 (tara5_ramp_up)
    【信息可用性】：提供已生成开篇的必要上下文、开篇末锚、高潮首锚、蓄水桥行动链、起止状态与篇幅预算。
    """
    title = chapter_data.get("chapter_title", "未命名章节")
    outline_list = chapter_data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    plans = chapter_data.get("stage_plans", {})
    ru_plan = plans.get(STAGE_RAMP_UP, {})

    # 获取锚点：优先从参数传入，次之从 chapter_data.segments 提取
    segs = chapter_data.get("segments", {})
    start_anchor = opening_anchor_end or segs.get(STAGE_OPENING, {}).get("anchor_end", "")
    target_anchor = climax_anchor_start or segs.get(STAGE_CLIMAX, {}).get("anchor_start", "")

    # 获取前置上下文
    op_ctx = opening_context or segs.get(STAGE_OPENING, {}).get("text", "")
    op_ctx = op_ctx.strip()
    if len(op_ctx) > 280:
        op_ctx = f"...{op_ctx[-280:]}"

    actions_str = format_action_chain(ru_plan.get("action_chain", []))
    start_state_str = format_narrative_state(ru_plan.get("start_state", {}))
    end_state_str = format_narrative_state(ru_plan.get("end_state", {}))
    target_outcome = ru_plan.get("target_outcome", "层层升级矛盾蓄水，使危机态势达到临界爆发点")
    constraints_str = format_constraints(chapter_data.get("constraints", {}))
    budget = ru_plan.get("length_budget", {"min_chars": 600, "max_chars": 900})

    prompt = (
        f"【章节标题】{title}\n\n"
        f"【故事总章纲】\n{outline_str}\n\n"
        f"【前置破题局势上下文】\n{op_ctx}\n\n"
        f"【起始铁锚 (破题定势句)】{start_anchor}\n"
        f"【终点铁锚 (Target Anchor · 高潮爆发首句)】{target_anchor}\n\n"
        f"【本阶段文学职责】前向蓄水引信桥 (Ramp Up) —— 从起始铁锚出发，推进铺垫蓄水行动链\n"
        f"【桥接起始态势】\n{start_state_str}\n"
        f"【推进行动执行链】\n{actions_str}\n"
        f"【临界引爆态势】\n{end_state_str}\n"
        f"【目标达成结果】{target_outcome}\n"
        f"【篇幅预算】{budget.get('min_chars', 600)} ~ {budget.get('max_chars', 900)} 字\n\n"
        f"【创作约束与边界】\n{constraints_str}\n"
        f"• 行动归属规定：本段专注铺垫与蓄力，不得抢先解决高潮对抗核心。\n"
        f"• 强制咬合约束：正文最后一句必须严丝合缝、完全咬入指定的【终点铁锚】！\n\n"
        f"【任务指示】\n"
        f"请从起始铁锚推进情节，层层加码矛盾，直至严丝合缝咬入终点铁锚，直接输出正文。"
    )
    return prompt


def build_aftershock_prompt(
    chapter_data: Dict[str, Any],
    climax_context: Optional[str] = None,
    climax_anchor_end: Optional[str] = None,
    cliffhanger_anchor_start: Optional[str] = None,
) -> str:
    """
    余波转折中继桥任务输入构造 (tara5_aftershock)
    【信息可用性】：提供已生成高潮的必要上下文、高潮末锚、结尾首锚、余波行动链、起止状态与篇幅预算。
    """
    title = chapter_data.get("chapter_title", "未命名章节")
    outline_list = chapter_data.get("chapter_outline", [])
    outline_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(outline_list)])

    plans = chapter_data.get("stage_plans", {})
    af_plan = plans.get(STAGE_AFTERSHOCK, {})

    segs = chapter_data.get("segments", {})
    start_anchor = climax_anchor_end or segs.get(STAGE_CLIMAX, {}).get("anchor_end", "")
    target_anchor = cliffhanger_anchor_start or segs.get(STAGE_CLIFFHANGER, {}).get("anchor_start", "")

    cl_ctx = climax_context or segs.get(STAGE_CLIMAX, {}).get("text", "")
    cl_ctx = cl_ctx.strip()
    if len(cl_ctx) > 280:
        cl_ctx = f"...{cl_ctx[-280:]}"

    actions_str = format_action_chain(af_plan.get("action_chain", []))
    start_state_str = format_narrative_state(af_plan.get("start_state", {}))
    end_state_str = format_narrative_state(af_plan.get("end_state", {}))
    target_outcome = af_plan.get("target_outcome", "承接对决余波，战局消化与复盘，过渡至断章引爆点")
    constraints_str = format_constraints(chapter_data.get("constraints", {}))
    budget = af_plan.get("length_budget", {"min_chars": 400, "max_chars": 600})

    prompt = (
        f"【章节标题】{title}\n\n"
        f"【故事总章纲】\n{outline_str}\n\n"
        f"【高潮战果局势上下文】\n{cl_ctx}\n\n"
        f"【起始铁锚 (高潮定势末句)】{start_anchor}\n"
        f"【终点铁锚 (Target Anchor · 断章引爆首句)】{target_anchor}\n\n"
        f"【本阶段文学职责】余波转折中继桥 (Aftershock) —— 战后喘息、心理震颤、消化战局与复盘\n"
        f"【高潮落定起始态势】\n{start_state_str}\n"
        f"【余波行动执行链】\n{actions_str}\n"
        f"【下一轮危机引爆前态势】\n{end_state_str}\n"
        f"【目标达成结果】{target_outcome}\n"
        f"【篇幅预算】{budget.get('min_chars', 400)} ~ {budget.get('max_chars', 600)} 字\n\n"
        f"【创作约束与边界】\n{constraints_str}\n"
        f"• 行动归属规定：本段专注余波复盘与中继，不得重复完成高潮已决出的对战，也不得以无关水字数充斥篇幅。\n"
        f"• 强制咬合约束：正文最后一句必须顺理成章、完全咬入指定的【终点铁锚】！\n\n"
        f"【任务指示】\n"
        f"请从高潮末锚平稳过渡，描写生理与心理余震，并顺理成章咬入终点铁锚，直接输出正文。"
    )
    return prompt


# ==============================================================================
# 统一入口函数
# ==============================================================================
def build_task_input(
    stage: str,
    chapter_data: Dict[str, Any],
    context: Optional[str] = None,
    start_anchor: Optional[str] = None,
    target_anchor: Optional[str] = None,
) -> Tuple[str, str]:
    """
    通用构造入口：返回 (system_prompt, user_prompt)
    无论用于微调训练集 ShareGPT 编译，还是用于在线反向装配推理，一律调用此函数！
    """
    st = normalize_stage_name(stage)
    sys_prompt = get_system_prompt(st)

    if st == STAGE_OPENING:
        user_prompt = build_opening_prompt(chapter_data)
    elif st == STAGE_RAMP_UP:
        user_prompt = build_ramp_up_prompt(
            chapter_data,
            opening_context=context,
            opening_anchor_end=start_anchor,
            climax_anchor_start=target_anchor,
        )
    elif st == STAGE_CLIMAX:
        user_prompt = build_climax_prompt(chapter_data)
    elif st == STAGE_AFTERSHOCK:
        user_prompt = build_aftershock_prompt(
            chapter_data,
            climax_context=context,
            climax_anchor_end=start_anchor,
            cliffhanger_anchor_start=target_anchor,
        )
    elif st == STAGE_CLIFFHANGER:
        user_prompt = build_cliffhanger_prompt(chapter_data)
    else:
        raise ValueError(f"不支持的阶段: {stage}")

    return sys_prompt, user_prompt
