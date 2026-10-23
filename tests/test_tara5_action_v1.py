#!/usr/bin/env python3
"""
TARA-5 (tara5_action_v1) 全面回归与协议合规性测试套件
"""

import sys
import json
import unittest
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
LORA_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(LORA_ROOT))

from src.schema_validator import validate_tara5_chapter
from src.prompt_builder import (
    STAGE_OPENING,
    STAGE_RAMP_UP,
    STAGE_CLIMAX,
    STAGE_AFTERSHOCK,
    STAGE_CLIFFHANGER,
    VALID_STAGES,
    build_task_input,
)
from src.migrate_to_tara5_action_v1 import (
    migrate_single_chapter,
    build_sharegpt_samples_from_v1,
)
from src.assembly import safe_assemble_chapter, strip_exact_anchor_prefix
from src.train_lora_local import split_dataset_by_chapter


class TestTara5ActionV1(unittest.TestCase):
    def setUp(self):
        # 构造标准合规测试章节
        self.sample_legacy = {
            "chapter_id": "ch_0010",
            "chapter_index": 10,
            "chapter_title": "第十章 幽谷探秘",
            "prev_chapter_ending": "雾气中传出一阵阴冷的呼啸，两人拔剑在手，死死盯住谷口裂隙。",
            "chapter_outline": [
                "两人小心翼翼穿越幽谷前哨，识破陷阱",
                "在毒瘴深处遭遇双头阴蟒，主角以符箓诱敌周旋",
                "阴蟒狂暴扑杀，主角抓住弱点飞身刺瞎其主眼，配合斩下毒囊",
                "战后二人脱力包扎，搜集蛇蜕灵材并复盘变局",
                "谷底石门轰然闭合，地底传来更为骇人的地吼，新危机卡点"
            ],
            "truth_base": {
                "protagonist_plan": "利用地形周旋，击杀妖蟒获取毒囊解毒",
                "antagonist_horizon": "阴蟒视二人为寻常猎物，恃毒瘴以力压制",
                "information_gap": "阴蟒未知主角怀有避毒宝珠与破甲雷符",
                "deadline": "毒瘴将在半柱香后彻底吞没此地"
            },
            "segments": {
                "opening": {
                    "anchor_start": "雾气中传出一阵阴冷的呼啸，两人拔剑在手，死死盯住谷口裂隙。",
                    "anchor_end": "脚下的碎石发出微弱的脆响，谷口的石碑已然被毒气蚀成了半截断桩。",
                    "text": "雾气中传出一阵阴冷的呼啸，两人拔剑在手，死死盯住谷口裂隙。\n\n脚下的碎石发出微弱的脆响，谷口的石碑已然被毒气蚀成了半截断桩。",
                    "word_count": 80
                },
                "ramp_up": {
                    "anchor_start": "脚下的碎石发出微弱的脆响，谷口的石碑已然被毒气蚀成了半截断桩。",
                    "anchor_end": "阴蟒庞大的躯体自浓雾深处轰然直立，猩红的蛇信吐出足有三尺长！",
                    "text": "脚下的碎石发出微弱的脆响，谷口的石碑已然被毒气蚀成了半截断桩。\n\n二人屏息凝神，踩着湿滑的苔藓缓缓推进。四周草木尽枯，腥臭的气味越发浓郁。忽然腥风大作，阴蟒庞大的躯体自浓雾深处轰然直立，猩红的蛇信吐出足有三尺长！",
                    "word_count": 650
                },
                "climax": {
                    "core_conflict": "主角以避毒珠硬顶毒雾，飞身刺瞎阴蟒主眼并斩杀",
                    "anchor_start": "阴蟒庞大的躯体自浓雾深处轰然直立，猩红的蛇信吐出足有三尺长！",
                    "anchor_end": "狂暴的气劲瞬间崩裂，阴蟒那颗狰狞的头颅重重砸在石壁上，毒血如暴雨般倾盆浇落。",
                    "text": "阴蟒庞大的躯体自浓雾深处轰然直立，猩红的蛇信吐出足有三尺长！\n\n轰然之间，毒雾呼啸扑面而来。主角脚踏迷踪步翻身而起，掌心破甲雷符瞬息亮起耀眼雷光，直贯蛇目！狂暴的气劲瞬间崩裂，阴蟒那颗狰狞的头颅重重砸在石壁上，毒血如暴雨般倾盆浇落。",
                    "word_count": 700
                },
                "aftershock": {
                    "anchor_start": "狂暴的气劲瞬间崩裂，阴蟒那颗狰狞的头颅重重砸在石壁上，毒血如暴雨般倾盆浇落。",
                    "anchor_end": "二人背靠背大口喘息，将灵剑缓缓插回鞘中，周围终于只剩下毒液腐蚀碎石的嗤嗤声。",
                    "text": "狂暴的气劲瞬间崩裂，阴蟒那颗狰狞的头颅重重砸在石壁上，毒血如暴雨般倾盆浇落。\n\n二人背靠背大口喘息，将灵剑缓缓插回鞘中，周围终于只剩下毒液腐蚀碎石的嗤嗤声。",
                    "word_count": 400
                },
                "cliffhanger": {
                    "anchor_start": "二人背靠背大口喘息，将灵剑缓缓插回鞘中，周围终于只剩下毒液腐蚀碎石的嗤嗤声。",
                    "anchor_end": "然而还不等两人松一口气，脚下的地脉骤然剧震，石门轰然紧闭，地底深处传出一声令人灵魂冻结的低吼！",
                    "text": "二人背靠背大口喘息，将灵剑缓缓插回鞘中，周围终于只剩下毒液腐蚀碎石的嗤嗤声。\n\n然而还不等两人松一口气，脚下的地脉骤然剧震，石门轰然紧闭，地底深处传出一声令人灵魂冻结的低吼！",
                    "word_count": 180
                }
            }
        }

    def test_01_schema_validation(self):
        """测试 Schema 校验：合规数据通过，违规数据准确报错"""
        migrated = migrate_single_chapter(self.sample_legacy, source_novel="test_novel")
        is_valid, errors = validate_tara5_chapter(migrated)
        self.assertTrue(is_valid, f"迁移生成的样本未能通过 Schema 校验: {errors}")

        # 测试异常：缺少必填字段
        bad_sample = dict(migrated)
        del bad_sample["stage_plans"]
        is_val, errs = validate_tara5_chapter(bad_sample)
        self.assertFalse(is_val)
        self.assertTrue(any("stage_plans" in e for e in errs))

        # 测试异常：预算 min > max
        bad_budget = json.loads(json.dumps(migrated))
        bad_budget["stage_plans"]["climax"]["length_budget"] = {"min_chars": 1000, "max_chars": 500}
        is_val, errs = validate_tara5_chapter(bad_budget)
        self.assertFalse(is_val)
        self.assertTrue(any("length_budget min_chars" in e for e in errs))

    def test_02_train_vs_infer_prompt_consistency(self):
        """测试同一阶段的训练输入与推理输入具有完全相同的提示词模板与语义"""
        migrated = migrate_single_chapter(self.sample_legacy, source_novel="test_novel")
        train_samples = build_sharegpt_samples_from_v1(migrated)
        self.assertEqual(len(train_samples), 5)

        for sample in train_samples:
            stage_task = sample["task_type"]
            # 找到对应 stage
            stage = [st for st in VALID_STAGES if f"tara5_{st}" == stage_task][0]

            train_system = sample["system"]
            train_user = sample["conversations"][0]["value"]

            # 调用推理入口
            infer_system, infer_user = build_task_input(stage, migrated)

            # 严格核对系统提示词与用户输入 100% 一致！
            self.assertEqual(train_system, infer_system, f"{stage} 训练与推理 system prompt 不一致！")
            self.assertEqual(train_user, infer_user, f"{stage} 训练与推理 user prompt 不一致！")

    def test_03_assembly_anchor_ownership_and_deduplication(self):
        """测试接缝装配：前段保留，后段剥离 (Prev-Retained, Next-Stripped)，禁止整行误删"""
        # 模拟 opening 与 ramp_up 共享 anchor_2
        anc_2 = "脚下的碎石发出微弱的脆响，谷口的石碑已然被毒气蚀成了半截断桩。"
        seg_opening = f"两人拔剑在手，死死盯住谷口裂隙。\n\n{anc_2}"
        # ramp_up 开头包含 anc_2，并在同一行紧接其他正文！
        seg_ramp_up = f"{anc_2}四周草木尽枯，腥臭的气味越发浓郁。\n\n二人屏息凝神。"
        seg_climax = "阴蟒庞大的躯体自浓雾深处轰然直立！主角飞身一刺，将其重创。"
        seg_aftershock = "主角大口喘息，将灵剑插回鞘中。周围只剩下腐蚀的嗤嗤声。"
        seg_cliff = "地底深处忽然传出一声低吼，石门轰然闭死！"

        segments = {
            "opening": seg_opening,
            "ramp_up": seg_ramp_up,
            "climax": seg_climax,
            "aftershock": seg_aftershock,
            "cliffhanger": seg_cliff
        }
        anchors = {"anchor_2": anc_2}

        assembled, success, logs = safe_assemble_chapter(
            title="测试章节",
            segments=segments,
            anchors=anchors
        )
        self.assertTrue(success)

        # 验证 1：全章中 anc_2 只出现恰好 1 次（前段保留，后段剥离重复）
        self.assertEqual(assembled.count(anc_2), 1, "anchor_2 应该且只应该在全章中保留一次！")

        # 验证 2：同一行紧接的正文必须完好保留，严禁误伤整行！
        self.assertIn("四周草木尽枯，腥臭的气味越发浓郁", assembled, "后段首行除锚点外的正文被错误丢弃！")

        # 验证 3：strip_exact_anchor_prefix 单元测试
        test_text = f"{anc_2}后面的同段重要正文。"
        stripped, did_strip = strip_exact_anchor_prefix(test_text, anc_2)
        self.assertTrue(did_strip)
        self.assertEqual(stripped, "后面的同段重要正文。")

    def test_04_dataset_split_no_leakage(self):
        """测试验证集切分：同章五段绝对不跨界，避免相邻上下文泄漏"""
        # 生成 5 个章节的数据 (每个章节 5 条样本，共 25 条)
        all_samples = []
        for ch_idx in range(1, 6):
            ch_data = dict(self.sample_legacy)
            ch_data["chapter_id"] = f"ch_{ch_idx:04d}"
            migrated = migrate_single_chapter(ch_data, source_novel="novel_a")
            samples = build_sharegpt_samples_from_v1(migrated)
            all_samples.extend(samples)

        self.assertEqual(len(all_samples), 25)

        # 按 20% 切分验证集 (应恰好为 1 章 = 5 条)
        train_split, val_split = split_dataset_by_chapter(all_samples, val_ratio=0.2)
        self.assertEqual(len(train_split), 20)
        self.assertEqual(len(val_split), 5)

        train_ch_ids = {s["chapter_id"] for s in train_split}
        val_ch_ids = {s["chapter_id"] for s in val_split}

        # 核心保证：训练集章节与验证集章节绝对没有交集！
        intersection = train_ch_ids.intersection(val_ch_ids)
        self.assertEqual(len(intersection), 0, f"发现跨集泄漏章节: {intersection}")


if __name__ == "__main__":
    unittest.main()
