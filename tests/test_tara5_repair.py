import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.repair_tara5_data import source_units, validate_annotation
from src.tara5_quality import validate_source_chapter
from src.schema_validator import validate_tara5_chapter
from src.migrate_to_tara5_action_v1 import build_sharegpt_samples_from_v1
from src.prompt_builder import VALID_STAGES, SYSTEM_PROMPTS
from src.train_lora_local import split_dataset_by_chapter, TaraShareGPTDataset
from src.repair_tara5_data import api, ServiceUnavailable, SERVICE_STOP

class RepairTests(unittest.TestCase):
    def test_insufficient_balance_stops_subsequent_requests(self):
        config={'DEEPSEEK_BASE_URL':'https://example.invalid/v1','DEEPSEEK_API_KEY':'test'}
        SERVICE_STOP.clear()
        self.addCleanup(SERVICE_STOP.clear)
        error=HTTPError('https://example.invalid',402,'Insufficient Balance',{},None)
        with patch('src.repair_tara5_data.urllib.request.urlopen',side_effect=error) as request:
            with self.assertRaises(ServiceUnavailable): api('system',{},config)
            with self.assertRaises(ServiceUnavailable): api('system',{},config)
            self.assertEqual(request.call_count,1)

    def fixture(self,root):
        raw='甲推开门。\n乙走进来。\n甲递出信。\n乙读完信。\n两人约定明日见。'
        folder=root/'data/book/raw';folder.mkdir(parents=True)
        path=folder/'ch_0002_会面.txt';path.write_text(raw)
        prev=folder/'ch_0001_约定.txt';prev.write_text('甲已收到信。')
        import hashlib
        d={'schema_version':'tara5_action_v1','chapter_id':'ch_0002','source_novel':'book','source_key':'book',
           'chapter_title':'会面','chapter_outline':['甲乙见面传信'],
           'prev_chapter_ending':'甲已收到信。',
           'confirmed_facts':[{'fact':'甲收到信','evidence':'甲已收到信。','source_scope':'previous_chapter','source_path':str(prev.relative_to(root))}],
           'characters':[{'character_id':'c1','name':'甲','goal':'交信'}],
           'constraints':{'pov':'第三人称','negative_constraints':[],'allowed_creative_freedom':[]},
           'annotation_status':{'needs_review':False},'segments':{},'stage_plans':{},
           'source_info':{'raw_path':str(path.relative_to(root)),'sha256':hashlib.sha256(raw.encode()).hexdigest(),'boundary_policy':'disjoint_source_spans'}}
        for stage,unit in zip(VALID_STAGES,source_units(raw)):
            text=unit['text'];n=len(text)
            d['segments'][stage]={'text':text,'anchor_start':text,'anchor_end':text,'char_count':n,
                                 'raw_char_offset_start':unit['start'],'raw_char_offset_end':unit['end']}
            d['stage_plans'][stage]={'stage_name':stage,'responsibility':'会面','preconditions':[],'target_outcome':'推进',
                'length_budget':{'min_chars':n,'max_chars':n},'start_state':{},'end_state':{},
                'action_chain':[{'action_id':stage,'actor_id':'c1','action':'传信','result':'推进','evidence':text,
                                 'evidence_spans':[{'start':unit['start'],'end':unit['end']}]}]}
        return d

    def test_source_offsets_detect_invented_text_and_omissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=self.fixture(root)
            self.assertEqual(validate_source_chapter(d,root),[])
            bad=copy.deepcopy(d);bad['segments']['aftershock']['text']='玄京城毁灭。'
            self.assertTrue(any('原文精确' in x for x in validate_source_chapter(bad,root)))
            bad=copy.deepcopy(d);bad['segments']['climax']['raw_char_offset_start']+=1
            self.assertTrue(any('遗漏' in x for x in validate_source_chapter(bad,root)))

    def test_reference_and_count_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.fixture(Path(tmp))
            d['stage_plans']['climax']['action_chain'][0]['actor_id']='not-a-character'
            d['segments']['climax']['char_count']=999
            ok,errors=validate_tara5_chapter(d)
            self.assertFalse(ok)
            self.assertTrue(any('actor_id' in x for x in errors))
            self.assertTrue(any('char_count' in x for x in errors))

    def test_future_evidence_is_not_accepted_as_previous_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=self.fixture(root)
            d['confirmed_facts'][0]['source_path']=d['source_info']['raw_path']
            self.assertTrue(any('先前章节' in x for x in validate_source_chapter(d,root)))

    def test_export_roles_and_ids_are_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=self.fixture(Path(tmp));rows=build_sharegpt_samples_from_v1(d)
            other=copy.deepcopy(d);other['source_key']='another-book'
            second=build_sharegpt_samples_from_v1(other)
            self.assertEqual(len({x['id'] for x in rows+second}),10)
            for stage,row in zip(VALID_STAGES,rows):
                self.assertEqual(row['system'],SYSTEM_PROMPTS[stage])
                self.assertIn('【章节标题】',row['conversations'][0]['value'])
                self.assertEqual(row['conversations'][1]['value'],d['segments'][stage]['text'])

    def test_same_chapter_number_in_different_books_splits_independently(self):
        rows=[{'source_key':book,'chapter_id':f'ch_{i:04d}','id':f'{book}:{i}'} for book in ['a','b'] for i in range(1,6)]
        train,val=split_dataset_by_chapter(rows,.2)
        self.assertEqual(len(train),8);self.assertEqual(len(val),2)
        self.assertEqual({(r['source_key'],r['chapter_id']) for r in val},{('a','ch_0005'),('b','ch_0005')})

    def test_unreviewed_samples_are_blocked_before_tokenization(self):
        row={'id':'bad','annotation_status':{'needs_review':True}}
        with self.assertRaisesRegex(ValueError,'内容复核'):
            TaraShareGPTDataset([row],None)

if __name__=='__main__': unittest.main()
