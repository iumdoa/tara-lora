#!/usr/bin/env python3
"""Validate all staged chapters, compile datasets, and publish a repair run.

Preserves excluded chapters in quarantine and long samples in a separate file.
Default is staging/verification only; --apply publishes the validated artifacts.
"""
import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.repair_tara5_data import WORK, dump
from src.tara5_quality import validate_source_chapter
from src.migrate_to_tara5_action_v1 import build_sharegpt_samples_from_v1

REPORT=ROOT/'reports/cleaning-repair-2026-10-24'
BASE=ROOT/'data/tara5_action_v1'

def compile_run(tokenizer, max_tokens=2560, allow_pending=False):
    expected=sorted(BASE.glob('*/sliced/*.json'))
    # Original manifest is frozen before publication, allowing repeat verification.
    manifest=WORK/'publication_input.json'
    if not manifest.exists(): dump(manifest,[str(p.relative_to(ROOT)) for p in expected])
    expected=[ROOT/p for p in json.loads(manifest.read_text())]
    candidates=[];quarantine=[];missing=[]
    for old_path in expected:
        book=old_path.parent.parent.name
        key=old_path.name.split('_',2)[:2];chapter_id='_'.join(key)
        record_path=WORK/'records'/book/(chapter_id+'.json')
        if not record_path.exists():
            if allow_pending:
                quarantine.append({'key':f'{book}/{chapter_id}','file':str(old_path.relative_to(ROOT)),
                    'status':'pending_service','reasons':['标注服务余额不足，尚未执行语义重标注；不代表已确认原文有错']})
            else: missing.append(str(old_path))
            continue
        record=json.loads(record_path.read_text())
        if record['status']=='pending_service' and not allow_pending:
            missing.append(str(old_path));continue
        if record['status']!='approved': quarantine.append(record);continue
        data=json.loads((ROOT/record['candidate']).read_text())
        # Keep trailing source punctuation (e.g. an ellipsis) even if the model
        # treated it as decoration. Never manufacture or silently discard it.
        import re
        suffix=data['source_info'].get('excluded_suffix','')
        if suffix.strip() and not re.search(r'[^\s.…·—\-*＊]',suffix):
            raw=(ROOT/data['source_info']['raw_path']).read_text()
            seg=data['segments']['cliffhanger'];hi=len(raw.rstrip())
            seg['raw_char_offset_end']=hi
            seg['text']=raw[seg['raw_char_offset_start']:hi]
            seg['char_count']=len(seg['text'])
            seg['anchor_end']=suffix.strip()
            data['stage_plans']['cliffhanger']['length_budget']={'min_chars':max(1,int(len(seg['text'])*.85)),'max_chars':int(len(seg['text'])*1.15)+1}
            data['source_info']['excluded_suffix']=raw[hi:]
        # Replace inherited combat-only constraints with honest writing boundaries.
        data['constraints']={'pov':'保持原作叙事视角',
            'author_hard_requirements':['遵守本阶段行动与结果，保持原作语气'],
            'negative_constraints':['不新增未经规划的关键角色或事件','不改写既定因果和结果','不重复相邻阶段正文'],
            'allowed_creative_freedom':['措辞、动作细节、对白语气与节奏']}
        errors=validate_source_chapter(data,ROOT)
        if errors:
            quarantine.append({**record,'status':'quarantined','reasons':errors});continue
        candidates.append((record,data))
    if missing: raise ValueError(f'{len(missing)} 章尚无处理记录，拒绝发布；例如 {missing[:3]}')
    datasets=collections.defaultdict(list);token_rows=[]
    for record,data in candidates:
        rows=build_sharegpt_samples_from_v1(data)
        for row in rows:
            messages=[{'role':'system','content':row['system']},{'role':'user','content':row['conversations'][0]['value']},
                      {'role':'assistant','content':row['conversations'][1]['value']}]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
            ids=tokenizer(rendered,add_special_tokens=False)['input_ids']
            token_rows.append({'id':row['id'],'source_key':row['source_key'],'chapter_id':row['chapter_id'],
                              'task_type':row['task_type'],'tokens':len(ids),'partition':'ready' if len(ids)<=max_tokens else 'long_context'})
        datasets[record['key'].split('/')[0]].extend(rows)
    all_rows=[row for book in sorted(datasets) for row in datasets[book]]
    assert len({r['id'] for r in all_rows})==len(all_rows),'sample IDs not unique'
    sizes={row['id']:row['tokens'] for row in token_rows}
    ready=[row for row in all_rows if sizes[row['id']]<=max_tokens]
    long=[row for row in all_rows if sizes[row['id']]>max_tokens]
    if not ready: raise ValueError('No training-ready samples; refusing publication')
    stage=Path(tempfile.mkdtemp(prefix='publication-',dir=WORK))
    for book in sorted({p.parent.parent.name for p in expected}):
        dump(stage/book/'dataset/tara5_action_v1_train.json',datasets[book])
        dump(stage/book/'dataset/tara5_action_v1_ready_2560.json',[r for r in datasets[book] if sizes[r['id']]<=max_tokens])
    dest=stage/'all_books/dataset'
    dump(dest/'tara5_action_v1_all_train.json',all_rows)
    dump(dest/'tara5_action_v1_ready_2560.json',ready)
    dump(dest/'tara5_action_v1_long_context.json',long)
    for record,data in candidates:
        target=stage/record['key'].split('/')[0]/'sliced'/Path(record['file']).name
        dump(target,data)
    dump(REPORT/'quarantine_manifest.json',quarantine)
    dump(REPORT/'sample_lengths.json',token_rows)
    counts={'original_chapters':len(expected),'approved_chapters':len(candidates),'quarantined_chapters':len(quarantine),'pending_service_chapters':sum(r['status']=='pending_service' for r in quarantine),
        'all_samples':len(all_rows),'ready_samples':len(ready),'long_context_samples':len(long),
        'max_tokens':max_tokens,'longest':max(sizes.values()),
        'by_task':{task:dict(collections.Counter(x['partition'] for x in token_rows if x['task_type']==task)) for task in sorted({x['task_type'] for x in token_rows})},
        'by_book':{book:{'chapters':len(rows)//5,'samples':len(rows)} for book,rows in datasets.items()}}
    dump(REPORT/'summary.json',counts)
    dump(REPORT/'staged_release.json',{'directory':str(stage.relative_to(ROOT)),
        'status':'partial_service_blocked' if counts['pending_service_chapters'] else 'reviewed'})
    return stage,quarantine,counts

def publish(stage,quarantine):
    if not (WORK/'before.tar.gz').exists(): raise ValueError('Missing original snapshot')
    for record in quarantine:
        original=ROOT/record['file'];book=original.parent.parent.name
        target=BASE/'quarantine'/book/'sliced'/original.name
        if original.exists():
            data=json.loads(original.read_text())
            data['annotation_status']={'needs_review':True,'annotator':'quarantined_by_source_repair_v2',
                'review_items':[{'field':'chapter','reason':r} for r in record.get('reasons',[])], 'direct_evidence_fields':[]}
            dump(target,data)
            original.unlink()  # Retained in quarantine and the immutable before.tar.gz snapshot.
    for source in sorted(stage.rglob('*.json')):
        dump(BASE/source.relative_to(stage),json.loads(source.read_text()))
    for book_dir in sorted(BASE.iterdir()):
        if book_dir.name in ('examples','quarantine') or not (book_dir/'dataset').is_dir(): continue
        info={}
        for name,file in [('ready_2560','tara5_action_v1_ready_2560.json'),('full','tara5_action_v1_all_train.json' if book_dir.name=='all_books' else 'tara5_action_v1_train.json')]:
            info[f'tara5_{book_dir.name}_{name}']={'file_name':file,'formatting':'sharegpt','columns':{'messages':'conversations','system':'system'}}
        dump(book_dir/'dataset/dataset_info_snippet.json',info)
    dump(REPORT/'publication.json',{'published':True,'backup':str((WORK/'before.tar.gz').relative_to(ROOT))})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');parser.add_argument('--allow-pending',action='store_true',help='Explicit partial release: preserve unprocessed chapters in quarantine');args=parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-1.5B-Instruct',local_files_only=True)
    stage,quarantine,summary=compile_run(tokenizer,allow_pending=args.allow_pending)
    if args.apply: publish(stage,quarantine)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
