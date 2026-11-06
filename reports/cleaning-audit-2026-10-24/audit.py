"""Read-only audit; writes sidecar issue markers, never rewrites training data."""
import collections, csv, hashlib, json, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.schema_validator import validate_tara5_chapter
from src.prompt_builder import VALID_STAGES, STAGE_TASK_MAP, build_task_input
OUT = pathlib.Path(__file__).resolve().parent
BASE = ROOT / 'data/tara5_action_v1'
issues, stats, manifests = [], collections.Counter(), []
def norm(t): return re.sub(r'\s+', '', t)
def mark(p, d, code, field, evidence, severity='P2', status='confirmed'):
    issues.append(dict(severity=severity,status=status,code=code,file=str(p.relative_to(ROOT)),chapter_id=d.get('chapter_id',''),field=field,evidence=str(evidence),needs_review=True))
all_rows=[]
for book in sorted(BASE.iterdir()):
    files=sorted((book/'sliced').glob('*.json'))
    if not files: continue
    stats['books']+=1
    rows=json.loads((book/'dataset/tara5_action_v1_train.json').read_text())
    all_rows.extend(rows)
    index=collections.defaultdict(list)
    for row in rows: index[(row['chapter_id'],row['task_type'])].append(row)
    for p in files:
        issue_start=len(issues)
        raw_bytes=p.read_bytes(); d=json.loads(raw_bytes); stats['chapters']+=1
        manifests.append({'file':str(p.relative_to(ROOT)), 'sha256':hashlib.sha256(raw_bytes).hexdigest()})
        ok, errors=validate_tara5_chapter(d)
        stats['schema_pass']+=ok
        if not ok: mark(p,d,'SCHEMA','',errors,'P1')
        candidates=list((ROOT/'data'/book.name/'raw').glob(d['chapter_id']+'_*.txt'))
        if not candidates and book.name=='daomu': candidates=list((ROOT/'data/raw').glob(d['chapter_id']+'_*.txt'))
        raw=candidates[0].read_text() if len(candidates)==1 else None
        stats['raw_matched']+=raw is not None
        if raw is None: mark(p,d,'SOURCE_UNRESOLVED','source_info','无法唯一匹配本地原始章节',status='needs_manual_review')
        ids={c['character_id'] for c in d['characters']}
        for i,f in enumerate(d['confirmed_facts']):
            if f['evidence'] in ('本阶段既定设定与局势信息差记录','本章开局死线与时间窗口要求') or '原著事实基座' in f['evidence']:
                mark(p,d,'PLACEHOLDER_EVIDENCE',f'confirmed_facts[{i}].evidence',f['evidence'],'P1')
        for st in VALID_STAGES:
            seg=d['segments'][st]; t=seg['text']; nt=norm(t); stats['segments']+=1
            if raw is not None and nt and nt not in norm(raw):
                mark(p,d,'TEXT_NOT_IN_SOURCE',f'segments.{st}.text',f'空白归一后不在原文连续出现；原文={candidates[0].relative_to(ROOT)}；片段开头={t[:100]}','P1','needs_manual_review')
            if seg.get('char_count')!=len(t): mark(p,d,'STALE_CHAR_COUNT',f'segments.{st}.char_count',f"stored={seg.get('char_count')}, actual={len(t)}")
            budget=d['stage_plans'][st]['length_budget']
            if not budget['min_chars']<=len(t)<=budget['max_chars']: mark(p,d,'LENGTH_OUTSIDE_BUDGET',f'stage_plans.{st}.length_budget',f'actual={len(t)}, budget={budget}',status='needs_manual_review')
            for a in ('anchor_start','anchor_end'):
                if not seg[a] or seg[a] not in t: mark(p,d,'ANCHOR_MISSING',f'segments.{st}.{a}',seg[a],'P1')
            for i,a in enumerate(d['stage_plans'][st]['action_chain']):
                if a['actor_id'] not in ids: mark(p,d,'ACTOR_ID_UNRESOLVED',f'stage_plans.{st}.action_chain[{i}].actor_id',f"actor_id={a['actor_id']}; valid={sorted(ids)}",'P1')
            for m in re.finditer(r'求月票|求推荐票|求订阅|求收藏|感谢.*打赏|(?:^|\n)\s*[Pp][Ss][:：]|未完待续|感觉一直在单机有人不',t):
                mark(p,d,'AUTHOR_NOTE_CANDIDATE',f'segments.{st}.text',t[max(0,m.start()-35):m.end()+75],'P1','confirmed')
            matches=index[(d['chapter_id'],STAGE_TASK_MAP[st])]
            if len(matches)!=1: mark(p,d,'TRAIN_ROW_COUNT',f'dataset.{st}',len(matches),'P1')
            else:
                prompt,system=build_task_input(stage=st,chapter_data=d)
                expected=[{'from':'human','value':prompt},{'from':'gpt','value':t.strip()}]
                if matches[0]['conversations']!=expected or matches[0]['system']!=system: mark(p,d,'TRAIN_SLICE_MISMATCH',f'dataset.{st}','训练数据与当前切片/提示构造不一致','P1')
        for i,st in enumerate(VALID_STAGES):
            for nxt in VALID_STAGES[i+1:]:
                a,b=norm(d['segments'][st]['text']),norm(d['segments'][nxt]['text'])
                if a and b and (a in b or b in a): mark(p,d,'NESTED_SEGMENTS',f'segments.{st}/{nxt}',f'整段包含；长度 {len(a)}/{len(b)}','P1')
        for st,nxt in zip(VALID_STAGES,VALID_STAGES[1:]):
            a,b=d['segments'][st]['text'].strip(),d['segments'][nxt]['text'].strip()
            anchor=d['segments'][st]['anchor_end']
            if len(anchor)>=8 and a.endswith(anchor) and b.startswith(anchor): mark(p,d,'SHARED_BOUNDARY_IN_TRAIN',f'segments.{st}/{nxt}',anchor,'P2')
        if book.name!='changsheng' and '玄京城，在刹那间变得支离破碎' in d['segments']['aftershock']['text']:
            mark(p,d,'CROSS_BOOK_CORRUPTION','segments.aftershock/cliffhanger; stage_plans','其他书目被写入《我的模拟长生路》的李凡、玄京城、还真剧情','P0')
        if book.name=='changsheng' and d['chapter_id']=='ch_0062':
            mark(p,d,'FUTURE_FACT_LEAK','confirmed_facts[1:4]; characters','开局已确认事实包含“至死不知”“导致捕仙失败”“道玄子祭出本命道基残剑”，原文均为本章推进后的事件','P1')
        if d.get('annotation_status',{}).get('needs_review') is False and len(issues)>issue_start:
            mark(p,d,'REVIEW_STATUS_CONFLICT','annotation_status','存在待处理问题，但原始 needs_review=false；本次旁路标记为 true','P1')
    stats['train_rows']+=len(rows)
global_path=BASE/'all_books/dataset/tara5_action_v1_all_train.json'
global_rows=json.loads(global_path.read_text())
canon=lambda row:json.dumps(row,ensure_ascii=False,sort_keys=True)
stats['global_rows']=len(global_rows)
stats['global_matches_books']=collections.Counter(map(canon,global_rows))==collections.Counter(map(canon,all_rows))
id_counts=collections.Counter(r['id'] for r in global_rows)
stats['duplicate_id_values']=sum(n>1 for n in id_counts.values())
stats['rows_with_duplicate_id']=sum(n for n in id_counts.values() if n>1)
duplicate_rows=[{'id':r['id'],'source_novel':r['source_novel'],'chapter_id':r['chapter_id'],'task_type':r['task_type']} for r in global_rows if id_counts[r['id']]>1]
(OUT/'duplicate_sample_ids.json').write_text(json.dumps(duplicate_rows,ensure_ascii=False,indent=2))
with (OUT/'issues.jsonl').open('w') as f:
    for n,x in enumerate(issues,1): x['issue_id']=f'AUDIT-{n:06d}'; f.write(json.dumps(x,ensure_ascii=False)+'\n')
with (OUT/'issues.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(issues[0]));writer.writeheader();writer.writerows(issues)
byfile=collections.defaultdict(list)
for x in issues: byfile[x['file']].append(x['issue_id'])
(OUT/'chapter_review_flags.json').write_text(json.dumps([{'file':p,'needs_review':True,'issue_ids':v} for p,v in byfile.items()],ensure_ascii=False,indent=2))
summary={'stats':dict(stats),'issue_counts':dict(collections.Counter(x['code'] for x in issues)),'affected_chapters':len(byfile),'issues_total':len(issues)}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(OUT/'input_manifest.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
