"""Read-only snapshot of a running repair job; only writes this review directory."""
import collections,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.tara5_quality import validate_source_chapter
OUT=Path(__file__).resolve().parent
WORK=ROOT/'output/cleaning-repair-2026-10-24'
records=[];unreadable=[]
for p in sorted((WORK/'records').glob('*/*.json')):
 try:
  raw=p.read_bytes();r=json.loads(raw);r['_record_path']=str(p.relative_to(ROOT));r['_record_hash']=hashlib.sha256(raw).hexdigest();records.append(r)
 except Exception as e: unreadable.append({'file':str(p),'error':type(e).__name__})
counts=collections.Counter(r['status'] for r in records);bybook={};issues=[];review_issues=[];shorts=[]
for r in records:
 book=r['key'].split('/')[0]
 bybook.setdefault(book,collections.Counter())[r['status']]+=1
 if r['status']!='approved':continue
 p=ROOT/r['candidate']
 try:
  raw=p.read_bytes();d=json.loads(raw);errors=validate_source_chapter(d,ROOT)
  if errors: issues.append({'key':r['key'],'candidate':r['candidate'],'errors':errors,'sha256':hashlib.sha256(raw).hexdigest()})
  if r.get('review')!='codex_source_review_v2':
   rev=WORK/'responses'/f"{r['key']}-review-{r.get('attempts',1)-1}.json"
   if not rev.exists():review_issues.append({'key':r['key'],'error':'review_response_missing','path':str(rev.relative_to(ROOT))})
   elif json.loads(rev.read_text()).get('approved') is not True:review_issues.append({'key':r['key'],'error':'review_response_not_approved','path':str(rev.relative_to(ROOT))})
  tiny={st:len(v['text']) for st,v in d['segments'].items() if len(v['text'])<20}
  if tiny:shorts.append({'key':r['key'],'stages':tiny})
 except Exception as e:issues.append({'key':r['key'],'error':str(e)})
errors=collections.Counter(e for row in issues for e in row.get('errors',[row.get('error','')]))
summary={'snapshot_time':datetime.datetime.now().astimezone().isoformat(), 'records':len(records),'counts':dict(counts),'by_book':bybook,
 'approved_failing_publish_checks':len(issues),'publish_errors':dict(errors),'review_provenance_failures':len(review_issues),'short_segment_chapters':len(shorts),'unreadable_records':unreadable}
for filename,data in [('summary.json',summary),('publish_check_failures.json',issues),('review_provenance_failures.json',review_issues),('short_segments_review_candidates.json',shorts),('records_snapshot.json',records)]:
 (OUT/filename).write_text(json.dumps(data,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
