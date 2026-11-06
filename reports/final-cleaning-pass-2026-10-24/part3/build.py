import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.repair_tara5_data import source_units,validate_annotation,build_chapter,dump,WORK
from src.tara5_quality import validate_source_chapter
HERE=Path(__file__).resolve().parent
records={r['key'].split('/')[1]:r for r in json.loads((HERE.parent/'assignment_3.json').read_text())}
def run():
 specs=json.loads((HERE/'annotations.json').read_text());out=[]
 for key,s in specs.items():
  r=records[key];p=ROOT/r['file'];oldpath=p if p.exists() else ROOT/'data/tara5_action_v1/quarantine/yaowu/sliced'/p.name
  old=json.loads(oldpath.read_text());rp=next((ROOT/'data/yaowu/raw').glob(key+'_*'));raw=rp.read_text();u=source_units(raw)
  pp=next((ROOT/'data/yaowu/raw').glob(f"ch_{int(key.split('_')[1])-1:04d}_*"),None)
  prev=[x['text'] for x in source_units(pp.read_text())][-6:] if pp else []
  names={n:f'c{i+1}' for i,(n,g) in enumerate(s['characters'])}
  a={'is_narrative':True,'body_start':s.get('start',3),'stage_ends':s['ends'],'exclusion_reason':s.get('exclusion','仅排除卷章标题；完整保留叙事正文。'),
   'chapter_outline':[x[1] for x in s['plans']], 'facts':[],
   'characters':[{'character_id':names[n],'name':n,'goal':g,'known_info':[],'unknown_info':[],'misjudgment':''} for n,g in s['characters']],
   'plans':[{'actor_id':names[n],'action':a,'result':z,'evidence_units':e} for n,a,z,e in s['plans']]}
  validate_annotation(a,u,prev);d=build_chapter(old,a,raw,u,rp,pp,prev,'yaowu')
  d['annotation_status']['annotator']='codex_source_review_v2';d['annotation_status']['review_method']='Codex子任务直接阅读完整原文，核对分段、行动主体与阶段证据；未使用外部API，非人类或独立外部模型审核。'
  errors=validate_source_chapter(d,ROOT)
  if errors: raise ValueError((key,errors))
  target=WORK/'candidates/yaowu'/p.name;dump(target,d)
  dump(WORK/'records/yaowu'/f'{key}.json',{'key':r['key'],'file':r['file'],'status':'approved','candidate':str(target.relative_to(ROOT)),'review':'codex_source_review_v2','raw_sha256':hashlib.sha256(raw.encode()).hexdigest()})
  dump(HERE/f'{key}.json',a);out.append(key)
 dump(HERE/'summary.json',{'approved':out,'count':len(out),'validation':'validate_annotation and validate_source_chapter passed'})
 print('approved',len(out))
if __name__=='__main__':run()
