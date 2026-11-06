import json,sys,re,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.repair_tara5_data import validate_annotation,build_chapter,dump,WORK
from src.tara5_quality import validate_source_chapter
HERE=Path(__file__).resolve().parent
assign={r['key'].split('/')[1]:r for r in json.loads((HERE.parent/'assignment_2.json').read_text())}
def source_units(raw):
    units=[]
    for line in re.finditer(r'[^\r\n]+',raw):
        for sentence in re.finditer(r'.+?(?:[。！？!?][”」』\"]*|$)',line.group()):
            a,b=line.start()+sentence.start(),line.start()+sentence.end()
            while a<b and raw[a].isspace():a+=1
            while b>a and raw[b-1].isspace():b-=1
            if a<b:units.append({'start':a,'end':b,'text':raw[a:b]})
    return units

def put(ch,ends,characters,plans,start=3):
    ch=f'ch_{ch:04d}';r=assign[ch];book=r['key'].split('/')[0]
    path=ROOT/r['file'];oldpath=path if path.exists() else ROOT/'data/tara5_action_v1/quarantine'/book/'sliced'/path.name
    old=json.loads(oldpath.read_text());rp=next((ROOT/'data'/book/'raw').glob(ch+'_*'));raw=rp.read_text();units=source_units(raw)
    a={'is_narrative':True,'body_start':start,'stage_ends':ends,'exclusion_reason':'仅略去原文开头的重复章节标题；完整保留全部叙事。','chapter_outline':[x[1] for x in plans],'facts':[],
       'characters':[{'character_id':f'c{i+1}','name':name,'goal':goal,'known_info':[],'unknown_info':[],'misjudgment':''} for i,(name,goal) in enumerate(characters)],
       'plans':[{'actor_id':f'c{who}','action':action,'result':result,'evidence_units':ev} for who,action,result,ev in plans]}
    prevpaths=list(rp.parent.glob(f'ch_{int(ch[3:])-1:04d}_*'));pp=prevpaths[0] if len(prevpaths)==1 else None
    prev=[u['text'] for u in source_units(pp.read_text())][-6:] if pp else []
    # No unsupported facts: existing previous context is retained, facts left empty.
    validate_annotation(a,units,prev)
    d=build_chapter(old,a,raw,units,rp,pp,prev,book)
    d['annotation_status']['annotator']='codex_source_review_v2'
    d['annotation_status']['review_method']='Codex subtask read the complete local source and directly annotated semantic stage boundaries, actors, actions and evidence; deterministic source validation passed. Not human or external independent-model review.'
    errors=validate_source_chapter(d,ROOT)
    if errors:raise ValueError((ch,errors))
    dest=WORK/'candidates'/book/path.name
    dump(HERE/(ch+'_annotation.json'),a);dump(dest,d)
    dump(WORK/'records'/book/(ch+'.json'),{'key':r['key'],'file':r['file'],'status':'approved','candidate':str(dest.relative_to(ROOT)),'review':'codex_source_review_v2','raw_sha256':hashlib.sha256(raw.encode()).hexdigest()})
    print('approved',r['key'])
