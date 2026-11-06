#!/usr/bin/env python3
"""Independent post-publication integrity check; exits nonzero on any regression."""
import collections
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.tara5_quality import validate_source_chapter
from src.migrate_to_tara5_action_v1 import build_sharegpt_samples_from_v1
from src.prompt_builder import SYSTEM_PROMPTS, VALID_STAGES

BASE=ROOT/'data/tara5_action_v1'
REPORT=ROOT/'reports/cleaning-repair-2026-10-24'

def main():
    failures=[];all_rows=[];chapters=0;book_counts={}
    for book in sorted(BASE.iterdir()):
        files=sorted((book/'sliced').glob('*.json'))
        if not files: continue
        expected=[]
        for path in files:
            d=json.loads(path.read_text());chapters+=1
            errors=validate_source_chapter(d,ROOT)
            if errors: failures.append({'file':str(path.relative_to(ROOT)),'errors':errors})
            if d['annotation_status']['needs_review']: failures.append({'file':str(path),'errors':['review pending in active slices']})
            rows=build_sharegpt_samples_from_v1(d);expected.extend(rows)
            for stage,row in zip(VALID_STAGES,rows):
                if row['system']!=SYSTEM_PROMPTS[stage] or '【章节标题】' not in row['conversations'][0]['value']:
                    failures.append({'file':str(path),'errors':['system/user roles reversed']})
        actual=json.loads((book/'dataset/tara5_action_v1_train.json').read_text())
        if actual!=expected: failures.append({'file':book.name,'errors':['per-book export mismatch']})
        all_rows.extend(expected);book_counts[book.name]=len(files)
    target=BASE/'all_books/dataset'
    full=json.loads((target/'tara5_action_v1_all_train.json').read_text())
    ready=json.loads((target/'tara5_action_v1_ready_2560.json').read_text())
    long=json.loads((target/'tara5_action_v1_long_context.json').read_text())
    if all_rows!=full: failures.append({'errors':['merged export mismatch']})
    if len({r['id'] for r in full})!=len(full): failures.append({'errors':['duplicate sample IDs']})
    serialize=lambda r:json.dumps(r,ensure_ascii=False,sort_keys=True)
    if collections.Counter(map(serialize,ready+long))!=collections.Counter(map(serialize,full)):
        failures.append({'errors':['ready/long partitions lost or duplicated samples']})
    # Check lengths recorded at compile-time; tokenizer preflight is a separate check.
    lengths={r['id']:r['tokens'] for r in json.loads((REPORT/'sample_lengths.json').read_text())}
    if any(lengths[r['id']]>2560 for r in ready) or any(lengths[r['id']]<=2560 for r in long): failures.append({'errors':['length partition invalid']})
    quarantine=json.loads((REPORT/'quarantine_manifest.json').read_text())
    for record in quarantine:
        old=Path(record['file']);book=old.parent.parent.name
        path=BASE/'quarantine'/book/'sliced'/old.name
        if not path.exists() or not json.loads(path.read_text())['annotation_status']['needs_review']:
            failures.append({'errors':['quarantine missing or status false'],'file':str(path)})
    summary=json.loads((REPORT/'summary.json').read_text())
    if chapters+len(quarantine)!=summary['original_chapters']: failures.append({'errors':['chapter accounting mismatch']})
    result={'passed':not failures,'chapters':chapters,'samples':len(full),'ready':len(ready),'long_context':len(long),
            'quarantined':len(quarantine),'books':book_counts,'failures':failures}
    (REPORT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
