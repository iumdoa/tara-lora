import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.repair_tara5_data import source_units
rows=json.loads((ROOT/'reports/final-cleaning-pass-2026-10-24/assignment_1.json').read_text())
for r in rows[int(sys.argv[1]):int(sys.argv[2])]:
 book,ch=r['key'].split('/');p=next((ROOT/'data'/book/'raw').glob(ch+'_*.txt'))
 print('\n===',r['key'],p.name,'===')
 for i,u in enumerate(source_units(p.read_text())):print(f'{i} {u["text"]}')
