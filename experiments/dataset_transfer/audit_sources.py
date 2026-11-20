"""Public-source feasibility audit. No model or credential calls."""
import csv, hashlib, io, json
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/dataset_transfer_v1'
DATA=OUT/'data'


def fetch(url,name):
    path=DATA/name
    if not path.exists():
        with urlopen(url,timeout=90) as response: content=response.read()
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(content)
    return path


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    tree=json.loads(fetch('https://api.github.com/repos/mguti97/xGHub/git/trees/main?recursive=1','xghub_tree.json').read_text())
    print('XGHUB FILES',[(x['path'],x.get('size',0)) for x in tree['tree']],flush=True)
    readme=fetch('https://raw.githubusercontent.com/mguti97/xGHub/main/README.md','xghub_README.md').read_text()
    print('XGHUB README',readme[:10000],flush=True)
    for game in (1,2):
        rel=f'data/Sample_Game_{game}/Sample_Game_{game}_RawEventsData.csv'
        path=fetch('https://raw.githubusercontent.com/metrica-sports/sample-data/master/'+rel,f'metrica_{game}_events.csv')
        rows=list(csv.DictReader(path.open())); shots=[x for x in rows if x['Type']=='SHOT']
        print('METRICA',game,{'shot_count':len(shots),'columns':list(rows[0]),'shot_subtypes':sorted({x['Subtype'] for x in shots}),'example':shots[:2]},flush=True)
    comps=json.loads(fetch('https://raw.githubusercontent.com/hudl/open-data/master/data/competitions.json','statsbomb_competitions.json').read_text())
    print('COMPETITIONS', [{k:x[k] for k in ('competition_id','season_id','competition_name','season_name','competition_gender')} for x in comps if 'Women' in x['competition_name'] or x['competition_name']=='Premier League'],flush=True)


if __name__=='__main__': main()
