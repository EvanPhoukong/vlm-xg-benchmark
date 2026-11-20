"""Read-only final provenance and score audit; no API or fitting."""
import json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn, openai
import run_experiment as e
out=e.ROOT/'task-division-v1'
selection,context,protocol,signature=e.prepare(e.ROOT,out)
cache=e.load_cache(out,protocol,signature)
assert len(cache)==600
e.verify_protected(protocol)
assert all(e.sha(p)==h for p,h in protocol['implementation_sha256'].items())
assert e.sha(out/'metadata_only_model.joblib')==protocol['metadata_model_sha256']
pred=pd.read_csv(out/'predictions.csv')
assert pred.shot_id.tolist()==selection.shot_id.tolist() and pred.shot_id.is_unique
assert len(pred)==200 and int(pred.goal.sum())==19 and pred.match_id.nunique()==34
scores=pd.read_csv(out/'scores.csv').set_index('method')
largest=0.
for name in e.LABELS:
    computed=e.robust.metrics(pred.goal,pred[name])
    for metric,value in computed.items():
        largest=max(largest,abs(value-float(scores.loc[name,metric])))
assert largest<1e-12
assert set(v['returned_model'] for v in cache.values())=={'gpt-6-luna'}
result={'complete':True,'responses':600,'shots':200,'goals':19,'matches':34,
'only_luna':True,'code_input_image_and_model_hashes_valid':True,'protected_inputs_unchanged':True,
'cached_request_signatures_valid':True,'max_score_recomputation_difference':largest,
'python':platform.python_version(),'scikit_learn':sklearn.__version__,'openai_sdk':openai.__version__,
'api_requests_in_this_audit':0,'model_fitting_in_this_audit':False}
e.save_json(out/'final_audit.json',result)
paths=sorted(p for p in out.iterdir() if p.is_file() and p.suffix in {'.md','.json','.jsonl','.csv','.png','.joblib'} and p.name!='artifact_manifest.json')
e.save_json(out/'artifact_manifest.json',{'files_sha256':{p.name:e.sha(p) for p in paths},
'authoring_files_sha256':{p.name:e.sha(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
'protocol_signature':signature})
print(json.dumps(result,indent=2))
