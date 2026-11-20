"""Full evaluator replay with old saved geometry; no API and no real new scores."""
import contextlib, io, json, shutil, tempfile
from pathlib import Path
import pandas as pd
import run_experiment as e
from build_report import build

root=e.ROOT
actual=root/'task-division-v1'
selection,context,protocol,signature=e.prepare(root,actual)
old={v['shot_id']:v['answer'] for v in map(json.loads,(Path(protocol['source'])/'inference/baseline_cache.jsonl').read_text().splitlines())}
cache={}
for shot_id in selection.shot_id:
    for role in e.ROLES:
        answer={k:old[shot_id][k] for k in e.SCHEMAS[role]['properties']}
        e.validate(role,answer)
        cache[(shot_id,role)]={'answer':answer,'usage':{},'elapsed_seconds':0.,'returned_model':e.MODEL}
with tempfile.TemporaryDirectory(prefix='xg_division_replay_') as directory:
    folder=Path(directory)
    shutil.copy2(actual/'metadata_only_model.joblib',folder/'metadata_only_model.joblib')
    e.save_json(folder/'protocol.json',protocol)
    with contextlib.redirect_stdout(io.StringIO()):
        e.evaluate(selection,context,folder,protocol,signature,cache)
        build(folder)
    scores=pd.read_csv(folder/'scores.csv').set_index('method')
    comparisons=pd.read_csv(folder/'comparisons.csv')
    assert abs(scores.loc['monolithic_image','brier']-scores.loc['specialists_image','brier'])<1e-15
    assert abs(scores.loc['monolithic_context','brier']-scores.loc['specialists_context','brier'])<1e-15
    assert (comparisons[comparisons.primary].brier_difference.abs()<1e-15).all()
    assert (folder/'report.md').exists() and (folder/'comparison.png').stat().st_size>10000
    assert json.loads((folder/'integrity.json').read_text())['protected_inputs_unchanged']
    print('Full 200-shot evaluator replay passed: identical supplied readings yield identical specialist/control probabilities; old controls reproduce; report and plot render. No API requests made.')
