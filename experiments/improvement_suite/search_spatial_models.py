"""Select richer spatial models using development tournaments only."""
import json
import os
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

import spatial_models as spatial

ROOT=Path(os.environ.get('XG_RESEARCH_ROOT',str(Path(__file__).resolve().parents[2]/'output/research-loop')))


def metrics(frame,p):
    return {'brier':float(brier_score_loss(frame.goal,p)),'auc':float(roc_auc_score(frame.goal,p)),
            'logloss':float(log_loss(frame.goal,p)),'mean_xg':float(np.mean(p))}


def main():
    frames=[]
    for name in ['wc2018','wc2022']:
        frame=pd.read_json(ROOT/'data'/f'{name}.jsonl',lines=True,precise_float=True)
        frame['tournament']=name
        frame['group_id']=name+'/'+frame.match_id.astype(str)
        frames.append(frame)
    frame=pd.concat(frames,ignore_index=True)
    if not frame.shot_id.is_unique:
        raise ValueError('Duplicate development shot IDs.')
    features=spatial.known_features(frame)
    features.to_csv(ROOT/'development_features.csv',index=False)
    methods={'baseline_image':{'family':'baseline','context':False},
             'baseline_context':{'family':'baseline','context':True},**spatial.catalogue()}
    protocol={'development':{'shots':len(frame),'matches':int(frame.group_id.nunique()),'goals':int(frame.goal.sum())},
              'candidates':methods,'folds':5,'selection':'Lowest development OOF Brier within each input condition',
              'external_validation':'Euro 2024, untouched until selected model is frozen',
              'code_sha256':hashlib.sha256(Path(spatial.__file__).read_bytes()).hexdigest(),
              'warning':'Development scores used for selection, not independent confirmation.'}
    path=ROOT/'spatial_search_protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Search protocol changed; preserve the previous experiment and use a new folder.')
    path.write_text(json.dumps(protocol,indent=2))
    folds=list(GroupKFold(5).split(frame,frame.goal,frame.group_id))
    result=frame[['shot_id','match_id','group_id','tournament','goal','sb_xg']].copy()
    result['fold']=-1
    audit=[]
    for fold,(train,test) in enumerate(folds):
        result.loc[test,'fold']=fold
        train_groups=set(frame.iloc[train].group_id)
        test_groups=set(frame.iloc[test].group_id)
        assert not train_groups.intersection(test_groups)
        audit.append({'fold':fold,'training_groups':sorted(train_groups),'testing_groups':sorted(test_groups)})
    scores=[]
    for name,spec in methods.items():
        predicted=np.full(len(frame),np.nan)
        for fold,(train,test) in enumerate(folds):
            columns,model=spatial.build_model(**spec)
            model.fit(features.iloc[train][columns],frame.iloc[train].goal)
            predicted[test]=model.predict_proba(features.iloc[test][columns])[:,1]
        assert np.isfinite(predicted).all()
        result[name]=predicted
        row={'method':name,**metrics(frame,predicted)}
        scores.append(row)
        print(json.dumps(row),flush=True)
    scores=pd.DataFrame(scores).sort_values('brier')
    scores.to_csv(ROOT/'development_scores.csv',index=False)
    result.to_csv(ROOT/'development_predictions.csv',index=False)
    (ROOT/'development_fold_audit.json').write_text(json.dumps(audit,indent=2))
    selected={}
    for condition in ['image','context']:
        candidates=scores[scores.method.str.endswith('_'+condition)&~scores.method.str.startswith('baseline_')]
        best=candidates.iloc[0].method
        selected[condition]={'method':best,**methods[best]}
    (ROOT/'selected_spatial_models.json').write_text(json.dumps(selected,indent=2))
    print('Selected for external validation:',json.dumps(selected),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
