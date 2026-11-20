"""Post-hoc error localization only; raw predictions are never adjusted."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/error_diagnosis_v1'
SOURCES=ROOT/'results/dataset_transfer_v1'


def mean(values): return sum(values)/len(values)


def analyze(rows,baseline):
    y=[float(r['goal']) for r in rows]; p=[float(r['vlm_xg']) for r in rows]; q=[float(r[baseline]) for r in rows]
    vp=mean(p); vq=mean(q); vy=mean(y)
    bp=mean([(a-b)**2 for a,b in zip(p,y)]); bq=mean([(a-b)**2 for a,b in zip(q,y)])
    mp=(vp-vy)**2; mq=(vq-vy)**2
    cp=mean([((a-vp)-(b-vy))**2 for a,b in zip(p,y)])
    cq=mean([((a-vq)-(b-vy))**2 for a,b in zip(q,y)])
    assert abs(bp-mp-cp)<1e-12 and abs(bq-mq-cq)<1e-12
    bins=[]
    for name,low,high in [('0-6',0,6),('6-12',6,12),('12-18',12,18),('18-30',18,30),('30+',30,math.inf)]:
        selected=[r for r in rows if low<=float(r['distance'])<high]
        if not selected: continue
        goal=mean([float(r['goal']) for r in selected]); pred=mean([float(r['vlm_xg']) for r in selected]); ref=mean([float(r[baseline]) for r in selected])
        total=sum((float(r['vlm_xg'])-float(r['goal']))**2-(float(r[baseline])-float(r['goal']))**2 for r in selected)
        bins.append({'distance_bin_normalized_pitch_units':name,'shots':len(selected),'goals':sum(int(r['goal']) for r in selected),
            'goal_rate':goal,'mean_vlm':pred,'mean_baseline':ref,'contribution_to_overall_brier_gap':total/len(rows)})
    assert abs(sum(b['contribution_to_overall_brier_gap'] for b in bins)-(bp-bq))<1e-12
    return {'baseline':baseline,'shots':len(rows),'goal_rate':vy,'mean_vlm':vp,'mean_baseline':vq,
        'vlm_brier':bp,'baseline_brier':bq,'brier_gap':bp-bq,'vlm_squared_overall_bias':mp,
        'baseline_squared_overall_bias':mq,'difference_in_squared_overall_bias':mp-mq,
        'difference_in_centered_error':cp-cq,
        'fraction_of_positive_gap_due_to_overall_bias_component':(mp-mq)/(bp-bq) if bp>bq else None,'distance_bins':bins}


def main():
    results={}
    for cohort in ('premier_league','womens_world_cup','metrica_v2'):
        with (SOURCES/cohort/'predictions.csv').open() as handle: rows=list(csv.DictReader(handle))
        results[cohort]={name:analyze(rows,name) for name in ('original_logistic','original_no_header')}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'diagnosis.json').write_text(json.dumps({'status':'post_hoc_descriptive','raw_predictions_unchanged':True,
        'note':'Exact algebraic Brier decomposition and error localization, not calibration or a causal attribution. Centered error contains probability spread and association with outcomes, and cannot be equated with visual measurement error.',
        'cohorts':results},indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__': main()
