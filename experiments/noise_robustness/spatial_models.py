"""Image-visible spatial features and fixed candidate probability models."""
import math
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

RICH = ['distance', 'angle', 'keeper_depth', 'keeper_offset', 'keeper_distance',
        'keeper_coverage', 'defender_cone', 'nearest_defender', 'close_defenders',
        'defender_coverage', 'open_angle']
BASE = ['distance', 'angle', 'paper_cone']


def in_triangle(point, a, b, c):
    def sign(p, q, r):
        return (p[0]-r[0])*(q[1]-r[1])-(q[0]-r[0])*(p[1]-r[1])
    values = [sign(point,a,b),sign(point,b,c),sign(point,c,a)]
    return not (any(v < 0 for v in values) and any(v > 0 for v in values))


def soft_coverage(shooter, players, sigma=.75):
    """Smooth estimated blocking over 41 evenly spaced goal-face rays."""
    if not players:
        return np.zeros(41)
    shooter = np.asarray(shooter, dtype=float)
    endpoints = np.column_stack([np.linspace(-4, 4, 41), np.zeros(41)])
    rays = endpoints-shooter
    lengths = np.maximum(np.linalg.norm(rays,axis=1), 1e-6)
    units = rays/lengths[:,None]
    offsets = np.asarray(players,dtype=float)[None,:,:]-shooter
    along = np.sum(offsets*units[:,None,:],axis=2)
    perpendicular = np.abs(offsets[:,:,0]*units[:,None,1]-offsets[:,:,1]*units[:,None,0])
    blocked = np.exp(-.5*(perpendicular/sigma)**2)*expit(along/sigma)*expit((lengths[:,None]-along)/sigma)
    return 1-np.prod(1-blocked,axis=1)


def geometry_features(shooter, keepers, defenders):
    sl,sd = [float(v) for v in shooter]
    distance = math.hypot(sl,sd)
    a,b = (-4-sl,-sd),(4-sl,-sd)
    angle = math.atan2(abs(a[0]*b[1]-a[1]*b[0]),a[0]*b[0]+a[1]*b[1])
    triangle = ((sl,sd),(-4,0),(4,0))
    kcoverage = soft_coverage((sl,sd),keepers)
    dcoverage = soft_coverage((sl,sd),defenders)
    combined = 1-(1-kcoverage)*(1-dcoverage)
    result = {'distance':distance,'angle':angle,
              'paper_cone':sum(in_triangle(p,*triangle) for p in keepers+defenders),
              'defender_cone':sum(in_triangle(p,*triangle) for p in defenders),
              'keeper_depth':np.nan,'keeper_offset':np.nan,'keeper_distance':np.nan,
              'keeper_coverage':float(kcoverage.mean()),'defender_coverage':float(dcoverage.mean()),
              'nearest_defender':80.0,'close_defenders':0,
              'open_angle':float(angle*(1-combined).mean())}
    if len(keepers)==1:
        kl,kd = keepers[0]
        result.update(keeper_depth=kd,keeper_offset=abs(sl*kd-sd*kl)/max(distance,1e-6),
                      keeper_distance=math.hypot(sl-kl,sd-kd))
    if defenders:
        distances = np.linalg.norm(np.asarray(defenders)-[sl,sd],axis=1)
        result.update(nearest_defender=float(distances.min()),close_defenders=int((distances<=3).sum()))
    return result


def known_features(frame):
    rows=[]
    for row in frame.itertuples():
        keepers,defenders=[],[]
        for p in row.ff:
            if p['teammate']:
                continue
            x,y=p['location'][:2]
            point=[float(y)-40,120-float(x)]
            if not (-44<=point[0]<=44 and -4<=point[1]<=64):
                continue
            target=keepers if (p.get('position') or {}).get('name')=='Goalkeeper' else defenders
            target.append(point)
        result=geometry_features((float(row.y)-40,120-float(row.x)),keepers,defenders)
        result['header']=int(row.header)
        # Retain the original published cone definition for the comparator.
        result['paper_cone']=int(row.defenders_in_cone)
        rows.append(result)
    return pd.DataFrame(rows,index=frame.index)


def catalogue():
    candidates={}
    for context in [False,True]:
        suffix='context' if context else 'image'
        for family in ['linear','spline','hgb7','hgb15']:
            candidates[f'{family}_{suffix}']={'family':family,'context':context}
    return candidates


def build_model(family,context):
    columns=RICH+(['header'] if context else [])
    if family=='baseline':
        columns=BASE+(['header'] if context else [])
        return columns,LogisticRegression(max_iter=2000)
    if family=='linear':
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),
                            StandardScaler(),LogisticRegression(C=.3,max_iter=3000))
    elif family=='spline':
        curved=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                             SplineTransformer(n_knots=4,degree=2,include_bias=False,extrapolation='constant'))
        other=[c for c in columns if c not in ['distance','angle']]
        transform=ColumnTransformer([('curved',curved,['distance','angle']),
                                     ('other',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),other)])
        model=make_pipeline(transform,StandardScaler(),LogisticRegression(C=.3,max_iter=3000))
    elif family in ['hgb7','hgb15']:
        constraints={column:0 for column in columns}
        constraints.update(distance=-1,angle=1,keeper_coverage=-1,defender_coverage=-1,
                           nearest_defender=1,open_angle=1)
        model=HistGradientBoostingClassifier(max_iter=180,learning_rate=.05,
            max_leaf_nodes=7 if family=='hgb7' else 15,min_samples_leaf=40,l2_regularization=10,
            monotonic_cst=constraints,early_stopping=False,random_state=7)
    else:
        raise ValueError(f'Unknown model family {family}.')
    return columns,model
