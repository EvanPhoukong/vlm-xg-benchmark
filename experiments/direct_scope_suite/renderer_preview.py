"""Outcome-blind QC of the already specified hollow-star renderer."""
import json
import numpy as np
import pandas as pd
from PIL import Image,ImageChops
import matplotlib.pyplot as plt
import common as c
from prepare_next import outline
row=c.read(c.OUT/'reasoning/selection.json')[0]
folder=c.OUT/'renderer_preview'; folder.mkdir(exist_ok=True)
standard=c.OUT/'images/reasoning'/(row['shot_id']+'.png'); target=folder/'hollow.png'
outline(pd.Series(row),target)
with Image.open(standard) as first,Image.open(target) as second:
    a=first.convert('RGB'); b=second.convert('RGB'); assert a.size==b.size
    box=ImageChops.difference(a,b).getbbox(); assert box and box[2]-box[0]<80 and box[3]-box[1]<80
    aa=np.asarray(a); bb=np.asarray(b)
    red=lambda x:int(((x[:,:,0]>180)&(x[:,:,1]<100)&(x[:,:,2]<100)).sum())
    c.save(folder/'qc.json',{'same_dimensions':True,'difference_bbox_pixels':box,'red_pixels_before':red(aa),'red_pixels_after':red(bb),
                            'note':'Only a localized shooter-star region changes. Visible-red count describes this example, not model accuracy. No outcomes or probabilities used.'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,img,title in zip(axes,[a,b],['Original filled shooter star','Hollow shooter star; positions unchanged']): ax.imshow(img); ax.axis('off'); ax.set_title(title,fontsize=11)
    fig.tight_layout(); fig.savefig(folder/'comparison.png',dpi=160); plt.close(fig)
print(json.dumps(c.read(folder/'qc.json'),indent=2))
