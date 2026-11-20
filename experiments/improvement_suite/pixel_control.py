"""A deterministic pixel-only geometry control for the existing diagrams."""
import json
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from threadpoolctl import threadpool_limits
import rich_vlm_eval as base
import robust_xg as robust
from vision_trials import ROOT, OUT, freeze, evaluate, sha


def calibration(folder):
    # A generic synthetic scene supplies the fixed renderer transform, never query coordinates.
    row = pd.Series({'x': 100., 'y': 40., 'ff': []})
    close = plt.close
    try:
        plt.close = lambda *a, **k: None
        base.render_grid(row, folder/'calibration.png')
        fig = plt.gcf(); axis = fig.axes[0]
        height = Image.open(folder/'calibration.png').height
        origin = axis.transData.transform((0, 0))*130/fig.dpi
        step = axis.transData.transform((1, 1))*130/fig.dpi-origin
        return {'origin_x': float(origin[0]), 'origin_y': float(height-origin[1]),
                'scale_x': float(step[0]), 'scale_y': float(-step[1]), 'image_height': height,
                'image_width': Image.open(folder/'calibration.png').width}
    finally:
        plt.close = close
        close('all')


def read_pixels(path, transform):
    a = np.asarray(Image.open(path).convert('RGB'))/255.
    if a.shape[:2] != (transform['image_height'], transform['image_width']):
        raise ValueError('Image dimensions differ from the fixed renderer.')
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    masks = {'shooter': (r > .75) & (g > .72) & (b < .25),
             'goalkeeper': (r > .75) & (g > .4) & (g < .72) & (b < .25),
             'defenders': (r > .75) & (g < .25) & (b < .25)}
    def centers(mask):
        labels, n = ndimage.label(mask)
        values = []
        for identifier, region in enumerate(ndimage.find_objects(labels), 1):
            if region is None:
                continue
            local = labels[region]
            yy, xx = np.nonzero(local == identifier)
            if len(xx) < 10:
                continue
            x, y = xx.mean()+region[1].start, yy.mean()+region[0].start
            p = {'lateral': float((x-transform['origin_x'])/transform['scale_x']),
                 'depth': float((y-transform['origin_y'])/transform['scale_y'])}
            if -44 <= p['lateral'] <= 44 and -4 <= p['depth'] <= 64:
                values.append((len(xx), p))
        # A label may occur in overlapping boxes; deduplicate by point.
        unique = {}
        for size, p in values:
            key = (round(p['lateral'], 7), round(p['depth'], 7))
            unique[key] = (size, p)
        return sorted(unique.values(), key=lambda item: -item[0])
    s, k, d = [centers(masks[role]) for role in ['shooter', 'goalkeeper', 'defenders']]
    return {'shooter': s[0][1] if s else None, 'goalkeeper': k[0][1] if k else None,
            'defenders': [p for size, p in d], 'xg': .1, 'reason': 'Pixel-only geometry control; placeholder xG unused.'}


def main():
    folder = OUT/'pixels'; folder.mkdir(parents=True, exist_ok=True)
    transform = calibration(folder)
    development = robust.load_development(ROOT)
    train = development.sort_values('shot_id').sample(n=40, random_state=20261023)
    if set(train.match_id) & set(pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True).match_id):
        raise ValueError('Pixel detector validation overlaps test matches.')
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, row in enumerate(train.itertuples(index=False)):
            path = Path(tmp)/f'{i}.png'; base.render_grid(row, path)
            answer = read_pixels(path, transform)
            true_s, true_k, true_d = robust.scene(row)
            error = np.nan if answer['shooter'] is None else float(np.linalg.norm(np.array(list(answer['shooter'].values()))-true_s))
            errors.append({'shot_id': row.shot_id, 'shooter_error': error,
                           'defender_count_error': len(answer['defenders'])-len(true_d)})
    pd.DataFrame(errors).to_csv(folder/'training_detection_checks.csv', index=False)
    protocol = {'implementation_sha256': sha(__file__), 'calibration': transform,
                'method': 'Fixed RGB color thresholds, connected components with area >=10 pixels, component centroids, affine mapping calibrated from one generic renderer image.',
                'training_check_shots': train.shot_id.tolist(), 'test_coordinates_used': False,
                'detector_tuned_on_external': False, 'input_images': 'Same existing single-panel images; hidden players cannot be recovered.',
                'external_status': 'Previously studied Euro 2024 cohort; separate non-VLM control.'}
    freeze(folder, protocol)
    selection = pd.read_json(ROOT/'external_vlm_selection.jsonl', lines=True, precise_float=True)
    answers = [read_pixels(ROOT/'rich_vlm_external/images'/f'{s}.png', transform) for s in selection.shot_id]
    (folder/'answers.json').write_text(json.dumps(answers), encoding='utf-8')
    evaluate(selection, answers, 'pixels', folder)
    print('Training pixel shooter error:', pd.DataFrame(errors).shooter_error.describe().to_json(), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
