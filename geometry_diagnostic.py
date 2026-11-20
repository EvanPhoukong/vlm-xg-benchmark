"""Check VLM geometry against saved shot and freeze-frame coordinates.

This is a diagnostic, not a new xG model. Images are unchanged. Geometry is
measured in the same normalized StatsBomb coordinate system as the benchmark.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsbombpy import sb


PROMPT = """Read the geometry of this football shot diagram, not its scoring probability.
The goal is at the top. Gold star = shooter; blue circles = teammates;
red circles = opponents; orange square = opposing goalkeeper.

Use normalized pitch units, not meters. The full pitch is 120 units long and
80 units wide; the picture shows the attacking half. Goal width is 8 units.
The penalty spot is 12 units from the goal line; penalty-area depth is 18 units;
six-yard-area depth is 6 units. Use marker CENTERS for locations.

Report locations relative to the CENTER of the goal on the goal line:
- lateral: 0 at goal center, negative to IMAGE LEFT, positive to IMAGE RIGHT;
  the touchlines are -40 and +40. The posts are at lateral -4 and +4.
- depth: 0 at the goal line, positive DOWN the image into the field;
  the halfway line is depth 60.

Estimate the shooter's lateral and depth coordinates, the straight-line distance
from the shooter to goal CENTER, and the goal-opening angle in degrees.
The goal-opening angle is the angle BETWEEN the two lines from the shooter to
the two posts (0 to 180 degrees), not the bearing toward the center of goal.
Also locate the opposing goalkeeper, if the orange square is visible.
Return null for a measurement that cannot be determined. If the goalkeeper is
not visible, set goalkeeper_visible=false and both goalkeeper coordinates=null.
Give JSON only, using the requested schema. Do not estimate xG.
"""


def nullable_number(low, high):
    return {'type': ['number', 'null'], 'minimum': low, 'maximum': high}


SCHEMA = {
    'type': 'object',
    'properties': {
        'shooter_lateral': nullable_number(-44, 44),
        'shooter_depth': nullable_number(-4, 64),
        'shot_distance': nullable_number(0, 100),
        'goal_angle_degrees': nullable_number(0, 180),
        'goalkeeper_visible': {'type': 'boolean'},
        'goalkeeper_lateral': nullable_number(-44, 44),
        'goalkeeper_depth': nullable_number(-4, 64),
    },
    'required': ['shooter_lateral', 'shooter_depth', 'shot_distance',
                 'goal_angle_degrees', 'goalkeeper_visible',
                 'goalkeeper_lateral', 'goalkeeper_depth'],
    'additionalProperties': False,
}
SETTINGS = {'reasoning_effort': 'low', 'image_detail': 'high', 'max_output_tokens': 2048}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative_coordinates(x, y):
    # VerticalPitch plots StatsBomb (x,y) at display (y,x), with y increasing rightward.
    return float(y) - 40.0, 120.0 - float(x)


def geometry_from_relative(lateral, depth):
    if lateral is None or depth is None or pd.isna(lateral) or pd.isna(depth):
        return np.nan, np.nan
    distance = math.hypot(lateral, depth)
    first, second = (-4.0 - lateral, -depth), (4.0 - lateral, -depth)
    if math.hypot(*first) == 0 or math.hypot(*second) == 0:
        return distance, np.nan  # A shot exactly on a post has no defined opening angle.
    cross = first[0] * second[1] - first[1] * second[0]
    dot = first[0] * second[0] + first[1] * second[1]
    angle = math.degrees(math.atan2(abs(cross), dot))
    return distance, angle


def select_shots(frame, n, seed):
    if frame.shot_id.duplicated().any():
        raise ValueError('Duplicate source shot IDs.')
    if not 1 <= n <= len(frame):
        raise ValueError('n must be between 1 and the number of saved shots.')
    # Selection is independent of outcomes and all existing model predictions.
    ids = frame[['shot_id']].sort_values('shot_id').sample(n=n, random_state=seed)
    return frame.set_index('shot_id').loc[ids.shot_id].reset_index()


def prepare_reference(input_csv, output, n, seed):
    frame = pd.read_csv(input_csv)
    selected = select_shots(frame, n, seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / 'selection.csv'
    if manifest.exists():
        previous = pd.read_csv(manifest)
        if previous.shot_id.tolist() != selected.shot_id.tolist():
            raise ValueError('Selection changed; choose a new output folder.')
    else:
        selected.to_csv(manifest, index=False)
    reference_path = output / 'reference.jsonl'
    if reference_path.exists():
        reference = pd.read_json(reference_path, lines=True)
        if reference.shot_id.tolist() != selected.shot_id.tolist():
            raise ValueError('Reference shot IDs do not match selection.')
        return reference

    events = {}
    match_ids = selected.match_id.unique()
    for index, match_id in enumerate(match_ids, 1):
        data = sb.events(match_id=int(match_id))
        events.update({row['id']: row for _, row in data[data.id.isin(selected.shot_id)].iterrows()})
        print(f'  Reference matches {index}/{len(match_ids)}', flush=True)

    references = []
    for row in selected.to_dict('records'):
        event = events[row['shot_id']]
        np.testing.assert_allclose(event['location'][:2], [row['x'], row['y']], atol=1e-8)
        ff = event['shot_freeze_frame']
        opponents = [p for p in ff if not p['teammate']]
        keepers = [p for p in opponents if (p.get('position') or {}).get('name') == 'Goalkeeper']
        lateral, depth = relative_coordinates(row['x'], row['y'])
        distance, angle = geometry_from_relative(lateral, depth)
        np.testing.assert_allclose(distance, row['distance'], atol=1e-8)
        np.testing.assert_allclose(angle, math.degrees(row['angle']), atol=1e-8)
        row.update({'ff': ff, 'true_shooter_lateral': lateral, 'true_shooter_depth': depth,
                    'true_distance': distance, 'true_angle_degrees': angle,
                    'reference_goalkeepers': len(keepers),
                    'true_goalkeeper_lateral': None, 'true_goalkeeper_depth': None,
                    'goalkeeper_in_view': False})
        if len(keepers) == 1:
            lateral, depth = relative_coordinates(*keepers[0]['location'][:2])
            row.update({'true_goalkeeper_lateral': lateral, 'true_goalkeeper_depth': depth,
                        'goalkeeper_in_view': -44 <= lateral <= 44 and -4 <= depth <= 64})
        references.append(row)
    reference = pd.DataFrame(references)
    reference.to_json(reference_path, orient='records', lines=True, double_precision=15)
    return reference


def validate_response(prediction):
    if set(prediction) != set(SCHEMA['properties']):
        raise ValueError('Geometry response has missing or extra fields.')
    for key, spec in SCHEMA['properties'].items():
        value = prediction[key]
        if key == 'goalkeeper_visible':
            if not isinstance(value, bool):
                raise ValueError('goalkeeper_visible must be boolean.')
        elif value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'Invalid number for {key}.')
            if not spec['minimum'] <= value <= spec['maximum']:
                raise ValueError(f'Out-of-range {key}.')
    gk = [prediction['goalkeeper_lateral'], prediction['goalkeeper_depth']]
    if not prediction['goalkeeper_visible'] and any(value is not None for value in gk):
        raise ValueError('An invisible keeper cannot have reported coordinates.')
    if prediction['goalkeeper_visible'] and any(value is None for value in gk):
        raise ValueError('A visible keeper needs both coordinates.')
    return prediction


def ask_geometry(client, image_bytes, model):
    image_b64 = base64.b64encode(image_bytes).decode()
    response = client.responses.create(
        model=model, store=False, reasoning={'effort': SETTINGS['reasoning_effort']},
        max_output_tokens=SETTINGS['max_output_tokens'],
        input=[{'role': 'user', 'content': [
            {'type': 'input_image', 'image_url': f'data:image/png;base64,{image_b64}',
             'detail': SETTINGS['image_detail']},
            {'type': 'input_text', 'text': PROMPT}]}],
        text={'format': {'type': 'json_schema', 'name': 'shot_geometry',
                         'strict': True, 'schema': SCHEMA}})
    if response.status != 'completed' or not response.output_text:
        raise RuntimeError(f'No complete geometry response (status={response.status}).')
    prediction = validate_response(json.loads(response.output_text))
    return prediction, {'response_id': response.id, 'returned_model': response.model,
                        'usage': response.usage.model_dump() if response.usage else None}


def request_signature(model):
    return digest(json.dumps({'model': model, 'prompt': PROMPT, 'schema': SCHEMA,
                              'settings': SETTINGS}, sort_keys=True).encode())


def load_cache(path, signature):
    cached = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            entry = json.loads(line)
            if entry['request_signature'] != signature:
                raise ValueError('Cache uses another prompt, model, or settings; use a new output folder.')
            if entry['shot_id'] in cached:
                raise ValueError('Duplicate shot IDs in geometry cache.')
            validate_response(entry['prediction'])
            cached[entry['shot_id']] = entry
    return cached


def infer_geometry(reference, images, output, model, client=None):
    signature = request_signature(model)
    cache_path = output / 'geometry_cache.jsonl'
    cache = load_cache(cache_path, signature)
    predictions = []
    for index, row in enumerate(reference.itertuples(), 1):
        image_bytes = (images / f'{row.shot_id}.png').read_bytes()
        image_hash = digest(image_bytes)
        if row.shot_id in cache:
            entry = cache[row.shot_id]
            if entry['image_sha256'] != image_hash:
                raise ValueError(f'Image changed for shot {row.shot_id}; use a new output folder.')
        else:
            if client is None:
                from openai import OpenAI
                client = OpenAI(max_retries=6, timeout=120.0)
            prediction, response_info = ask_geometry(client, image_bytes, model)
            entry = {'shot_id': row.shot_id, 'request_signature': signature,
                     'image_sha256': image_hash, 'prediction': prediction, **response_info}
            with cache_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            cache[row.shot_id] = entry
        predictions.append({'shot_id': row.shot_id, **entry['prediction']})
        print(f'  Geometry {index}/{len(reference)}', flush=True)
    return pd.DataFrame(predictions)


def error_tables(reference, predictions):
    frame = reference.drop(columns=['ff']).merge(predictions, on='shot_id', validate='one_to_one')
    if len(frame) != len(reference):
        raise ValueError('Missing predictions: all selected shots must be accounted for.')
    for column in [key for key in SCHEMA['properties'] if key != 'goalkeeper_visible'] + [
        'true_goalkeeper_lateral', 'true_goalkeeper_depth',
    ]:
        frame[column] = pd.to_numeric(frame[column], errors='raise').astype(float)
    derived = [geometry_from_relative(row.shooter_lateral, row.shooter_depth)
               for row in frame.itertuples()]
    frame['computed_distance'], frame['computed_angle_degrees'] = np.asarray(derived).T
    frame['shooter_position_error'] = np.hypot(
        frame.shooter_lateral - frame.true_shooter_lateral,
        frame.shooter_depth - frame.true_shooter_depth)
    frame['goalkeeper_position_error'] = np.hypot(
        frame.goalkeeper_lateral - frame.true_goalkeeper_lateral,
        frame.goalkeeper_depth - frame.true_goalkeeper_depth)

    definitions = [
        ('Shot distance, direct answer', 'shot_distance', 'true_distance', 2.0, 'pitch units'),
        ('Goal-opening angle, direct answer', 'goal_angle_degrees', 'true_angle_degrees', 5.0, 'degrees'),
        ('Shooter position', 'shooter_position_error', None, 2.0, 'pitch units'),
        ('Goalkeeper position', 'goalkeeper_position_error', None, 2.0, 'pitch units'),
        ('Distance calculated from VLM coordinates', 'computed_distance', 'true_distance', 2.0, 'pitch units'),
        ('Angle calculated from VLM coordinates', 'computed_angle_degrees', 'true_angle_degrees', 5.0, 'degrees'),
    ]
    metrics = []
    for label, column, truth, tolerance, unit in definitions:
        eligible = frame.goalkeeper_in_view.astype(bool) if column == 'goalkeeper_position_error' else pd.Series(True, index=frame.index)
        error = frame[column] - frame[truth] if truth else frame[column]
        error = error[eligible]
        absolute = error.abs().dropna()
        metrics.append({'measurement': label, 'column': column, 'unit': unit,
                        'eligible': int(eligible.sum()), 'answered': len(absolute),
                        'mean_absolute_error': float(absolute.mean()),
                        'median_absolute_error': float(absolute.median()),
                        'p90_absolute_error': float(absolute.quantile(0.9)),
                        'mean_signed_error': float(error.mean()) if truth else np.nan,
                        'tolerance': tolerance, 'within_tolerance': int((absolute <= tolerance).sum()),
                        'within_tolerance_fraction': float((absolute <= tolerance).sum() / eligible.sum()) if eligible.sum() else np.nan})
        if truth:
            frame[f'{column}_error'] = error.reindex(frame.index)
    return frame, pd.DataFrame(metrics)


def save_figures(frame, reference, output):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, truth, direct, derived, label in [
        (axes[0], 'true_distance', 'shot_distance', 'computed_distance', 'Distance (pitch units)'),
        (axes[1], 'true_angle_degrees', 'goal_angle_degrees', 'computed_angle_degrees', 'Goal-opening angle (degrees)'),
    ]:
        ax.scatter(frame[truth], frame[direct], color='#bd6331', alpha=.7, s=20, label='VLM direct answer')
        ax.scatter(frame[truth], frame[derived], color='#2674b8', alpha=.65, s=20, label='Code using VLM coordinates')
        upper = float(frame[[truth, direct, derived]].max().max()) * 1.05
        ax.plot([0, upper], [0, upper], '--', color='#777777', lw=1)
        ax.set(xlabel='Known ' + label.lower(), ylabel='Estimated ' + label.lower(), xlim=(0, upper), ylim=(0, upper))
        ax.legend(fontsize=7, loc='upper left')
    position_max = frame[['shooter_position_error', 'goalkeeper_position_error']].max().max()
    position_max = max(6, math.ceil(position_max) + 1) if pd.notna(position_max) else 6
    axes[2].hist([frame.shooter_position_error.dropna(),
                  frame.loc[frame.goalkeeper_in_view.astype(bool), 'goalkeeper_position_error'].dropna()],
                 bins=np.arange(0, position_max + .5, .5),
                 label=['Shooter', 'Goalkeeper'], color=['#bd6331', '#2674b8'])
    axes[2].axvline(2, color='#777777', linestyle='--', lw=1)
    axes[2].set(xlabel='Position error (pitch units)', ylabel='Number of shots')
    axes[2].legend(fontsize=8)
    fig.suptitle(f'Luna geometry diagnostic: {len(frame)} unchanged shot images')
    fig.tight_layout()
    fig.savefig(output / 'geometry_errors.png', dpi=160)
    plt.close(fig)

    # Select examples only for illustration after scoring; they do not change the test set.
    chosen = []
    for column in ['shot_distance_error', 'goal_angle_degrees_error', 'goalkeeper_position_error']:
        for shot_id in frame.assign(_error=frame[column].abs()).nlargest(2, '_error').shot_id:
            if shot_id not in chosen:
                chosen.append(shot_id)
    chosen += [shot_id for shot_id in frame.shot_id if shot_id not in chosen]
    from mplsoccer import VerticalPitch
    fig, axes = plt.subplots(2, 3, figsize=(12, 9))
    for ax, shot_id in zip(axes.flat, chosen[:6]):
        row = frame.set_index('shot_id').loc[shot_id]
        original = reference.set_index('shot_id').loc[shot_id]
        pitch = VerticalPitch(half=True, pitch_type='statsbomb', line_color='#cccccc')
        pitch.draw(ax=ax)
        for player in original.ff:
            x, y = player['location'][:2]
            keeper = not player['teammate'] and (player.get('position') or {}).get('name') == 'Goalkeeper'
            pitch.scatter(x, y, ax=ax, s=45 if not keeper else 75,
                          c='orange' if keeper else ('royalblue' if player['teammate'] else 'red'),
                          marker='s' if keeper else 'o', edgecolors='black', linewidths=.5)
        pitch.scatter(row.x, row.y, ax=ax, s=180, c='gold', marker='*', edgecolors='black')
        for lateral_col, depth_col, color in [
            ('shooter_lateral', 'shooter_depth', '#7935a4'),
            ('goalkeeper_lateral', 'goalkeeper_depth', '#0994a0'),
        ]:
            if pd.notna(row[lateral_col]) and pd.notna(row[depth_col]):
                pitch.scatter(120 - row[depth_col], row[lateral_col] + 40, ax=ax,
                              s=100, c=color, marker='x', linewidths=2)
        ax.set_title(f'Distance: known {row.true_distance:.1f}, VLM {row.shot_distance:.1f}\n'
                     f'Angle: known {row.true_angle_degrees:.1f}°, VLM {row.goal_angle_degrees:.1f}°', fontsize=9)
    fig.suptitle('Illustrative errors: star/square = known locations; purple x = VLM shooter; teal x = VLM keeper', fontsize=11)
    fig.tight_layout()
    fig.savefig(output / 'examples.png', dpi=160)
    plt.close(fig)


def write_report(frame, metrics, output, metadata):
    indexed = metrics.set_index('column')
    lines = ['# Luna geometry diagnostic — simple report', '',
             f"Tested {len(frame)} randomly selected shots from the saved 300-shot run, across {frame.match_id.nunique()} matches. "
             'These are the unchanged benchmark images, with a new prompt asking explicitly for geometry.', '',
             '| Measurement | Mean error | Median error | 90% of errors below | Within tolerance |',
             '|---|---:|---:|---:|---:|']
    for row in metrics.itertuples():
        lines.append(f'| {row.measurement} | {row.mean_absolute_error:.2f} {row.unit} | '
                     f'{row.median_absolute_error:.2f} | {row.p90_absolute_error:.2f} | '
                     f'{row.within_tolerance}/{row.eligible} within {row.tolerance:g} {row.unit} |')
    distance = indexed.loc['shot_distance']
    angle = indexed.loc['goal_angle_degrees']
    keeper = indexed.loc['goalkeeper_position_error']
    lines += ['', 'What these numbers mean:', '',
              f'- Luna directly estimated distance within 2 pitch units on {int(distance.within_tolerance)}/{int(distance.eligible)} shots.',
              f'- It directly estimated the goal-opening angle within 5 degrees on {int(angle.within_tolerance)}/{int(angle.eligible)} shots.',
              f'- It located the goalkeeper within 2 pitch units on {int(keeper.within_tolerance)}/{int(keeper.eligible)} eligible shots.',
              '- The coordinate-based rows use ordinary code to calculate distance and angle from Luna’s reported shooter location. '
              'They show whether calculation in code helps, without giving it the true location.', '',
              'Coverage:', '']
    for row in metrics.itertuples():
        lines.append(f'- {row.measurement}: {row.answered}/{row.eligible} eligible measurements returned.')
    lines += ['', 'How this was tested:', '',
              f"- Fixed random sample, seed {metadata['seed']}, selected by shot ID before any geometry requests. No selection by goals, xG, or model error.",
              '- One fixed prompt, one fresh request per image, GPT-6 Luna, low reasoning, high image detail, structured JSON. Requests run sequentially.',
              '- The model sees only the original image and fixed coordinate/angle definitions. It never receives shot coordinates, outcomes, xG, or previous answers.',
              '- Ground truth uses the original shot location and the opposing goalkeeper in its StatsBomb shot freeze frame.',
              '- Position error is straight-line displacement. Units match the benchmark’s normalized 120×80 StatsBomb pitch; they are not measured stadium meters or verified yards.',
              '- Goal-opening angle is measured between the shooter-to-post lines, not the direction toward goal center.',
              '- Tolerances of 2 pitch units and 5 degrees were fixed before requests. Unanswered eligible measurements count as failures in tolerance rates.',
              '- Each response is saved with hashes of the image, prompt, model settings, and schema to prevent accidental mixing on resume.', '',
              'Limits:', '',
              '- This is an exploratory geometry diagnostic, not a new xG performance test or fine-tuning.',
              '- An explicit geometry prompt can behave differently from the earlier xG prompt. This does not prove the cause of any earlier xG error.',
              '- A goalkeeper coordinate can be present in the data but hard to see because markers overlap. Missing, multiple, or out-of-view keeper references are excluded from keeper position metrics and counted below.',
              f"- Reference keeper unavailable/out of view: {int((~frame.goalkeeper_in_view.astype(bool)).sum())}/{len(frame)} shots.",
              f'- Errors are descriptive for these {len(frame)} images. No thresholds or prompts were tuned after seeing the answers.', '',
              '![Geometry errors](geometry_errors.png)', '',
              '![Illustrative errors](examples.png)', '',
              'Reproduce from the repository with OPENAI_API_KEY set:', '',
              '```bash',
              'python geometry_diagnostic.py --input results/gpt_6_luna/shots.csv --n 50 --seed 7 '
              '--output results/geometry_gpt_6_luna_50',
              '```', '',
              'Sources: [OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna), '
              '[mplsoccer coordinate standardization](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_plots/plot_standardize.html).', '']
    text = '\n'.join(lines)
    (output / 'report.md').write_text(text, encoding='utf-8')
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('results/gpt_6_luna/shots.csv'))
    parser.add_argument('--images', type=Path, default=Path('results/images'))
    parser.add_argument('--output', type=Path, default=Path('results/geometry_gpt_6_luna_50'))
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare and inspect reference data; no model calls')
    args = parser.parse_args()
    reference = prepare_reference(args.input, args.output, args.n, args.seed)
    metadata = {'input': str(args.input.resolve()), 'input_sha256': digest(args.input.read_bytes()),
                'shots': len(reference), 'seed': args.seed, 'model': args.model,
                'prompt': PROMPT, 'schema': SCHEMA, 'settings': SETTINGS,
                'request_signature': request_signature(args.model),
                'selection': reference.shot_id.tolist(),
                'tolerances': {'distance_and_position_units': 2, 'angle_degrees': 5}}
    metadata_path = args.output / 'metadata.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        for key in ['input_sha256', 'request_signature', 'selection']:
            if old[key] != metadata[key]:
                raise ValueError(f'Changed {key}; choose a new output folder.')
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Reference ready: {len(reference)} shots; {reference.match_id.nunique()} matches; '
          f'{int(reference.goalkeeper_in_view.sum())} in-view keeper references.', flush=True)
    for shot_id in reference.shot_id:
        if not (args.images / f'{shot_id}.png').is_file():
            raise FileNotFoundError(f'Missing original image for {shot_id}.')
    if args.prepare_only:
        return
    predictions = infer_geometry(reference, args.images, args.output, args.model)
    frame, metrics = error_tables(reference, predictions)
    frame.to_csv(args.output / 'predictions.csv', index=False)
    metrics.to_csv(args.output / 'metrics.csv', index=False)
    save_figures(frame, reference, args.output)
    print(write_report(frame, metrics, args.output, metadata))


if __name__ == '__main__':
    main()
