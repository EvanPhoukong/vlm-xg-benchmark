"""Fixed training-rate reference; never estimate its probability on test outcomes."""
import argparse
import pandas as pd
import robust_xg as robust
from vision_trials import ROOT, OUT, freeze, sha


def prepare():
    development = robust.load_development(ROOT)
    rate = float(development.goal.mean())
    folder = OUT/'sanity_baselines'
    protocol = {'implementation_sha256': sha(__file__), 'probability': rate,
                'training_shots': len(development), 'training_goals': int(development.goal.sum()),
                'definition': 'Predict the World Cup training scoring rate for every test shot. No test-outcome fitting.',
                'purpose': 'Additional descriptive control, added after Euro screening and before fresh baseline results.',
                'inference_requests': 0}
    if (OUT/'confirmation_bundesliga/comparisons.csv').exists() and not (folder/'protocol.json').exists():
        raise ValueError('Training-rate reference must be recorded before fresh results.')
    freeze(folder, protocol)
    return folder, rate


def main(prepare_only=False):
    folder, rate = prepare()
    if prepare_only:
        print(f'Fixed training-only constant probability {rate:.6f} before fresh results.', flush=True); return
    screen = pd.read_csv(OUT.parent/'noise-robustness-v1/external_predictions.csv')
    for name in ['multiview', 'fewshot']:
        values = pd.read_csv(OUT/'vision'/f'{name}_predictions.csv').set_index('shot_id').loc[screen.shot_id]
        screen[name] = values[name].to_numpy(); screen[name+'_noise_head'] = values[name+'_noise_head'].to_numpy()
    fresh = pd.read_csv(OUT/'confirmation_bundesliga/best_vision/predictions.csv')
    for condition, frame, methods in [
        ('screening', screen, ['noise_trained', 'multiview_noise_head', 'fewshot_noise_head']),
        ('fresh', fresh, ['original_hybrid', 'noise_trained', 'best_vision', 'pixels_noise_head'])]:
        frame['training_rate_constant'] = rate
        pd.DataFrame([{'method': 'training_rate_constant', **robust.metrics(frame.goal, frame.training_rate_constant)}]).to_csv(folder/f'{condition}_scores.csv', index=False)
        pd.DataFrame([robust.comparison(frame, method, 'training_rate_constant') for method in methods]).to_csv(folder/f'{condition}_comparisons.csv', index=False)
        frame.to_csv(folder/f'{condition}_predictions.csv', index=False)
    print('Training-only constant baseline scored on both cohorts; no new requests.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    main(parser.parse_args().prepare_only)
