"""Score the frozen constant on complete cohorts; leave partial trials unscored."""
import pandas as pd
import robust_xg as robust
from sanity_baselines import prepare
from vision_trials import OUT


def main():
    folder, rate = prepare()
    screen = pd.read_csv(OUT.parent/'noise-robustness-v1/external_predictions.csv')
    for name in ['multiview', 'fewshot']:
        values = pd.read_csv(OUT/'vision'/f'{name}_predictions.csv').set_index('shot_id').loc[screen.shot_id]
        screen[name] = values[name].to_numpy(); screen[name+'_noise_head'] = values[name+'_noise_head'].to_numpy()
    best = OUT/'confirmation_bundesliga/best_vision/predictions.csv'
    fresh = pd.read_csv(best if best.exists() else OUT/'confirmation_bundesliga/predictions.csv')
    for condition, frame, methods in [
        ('screening', screen, ['noise_trained', 'multiview_noise_head', 'fewshot_noise_head']),
        ('fresh', fresh, ['original_hybrid', 'noise_trained', 'best_vision', 'pixels_noise_head'])]:
        frame['training_rate_constant'] = rate
        pd.DataFrame([{'method': 'training_rate_constant', **robust.metrics(frame.goal, frame.training_rate_constant)}]).to_csv(folder/f'{condition}_scores.csv', index=False)
        pd.DataFrame([robust.comparison(frame, name, 'training_rate_constant') for name in methods if name in frame]).to_csv(folder/f'{condition}_comparisons.csv', index=False)
        frame.to_csv(folder/f'{condition}_predictions.csv', index=False)
    print('Frozen training-rate reference scored on complete 200-shot cohorts only.', flush=True)


if __name__ == '__main__':
    main()
