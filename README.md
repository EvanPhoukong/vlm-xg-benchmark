# Seeing Is Not Scoring: Benchmarking Vision-Language Models Against Expected Goals

Code and results for our MIT Sloan Sports Analytics Conference 2027 research abstract.

Latest SoccerTrack v2 diagrams plus ICL (20 November 2026): [completed report](results/soccertrack_spatial_icl_v1/REPORT.md). Sonnet 5.5 completed five conditions on all 304 eligible shots and ten matches. Fresh diagram-only Brier/AUC was 0.1147/0.706; adding 32 outcome examples gave 0.1187/0.681 and spatial-teacher examples 0.1152/0.684. All four adjusted comparisons against no examples and shuffled controls include zero: no additional ICL benefit was established. All 1,520 requests and 320 example sheets passed the 33-check audit. Estimated new spend $6.32 ($7.45 conservative), below $10; no paid runner is active.

Latest SoccerTrack v2 spatial-information experiment (20 November 2026): [completed report](results/soccertrack_spatial_v1/REPORT.md). Using released player positions and roles in pitch diagrams reduced Sonnet 5.5 Brier from 0.1293 to 0.1148 (11.2%) and raised AUC from 0.599 to 0.706 on 304 identical shots across ten matches; the adjusted match interval supports improvement over original photographs. Marked/cropped photos showed no supported benefit. Statistical references still have lower Brier point estimates. All 1,216 requests passed the audit; estimated spend $5.53 ($6.68 conservative), under $10. This is annotation-assisted exploratory evaluation on previously studied matches, not automatic player detection or untouched confirmation.

Latest SoccerTrack v2 in-context learning: [completed report](results/soccertrack_icl_completed_v1/REPORT.md). GPT-6.2 Sol completed six conditions on 80 identical shots across ten matches, with query-match exclusions for all examples and baseline fitting. Outcome-labelled and logistic-labelled examples did not establish improvement over zero-shot predictions. The investigation cost an estimated $5.49 ($7.07 conservative), including the credit-interruption recovery and four unused Sonnet startup calls, under the same $10 cap. All 480 Sol predictions passed the final request audit.

Latest StatsBomb in-context learning investigation: [combined report](results/statsbomb_icl_v1/FINAL_REPORT.md). GPT-6.2 Sol tested outcome-labelled examples and a separate StatsBomb teacher-probability condition, with shuffled-label controls and disjoint example/query matches. All 1,100 requests completed for an estimated $6.19 ($7.62 conservative), under one $10 combined cap. Outcome examples did not establish improvement. Teacher examples improved Brier over zero-shot predictions, but did not establish an advantage over shuffled teacher labels. No model weights were updated.

Latest autonomous continuation (29 October 2026): [changes aimed at the core errors](results/core_fixes_v1/REPORT.md). Completed 945 new calls, estimated $2.5637. On 150 new shots from 34 unused matches, zero-shot Astra reduced raw Brier from Luna's 0.1382 to 0.1231; the prespecified 99% match interval supports improvement. Ranking and repeat stability also improved by point estimate. Its small point advantage over logistic remains inconclusive. Correct outcome examples and past tracking did not establish fixes. No recalibration or other output correction. The exact [Astra candidate](results/core_fixes_v1/candidate.json) is saved separately from the original benchmark.

Previous continuations: [different datasets](results/dataset_transfer_v1/REPORT.md), 1,046 calls, estimated $0.2049; and [direct model search and visual controls](results/direct_model_search_v1/ROUND_SUMMARY.md), 722 calls, estimated $0.3986. See [current research status](CURRENT_RESEARCH_STATUS.md) for the complete history and used-match exclusions.

We ask whether general-purpose vision-language models (VLMs) can estimate expected goals (xG)
from an image of a shot situation alone, and where they diverge from established xG models.

## Data
StatsBomb open data, 2022 FIFA World Cup (competition 43, season 106), loaded via `statsbombpy`.
Each shot's freeze frame is rendered as a half-pitch image (shooter, teammates, opponents, goalkeeper).

## Models
- Gemma 4 (26B), open-weight, run locally via Ollama
- Claude Sonnet 5.5 and Claude Opus 5.5 (Anthropic API)
- Gemini 3.8 Flash (Gemini API)

The script also supports GPT-6 Luna (`gpt-6-luna`) through the OpenAI API for new runs;
OpenAI results are not included in the original published benchmark files.

All models receive the same zero-shot, image-only prompt (see `PROMPT` in `xg_vlm_pilot.py`).
Comparisons: StatsBomb xG and a logistic baseline (distance, angle, header, defenders in shooting cone)
trained with match-grouped cross-validation.

## Reproduce
```bash
python -m pip install statsbombpy mplsoccer scikit-learn scipy anthropic google-genai requests openai

python xg_vlm_pilot.py --backend ollama --model gemma4:26b --n 300
python xg_vlm_pilot.py --backend anthropic --model claude-sonnet-5-5 --n 300
python xg_vlm_pilot.py --backend anthropic --model claude-opus-5-5 --n 300
python xg_vlm_pilot.py --backend gemini --model gemini-3.8-flash --n 300
python xg_vlm_pilot.py --backend openai --model gpt-6-luna --n 300
```

API keys are read from the `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, and `OPENAI_API_KEY`
environment variables. The OpenAI backend uses the Responses API with the same image
and prompt, strict JSON output, high image detail, and low reasoning effort.

For a first OpenAI run in Ubuntu/WSL:

```bash
export OPENAI_API_KEY="your-api-key"
python xg_vlm_pilot.py --backend openai --n 50
```

`--backend openai` defaults to `gpt-6-luna`; `--model` can override it. The SDK retries
temporary API errors. If a request still fails, the run stops with an error and can
be resumed with the same command, reusing successful cached predictions.

OpenAI outputs are saved to `results/gpt_6_luna/`: `shots.csv` contains the predictions
and explanations, `summary.txt` contains the metrics, and `calibration.png` shows the
reliability curves. The shot images are in `results/images/`, named by `shot_id`.
Adding `--context` supplies body-part information and uses a separate `_context`
results folder. Do not reuse a results folder after changing the prompt or model settings.

To run the offline OpenAI integration checks (no API key or paid requests required):

```bash
python -m unittest discover -s tests -v
```

## Files
- `xg_vlm_pilot.py`: full pipeline (data loading, rendering, VLM queries, evaluation)
- `*_summary.txt`: per-model results
- `*_shots.csv`: per-shot predictions, StatsBomb xG, baseline xG, and model rationales
- `calibration_all_models.png`: calibration of all models vs StatsBomb xG

## Authors
[Your name], Northeastern University
[Co-author name], [Institution]
