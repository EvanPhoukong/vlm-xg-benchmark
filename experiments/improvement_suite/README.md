# xG improvement experiments, 22 October 2026

This suite tests the five proposed improvements separately, then evaluates fixed
noise-trained and pixel methods on unused matches. Model requests are sequential.
The original 300-shot study and all earlier saved predictions remain preserved.

Current completion status: all five 200-shot screening experiments and the
200-shot fresh baseline/noise/pixel comparison are complete. The selected best
VLM's extra fresh confirmation stopped at 46/200 because the API credit balance
was exhausted. Its cache and frozen protocol are preserved; partial xG scores
are not reported. After credits are available, the unchanged
`confirm_best_vision.py` resumes using the saved readings. The final report
clearly distinguishes this incomplete secondary trial from completed results.

## Experiments

1. `../noise-robustness/robust_xg.py`: train the existing image-only spline head
   using imperfect positions. The suite includes an unchanged copy of its source.
2. `vision_trials.py --phase multiview`: original cached reading plus a new
   full-image/fixed-zoom reading; one extra reading on disagreement. Coordinates
   are merged without accessing true test positions or outcomes.
3. `vision_trials.py --phase fewshot`: four fixed World Cup 2018 worked geometry
   examples precede each new image. Example outcomes, xG and body part are absent.
4. `pixel_control.py`: color/component centroids plus a generic renderer affine
   calibration. This is a separate deterministic computer-vision control.
5. `fetch_context.py` and `context_trial.py`: strictly pre-shot same-team pass
   context within 15 seconds, shot type, play pattern and pressure. This is an
   additional-input condition, with a comparator receiving the same metadata.

`prepare_confirmation.py` reserves 200 shots from the 34 publicly available
Leverkusen 2023/24 Bundesliga fixtures. `confirm_frozen.py` uses the unchanged
single-panel renderer and frozen models. Two predeclared primary comparisons
receive 97.5% Bonferroni bootstrap intervals; other comparisons are secondary.
The initial 40-match preparation attempt stopped before loading event data,
because only 34 matches were publicly available. Its protocol is retained.

`confirm_best_vision.py` selects the lowest-Brier prompt/ensemble VLM pipeline
from the completed screening scores before fresh baseline results are released.
It waits for the original fresh API work to finish, then makes a secondary
prospective check on that same cohort. Three secondary comparisons receive
98.333% intervals; they do not replace the original two primary comparisons.

The Euro 2024 screening cohort was studied previously: all its results are
exploratory. The partial Euro 2020 image experiment remains stopped at 62/200.
There is no VLM fine-tuning. Added image examples change the prompt, and geometry
augmentation trains only the statistical probability model.

## Runtime and inputs

The tested environment is the repository's Ubuntu virtual environment, Python
3.14.7 and scikit-learn 1.9.1. Exact source/input/image hashes are saved in each
protocol. Scripts currently use the saved local research data locations:

```
/mnt/c/Users/ephou/Downloads/xG/output/research-loop
/mnt/c/Users/ephou/Downloads/xG/output/noise-robustness-v1
/mnt/c/Users/ephou/Downloads/xG/output/improvement-suite
/home/evan/xG_Research/vlm-xg-benchmark
```

Run scripts with the repository virtual environment's Python. API keys are read
from the existing authorized handoff or environment and are never printed or
saved in artifacts. All successful new responses are saved as they complete;
unchanged commands reuse the cache. A changed protocol or implementation needs
a new experiment folder. Do not launch two API runners at the same time.

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python vision_trials.py --prepare-only
python vision_trials.py --phase all
python fetch_context.py
python context_trial.py
python pixel_control.py
python prepare_confirmation.py
python confirm_frozen.py --prepare-only
python confirm_frozen.py
python diagnose_geometry.py
```

Completed fresh inference can be rescored with `confirm_frozen.py --offline`.
`continue_confirmation.py` is only a sequencing helper for this run; it waits for
the screening score file before starting confirmation. Never start that helper
again when a confirmation process is already running.

Primary data: StatsBomb open data, https://github.com/statsbomb/open-data.
Image request format was checked against official OpenAI documentation:
https://developers.openai.com/api/docs/guides/images-vision?api-mode=responses.
