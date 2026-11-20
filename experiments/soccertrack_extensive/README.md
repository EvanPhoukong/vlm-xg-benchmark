# SoccerTrack v2 extensive comparison

The fixed complete cohort has 306 unique audited SHOT/GOAL events from ten matches, including 44 goals. All four models receive the same full-pitch PNG per shot and the same original zero-shot prompt. Models are GPT-6 Luna, GPT-6.1 Terra, GPT-6.1 Sol and Claude Opus 5.5, each at low reasoning effort with a 1,024-token output ceiling. Thirty exact earlier Sol predictions are reused after verifying their image and prompt fingerprints.

The main comparator is fixed L2 logistic regression on distance and goal-opening angle, trained on the other nine matches for every evaluated match. The complete source audit and training-only scaling are reused from ../soccertrack_comparison. The 64-shot official test subset separately uses the six official training matches. Header and training-constant controls are secondary.

Budget is $10 combined across new requests and the earlier conservative SoccerTrack Sol cost. Reservations cover concurrent requests and maximum output, including reasoning. An unresolved request stops dispatch. There are no automatic generation retries. This is experiment-level accounting and does not claim a live account balance.

From the repository root, run the following stages in order.

    python experiments/soccertrack_extensive/frames.py
    python experiments/soccertrack_extensive/study.py prepare
    python experiments/soccertrack_extensive/resume.py run
    python experiments/soccertrack_extensive/resume.py analyze
    python experiments/soccertrack_extensive/report.py

Frames are extracted 0.32 seconds before the released event frame. Original pilot PNGs are retained, and additional frames use LANCZOS resizing to at most 1536 pixels on the long side. The shooter is unmarked. The full cohort is exploratory because the data and prior 30-shot results were already inspected. The prompt retains its average-professional-finisher assumption on university matches. Some penalties may remain and the SHOT/GOAL selection can omit blocked or headed attempts assigned other event labels.

Outputs live in results/soccertrack_extensive_v1, including the frozen protocol, frame audit, raw API responses, durable usage ledger, predictions, report, and comparison figure. SoccerTrack v2 by Scott et al. is used under CC BY 4.0, https://huggingface.co/datasets/atomscott/soccertrack-v2 and https://creativecommons.org/licenses/by/4.0/ .

The parser amendment accepts a single valid JSON answer surrounded by prose. One completed Claude response was recovered from its saved raw answer and usage without a further model call. The amendment and reconciliation are recorded in the output directory.

