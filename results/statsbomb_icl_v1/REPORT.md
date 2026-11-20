# StatsBomb visual in-context learning

GPT-6.2 Sol, low reasoning. 200 confirmation shots,28 goals,48 matches.
Selected 64 outcome-labelled examples using40 separate development shots. No weight updates.

| Confirmation method | Brier (lower) | AUC (higher) | Mean xG |
|---|---:|---:|---:|
| icl64 | 0.103870 | 0.7732 | 0.1042 |
| shuffled64 | 0.106253 | 0.7501 | 0.1021 |
| zero | 0.105254 | 0.7497 | 0.1409 |
| logistic_image | 0.106537 | 0.7236 | 0.1059 |
| logistic_header | 0.104936 | 0.7253 | 0.1104 |
| training_constant | 0.122850 | 0.5000 | 0.0905 |
| statsbomb | 0.096492 | 0.8021 | 0.1125 |

Two primary contrasts:97.5% whole-match bootstrap intervals,20,000 draws. Negative Brier difference favors correct examples.

- icl64_minus_zero: -0.001384, interval [-0.009134,+0.006318].
- icl64_minus_shuffled64: -0.002383, interval [-0.006500,+0.000863].

## Development count comparison

| Examples | Brier | AUC |
|---|---:|---:|
| icl32 | 0.076660 | 0.2961 |
| icl64 | 0.065388 | 0.2895 |
| icl8 | 0.082104 | 0.3026 |
| shuffled64 | 0.063991 | 0.2500 |
| zero | 0.087136 | 0.3026 |

New API calls:800. Estimated cost $4.4658; conservative $5.4950,within the$10 combined cap.

Example,development and confirmation match groups are disjoint. Source matches have prior research exposure; this is exploratory, not untouched external confirmation. Query outcomes,IDs,coordinates and reference predictions never enter the prompt.

Examples are natural random shots with noisy binary outcomes. Diagrams are192px in numbered contact sheets to fit the budget; queries remain512px. This tests this format and one fixed example sample. A null result does not rule out other ICL designs. Only positions are visible; body part,height and motion are absent.

Logistic controls train only on the2000-shot training split. The image-feature control uses distance,angle and opponents in the cone; the header control has extra information. StatsBomb is a separate reference with richer inputs. No calibration,averaging or teacher probabilities.

Raw responses,usage,exact request hashes,images,frozen source hashes and per-shot predictions accompany this report. No account-wide balance is claimed. No paid runner remains active after completion.

StatsBomb Open Data: https://github.com/hudl/open-data. Retain attribution and follow the source license.
