"""Finish analysis and archive artifacts after every planned request is complete."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from vision_trials import OUT, REPO


def main():
    code = Path(__file__).resolve().parent
    deadline = time.monotonic()+4*3600
    target = OUT/'confirmation_bundesliga/best_vision/comparisons.csv'
    print('Waiting for all fixed experiments before producing the combined report.', flush=True)
    while not target.exists():
        if time.monotonic() > deadline:
            raise SystemExit('Experiments did not finish; no final report was labeled complete.')
        time.sleep(10)
    for name in ['diagnose_geometry.py', 'direct_prompt_diagnostics.py', 'audit_integrity.py', 'final_report.py']:
        result = subprocess.run([sys.executable, str(code/name)])
        if result.returncode:
            raise SystemExit(f'Finalization stopped at {name}.')
    source_target = REPO/'experiments/improvement_suite'
    source_target.mkdir(parents=True, exist_ok=True)
    for path in code.iterdir():
        if path.suffix == '.py' or path.name == 'README.md':
            shutil.copy2(path, source_target/path.name)
    shutil.copytree(OUT, REPO/'results/improvement_suite', dirs_exist_ok=True)
    (OUT/'ready_for_review.json').write_text(json.dumps({'experiments_complete': True,
        'report': str(OUT/'final_report.md'), 'repository_report': str(REPO/'results/improvement_suite/final_report.md'),
        'remaining': 'Visually inspect figures and report; update the research log and notify user.'}, indent=2))
    print('All experiments and report are complete; artifacts copied to the repository. Ready for visual review.', flush=True)


if __name__ == '__main__':
    main()
