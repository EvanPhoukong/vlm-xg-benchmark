import time
from pathlib import Path
import subprocess
import sys
root = Path('/mnt/c/Users/ephou/Downloads/xG/output/improvement-suite')
ready = root/'vision/fewshot_comparisons.csv'
deadline = time.monotonic()+3*3600
print('Waiting for the sequential vision screening to finish before confirmation requests.', flush=True)
while not ready.exists():
    if time.monotonic() > deadline:
        raise SystemExit('Screening did not finish within three hours; confirmation not started.')
    time.sleep(10)
print('Vision screening finished. Starting frozen fresh confirmation sequentially.', flush=True)
script = Path(__file__).resolve().parent/'confirm_frozen.py'
raise SystemExit(subprocess.run([sys.executable, str(script)]).returncode)
