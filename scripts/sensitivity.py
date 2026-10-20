"""Run explicit synthetic assumption sensitivity without changing baseline configs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from zb202.sensitivity import run

if __name__ == '__main__':
    output, summary = run(ROOT, ROOT/'configs/scenarios/sensitivity.json')
    print(output)
    for item in summary:
        print(item)
