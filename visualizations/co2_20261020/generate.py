"""重画全部图：python generate.py；不重训、不修改原始数据。"""
from pathlib import Path
import subprocess,sys
BASE=Path(__file__).resolve().parent
for script in sorted((BASE/'scripts').glob('gen_fig_*.py')):
    subprocess.run([sys.executable,str(script)],cwd=BASE,check=True)
subprocess.run([sys.executable,str(BASE/'scripts/export_metrics.py')],cwd=BASE,check=True)
print('全部完成：', BASE/'README.md')
