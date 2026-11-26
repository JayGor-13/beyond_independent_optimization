# Paste this whole file into a NEW cell in the failed Kaggle notebook.
from pathlib import Path
import re
import subprocess

CODE = Path('/kaggle/working/deepseek_accuracy_code')
PYTHON = '/kaggle/working/deepseek_accuracy_py311/bin/python'
requirements = CODE / 'requirements-kaggle.txt'
assert requirements.is_file(), 'Run the notebook extraction cell first.'
text = requirements.read_text()
text = re.sub(r'(?m)^\s*pyarrow[^\n]*\n?', '', text)
requirements.write_text(text.rstrip() + '\npyarrow==17.0.0\n')
subprocess.run([
    PYTHON, '-m', 'pip', 'install', '--only-binary=:all:', '--no-deps',
    'numpy==1.26.4', 'pyarrow==17.0.0'
], check=True)
subprocess.run([
    PYTHON, '-c',
    "import numpy, pyarrow, datasets; "
    "assert numpy.__version__ == '1.26.4' and pyarrow.__version__ == '17.0.0'; "
    "assert len(datasets.Dataset.from_dict({'answer':[1,0]})) == 2; "
    "print('Data imports OK:', numpy.__version__, pyarrow.__version__, datasets.__version__)"
], check=True)
subprocess.run([PYTHON, '-u', str(CODE / 'setup_runtime.py')], check=True)
print('Now run the configuration and inference cells. Do not rerun the old extraction cell, which restores the old requirements.')
