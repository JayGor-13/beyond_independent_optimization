"""Use a separate Python 3.11/CUDA environment for the legacy DeepSeek stack."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

CODE = Path(__file__).resolve().parent
ENV = Path('/kaggle/working/deepseek_accuracy_py311')
PYTHON = ENV / 'bin/python'


def run(command, **kwargs):
    print('Setup:', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    if sys.platform != 'linux':
        raise RuntimeError('This setup targets the Linux Kaggle GPU runtime.')
    env = os.environ.copy()
    env.update(USE_TORCH='1', USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false',
               UV_PYTHON_INSTALL_DIR='/kaggle/working/deepseek_managed_python',
               UV_CACHE_DIR='/kaggle/working/deepseek_uv_cache')
    uv = shutil.which('uv')
    if not uv:
        uv_root = Path('/kaggle/working/deepseek_uv_bootstrap')
        run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:',
             '--target', uv_root, 'uv>=0.8,<1'])
        uv = str(uv_root / 'bin/uv')
        if not Path(uv).is_file():
            raise RuntimeError(f'uv executable not found at {uv}')
    if not PYTHON.exists():
        run([uv, 'venv', '--python', '3.11', '--seed', ENV], env=env)
    run([PYTHON, '-c', 'import sys; assert sys.version_info[:2] == (3,11); print(sys.version)'], env=env)
    run([PYTHON, '-m', 'pip', 'install', '--only-binary=:all:',
         'torch==2.5.1', 'torchvision==0.20.1',
         '--index-url', 'https://download.pytorch.org/whl/cu121'], env=env)
    run([PYTHON, '-m', 'pip', 'install', '--only-binary=:all:',
         '-r', CODE / 'requirements-kaggle.txt'], env=env)
    deepseek = Path('/kaggle/working/DeepSeek-VL2')
    if not deepseek.exists():
        run(['git', 'clone', '--depth', '1', 'https://github.com/deepseek-ai/DeepSeek-VL2.git', deepseek], env=env)
    run([PYTHON, '-m', 'pip', 'install', '-e', deepseek, '--no-deps', '--no-build-isolation'], env=env)
    # Upstream metadata pins torch 2.0.1 / Transformers 4.38.2 and requires
    # xformers. This runner uses the saved 4.41.0 stack and its PyTorch SDPA
    # attention patch instead; verify actual imports rather than that metadata.
    run([PYTHON, '-c',
         "import sys, torch, torchvision, transformers, tokenizers, cv2, datasets, numpy, pyarrow; "
         "from deepseek_vl2.models import DeepseekVLV2Processor; "
         "print('Python',sys.version,'torch',torch.__version__,'transformers',transformers.__version__,'tokenizers',tokenizers.__version__); "
         "assert numpy.__version__ == '1.26.4' and pyarrow.__version__ == '17.0.0'; "
         "sample = datasets.Dataset.from_dict({'answer':[1,0]}); assert len(sample) == 2; "
         "print('NumPy',numpy.__version__,'PyArrow',pyarrow.__version__,'datasets',datasets.__version__); "
         "assert torch.cuda.is_available(), 'Enable a GPU accelerator'; "
         "print('GPU',torch.cuda.get_device_name(0))"], env=env)
    run([PYTHON, '-m', 'unittest', 'discover', '-s', CODE, '-p', 'test_core.py'], env=env)
    print(f'Setup complete. Run inference with {PYTHON}', flush=True)


if __name__ == '__main__':
    main()
