import os
import importlib
from .prints import aargs_print
from .main import cli

def pip(module: str, command='install'):
    cmd = f"pip3 {command} {module}"
    aargs_print(cmd, color='red')
    os.system(cmd)


def safe_import(module: str, pip_install_modules=None):
    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError:
        pip(pip_install_modules or module)
        module = importlib.import_module(module)
    if hasattr(module, '__version__'):
        aargs_print(module.__name__, module.__version__, once=module.__name__)
    return module

@cli
def update_cli(**kwargs):
    aargs_print(
        "KOGITUNEを最新の安定版に更新します。"
    )
    os.system("pip3 uninstall -y kogitune")
    pip('git+https://github.com/kuramitsulab/kogitune.git', command='install -U -q')

@cli
def update_beta_cli(**kwargs):
    aargs_print(
        "KOGITUNEを研究室内ベータ版に更新します。"
    )
    os.system("pip3 uninstall -y kogitune")
    pip('git+https://github.com/kkuramitsu/kogitune.git', command='install -U -q')
