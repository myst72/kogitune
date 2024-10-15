__version__ = "1.0.0"

from .adhocs import load
from .loads import Model

import kogitune.cli as cli

def eval(**kwargs):
    from .cli import eval_cli
    eval_cli(**kwargs)
    
