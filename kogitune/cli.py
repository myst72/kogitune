import sys
import kogitune.adhocs as adhoc

from .adhocs.modules import (
    update_beta_cli as update_beta, 
    update_cli as update,
)

from .loads.cli import (
    texteval_cli as texteval,
)

from .datasets.cli import (
    train_bpe_cli as train_bpe,
    add_vocab_cli as add_vocab,
    get_cli as get,
    get_cli as get_split,
    store_cli as store,
)

from .trainers.cli import (
    scratch_cli as scratch,
    pretrain_cli as pretrain,    
)

from .metrics.cli import (
    eval_cli as eval,
    leaderboard_cli as leaderboard,
)

def main():
    with adhoc.kwargs_from_main(sys.argv, use_subcommand=True) as kwargs:
        subcommand = kwargs.pop("subcommand")
        adhoc.load('cli', subcommand, **kwargs)

if __name__ == "__main__":
    main()
