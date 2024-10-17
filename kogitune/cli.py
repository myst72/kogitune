import kogitune.adhocs as adhoc

from .adhocs.modules import (
    update_beta_cli as update_beta, 
    update_cli as update,
)

from .loads.cli import (
    texteval_cli as texteval,
)

from .datasets.cli import (
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

def main(subcommand=None, /, **kwargs):
    if subcommand is None:
        with adhoc.kwargs_from_main(use_subcommand=True, **kwargs) as kwargs:
            subcommand = kwargs["subcommand"]
            adhoc.load('cli', subcommand, **kwargs)
    else:
        with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
            adhoc.load('cli', subcommand, **kwargs)

if __name__ == "__main__":
    main()
