import kogitune.adhocs as adhoc

from .adhocs.modules import (
    update_beta_cli, 
    update_cli
)

from .loads.cli import (
    texteval_cli
)

from .datasets.cli import (
    add_vocab_cli,
)

from .trainers.cli import (
    scratch_cli,
    pretrain_cli,    
)

from .metrics.cli import (
    eval_cli,
    chaineval_cli, 
    leaderboard_cli
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
