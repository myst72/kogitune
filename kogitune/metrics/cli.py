import kogitune.metrics

from ..loads.commons import adhoc, listfy

@adhoc.cli
def fileeval_cli(**kwargs):
    from .chaineval import file_eval
    with adhoc.aargs_from(**kwargs) as aargs:
        files = aargs["files|!!"]
        metric_list = adhoc.list_values(aargs["metric_list|metrics|metric"])
        file_eval(files, metric_list, **aargs)


@adhoc.cli
def chaineval_cli(**kwargs):
    from .chaineval import chain_eval
    with adhoc.aargs_from(**kwargs) as aargs:
        model_list = listfy(aargs["model_list|model_path|!!"])
        metric_list = adhoc.list_values(aargs["metric_list|metrics|metric"])
        chain_eval(model_list, aargs["eval_type|=gen"], metric_list, **aargs)

@adhoc.cli
def selfcheck_cli(**kwargs):
    from .chaineval import chain_eval
    kwargs = dict(max_new_tokens=128) | kwargs | dict(selfcheck=True)
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)


@adhoc.cli
def eval_loss_cli(**kwargs):
    from .chaineval import chain_eval
    kwargs = kwargs | dict(eval_type="loss", metric="perplexity")
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)


@adhoc.cli
def eval_choice_cli(**kwargs):
    from .chaineval import chain_eval
    kwargs = kwargs | dict(eval_type="choice", metric="exact_match")
    with adhoc.aargs_from(**kwargs) as aargs:
        chain_eval(aargs)
