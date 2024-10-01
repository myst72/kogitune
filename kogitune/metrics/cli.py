import kogitune.metrics

from ..loads.commons import adhoc, listfy

@adhoc.cli
def chaineval_cli(**kwargs):
    from .chaineval import chain_eval
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        model_list = adhoc.get_list(kwargs, "model_list|model_path|!!")
        metric_list = adhoc.get_list(kwargs, "metric_list|metrics|metric")
        chain_eval(model_list, "gen", metric_list, **kwargs)

@adhoc.cli
def fileeval_cli(**kwargs):
    from .chaineval import file_eval
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        files = adhoc.get(kwargs, "files|!!")
        metric_list = adhoc.get_list(kwargs, "metric_list|metrics|metric")
        file_eval(files, metric_list, **kwargs)

@adhoc.cli
def leaderboard_cli(**kwargs):
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
        for filepath in adhoc.get_list(kwargs, "files|!!"):
            testdata = adhoc.load("testdata", filepath)
            metric_list = adhoc.get_list(kwargs, "metric_list|metrics|metric")
            groupby = adhoc.get(kwargs, 'groupby')
            index = adhoc.get(kwargs, 'index')
            for name in metric_list:
                board.pivot_table(testdata.samples(), name, index=index, groupby=groupby)
        board.show()

@adhoc.cli
def selfcheck_cli(**kwargs):
    from .chaineval import chain_eval
    kwargs = dict(max_new_tokens=128) | kwargs | dict(selfcheck=True)
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        model_list = adhoc.get_list(kwargs, "model_list|model_path|!!")
        metric_list = adhoc.get_list(kwargs, "metric_list|metrics|metric")
        chain_eval(model_list, "gen", metric_list, **kwargs)

@adhoc.cli
def eval_choice_cli(**kwargs):
    from .chaineval import eval_choice
    kwargs = kwargs | dict(eval_type="choice", metric="exact_match")
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        model_list = adhoc.get_list(kwargs, "model_list|model_path|!!")
        eval_choice(model_list, **kwargs)


@adhoc.cli
def eval_mia_cli(**kwargs):
    from .mia import eval_mia
    kwargs = kwargs | dict(eval_type="mia")
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        model_list = adhoc.get_list(kwargs, "model_list|model_path|!!")
        eval_mia(model_list, **kwargs)


