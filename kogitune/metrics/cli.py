from ..loads.commons import adhoc, listfy

@adhoc.cli
def eval_cli(**kwargs):
    from .tasks import task_eval
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        model_list = adhoc.get_list(kwargs, "model_list|model_path|!!")
        metric_list = adhoc.get_list(kwargs, "metric_list|metrics|metric")
        task_eval(model_list, metric_list, **kwargs)
    adhoc.lazy_print() # 最後の出力

@adhoc.cli
def chaineval_cli(**kwargs):
    adhoc.print("【注意】chain_eval は廃止されます。eval を使って、`task=pass@1`を指定してください", color="red")
    eval_cli(**kwargs)

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

