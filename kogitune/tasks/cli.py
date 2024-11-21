from ..loads.commons import adhoc

@adhoc.cli
def eval_cli(**kwargs):
    """
    eval サブコマンド

    - task 評価タスク
    - model_list|model_path: 対処とするモデル
    - metric_list|metrics: 評価尺度
    """
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
    """
    リーダーボードの作成
    評価タスクのデータからリーダーボードを作成します。

    leaderboard: リーダーボードファイル (leaderboard.csv)
    files: 評価タスクの結果ファイル
    names: 成績表に集計する項目名
    """
    from .tasks import calc_leaderboard
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        files = adhoc.get_list(kwargs, "files|!!")
        names = adhoc.get_list(kwargs, "names")
        calc_leaderboard(files, names, **kwargs)

