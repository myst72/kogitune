from typing import List

from ..loads.commons import *
from .leaderboard import LeaderBoard

import kogitune.metrics.metrics
import kogitune.metrics.metrics_python
import kogitune.metrics.metrics_textsim

def evaluate(testdata, metric_list:List[str], board:LeaderBoard, **kwargs):
    adhoc.notice("Starting model eval//モデル評価を始めるよ", metric_list, kwargs)
    force_eval = kwargs.get('force_eval', True)
    for metric_path in listfy(metric_list):
        metric = adhoc.load("metric", metric_path, **kwargs)
        result = metric.evaluate(
            testdata.samples(), force_eval=force_eval
        )
        testdata.save()
        if result:
            groupby = kwargs.get('groupby', None)
            board.score(testdata, metric.path, groupby=groupby)

def file_eval(files: List[str], metric_list: List[str], **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for filepath in files:
        testdata = adhoc.load("testdata", filepath)
        evaluate(testdata, metric_list, board, **kwargs)
    board.show()


def guess_max_new_tokens(testdata, model, keys="reference", q=95):
    lengths = []
    if hasattr(model, "tokenizer"):
        tokenizer = model.tokenizer
    else:
        tokenizer = adhoc.load("tokenizer", "default")
    for sample in testdata.samples():
        text = sample.get(keys, "")
        lengths.append(len(tokenizer.encode(text)))
    max_new_tokens = int(max(lengths) * 1.05)
    if max_new_tokens < 512:
        return max_new_tokens
    if q < 1:
        q = int(q * 100)
    return int(np.percentile(lengths, q=q))


def check_generater_args(testdata, model, kwargs):
    # print("@check_generater_args", args)
    if "max_new_tokens" not in kwargs and "max_length" not in kwargs:
        kwargs["max_new_tokens"] = guess_max_new_tokens(testdata, model)
    if "max_new_tokens" in kwargs and "max_length" in kwargs:
        kwargs.pop("max_length", None)
    #print("@check_generater_args", kwargs)

def selfcheck(testdata, model, **kwargs):
    test_list = [sample for sample in testdata.samples() if "selfcheck" not in sample]
    if len(test_list) > 0:
        adhoc.notice(
            "Preparing data for SelfCheck//SelfCheck用のデータを準備します",
            model=model,
            gen_args=model.gen_args,
        )
        model.eval(
            test_list,
            batch_size=adhoc.get(kwargs, "eval_batch_size|batch_size|=2"),
        )
        testdata.save()
    
    n = adhoc.get(kwargs, "num_return_sequences|n|N|=1")
    if n == 1:
        adhoc.notice("num_return_sequencesを設定してね. num_return_sequences=6")
        kwargs["num_return_sequences"] = 6
    if "temperature" not in kwargs:
        adhoc.notice("temperatureを設定してね. temperature=0.8")
        kwargs["temperature"] = 0.8
    if "do_sample" not in kwargs:
        kwargs["do_sample"] = True
    return kwargs


def generate(testdata, model, eval_type, **kwargs):
    n = adhoc.get(kwargs, "num_return_sequences|n|N|=1")
    test_run = adhoc.get(kwargs, f"test_run|head|={len(testdata.samples())}")
    test_list = [sample for sample in testdata.samples() if "output" not in sample]
    if len(test_list) == 0:
        return testdata

    adhoc.notice(
        "Start generation//生成をはじめます",
        model=model,
        eval_type=eval_type,
        n=n,
        gen_args=model.gen_args,
    )
    if test_run < len(test_list):
        adhoc.print(f"Test running head={test_run}//先頭のhead={test_run}件のみ、テストしてみます")
        test_list = test_list[:test_run]
    try:
        with adhoc.start_timer() as timer:
            batch_size = adhoc.get(kwargs, "eval_batch_size|batch_size|=2")
            model.eval(
                test_list,
                eval_type=eval_type,
                n=n,
                batch_size=batch_size,
            )
            timer.notice("お疲れ様！！ 生成終わりました", total=len(test_list))
    finally:
        testdata.save()


def chain_eval(model_list: List[str], eval_type, metric_list:List[str], /, **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for model_path in model_list:
        model = adhoc.load("model", model_path, extract_prefix="model", **kwargs)
        kwargs['modeltag'] = model.modeltag
        dataset = adhoc.get(kwargs, "dataset|!!")
        for subset in adhoc.get_list(kwargs, "dataset_subset|dataset_name|="):
            kwargs['name'] = subset
            testdata = adhoc.load("testdata", dataset, **kwargs)

            check_generater_args(testdata, model, model.gen_args)
            if adhoc.get(kwargs, "selfcheck|self_check|=False"):
                kwargs = selfcheck(testdata, model, **kwargs)
                testdata.save()

            generate(testdata, model, eval_type, **kwargs)
            testdata.save()

            evaluate(testdata, metric_list, board, **kwargs)

    board.show()



