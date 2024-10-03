from typing import List
import numpy as np

from ..loads.commons import *
from .leaderboard import LeaderBoard

import kogitune.metrics.metrics
import kogitune.metrics.metrics_python
import kogitune.metrics.metrics_textsim


class TestDataLoader(adhoc.AdhocLoader):

    def load(self, path, tag, kwargs):
        from ..loads import RecordData
        import kogitune.datasets.templates

        stream = adhoc.load("datastream", path, **kwargs)
        samples = [sample for sample in stream.samples()]
        if path.endswith('.jsonl'):
            sample = samples[0]
            if 'dataset' in sample and 'model' in sample:
                record = RecordData(path, samples, **kwargs)
                record.tags = (sample['model'], sample['dataset'])
                return record

        transform = adhoc.load('from_kwargs', 'Transform', **kwargs)
        transform.transform(samples)

        datatag = tag if tag != '' else stream.datatag()
        samples = [{"dataset": datatag} | sample for sample in samples]
        modeltag = kwargs.get('modeltag', '(model)')

        eval_type = kwargs.get('eval_type', 'gen')
        template = adhoc.load("from_kwargs", "Template", _sample=samples[0], **kwargs)
        for sample in samples:
            template.apply(eval_type, sample)

        save_path = get_save_path(modeltag, datatag, eval_type, **kwargs)

        record = RecordData(save_path, samples)
        record.tags = (modeltag, datatag)
        record.save()
        adhoc.saved(save_path, 'Results///テストデータ')
        return record

TestDataLoader().register("testdata")

def get_save_path(modeltag, datatag, task, /, **kwargs):
    if adhoc.get(kwargs, "selfcheck|self_check|=False"):
        task = 'selfcheck'
    
    if task is None or task == "gen":
        save_path = f"{modeltag}/{datatag}_x_{modeltag}.jsonl"
    else:
        save_path = f"{modeltag}/{datatag}_{task}_x_{modeltag}.jsonl"
    
    output_path = adhoc.get(kwargs, "output_path")
    if output_path:
        save_path = os.path.join(output_path, save_path)

    return save_path


def list_testdata(modeltag, eval_type, /, **kwargs):
    if 'modeltag' not in kwargs:
        kwargs['modeltag'] = modeltag
    dataset_list = adhoc.get_list(kwargs, "dataset_list|dataset|!!")
    for dataset in dataset_list:
        for subset in adhoc.get_list(kwargs, "dataset_subset|="):
            kwargs['name'] = subset
            testdata = adhoc.load("testdata", dataset, **kwargs)
            yield testdata


def chain_eval(model_list: List[str], eval_type, metric_list:List[str], /, **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for model_path in model_list:
        model = adhoc.load("model", model_path, extract_prefix="model", **kwargs)
        for testdata in list_testdata(model.modeltag, "gen", **kwargs):
            _check_generater_args(testdata, model, model.gen_args)

            generate(testdata, model, eval_type, **kwargs)
            testdata.save()

            evaluate(testdata, metric_list, board, **kwargs)

    board.show()

def eval_choice(model_list: List[str], /, **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for model_path in model_list:
        model = adhoc.load("model", model_path, extract_prefix="model", **kwargs)
        for testdata in list_testdata(model.modeltag, "choice", **kwargs):

            generate(testdata, model, "choice", **kwargs)
            testdata.save()

            evaluate(testdata, "exact_match", board, **kwargs)

    board.show()

def _check_generater_args(testdata, model, kwargs):
    # print("@check_generater_args", args)
    if "max_new_tokens" not in kwargs and "max_length" not in kwargs:
        kwargs["max_new_tokens"] = _guess_max_new_tokens(testdata, model)
    if "max_new_tokens" in kwargs and "max_length" in kwargs:
        kwargs.pop("max_length", None)
    #print("@check_generater_args", kwargs)

def _guess_max_new_tokens(testdata, model, keys="reference", q=95):
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

def generate(testdata, model, eval_type, **kwargs):
    n = adhoc.get(kwargs, "num_return_sequences|n|N|=1")
    test_run = adhoc.get(kwargs, f"test_run|head|={len(testdata.samples())}")
    test_list = [sample for sample in testdata.samples() if "output" not in sample]
    if len(test_list) == 0:
        return testdata

    adhoc.notice(
        "Start generation///生成をはじめます",
        model=model,
        eval_type=eval_type,
        n=n,
        gen_args=model.gen_args,
    )
    if test_run < len(test_list):
        adhoc.print(f"Test running head={test_run}///先頭のhead={test_run}件のみ、テストしてみます")
        test_list = test_list[:test_run]
    try:
        with adhoc.start_timer() as timer:
            # batch_size = adhoc.get(kwargs, "eval_batch_size|batch_size|=2")
            model.eval(
                test_list,
                eval_type=eval_type,
                n=n,
                # batch_size=batch_size,
            )
            timer.notice("お疲れ様！！ 生成終わりました", total=len(test_list))
    finally:
        testdata.save()

def evaluate(testdata, metric_list:List[str], board:LeaderBoard, **kwargs):
    adhoc.notice("Starting model eval///モデル評価を始めるよ", metric_list, kwargs)
    force_eval = kwargs.get('force_eval', True)
    for metric_path in listfy(metric_list):
        metric = adhoc.load("metric", metric_path, **kwargs)
        result = metric.evaluate(
            testdata.samples(), force_eval=force_eval
        )
        testdata.save()
        board.score_testdata(testdata, metric.path, **kwargs)

def _selfcheck(testdata, model, **kwargs):
    test_list = [sample for sample in testdata.samples() if "selfcheck" not in sample]
    if len(test_list) > 0:
        adhoc.notice(
            "Preparing data for SelfCheck///SelfCheck用のデータを準備します",
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
        kwargs["temperature"] = 1.0
    if "do_sample" not in kwargs:
        kwargs["do_sample"] = True
    return kwargs


def file_eval(files: List[str], metric_list: List[str], **kwargs):
    board = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    for filepath in files:
        testdata = adhoc.load("testdata", filepath)
        evaluate(testdata, metric_list, board, **kwargs)
    board.show()


