from typing import List, Union, Any

from ..loads.commons import *
from ..loads import Model, LeaderBoard
from .loads import Task

def task_eval(model_list: List[str], metrics:List[str], /, **kwargs):
    # タスクをロードしておきます。
    from .tasks_textgen import TextGeneration
    from .tasks_choice import QAChoice
    from .metrics import Metric

    board:LeaderBoard = adhoc.load('from_kwargs', 'leaderboard', **kwargs)
    save_step = adhoc.get(kwargs, "save_steps|save_step")
    if save_step:
        kwargs["_resume"] = True # 前の続きから実行する (ただし、resumeが優先)
    task:Task = adhoc.load('task', adhoc.get(kwargs, 'task|='))

    for model_path in model_list:
        model : Model = adhoc.load("model", model_path, _lazy = True, **kwargs)
        for taskdata in list_testdata(model.modeltag, task.tasktag, **kwargs):
            # 未処理のサンプルのみ対象にする
            samples = [sample for sample in taskdata.samples() 
                       if sample.get("_model") != model.modeltag]
            if len(samples) > 0:   
                head = adhoc.get(kwargs, "test_run|head")
                if head:
                    samples = samples[:head]
                
                # save_stepごとに保存する
                save_step = adhoc.get(kwargs, "save_steps|save_step|=4")
                with VerboseCounter(**kwargs) as verbose:
                    task.start_progress_bar(total=len(samples), desc=model.modeltag)
                    for start in range(0, len(samples), save_step):
                        splited_samples = samples[start:start+save_step]
                        task.prepare(splited_samples)
                        task.eval(model, splited_samples)
                        taskdata.save()
                        verbose.print_sample(splited_samples)
                    task.end_progress_bar()
            
            if len(metrics) == 0:
                metrics = task.default_metrics

            # 処理済みのサンプルのみ扱う
            samples = [sample for sample in taskdata.samples() if sample.get("_model") == model.modeltag]
            if len(samples) == 0:
                adhoc.verbose_print("ひとつも生成されてないね")
                break
            
            for metric_path in listfy(metrics):
                if metric_path == "none": 
                    break
                metric = adhoc.load("metric", metric_path, **kwargs)
                if metric.check(samples):
                    results = task.calc(metric, samples)
                    board.pivot_table(samples, results, **kwargs)
                taskdata.save()
        model = None
    board.show()

def list_testdata(modeltag, tasktag, /, **kwargs):
    dataset_list = adhoc.get_list(kwargs, "dataset_list|dataset|!!")
    kwargs['_modeltag'] = modeltag
    kwargs['_tasktag'] = tasktag
    for path in dataset_list:
        if 'dataset_subset' in kwargs:
            path, largs, _ = adhoc.parse_path(path, parent_args=kwargs)
            if kwargs['dataset_subset'] == '*':
                from ..loads.datasets import load_dataset_names
                kwargs['dataset_subset'] = load_dataset_names(path)
            if largs.get('name') != '*':
                for subset_name in adhoc.get_list(kwargs, f"dataset_subset"):
                    taskdata = load_taskdata(path, **(kwargs|{'_name': subset_name}))
                    yield taskdata
                continue
        taskdata = load_taskdata(path, **kwargs)
        yield taskdata


def load_taskdata(path, /, **kwargs):
    from ..loads import RecordData, Transform

    stream = adhoc.load("datastream", path, **kwargs)
    samples = [sample for sample in stream.samples()]
    if path.endswith('.jsonl'):
        sample = samples[0]
        if '_dataset' in sample and '_model' in sample and '_task' in sample:
            record = RecordData(path, samples, **kwargs)
            record.tags = (sample['_model'], sample['_dataset'], sample['_task'])
            return record

    transform: Transform = adhoc.load('from_kwargs', 'Transform', **kwargs)
    if not transform.isNullObject():
        samples = transform.transform(samples)
        adhoc.verbose_print("Transform//簡易的な変形:", adhoc.dump(samples[0]))

    modeltag = kwargs.get('_modeltag', '(model)')
    datatag = stream.datatag
    task = adhoc.get(kwargs, "_tasktag|task|=")
    samples = [(sample | {"_dataset": datatag}) for sample in samples]

    # save_path の決定
    if task == "":
        save_path = f"{modeltag}/{datatag}_x_{modeltag}.jsonl"
    else:
        save_path = f"{modeltag}/{datatag}_{task}_x_{modeltag}.jsonl"
    output_path = adhoc.get(kwargs, "output_path|output_dir")
    if output_path:
        save_path = os.path.join(output_path, save_path)

    record = RecordData(save_path, samples, **kwargs)
    record.tags = (modeltag, datatag, task)
    record.save()
    adhoc.saved(save_path, f'Results//実験結果 {(modeltag, datatag, task)}')
    return record






