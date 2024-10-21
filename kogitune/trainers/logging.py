from ..loads.commons import adhoc
import os

class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

WANDB_INIT_ARGS =[
    'entity|wandb_team|WANDB_TEAM',
    'project|wandb_project|WANDB_PROJECT',
    'name|wandb_name|run_name|WANDB_NAME',
]

def wandb_init(kwargs: dict):
    if 'use_wandb' not in kwargs:
        project = adhoc.get(kwargs, 'project|wandb_project|WANDB_PROJECT')
        if project:
            kwargs['_use_wandb'] = True
        else:
            adhoc.print("wandb を使いたいときは、`project`か`wandb_project`を設定してね")
            kwargs['_use_wandb'] = False
    if adhoc.get(kwargs, 'use_wandb|_use_wandb|=True') == False:
        os.environ['WANDB_DISABLED'] = "true"
        kwargs['_report_to'] = 'none'
        return DummyWandb()
    try:
        wandb = adhoc.safe_import("wandb")
    except ModuleNotFoundError:
        adhoc.print("wandb は入れた方がいいよ")
        return DummyWandb()
    kwargs = adhoc.safe_kwargs(kwargs, WANDB_INIT_ARGS, unsafe='WANDB')
    return wandb.init(**kwargs)
