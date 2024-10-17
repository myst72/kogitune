from .commons import adhoc

class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def load_wandb(**kwargs):
    try:
        import wandb
        with adhoc.kwargs_from_stacked(**kwargs) as aargs:
            if "wandb_team" in aargs:
                wandb.init(
                    entity=adhoc.get(kwargs, "wandb_team"),
                    project=adhoc.get(kwargs, "project"),
                    name=adhoc.get(kwargs, "run_name"),
                )
            else:
                wandb.init(
                    project=adhoc.get(kwargs, "project"),
                    name=adhoc.get(kwargs, "run_name"),
                )
            return wandb
    except ModuleNotFoundError:
        adhoc.print("wandb は入れた方がいいよ")
    return DummyWandb()
