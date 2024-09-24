from .commons import adhoc

class DummyWandb:
    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def load_wandb(**kwargs):
    try:
        import wandb
        with adhoc.aargs_from(**kwargs) as aargs:
            if "wandb_team" in aargs:
                wandb.init(
                    entity=aargs["wandb_team"],
                    project=aargs["project"],
                    name=aargs["run_name"],
                )
            else:
                wandb.init(
                    project=aargs["project"],
                    name=aargs["run_name"],
                )
            return wandb
    except ModuleNotFoundError:
        adhoc.print("wandb は入れた方がいいよ")
    return DummyWandb()
