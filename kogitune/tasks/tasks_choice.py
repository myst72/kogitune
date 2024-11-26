from ..loads.commons import *
from .tasks import Task

@adhoc.reg('choice')
class QAChoice(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'choice'
        self.template = None

    def apply_template(self, sample:dict, template:dict):
        sample["_choice"] = template['choice']
        sample["_reference"] = self.format(template, "reference", sample)
        for c in template["choice"]:
            sample[f"_choice_{c}"] = self.format(template, f"choice_{c}", sample)

    def eval(self, model, samples: List[dict]):
        for sample in samples:
            sample['_model'] = model.modeltag
            sample['_task'] = self.tasktag
            choices = [sample[f"_choice_{c}"] for c in sample["_choice"]]
            sample["_losses"] = scores = model.compute_loss(choices)
            predicted_idx = scores.index(min(scores))
            sample["_output"] = sample["_choice"][predicted_idx]
            if self.progress_bar:
                self.progress_bar.update(1)

    @property
    def default_metrics(self):
        return ['exact_match']

