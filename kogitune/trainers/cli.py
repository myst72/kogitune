from ..loads.commons import adhoc

@adhoc.cli
def scratch_cli(**kwargs):
    with adhoc.aargs_from(kwargs) as aargs:
        model_type = aargs['model_type|!llama2']
        model = adhoc.load('scratch', model_type, **kwargs)
        save_path = aargs['save_path|!scratch']
        if save_path:
            model.save(save_path, use_fp16=aargs['save_fp16|=False'])


@adhoc.cli
def pretrain_cli(**kwargs):
    from transformers import Trainer
    from .recipe import DatasetRecipe
    from .pretrain import get_trainer_args
    from .callbacks import TimeoutStoppingCallback

    with adhoc.aargs_from(**kwargs) as aargs:
        wandb = load_wandb(aargs)
        kgmodel = adhoc.load("model", aargs["model_path|model|!!"])
        batch_size = aargs["global_batch_size|batch_size|!1024"]
        block_size = aargs['model_size|max_length|!512']
        dataset = DatasetRecipe(aargs["recipie|!!"], batch_size=batch_size, block_size=block_size)
        trainer_args = get_trainer_args(**aargs)
        resume_from_checkpoint = aargs["resume_from_checkpoint|=False"]
        callbacks=[]
        if "max_time" in aargs or "sge_walltime_sec" in aargs:
            max_time = aargs["max_time|sge_walltime_sec"]
            callbacks=[TimeoutStoppingCallback(max_time=max_time)],
        trainer = Trainer(
            model=kgmodel.model,
            data_collator = get_collator(kgmodel.model),
            train_dataset=dataset,
            args=trainer_args,
            callbacks=callbacks,
        )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print(adhoc.dump(result))
        save_path = aargs["save_path|!model"]
        if save_path:
            kgmodel.tokenizer.save_pretrained(save_path)
            kgmodel.model.save_pretrained(save_path)
        wandb.finish()
        return result


