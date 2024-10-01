from ..loads.commons import adhoc

@adhoc.cli
def scratch_cli(**kwargs):
    with adhoc.kwargs_from_stacked(kwargs) as kwargs:
        model_type = adhoc.get(kwargs, 'model_type|!llama2')
        model = adhoc.load('scratch', model_type, **kwargs)
        save_path = adhoc.get(kwargs, 'save_path|!scratch')
        if save_path:
            use_fp16=adhoc.get(kwargs, 'save_fp16|=False')
            model.save(save_path, use_fp16=use_fp16)


@adhoc.cli
def pretrain_cli(**kwargs):
    from transformers import Trainer
    from .recipe import DatasetRecipe
    from .pretrain import get_trainer_args
    from .callbacks import TimeoutStoppingCallback

    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        wandb = load_wandb(kwargs)
        kgmodel = adhoc.load("model", adhoc.get(kwargs, "model_path|model|!!"))
        batch_size = adhoc.get(kwargs, "global_batch_size|batch_size|!1024")
        block_size = adhoc.get(kwargs, 'model_size|max_length|!512')
        dataset = DatasetRecipe(adhoc.get(kwargs, "recipie|!!"), batch_size=batch_size, block_size=block_size)
        trainer_args = get_trainer_args(**kwargs)
        resume_from_checkpoint = adhoc.get(kwargs, "resume_from_checkpoint|=False")
        callbacks=[]
        if "max_time" in kwargs or "sge_walltime_sec" in kwargs:
            max_time = adhoc.get(kwargs, "max_time|sge_walltime_sec")
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
        save_path = adhoc.get(kwargs, "save_path|!model")
        if save_path:
            kgmodel.tokenizer.save_pretrained(save_path)
            kgmodel.model.save_pretrained(save_path)
        wandb.finish()
        return result


