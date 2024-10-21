
from ..loads.commons import adhoc

def pretrain(**kwargs):
    from .trainer_args import (
        check_batch_size, check_resume_step, check_gpus, 
        get_pretrain_args
    )
    from transformers import Trainer
    from .recipe import DatasetRecipe
    from .callbacks import load_callbacks
    from .logging import wandb_init

    with wandb_init(kwargs):
        model = adhoc.load("_model", adhoc.get(kwargs, "model_path|model|!!"), **kwargs)
        tokenizer = load_train_tokenizer(**kwargs)
        save_path = adhoc.get(kwargs, "save_path|!model")

        batch_size = check_batch_size(kwargs, default_batch_size=1024)
        block_size = adhoc.get(kwargs, 'block_size|max_length|max_seq_length|!512')
        dataset = DatasetRecipe(adhoc.get(kwargs, "recipe|url_list|!!"), 
                                batch_size=batch_size, 
                                block_size=block_size)
        resume_step = check_resume_step(kwargs)
        dataset.skip(resume_step * batch_size)

        check_gpus(kwargs)
        trainer_args = get_pretrain_args(**kwargs)

        trainer = Trainer(
            model=model,
            data_collator = get_collator(tokenizer),
            train_dataset=dataset,
            args=trainer_args,
            callbacks=load_callbacks(**kwargs),
        )
        resume_from_checkpoint = adhoc.get(kwargs, "resume_from_checkpoint|=False")
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        adhoc.verbose_print('トレーニング完了', dump=result)
        save_path = adhoc.get(kwargs, "save_path|!model")
        if save_path:
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        return result

def load_train_tokenizer(**kwargs):
    tokenizer = adhoc.load("_tokenizer", adhoc.get(kwargs, "tokenizer_path|model_path|model|!!"))
    if not hasattr(tokenizer, 'pad_token_id'):
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def get_collator(tokenizer):
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(
        tokenizer, 
        pad_to_multiple_of=8, 
        mlm=False
    )

