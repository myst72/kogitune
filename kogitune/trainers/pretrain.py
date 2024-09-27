import numpy as np
import torch

from .commons import adhoc
from .recipe import DatasetRecipe
from .gpus import *
from .logging import load_wandb

def get_trainer_args(**kwargs):
    from transformers import TrainingArguments

    with adhoc.aargs_from(**kwargs) as aargs:
        global_batch_size = aargs["global_batch_size|batch_size|=1024"]
        device_batch_size = aargs["device_batch_size|=16"]
        gas = global_batch_size // device_batch_size
        adhoc.print(
            f"batch_size global={global_batch_size} device={device_batch_size} gradient_accumulation_steps={gas}"
        )
        overwrite_output_dir = "resume_from_checkpoint" not in aargs
        bf16_enabled = aargs[f"bf16|={bf16_is_available()}"]
        fp16_enabled = False
        optim = "adamw_torch"
        if torch.cuda.is_available():
            if not bf16_enabled:
                fp16_enabled = True
            optim = "adamw_torch_fused"
        train_args = dict(
            output_dir=aargs["output_dir|=output"],
            overwrite_output_dir=aargs[
                f"overwrite_output_dir|={overwrite_output_dir}"
            ],
            per_device_train_batch_size=aargs[
                f"per_device_train_batch_size|={device_batch_size}"
            ],
            gradient_accumulation_steps=aargs[
                f"gradient_accumulation_steps|={gas}"
            ],
            # per_device_eval_batch_size=64,
            auto_find_batch_size=aargs[
                "auto_find_batch_size|=False"
            ],  # バッチサイズ自動
            do_eval=aargs["do_eval|=False"],
            # evaluation_strategy='steps',
            # eval_steps=50,
            optim=aargs[f"optim|={optim}"],
            learning_rate=aargs["learning_rate|=4e-4"],
            weight_decay=aargs["weight_decay|=0.1"],
            adam_beta1=aargs["adam_beta1|=0.9"],
            adam_beta2=aargs["adam_beta2|=0.999"],
            adam_epsilon=aargs["adam_epsilon|=1e-8"],
            max_grad_norm=aargs["max_grad_norm|=1.0"],
            num_train_epochs=aargs["num_train_epochs|=2"],
            max_steps=aargs["max_steps|=-1"],
            lr_scheduler_type=aargs["lr_scheduler_type|=cosine"],
            logging_steps=aargs["logging_steps|=10"],
            dataloader_pin_memory=False,
            save_steps=aargs["save_steps|=1000"],
            save_total_limit=aargs["save_total_limit"],
            save_only_model=aargs["save_only_model|=False"],
            neftune_noise_alpha=aargs["neftune_noise_alpha"],
            torch_compile=aargs["torch_compile|=False"],
            bf16=bf16_enabled,
            fp16=fp16_enabled,
        )
        # adhoc.setlog('train', trainer_args=train_args)
        return TrainingArguments(**train_args)

def get_collator(tokenizer):
    from transformers import DataCollatorForLanguageModeling
    return DataCollatorForLanguageModeling(
        tokenizer, pad_to_multiple_of=8, mlm=False
    )

