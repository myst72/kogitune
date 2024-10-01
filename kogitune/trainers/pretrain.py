import numpy as np
import torch

from .commons import adhoc
from .recipe import DatasetRecipe
from .gpus import *
from .logging import load_wandb

def get_trainer_args(**kwargs):
    from transformers import TrainingArguments

    with adhoc.kwargs_from_stacked(**kwargs) as aargs:
        global_batch_size = adhoc.get(kwargs, "global_batch_size|batch_size|=1024"]
        device_batch_size = adhoc.get(kwargs, "device_batch_size|=16"]
        gas = global_batch_size // device_batch_size
        adhoc.print(
            f"batch_size global={global_batch_size} device={device_batch_size} gradient_accumulation_steps={gas}"
        )
        overwrite_output_dir = "resume_from_checkpoint" not in aargs
        bf16_enabled = adhoc.get(kwargs, f"bf16|={bf16_is_available()}"]
        fp16_enabled = False
        optim = "adamw_torch"
        if torch.cuda.is_available():
            if not bf16_enabled:
                fp16_enabled = True
            optim = "adamw_torch_fused"
        train_args = dict(
            output_dir=adhoc.get(kwargs, "output_dir|=output"],
            overwrite_output_dir=adhoc.get(kwargs, 
                f"overwrite_output_dir|={overwrite_output_dir}"
            ],
            per_device_train_batch_size=adhoc.get(kwargs, 
                f"per_device_train_batch_size|={device_batch_size}"
            ],
            gradient_accumulation_steps=adhoc.get(kwargs, 
                f"gradient_accumulation_steps|={gas}"
            ],
            # per_device_eval_batch_size=64,
            auto_find_batch_size=adhoc.get(kwargs, 
                "auto_find_batch_size|=False"
            ],  # バッチサイズ自動
            do_eval=adhoc.get(kwargs, "do_eval|=False"],
            # evaluation_strategy='steps',
            # eval_steps=50,
            optim=adhoc.get(kwargs, f"optim|={optim}"],
            learning_rate=adhoc.get(kwargs, "learning_rate|=4e-4"],
            weight_decay=adhoc.get(kwargs, "weight_decay|=0.1"],
            adam_beta1=adhoc.get(kwargs, "adam_beta1|=0.9"],
            adam_beta2=adhoc.get(kwargs, "adam_beta2|=0.999"],
            adam_epsilon=adhoc.get(kwargs, "adam_epsilon|=1e-8"],
            max_grad_norm=adhoc.get(kwargs, "max_grad_norm|=1.0"],
            num_train_epochs=adhoc.get(kwargs, "num_train_epochs|=2"],
            max_steps=adhoc.get(kwargs, "max_steps|=-1"],
            lr_scheduler_type=adhoc.get(kwargs, "lr_scheduler_type|=cosine"],
            logging_steps=adhoc.get(kwargs, "logging_steps|=10"],
            dataloader_pin_memory=False,
            save_steps=adhoc.get(kwargs, "save_steps|=1000"],
            save_total_limit=adhoc.get(kwargs, "save_total_limit"],
            save_only_model=adhoc.get(kwargs, "save_only_model|=False"],
            neftune_noise_alpha=adhoc.get(kwargs, "neftune_noise_alpha"],
            torch_compile=adhoc.get(kwargs, "torch_compile|=False"],
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

