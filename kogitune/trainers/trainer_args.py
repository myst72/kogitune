from ..loads.commons import *
import json

def check_batch_size(kwargs, default_batch_size=1024):
    batch_size = adhoc.get(kwargs, f"global_batch_size|batch_size|!{default_batch_size}")
    device_batch_size = adhoc.get(kwargs, "per_device_train_batch_size|device_batch_size") # |=16
    if device_batch_size:
        kwargs['gradient_accumulation_steps'] = batch_size // device_batch_size
        kwargs['auto_find_batch_size'] = False
    else:
        kwargs['gradient_accumulation_steps'] = adhoc.get(kwargs, "gradient_accumulation_steps|!8")
        kwargs['auto_find_batch_size'] = True
        kwargs["per_device_train_batch_size"] = 1
    return batch_size

def check_resume_step(kwargs):
    kwargs['output_dir'] = adhoc.get(kwargs, "output_dir|project|=output")
    if "resume_from_checkpoint" in kwargs:
        kwargs["overwrite_output_dir"] = False
        resume_step = get_trained_global_step(kwargs['output_dir'])
        if resume_step == 0:
            adhoc.verbose_print(f'チェックポイントが見つかりません')
            kwargs['resume_from_checkpoint'] = False
            kwargs["overwrite_output_dir"] = True
        if resume_step > 0:
            adhoc.print(f'チェックポイント step={resume_step} から再開します。')
        return resume_step
    else:
        kwargs["overwrite_output_dir"] = True
        return 0

def get_trained_global_step(path: str):
    state_file = os.path.join(path, 'trainer_state.json')
    if os.path.exists(state_file):
        try:
            with open(state_file) as f:
                data = json.load(f)
                return data['global_step']
        except:
            pass

    if not os.path.isdir(path):
        return 0
    # 指定されたパス内のすべてのファイルとディレクトリのリストを取得
    dirs = [os.path.join(path, item) for item in os.listdir(path) 
            if os.path.isdir(os.path.join(path, item))]
    if len(dirs) == 0:
        return 0
    # 最も新しいディレクトリを見つける
    newest = max(dirs, key=lambda dir: os.path.getmtime(dir))
    return get_trained_global_step(newest)

def check_gpus(kwargs):
    from .gpus import is_bf16_supported, cuda_is_available
    if 'bf16' not in kwargs and 'fp16' not in kwargs:
        if is_bf16_supported():
            kwargs['bf16'] = True
        elif cuda_is_available():
            kwargs['fp16'] = True
    if 'optim' not in kwargs:
        if cuda_is_available():
            kwargs['optim'] = adhoc.get(kwargs, "optim|=adamw_torch_fused")
        else:
            kwargs['optim'] = adhoc.get(kwargs, "optim|=adamw_torch")

PRETRAIN_ARGS = [
    "output_dir",
    "overwrite_output_dir",
    "per_device_train_batch_size",
    "num_train_epochs",
    "do_eval|=False",
    # "evaluation_strategy|=steps",
    # "eval_steps|=50",
    "gradient_accumulation_steps",
    "auto_find_batch_size",
    "dataloader_pin_memory|=False",
    "optim",
    "learning_rate|=4e-4",
    "weight_decay|=0.1",
    "adam_beta1|=0.9",
    "adam_beta2|=0.999",
    "adam_epsilon|=1e-8",
    "max_grad_norm|=1.0",
    "num_train_epochs|=2",
    "max_steps|=-1",
    "lr_scheduler_type|=cosine",
    "logging_steps|=10",
    "dataloader_pin_memory=False",
    "save_steps|=1000",
    "save_total_limit",
    "save_only_model|=False",
    "neftune_noise_alpha",
    "torch_compile|=False",
    "bf16",
    "fp16",
    "report_to",
]

def get_pretrain_args(**kwargs):
    from transformers import TrainingArguments
    train_args = adhoc.safe_kwargs(kwargs, PRETRAIN_ARGS, unsafe='TRAIN')
    training_args = TrainingArguments(
        **train_args
    )
    return training_args


