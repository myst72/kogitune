import os, json
import torch

from .commons import adhoc
from .gpus import bf16_is_available
from ..datasets import load_train_dataset, load_template
from .callbacks import load_callbacks


def inspect_model(model):
    # # モデルの構造を表示
    # print("Model Structure:")
    # print(model)
    
    # データタイプを表示
    print("Parameter Data Types:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.dtype}")
    
    # 全パラメータ数と学習可能なパラメータ数を表示
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    device = next(model.parameters()).device
    # デバイスを表示
    print(f"\nModel is loaded on device: {device}")

    # # 特定のレイヤーや属性の情報を表示（例: 埋め込み層、出力層）
    # print("\nEmbedding Layer Information:")
    # print(model.transformer.wte)
    # print("\nOutput Layer Information:")
    # print(model.lm_head)


def load_train_model(**kwargs):
    from transformers import AutoModelForCausalLM
    with adhoc.kwargs_from_stacked(**kwargs) as aargs:
        model_path = adhoc.get(kwargs, 'model_path|!!']
        model_args = adhoc.get(kwargs, 'model_args|model_config']
        if model_args is None:
            model_path, model_args = adhoc.parse_path_args(model_path)
        if 'use_auth_token' not in model_args:
            model_args['use_auth_token'] = adhoc.get(kwargs, 'hf_token']
        if 'trust_remote_code' not in model_args:
            model_args['trust_remote_code'] = True
        # MacOS 上でエラーになる
        if 'device_map' not in model_args:
            model_args['device_map'] = "auto"
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)
        inspect_model(model)
        return model


def get_checkpoint_global_step(path: str):
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
    # print('@', newest)
    return get_checkpoint_global_step(newest)

def check_resume_from_checkpoint(aargs):
    output_dir=adhoc.get(kwargs, 'output_dir|=output']
    overwrite_output_dir = adhoc.get(kwargs, f'overwrite_output_dir|={False}']
    resume_from_checkpoint = adhoc.get(kwargs, f'resume_from_checkpoint']
    if overwrite_output_dir:
        if resume_from_checkpoint:
            adhoc.notice(f'overwrite_output_dir={overwrite_output_dir}とresume_from_checkpoint={resume_from_checkpoint}が矛盾するね。さてさて')
        else:
            return resume_from_checkpoint, output_dir, overwrite_output_dir
    if isinstance(resume_from_checkpoint, str):
        global_steps = get_checkpoint_global_step(resume_from_checkpoint)
        if global_steps == 0:
            resume_from_checkpoint = True
            global_steps = get_checkpoint_global_step(output_dir)
    else:
        global_steps = get_checkpoint_global_step(output_dir)
    if global_steps > 0:
        adhoc.notice(f'チェックポイント(steps={global_steps}から学習するね')
        if not isinstance(resume_from_checkpoint, str):
            resume_from_checkpoint = True
        overwrite_output_dir = False
    else:
        adhoc.notice(f'チェックポイントが見つからないよ（最初からの学習になるよ）')
        # overwrite_output_dir = True
        resume_from_checkpoint = False
    return resume_from_checkpoint, output_dir, overwrite_output_dir

def configure_train_args(**kwargs):
    with adhoc.kwargs_from_stacked(**kwargs) as aargs:
        global_batch_size = adhoc.get(kwargs, 'global_batch_size|batch_size|=8']
        device_batch_size = adhoc.get(kwargs, 'device_batch_size|=8']
        gas = global_batch_size // device_batch_size
        adhoc.notice('バッチサイズ', 
                        global_batch_size=global_batch_size, 
                        device_batch_size=device_batch_size,
                        gradient_accumulation_steps=gas,
        )
        resume_from_checkpoint, output_dir, overwrite_output_dir = check_resume_from_checkpoint(aargs=aargs)
        bf16_enabled = adhoc.get(kwargs, f'bf16|={bf16_is_available()}']
        fp16_enabled = False
        optim='adamw_torch'
        if torch.cuda.is_available():
            if not bf16_enabled:
                fp16_enabled=True
            optim='adamw_torch_fused'
        train_args = dict(
            output_dir=output_dir,
            overwrite_output_dir=overwrite_output_dir,
            per_device_train_batch_size=adhoc.get(kwargs, f'per_device_train_batch_size|={device_batch_size}'],
            gradient_accumulation_steps=adhoc.get(kwargs, f'gradient_accumulation_steps|={gas}'],
            # per_device_eval_batch_size=64,
            auto_find_batch_size=adhoc.get(kwargs, 'auto_find_batch_size|=False'],  # バッチサイズ自動
            do_eval=adhoc.get(kwargs, 'do_eval|=False'],
            # evaluation_strategy='steps',
            # eval_steps=50,
            optim=adhoc.get(kwargs, f'optim|={optim}'],
            learning_rate=adhoc.get(kwargs, 'learning_rate|=4e-4'], 
            weight_decay=adhoc.get(kwargs, 'weight_decay|=0.1'],
            adam_beta1=adhoc.get(kwargs, 'adam_beta1|=0.9'],
            adam_beta2=adhoc.get(kwargs, 'adam_beta2|=0.999'],
            adam_epsilon=adhoc.get(kwargs, 'adam_epsilon|=1e-8'],
            max_grad_norm=adhoc.get(kwargs, 'max_grad_norm|=1.0'],
            num_train_epochs=adhoc.get(kwargs, 'num_train_epochs|=2'],
            max_steps=adhoc.get(kwargs, 'max_steps|=-1'],
            lr_scheduler_type=adhoc.get(kwargs, 'lr_scheduler_type|=cosine'],
            logging_steps=adhoc.get(kwargs, 'logging_steps|=10'],
            dataloader_pin_memory=False,
            save_steps=adhoc.get(kwargs, 'save_steps|=1000'],
            save_total_limit=adhoc.get(kwargs, 'save_total_limit'],
            save_only_model=adhoc.get(kwargs, 'save_only_model|=False'],
            neftune_noise_alpha=adhoc.get(kwargs, 'neftune_noise_alpha'],
            torch_compile=adhoc.get(kwargs, 'torch_compile|=False'],
            bf16=bf16_enabled, 
            fp16=fp16_enabled,
        )
        return train_args, resume_from_checkpoint

## これはサンプルで残しておこう

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        combined_text = f"{example['instruction'][i]} {example['input'][i]}"
        text = f"### instruction\n{combined_text}\n### output\n{example['output'][i]}"
        output_texts.append(text)
    return output_texts

@adhoc.cli
def finetune_cli(**kwargs):
    from transformers import TrainingArguments
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
    with adhoc.kwargs_from_stacked(**kwargs) as aargs:
        dataset = load_train_dataset()
        template = load_template(sample=dataset[0])

        tokenizer = adhoc.load_tokenizer()
        max_seq_length = adhoc.get(kwargs, 'max_seq_length|max_length']
        if max_seq_length is None:
            max_seq_length = template.calc_length(dataset, tokenizer=tokenizer)
            adhoc.notice(f'max_seq_length={max_seq_length}に設定したよ。95%辺り。お気に召さないなら自分で設定してね')
        if adhoc.get(kwargs, 'safe_finetune|={True}'] or adhoc.get(kwargs, 'test_run|head']:
            dataset2 = template.filter(dataset, 
                                        tokenizer=tokenizer, 
                                        max_length=max_seq_length, head=adhoc.get(kwargs, 'test_run|head'])
            adhoc.notice(f'データセットを{len(dataset)}件から{len(dataset2)}件にフィルタ。(これで学習しやすくなるね）')
            dataset = dataset2

        model = load_train_model()
        train_args, resume_from_checkpoint = configure_train_args()

        # We added context here: "\n". This is enough for this tokenizer
        response_template_with_context = template.SEC_OUT
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:] 

        data_collator = DataCollatorForCompletionOnlyLM(response_template_ids, 
                                                        tokenizer=tokenizer)
        trainer = SFTTrainer(
            model=model,
            args=TrainingArguments(**train_args),
            train_dataset=dataset,
            formatting_func=template.formatting_for_trainer,
            # formatting_func=formatting_prompts_func,
            data_collator=data_collator,
            max_seq_length=max_seq_length,
            callbacks = load_callbacks(),
        )
        result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        adhoc.notice('お疲れ様！ファインチューン完了です', result=result)
        save_trained_model(model, tokenizer, 
                            save_path=adhoc.get(kwargs, 'save_model_path|save_path'])

def save_trained_model(model, tokenizer=None, save_path=None, default_path='model'):
    # save_path = adhoc.get(kwargs, 'save_model_path|save_path']
    if save_path is None:
        save_path = default_path
        if save_path and not os.path.exists(save_path):
            adhoc.notice(f'{save_path}に出力するよ。嫌ならsave_model_pathで直してね！')
    if save_path:
        model.save_pretrained(save_path)
        if tokenizer:
            tokenizer.save_pretrained(save_path)
