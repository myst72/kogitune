from ..loads.commons import *

def finetune(**kwargs):
    adhoc.safe_import('trl')
    import trl
    dataset = load_dataset(**kwargs)
    model = adhoc.load('_model', adhoc.get(kwargs, 'model_path|!!'), **kwargs)
    tokenizer = adhoc.load('_tokenizer', adhoc.get(kwargs, 'tokenizer_path|model_path|!!'))
    
    training_args = get_finetune_args(**kwargs)
    trainer = trl.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

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


def load_dataset(**kwargs):
    # ところでSFT Trainerに渡したdatasetってどうなるの？
    # https://zenn.dev/spiralai/articles/cb8606fa0700f0
    from datasets import Dataset
    prompts=[]
    completions=[]
    for path in adhoc.get_list(kwargs, "dataset|dataset_list|files|!!"):
        stream = adhoc.load('datastream', path, **kwargs)
        samples = [sample for sample in stream.samples()]
        # if 'messages' in samples[0]:
        #     # FIXME
        #     data = {
        #         "messages": adhoc.extract_values("messages")
        #     }
        #     return Dataset.from_dict(data)
        template = guess_template(samples[0])
        apply_template(samples, template, prompts, completions)
    return Dataset.from_dict({
        'prompt': prompts,
        'completion': completions
    })

def apply_template(samples:List[dict], template:str, prompts:List[str], completions:List[str]):
    prompt_fmt = template['prompt']
    completion_fmt = template['completion']
    try:
        for sample in samples:
            prompts.append(adhoc.format(prompt_fmt, **sample))
            completions.append(adhoc.format(completion_fmt, **sample))
    except KeyError as e:
        adhoc.exit(throw=e)


def has_schema(data: dict, keys:str):
    for key in keys.split('|'):
        if key not in data:
            return False
    return True

def contains_japanese(text: str) -> bool:
    for char in text:
        if '\u3040' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FFF' or '\uFF66' <= char <= '\uFF9D':
            return True
    return False

def guess_template(sample: dict):
    if has_schema(sample, 'instruction|input|output'):
        # Alpaca形式
        return {
            "prompt": "{instruction}\n{input}",
            "completion": "{output}",
        }
    return {
        'prompt': '{prompt}',
        'completion': '{completion}'
    }




def format_chatml(messages: List[str]) -> str:
    """
    チャット履歴をChatML形式でフォーマットする。

    Args:
        messages (list): チャット履歴のリスト。
        各要素は {role: "user" or "assistant", content: "メッセージ"} の形式。

    Returns:
        str: ChatML形式でフォーマット済みのチャット履歴文字列。
    """
    # 各メッセージをChatML形式で追加
    ss=[]
    for message in messages:
        role = message["role"]
        content = message["content"]
        ss.append(f"<|im_start|>{role}\n{content}\n<|im_end|>")
    return '\n'.join(ss)

# サンプルデータ

FINETUNE_ARGS = [
    "output_dir",
    "per_device_train_batch_size",
    "num_train_epochs",
    "do_eval|=False",
    "evaluation_strategy|=steps",
    "eval_steps|=50",
    "gradient_accumulation_steps|={gas}",
    "auto_find_batch_size|=False"
    "optim|={optim}",
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
]

def get_finetune_args(**kwargs):
    from transformers import TrainingArguments
    train_args = adhoc.safe_kwargs(kwargs, FINETUNE_ARGS, unsafe='TRAIN')
    training_args = TrainingArguments(
        **train_args
    )
    return training_args


