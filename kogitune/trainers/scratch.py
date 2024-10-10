import os
from ..loads.commons import adhoc

SCRATCH_MAP = {}

class ScratchModel(adhoc.AdhocObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pathargs = {}

        self.tokenizer = adhoc.load("from_kwargs", "_tokenizer", use_default=True, **kwargs)
        config = self.build_config(kwargs)
        self.model = self.build(**config)
        self.print_model(self.model)

    def unwrap(self):
        return self.model

    def build_config(self, kwargs):
        tokenizer = self.tokenizer
        with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
            num_attention_heads = adhoc.get(kwargs, "num_attention_heads|n_heads|=4")
            if "hidden_size" in kwargs:
                hidden_size = adhoc.get(kwargs, "hidden_size|=1024")
            else:
                hidden_size = adhoc.get(kwargs, "head_dim|n_dims|=32") * num_attention_heads
            return dict(
                # model_type=adhoc.get(kwargs, "model_type|=llama2"],
                vocab_size=adhoc.get(kwargs, f"vocab_size|={tokenizer.vocab_size}"),
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_attention_heads=num_attention_heads,
                hidden_size=hidden_size,
                intermediate_size=adhoc.get(kwargs, "intermediate_size|=512"),
                num_hidden_layers=adhoc.get(kwargs, "num_hidden_layers|n_layers|=12"),
                num_key_value_heads=adhoc.get(kwargs, "num_key_value_heads|head_groups|n_groups|=4"),
                max_position_embeddings=adhoc.get(kwargs, "max_position_embeddings|=4096"),
            )
            # adhoc.safe_kwargs(
            #     kwargs,
            #     "num_key_value_heads|group_heads|n_groups",
            #     "hidden_act",
            #     "rms_norm_eps",
            #     "rope_theta",
            #     "tie_word_embeddings",
            #     "attention_dropout",
            #     "attention_bias",
            #     "sliding_window",
            #     "partial_rotary_factor",
            # )
            # return scratch_config

    def build(self, **kwargs):
        from transformers import LlamaForCausalLM, LlamaConfig

        config = LlamaConfig(**kwargs)
        model = LlamaForCausalLM(config)
        return model

    def print_model(self, model):
        print_model(model)
        print_model_structure(model)

    def save(self, save_path:str, use_fp16=True):
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        if use_fp16:
            self.model = reduce_model_using_float16(save_path)

    @classmethod
    def register(cls, names: str):
        global SCRATCH_MAP
        for name in adhoc.list_keys(names):
            SCRATCH_MAP[name] = cls


class ScratchBuilder(adhoc.AdhocLoader):
    def add_kwargs(self, path, kwargs):
        extra_config = adhoc.get(kwargs, "scratch_config|scratch_kwargs")
        if extra_config:
            kwargs = kwargs | extra_config
        return path, kwargs

ScratchBuilder(SCRATCH_MAP).register("scratch")




def count_parameters(model) -> int:
    """
    モデルのパラメータ数を数える

    model: モデル
    return パラメータ数
    """
    return sum(p.numel() for p in model.parameters())


def print_model(model):
    n_parameters = count_parameters(model)
    config = model.config
    adhoc.print(
        f"Parameters: {n_parameters} {adhoc.format_unit(n_parameters)}",
        end=" ",
        face="",
    )
    if hasattr(config, "max_position_embeddings"):
        adhoc.print(f"max_length: {config.max_position_embeddings}", end=" ", face="")
    elif hasattr(config, "n_positions"):
        adhoc.print(f"max_length: {config.n_positions}", end=" ", face="")
    adhoc.print(f"vocab_size: {config.vocab_size}", face="")

    if hasattr(config, "d_kv"):  # T5
        adhoc.print(f"d_model: {model.config.d_model}", end=" ", face="")
        adhoc.print(f"d_kv: {model.config.d_kv}", end=" ", face="")
        adhoc.print(f"d_ff: {model.config.d_ff}", end=" ", face="")
        adhoc.print(f"num_heads: {model.config.num_heads}", end=" ", face="")
        adhoc.print(
            f"num_layers: {model.config.num_layers}+{model.config.num_decoder_layers}",
            face="",
        )
        adhoc.print(config)
    elif hasattr(config, "n_embd"):  # GPT-2
        adhoc.print(f"hidden_size: {config.n_embd}", end=" ", face="")
        adhoc.print(f"intermediate_size: {config.n_inner}", end=" ", face="")
        adhoc.print(f"n_dims: {config.n_embd//config.n_head}", end=" ", face="")
        adhoc.print(f"n_head: {config.n_head}", end=" ", face="")
        adhoc.print(f"n_layers: {config.n_layer}", face="")
        adhoc.print(config)
    elif hasattr(config, "hidden_size"):  # GPT-NeoX
        adhoc.print(f"hidden_size: {config.hidden_size}", end=" ", face="")
        adhoc.print(f"intermediate_size: {config.intermediate_size}", end=" ", face="")
        adhoc.print(
            f"n_dims: {config.hidden_size//model.config.num_attention_heads}",
            end=" ",
            face="",
        )
        adhoc.print(
            f"num_attention_heads: {config.num_attention_heads}", end=" ", face=""
        )
        adhoc.print(f"num_hidden_layers: {config.num_hidden_layers}", face="")
    else:
        adhoc.print(config, face="")


def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def print_model_structure(model):
    num_dict = {}
    for name, param in model.named_parameters():
        num_dict[name] = param.numel()
    repl_parts = [
        "model",
        "layers",
        "weight",
        "mlp",
        "self_attn",
    ]
    name_set = []
    for original_name in num_dict:
        name = original_name.split(".")
        name = [i for i in name if i not in repl_parts]
        name = [i for i in name if not is_integer(i)]
        name_set.append(".".join(name))

    # 主要なレイヤーグループの表示
    # adhoc.print(set(name_set))

    # パラメータ数の計算
    name_group_dict = {}
    all_params = 0
    for k, v in num_dict.items():
        found = False
        for group_name in name_set:
            if group_name in k:
                if group_name not in name_group_dict:
                    name_group_dict[group_name] = v
                else:
                    name_group_dict[group_name] += v
                all_params += v
                found = True
                break
        if not found:
            print(k, " not found")
    # print(name_group_dict)
    import pandas as pd

    df = pd.DataFrame.from_dict(name_group_dict, orient="index")
    df.columns = ["params"]
    df["ratio"] = df["params"] / all_params * 100
    adhoc.print(df, face="")


### new version

def reduce_model_using_float16(model_path):
    import torch
    from transformers import AutoModelForCausalLM

    ## なんか馬鹿らしいコードだけど、float16に変換　サイズが半分になる
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    model.save_pretrained(model_path)
    return model


class GPT2Model(ScratchModel):

    def build(self, **kwargs):
        from transformers import GPT2LMHeadModel, GPT2Config

        config = GPT2Config(
            vocab_size=kwargs["vocab_size"],
            bos_token_id=kwargs["bos_token_id"],
            eos_token_id=kwargs["eos_token_id"],
            pad_token_id=kwargs["pad_token_id"],
            n_positions=kwargs["max_position_embeddings"],
            n_ctx=kwargs["max_position_embeddings"],
            n_embd=kwargs["hidden_size"],
            n_head=kwargs["num_attention_heads"],
            n_layer=kwargs["num_hidden_layers"],
            n_inner=kwargs["intermediate_size"],
        )
        model = GPT2LMHeadModel(config)
        return model


class Llama2Model(ScratchModel):

    def build(self, **kwargs):
        from transformers import LlamaForCausalLM, LlamaConfig

        config = LlamaConfig(**kwargs)
        model = LlamaForCausalLM(config)
        return model


Llama2Model.register('llama|llama2|llama3')

class MistralModel(ScratchModel):

    def build(self, **kwargs):
        from transformers import MistralForCausalLM, MistralConfig

        # adhoc.check_kwargs(kwargs, MistralConfig)
        config = MistralConfig(**kwargs)
        model = MistralForCausalLM(config)
        return model

MistralModel.register('mistral')

class GemmaModel(ScratchModel):

    def build(self, **kwargs):
        from transformers import GemmaForCausalLM, GemmaConfig

        config = GemmaConfig(**kwargs)
        model = GemmaForCausalLM(config)
        return model

GemmaModel.register('gemma|gemma2')

# def generate_scratch_stablelm(**kwargs):
#     from transformers import StableLmForCausalLM, StableLmConfig

#     config = StableLmConfig(**kwargs)
#     model = StableLmForCausalLM(config)
#     print_model(model)
#     print_model_structure(model)
#     return model



# def generate_scratch_gptneox(**kwargs):
#     from transformers import GPTNeoXForCausalLM, GPTNeoXConfig

#     config = GPTNeoXConfig(**kwargs)
#     model = GPTNeoXForCausalLM(config)
#     print_model(model)
#     print_model_structure(model)
#     return model
