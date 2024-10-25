import math
from .commons import *
from .models_ import TokenizerModel 

class vLLMModel(TokenizerModel):
    def __init__(self, **kwargs):
        vllm = adhoc.safe_import('vllm')
        self.SamplingParams = vllm.SamplingParams
        super().__init__(**kwargs)
        if not kwargs.get('_lazy', False):
            self.lazy_load()

    def lazy_load(self):
        if self.lazy_kwargs is None:
            return
        kwargs = self.lazy_kwargs
        self.lazy_kwargs = None

        self._vllm = load_model(**kwargs)

    @property
    def vllm(self):
        self.lazy_load()
        return self._vllm

    def unwrap(self):
        self.lazy_load()
        return self._vllm

    def supported_gen_args(self) -> List[str]:
        return [
            "_n|num_return_sequences|n",
            "_do_sample|do_sample",  # (bool, optional, defaults to False) — Whether or not to use sampling ; use greedy decoding otherwise.
            "_max_tokens|max_new_tokens|max_tokens|=256",  # (int, optional) — The maximum numbers of tokens to generate
            "_temperature|temperature",  # (float, optional, defaults to 1.0) — The value used to modulate the next token probabilities.
            "top_k",  # (int, optional, defaults to 50) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
            "top_p",  # (float, optional, defaults to 1.0) — If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            "repetition_penalty",  # (float, optional, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details.
#            "max_length",  # (int, optional, defaults to 20) — The maximum length the generated tokens can have.
        ]

    def filter_gen_args(self, **kwargs):
        gen_args = super().filter_gen_args(**kwargs)
        # if "return_full_text" not in gen_args:
        #     gen_args["return_full_text"] = False
        # if "max_length" in gen_args and "max_new_tokens" in gen_args:
        #     gen_args.pop("max_length")
        return gen_args

    def generate(self, input_texts: Union[List[str],str], progress_bar=None, /, **kwargs) -> Union[List[str], str]:
        self.lazy_load()
        gen_args = self.filter_gen_args(**kwargs)
        sampling_params = self.SamplingParams(gen_args)
        
        input_texts = [self.format_text_prompt(s) for s in self.listfy_prompt(input_texts)]
        adhoc.verbose_print('[Prompt]', dump=input_texts, once="formatted_prompt")

        outputs = self._vllm.generate(input_texts, sampling_params)

        output_texts = []
        for output in outputs:
            generated_texts = [output.outputs[0].text for i in len(output.outputs)]
            output_texts.append(singlefy_if_single(generated_texts))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(output_texts)
        
    def compute_loss(self, input_texts: Union[List[str],str], progress_bar=None) -> Union[List[float], float]:
        self.lazy_load()
        sampling_params = self.SamplingParams()
        outputs = self.llm.generate(listfy(input_texts), sampling_params)
        values = []
        for i, output in enumerate(outputs):
            values.append(math.log(output.outputs[0].perplexity))
            if progress_bar:
                progress_bar.update(1)
        return singlefy_if_single(values)

vLLMModel.regiser("vllm")

"""
・n : プロンプトに対して返される出力シーケンス数
・best_of : プロンプトから生成される出力シーケンスの数。best_ofシーケンスから、上位nシーケンスが返される
・frequency_penalty : 生成されたテキスト内の頻度に基づいて、新しいトークンにペナルティを与える浮動小数点数。値 > 0 の場合は新しいトークンの使用を推奨、値 < 0 の場合はトークン繰り返しを推奨
・repetition_penalty : 新しいトークンがプロンプトおよびこれまでに生成されたテキストに表示されるかどうかに基づいて、新しいトークンにペナルティを与える浮動小数点数。値 > 1 の場合は新しいトークンの使用を推奨、値 < 1 の場合はトークン繰り返しを推奨
・temperature : サンプリングのランダム性を制御する浮動小数点数。値が低いほどより決定的、値が高いほどよりランダム。0は貪欲なサンプリング
・top_p : 考慮する上位トークンの累積確率を制御する浮動小数点数。 (0, 1] でなければならない。すべてのトークンを考慮するには 1 に設定
・top_k : 考慮する上位トークンの数を制御する整数。すべてのトークンを考慮するには、-1 に設定
・min_p : 最も可能性の高いトークンの確率と比較して、考慮されるトークンの最小確率を表す浮動小数点数。[0, 1] になければならない。これを無効にするには 0 に設定
・use_beam_search : サンプリングの代わりにBeam Searchを使用するかどうか
・length_penalty : 長さに基づいてシーケンスにペナルティを与える浮動小数。Beam Searchに使用
・early_stopping : Beam Searchの停止条件を制御
　・True : best_of の完全な候補が存在するとすぐに生成が停止
　・False : ヒューリスティックが適用され、より適切な候補が見つかる可能性が非常に低い場合に生成が停止
　・never : Beam Search手順は、より良い候補が存在しない場合にのみ停止
・stop : 生成時に生成を停止する文字列のリスト。返される出力には停止文字列は含まれない
・stop_token_ids : 生成時に生成を停止するトークンのリスト。返される出力には、ストップトークンがスペシャルトークンでない限り、ストップトークンが含まれます
・include_stop_str_in_output : 出力テキストに停止文字列を含めるかどうか。デフォルトはFalse
・ignore_eos : EOS トークンが生成された後、EOS トークンを無視してトークンの生成を続行するかどうか
・max_tokens : 出力シーケンスごとに生成するトークンの最大数
・logprobs : 出力トークンごとに返されるログの確率の数。 実装は OpenAI API に従っていることに注意。返される結果には、最も可能性の高いlogprobsトークンのログ確率と、選択されたトークンが含まれる。API は常にサンプリングされたトークンの対数確率を返すため、応答には最大 logprobs+1 要素が含まれる可能性がある
・prompt_logprobs : プロンプト トークンごとに返されるログの確率の数。
・skip_special_tokens : 出力内の特別なトークンをスキップするかどうか
・space_between_special_tokens : 出力内の特別なトークンの間にスペースを追加するかどうか。デフォルトはTrue
・logits_processors : 以前に生成されたトークンに基づいてロジットを変更する関数のリスト

"""

def load_model(**kwargs):
    from vllm import LLM
    try:
        llm = LLM(model=adhoc.get(kwargs, '_subpath|model_path|model|!!'), 
                #   gpu_memory_utilization=float(0.8),
                #   tensor_parallel_size=torch.cuda.device_count(),
                #   max_model_len=adhoc.get(kwargs, "max_model_length|=4096",
                trust_remote_code=True)
        return llm
    except BaseException as e:
        adhoc.print('vLLMは利用できないようです。(理由)', e, color='red')
        adhoc.exit(throw=e)
