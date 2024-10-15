###
# OpenAI
import json

from .commons import *
from .models import Model

class OpenAIModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = adhoc.get(kwargs, "_subpath|model_path|model")
        openai = adhoc.safe_import('openai')
        api_key = adhoc.get(kwargs, "api_key|OPENAI_API_KEY|!!")
        self.client = openai.OpenAI(api_key=api_key)

    def supported_gen_args(self) -> List[str]:
        return [
            ## https://platform.openai.com/docs/api-reference/chat/create
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens|max_new_tokens",
            "presence_penalty",
            "response_format",
            "seed",
            "service_tier",
            "stop",
            # "stream",
            "temperature",
            "top_p",
        ]

    def generate(self, input_texts: Union[List[str], str], n=1, progress_bar=None, /, **kwargs):
        gen_args = adhoc.parse_value_of_args(kwargs)
        if "max_new_tokens" in gen_args:
            gen_args["max_tokens"] = gen_args.pop("max_new_tokens")
        if "num_return_sequences" in gen_args:
            gen_args.pop("num_return_sequences")
        return super().generate(input_texts, n, progress_bar, **gen_args)

    def generate_s(self, input_text: str, n=1, /, **gen_args):
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=[{"role": "user", "content": input_text}],
            n=n,
            **gen_args,
        )
        responses = [choice.message.content for choice in response.choices]
        return singlefy_if_single(responses)

OpenAIModel.regiser("openai")


class BedrockModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = adhoc.get(kwargs, "_subpath|model_path|model")
        boto3 = adhoc.safe_import('boto3')
        self.client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=adhoc.get(kwargs, 'region_name|=us-west-2')
        )

    def supported_gen_args(self) -> List[str]:
        return [
            "temperature",
            "max_new_tokens|max_tokens|=256",
            "top_p",
        ]

    def generate_s(self, input_text: Union[List, str], n=1, /, **kwargs):
        gen_args = self.filter_gen_args(n, kwargs)

        if isinstance(input_text, str):
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "prompt": input_text,
                    **gen_args,
                }),
                contentType='application/json'
            )
            # レスポンスの読み取り
            response_body = response['body'].read().decode('utf-8')
            response_json = json.loads(response_body)
            generated_text = response_json.get('completion', '')
            return generated_text
        else:
            response = self.client.invoke_model(
                modelId=self.model_path, 
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": input_text
                    **gen_args
                }), 
                accept='application/json', 
                contentType='application/json'
            )
            response_body = json.loads(response.get('body').read())
            output_text = response_body["content"][0]["text"]
            return output_text

BedrockModel.regiser("bedrock")

# class BedrockModel(Model):
#     def __init__(self, model_path, kwargs):
#         super().__init__(model_path, kwargs)
#         try:
#             import boto3

#             self.bedrock = boto3.client(
#                 "bedrock-runtime",
#                 aws_access_key_id=adhoc.get(kwargs, "aws_access_key_id"],
#                 aws_secret_access_key=adhoc.get(kwargs, "aws_secret_access_key"],
#                 region_name=adhoc.get(kwargs, "region_name|=ap-northeast-1"],
#             )
#         except ModuleNotFoundError as e:
#             raise e
#         default_args = {
#             "max_tokens_to_sample": adhoc.get(kwargs, "max_tokens|max_length|=512"],
#             "temperature": adhoc.get(kwargs, "temperature|=0.2"],
#             "top_p": adhoc.get(kwargs, "top_p|=0.95"],
#         }
#         self.generate_args = default_args

#     def check_and_append_claude_format(self, prompt: str) -> str:
#         ## FIXME: 改行の位置はここでいいのか？
#         human_str = "\n\nHuman:"
#         assistant_str = "\n\nAssistant:"

#         if human_str not in prompt:
#             prompt = human_str + prompt

#         if assistant_str not in prompt:
#             prompt += assistant_str

#         return prompt

#     def generate_text(self, prompt: str) -> str:
#         prompt = self.check_and_append_claude_format(prompt)
#         body = json.dumps(
#             {
#                 "prompt": prompt,
#                 "anthropic_version": "bedrock-2023-05-31",
#                 **self.generate_args,
#             }
#         )
#         response = self.bedrock.invoke_model(body=body, modelId=self.model_path)
#         response_body = json.loads(response.get("body").read())
#         return response_body.get("completion")
