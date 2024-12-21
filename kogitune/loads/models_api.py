###
# OpenAI
import json

from .commons import *
from .models_ import Model

@adhoc.reg('openai')
class OpenAIModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = adhoc.get(kwargs, "_subpath|model_path|model")
        openai = adhoc.safe_import('openai')
        api_key = adhoc.get(kwargs, "api_key|OPENAI_API_KEY|!!")
        try:
            self.client = openai.OpenAI(api_key=api_key)
        except BaseException as e:
            adhoc.print('環境変数 OPENAI_API_KEY を設定してね', repr(e))
            adhoc.exit(throw=e)

    def supported_gen_args(self) -> List[str]:
        return [
            ## https://platform.openai.com/docs/api-reference/chat/create
            "_max_tokens|max_tokens|max_new_tokens|=256",
            "_n|n|num_return_sequences",
            "temperature",
            "top_p",
            "frequency_penalty",
            # "logit_bias",
            # "logprobs",
            # "top_logprobs",
            # "presence_penalty",
            # "response_format",
            # "seed",
            # "service_tier",
            "stop",
            # "stream",
        ]
    
    def unwrap(self):
        return self.client

    def generate_s(self, input_text: str, /, **gen_args):
        gen_args = self.filter_gen_args(**gen_args)
        if isinstance(input_text, str):
            input_text = self.get_default_messages(input_text)
        response = self.client.chat.completions.create(
            model=self.model_path,
            messages=input_text,
            **gen_args,
        )
        responses = [choice.message.content for choice in response.choices]
        return singlefy_if_single(responses)



@adhoc.reg('anthropic')
class BedrockModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = adhoc.get(kwargs, "_subpath|model_path|model")
        boto3 = adhoc.safe_import('boto3')
        try:
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=adhoc.get(kwargs, "_region_name|region_name|AWS_REGION"),
                aws_secret_access_key=adhoc.get(kwargs, "_aws_secret_access_key|aws_secret_access_key|secret_key|AWS_SECRET_KEY"),
                aws_access_key_id=adhoc.get(kwargs, "_aws_access_key_id|aws_access_key_id|access_key|AWS_ACCESS_KEY"),
            )
        except BaseException as e:
            adhoc.print('環境変数 AWSの認証情報(AWS_REGION, AWS_SECRET_KEY, AWS_ACCESS_KEY)を設定してね', repr(e))
            adhoc.exit(throw=e)

    def unwrap(self):
        return self.client

    def supported_gen_args(self) -> List[str]:
        return [
            ## https://docs.anthropic.com/en/api/messages
            "_max_tokens|max_tokens|max_new_tokens|=256",
            # "_n|n|num_return_sequences", # サポートされていない
            "_temperature|temperature|=1.0",
            # "top_p",
            # "top_k",
            # "stop_sequences",
        ]
    
    def generate_s(self, input_text: Union[List, str], /, **kwargs):
        gen_args = self.filter_gen_args(**kwargs)
        if isinstance(input_text, str):
            input_text = self.get_default_messages(input_text)
        modelId = self.model_path
        body = json.dumps({
            "max_tokens": gen_args.get("max_tokens", 256),
            "temperature": gen_args.get("temperature", 1.0),
            "messages": input_text,
            "anthropic_version": "bedrock-2023-05-31",
        })
        response = self.client.invoke_model(modelId=modelId, body=body, accept='application/json', contentType='application/json')
        response_body = json.loads(response.get("body").read())
        return response_body.get("content")[0]["text"]
            

    
    


@adhoc.reg('google')
class GoogleModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_path = adhoc.get(kwargs, "_subpath|model_path|model")
        self.genai = adhoc.safe_import('google.generativeai', 'google-generativeai') #引数1:import, 引数2:pip
        api_key = adhoc.get(kwargs, "api_key|GOOGLE_API_KEY|!!")
        try:
            self.client = self.genai.configure(api_key=api_key)
        except BaseException as e:
            adhoc.print('環境変数 GOOGLE_API_KEY を設定してね', repr(e))
            adhoc.exit(throw=e)

    def unwrap(self):
        return self.client
    
    def supported_gen_args(self) -> List[str]: # defaultの値はモデルによって異なる
        return [
            "_candidate_count|candidate_count|n|num_return_sequences|=1", 
            "_max_output_tokens|max_output_tokens|max_tokens|max_new_tokens|=256",
            "temperature|=0.0",
            # "_topK|",
            # "_topP|=0.95",
            # "_stop_sequences|",
        ]
    
    def generate_s(self, input_text: Union[List, str], /, **kwargs):
        gen_args = self.filter_gen_args(**kwargs)
        model = self.genai.GenerativeModel(self.model_path)

        if isinstance(input_text, str): #input_textがstr型の場合->textversion
            input_text = self.get_default_messages(input_text)
        print("input_text", input_text)
        print("type(input_text)", type(input_text))
        response = model.generate_content(
            input_text[0]['content'],
            generation_config = self.genai.GenerationConfig(**gen_args)
            )
        return response.text
    
    
