from .commons import *

# CLI

IPSUM = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."

@adhoc.cli
def model_cli(**kwargs):
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        if "max_new_tokens" not in kwargs:
            kwargs["max_new_tokens"] = adhoc.get(kwargs, "max_new_tokens|!256")
        model_path = adhoc.get("model_path|model|!!")
        model = adhoc.load("model", model_path, **kwargs)
        adhoc.verbose_print(model)
        prompt = adhoc.get(kwargs, "test_prompt|prompt")
        if prompt is None:
            adhoc.print("プロンプトは、test_prompt='Lorem ipsum ...'で変更できるよ")
            prompt = IPSUM
        sample_list = [{"input": prompt}]
        model.eval_gen(sample_list)
        for sample in sample_list:
            input = sample["input"]
            output = sample["output"]
            adhoc.print(f"[INPUT]\n{input}")
            adhoc.print(f"[OUTPUT]\n{output}")


@adhoc.cli
def texteval_cli(**kwargs):
    """
    テキスト評価

    - files（必須）: JSONLファイル
    - texteval（必須）: 評価関数 例. "alpha-fraction"
    - input_key='text': 評価の対象となるキー
    - transform, columns: JSONを変形したいとき
    - head: 先頭だけをテストいたいとき
    - output_file: 出力先を指定したいとき
    - overwrite=False: 上書き

    例:
    ```python
    from kogitune.cli import texteval_cli

    texteval_cli(
        texteval='alpha-fraction',
        files=['jhumaneval.jsonl'],
        input_key='prompt',
        head=5,
    )
    ```
    """
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        files = adhoc.get_list(kwargs, "files|!!")
        textevals = adhoc.get_list(kwargs, "texteval|!!")
        transform = adhoc.load("from_kwargs", "transform", **kwargs)
        format_key = adhoc.get(kwargs, "text_key|input_key|format|=text")

        for filepath in files:
            record = adhoc.load("record", filepath)
            head = record.rename_save_path(**kwargs)
            for i, texteval in enumerate(textevals, start=1):
                texteval = adhoc.load("texteval", texteval, **kwargs)
                record_key = texteval.record_key()
                record_key = adhoc.get(kwargs, f"output_key|record_key|={record_key}")
                if i > 1:
                    record_key = f"{record_key}.{i}"
                try:
                    verbose = VerboseCounter(head, **kwargs)
                    for sample in record.samples(0, head):
                        sample = transform.transform(sample)
                        text = adhoc.get_formatted_text(sample, format_key)
                        sample[record_key] = texteval(text)
                        verbose.print_sample(sample)
                except KeyError as e:
                    report_KeyError(e, sample)
            record.save()
