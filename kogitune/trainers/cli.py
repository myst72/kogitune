from ..loads.commons import adhoc

@adhoc.cli
def scratch_cli(**kwargs):
    from .scratch import reduce_model_using_float16
    with adhoc.kwargs_from_stacked(kwargs) as kwargs:
        model_type = adhoc.get(kwargs, 'model_type|!llama2')
        model = adhoc.load('scratch', model_type, **kwargs)
        save_path = adhoc.get(kwargs, 'save_path|!scratch')
        if save_path:
            use_fp16=adhoc.get(kwargs, 'save_fp16|=False')
            model.save(save_path, use_fp16=use_fp16)


@adhoc.cli
def pretrain_cli(**kwargs):
    from .pretrain import pretrain
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        return pretrain(**kwargs)

@adhoc.cli
def finetune_cli(**kwargs):
    from .sft import finetune
    with adhoc.kwargs_from_stacked(**kwargs) as kwargs:
        return finetune(**kwargs)
