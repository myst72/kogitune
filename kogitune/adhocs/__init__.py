from .prints import (
    format_unit,
    dump_dict_as_json as dump,
    aargs_print as print,
    open_log_file,
    notice,
    warn,
    saved,
    report_saved_files,
    start_timer,
    describe_counters,
)

from .dicts import (
    edit_distance,
    find_simkey,
    list_keys,
    list_values,
    find_dict_from_keys,
    copy_dict_from_keys,
    move_dict_from_keys,
    get_formatted_text,
    extract_dict_with_keys,
    extract_dict_with_prefix,
    safe_kwargs,
    parse_path,
    encode_path,
    parse_path_args,
    ChainMap,
)

from .main import (
    AdhocArguments,
    AdhocArguments as Arguments,
    parse_main_args,
    load_class,
    instantiate_from_dict,
    launch_subcommand,
    aargs_from,
    get,
    get_list, 
    verbose_print,
    load,
    AdhocLoader,
    LoaderObject,
    cli,
    from_kwargs,
    wordlist,
)

from .inspects import extract_kwargs, check_kwargs, get_parameters, get_version  # OLD

from .adhoc_tqdm import (
    adhoc_progress_bar as progress_bar,
    adhoc_tqdm as tqdm,
)


def safe_import(module: str, pip_install_modules=None):
    import importlib, os

    try:
        module = importlib.import_module(module)
    except ModuleNotFoundError:
        cmd = f"pip3 install {pip_install_modules or module}"
        print(cmd)
        os.system(cmd)
        module = importlib.import_module(module)
    if hasattr(module, '__version__'):
        verbose_print(module.__name__, module.__version__)
    return module


def pip(module: str):
    import os

    print(f"pip3 install {module}")
    os.system(f"pip3 install {module}")


def load_tokenizer(tokenizer=None, **kwargs):
    from ..stores.tokenizers import load_tokenizer

    return load_tokenizer(tokenizer, **kwargs)
