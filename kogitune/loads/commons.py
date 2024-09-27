import kogitune.adhocs as adhoc
from .datasets_file import basename, zopen, safe_makedirs, write_config, read_config, join_name


def singlefy(v):
    if isinstance(v, list):
        return None if len(v) == 0 else singlefy(v[0])
    return v


def listfy(v):
    """
    常にリスト化する
    """
    if not isinstance(v, list):
        return [v]
    if v is None:
        return []
    return v

def list_tqdm(list_or_value, desc=None):
    if not isinstance(list_or_value, (list, tuple)):
        list_or_value = [list_or_value]
    if len(list_or_value) == 1:
        return list_or_value
    return adhoc.tqdm(list_or_value, desc=desc)


def report_KeyError(e: KeyError, sample: dict):
    adhoc.print(repr(e), face="🙈")
    adhoc.print(adhoc.dump(sample), face="")
    raise e
