import kogitune
import kogitune.adhocs as adhoc

def test_version():
    assert kogitune.__version__ >= "1.0.0"


def test_adhoc_get():
    kwargs = {
        'aaa': 1, 'bbb': 2
    }
    assert adhoc.get(kwargs, 'aaa|bbb') == 1
    assert adhoc.get(kwargs, 'bbb|aaa') == 2
    assert adhoc.get(kwargs, 'ccc|aaa|bbb') == 1
    assert adhoc.get(kwargs, 'ccc|bbb|aaa') == 2
    assert adhoc.get(kwargs, 'ccc|=3') == 3
    assert adhoc.get(kwargs, 'ccc|!3') == 3
    assert adhoc.get(kwargs, 'ccc') == None

def test_adhoc_get_simkey():
    kwargs = {
        'metric': 1,
        'files': 2
    }
    assert adhoc.get(kwargs, 'metric') == 1
    assert adhoc.get(kwargs, 'metrics') == None
    assert adhoc.get(kwargs, 'metrics*') == 1
    assert adhoc.get(kwargs, 'metric*') == 1

def test_adhoc_get_ENV():
    kwargs = { }
    assert adhoc.get(kwargs, 'PATH') != None
    assert adhoc.get(kwargs, 'HOME_HOME') == None
