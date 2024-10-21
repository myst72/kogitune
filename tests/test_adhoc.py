import kogitune
import kogitune.adhocs as adhoc
import pytest

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


def test_adhoc_load_error():
    with pytest.raises(KeyError) as e:
        adhoc.load('dummy', 'dummy')


def test_adhoc_main():
    with adhoc.kwargs_from_main(['cmd', '--a=1', 'b=2', '--c', '3']) as kwargs:
        assert(kwargs['a']==1)
        assert(kwargs['b']==2)
        assert(kwargs['c']==3)


def test_adhoc_safe_import():
    with adhoc.kwargs_from_stacked(auto_import=False) as kwargs:        
        with adhoc.progress_bar(total=10) as pbar:
            for n in range(10):
                pbar.update(1)


def test_adhoc_dummy_tqdm():
    with adhoc.kwargs_from_stacked(use_tqdm=False) as kwargs:
        with adhoc.progress_bar(total=10) as pbar:
            for n in range(10):
                pbar.update(1)



def test_adhoc_format_unit():
    with adhoc.start_timer() as timer:
        for i in range(16):
            print(adhoc.format_unit(10*i, scale=1000))
            print(adhoc.format_unit(10*i, scale=60))
            print(adhoc.format_unit(10*i, scale=1024))
        timer.notice('?')