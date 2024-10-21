import kogitune
from kogitune.loads import TextFilter

def test_filter():
    f = kogitune.load('filter', "maxmin:alpha-fraction")
    assert str(f) == "maxmin:alpha-fraction"

def test_filter_maxmin():
    # 適切なフィルターが見つからない場合は、texteval の maxmin
    f = kogitune.load('filter', "maxmin:alpha-fraction")
    f2 = kogitune.load('filter', "alpha-fraction")
    assert str(f) == str(f2)

def test_filter_contains():
    samples = [
        {"text": "Hello, World"},
        {"text": "こんにちは世界"},
    ]
    f = kogitune.load('filter', "contains:ja")
    results = [sample for sample in f(samples)]
    # results = f.filter_list(samples)
    assert len(results) == 1