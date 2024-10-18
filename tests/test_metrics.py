import kogitune
import os,tempfile


def test_metrics_editsim():
    m = kogitune.load('metric', "editsim")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result

def test_metrics_jaccard():
    m = kogitune.load('metric', "jaccard")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result

def test_metrics_sacrebleu():
    m = kogitune.load('metric', "sacrebleu")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result

def test_metrics_bleu4():
    m = kogitune.load('metric', "bleu-4")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result


def test_metrics_rouge_l():
    m = kogitune.load('metric', "rouge-l")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result

def test_metrics_embsim():
    m = kogitune.load('metric', "embsim")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result

def test_metrics_rouge_l():
    m = kogitune.load('metric', "bertscore")
    candidates = [
        ["A B", "A B C D"],
        ["X X", "X X X X"],
        ["X Y", "X Y X Y"],
    ]
    references = [
        "A B C", "X X X", "X Y",
    ]
    result = m.calc(candidates, references)
    assert m.nametag in result


def test_metrics_pass_at_1():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        m = kogitune.load('metric', "pass@1")
        print(m)
        candidates = [
            "def f(n): return 1",
            "def f(n): return 0",
        ]
        testcases = [
            "assert f(0) == 1",
            "assert f(0) == 1",
        ]
        result = m.calc(candidates, testcases)
        # {'pass@1': ('mean', [100.0, 0.0]),
        #  'pass@1_passed': ('sum', [1, 0]),
        #  'pass@1_result': ['passed', 'failed: ']}
        assert result[m.nametag][1][0] == 100.0
        assert result[m.nametag][1][1] == 0.0


def test_metrics_pass_at_2():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        m = kogitune.load('metric', "pass@2")
        print(m)
        candidates = [
            ["def f(n): return 1", "def f(n): return 0"],
            ["def f(n): return 0", "def f(n): return 0"]
        ]
        testcases = [
            "assert f(0) == 1",
            "assert f(0) == 1",
        ]
        result = m.calc(candidates, testcases)
        # {'pass@2': ('mean', [50.0]),
        #  'pass@1_passed': ('sum', [1, 0]),
        #  'pass@1_result': ['passed', 'failed: ']}
    assert m.nametag in result
