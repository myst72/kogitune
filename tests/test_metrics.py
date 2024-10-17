import kogitune
import os,tempfile

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
        assert result['pass@1'][1][0] == 100.0
        assert result['pass@1'][1][1] == 0.0
