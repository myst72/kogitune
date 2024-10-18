import kogitune
import os,tempfile


def test_leaderboard():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        from kogitune.loads import LeaderBoard
        board: LeaderBoard = kogitune.load('from_kwargs', "leaderboard")
        samples = [
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.01},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.1},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.2},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.3},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.4},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.5},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.6},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.7},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.8},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.9},
            {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.99},
        ]
        assert board.pivot_table(samples, 'value', 'mean') == 0.5


# def test_leaderboard_AUC():
#     with tempfile.TemporaryDirectory() as tmp_dir:
#         os.chdir(tmp_dir)
#         from kogitune.loads import LeaderBoard
#         board: LeaderBoard = kogitune.load('from_kwargs', "leaderboard")
#         samples = [
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.01},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.1},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.2},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.3},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.4},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.5},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 0, 'value': 0.6},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.7},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.8},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.9},
#             {'_dataset': 'A', '_task': 'T', '_model': 'M', 'label': 1, 'value': 0.99},
#         ]
#         assert board.pivot_table(samples, 'value', 'AUC:sample["label"]==1') == 0.5
