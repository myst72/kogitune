import kogitune
import os, tempfile

def test_eval_humaneval():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.eval(
            dataset_list=['openai_humaneval', 'kogi-jwu/jhumaneval'],
            task='code_eval',
            head=4,
            test_run=2,
            model_list=['kkuramitsu/chico-0.03b'],
            save_steps=3, # 巨大モデルに対応 落ちても途中からできるように
            # padding_side='left', # 警告が出たら追加
            max_new_tokens=256,
            metric='pass@1',
            output_path = 'code', # 作業用のサブフォルダを指定できるようになった
        )

def test_eval_mia():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.eval(
            task='mia',
            dataset_list=['openai_humaneval', 'kogi-jwu/jhumaneval'],
            test_run=2, 
            model_list=['kkuramitsu/chico-0.03b'],
            #model_list=['Qwen/Qwen2.5-0.5B'],
            save_steps=3, # 巨大モデルに対応 落ちても途中からできるように
            overwrite=False, # 既存の実験結果に追記する場合
            metric='min10_prob|min10++',
            output_path = 'mia', # 作業用のサブフォルダを指定できるようになった
        )

def test_eval_choice_jmmlu():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.eval(
            task='choice',
            model_list=['kkuramitsu/chico-0.03b'], #, 'Qwen/Qwen2.5-0.5B'],
            test_run=2,
            #dataset='nlp-waseda/JMMLU?name=japanese_history',
            dataset_subset = ['japanese_history', 'miscellaneous'],
            dataset='nlp-waseda/JMMLU',
            trust_remote_code=True,
            output_path='jmmlu',
        )

def test_eval_selfcheck():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.eval(
            task='selfcheck',
            model_list=['kkuramitsu/chico-0.03b'], 
            head=4,
            dataset='kogi-jwu/jhumaneval',
            output_path='kogi',
            max_new_tokens=256,
            temperature=0.9,
            metric='editsim_maxmean|jaccard',
        )

def test_eval_mgsm():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.eval(
            model_list=['kkuramitsu/chico-0.03b'], 
            head=4,
            test_run=2,
            shots=4,
            dataset='juletxara/mgsm?name=ja',
            output_path='mgsm',
            max_new_tokens=8,
            metrics='exact_match',
        )