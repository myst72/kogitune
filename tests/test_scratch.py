import kogitune
import os
import tempfile

def test_scratch():
    with tempfile.TemporaryDirectory() as tmp_dir:
        os.chdir(tmp_dir)
        kogitune.cli.scratch()
