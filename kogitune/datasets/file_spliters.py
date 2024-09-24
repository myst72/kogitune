import os
import json
import random
import tempfile
from ..loads.commons import *

class FileSpliter(object):
    def __init__(self, output_file: str, config: dict, max_items=1000000):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_file = output_file
        output_path = basename(output_file, split_dir=False)
        self.ext = output_file[len(output_path) :]
        self.config = config
        self.max_items = max_items
        self.random_seed = random.randint(100000, 999999)
        self.tempfiles = []
        self.wfile = None
        self.count = 0

    def __enter__(self):
        self.temp_dir.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.wfile is not None:
            self.wfile.close()
        if exc_type is not None:
            adhoc.notice("途中で中断しました", count=self.count)
        self.rename(self.output_file, self.config)
        self.temp_dir.__exit__(exc_type, exc_val, exc_tb)

    def random_file(self):
        while True:
            r = random.randint(100000, 999999)
            tempfile = os.path.join(self.temp_dir.name, f"_{r}_{self.ext}")
            if not os.path.exists(tempfile):
                return tempfile

    def write(self, line: str):
        if self.count % self.max_items == 0:
            if self.wfile is not None:
                self.wfile.close()
            self.tempfiles.append(self.random_file())
            self.wfile = zopen(self.tempfiles[-1], "wt")
        if not line.endswith("\n"):
            line = line + "\n"
        self.wfile.write(line)
        self.count += 1

    def rename(self, output_file: str, config: dict):
        output_path = basename(output_file, split_dir=False)
        ext = output_file[len(output_path) :]
        total = len(self.tempfiles)
        if total == 1:
            safe_makedirs(output_file)
            os.rename(self.tempfiles[0], output_file)
            if isinstance(config, dict):
                config["num_of_lines"] = self.count
                write_config(output_file, config)
            return
        for index, tempfile in enumerate(self.tempfiles):
            output_file = f"{output_path}_{index:04d}_of_{total:04d}{ext}"
            safe_makedirs(output_file)
            os.rename(tempfile, output_file)
            adhoc.print("splitted", os.path.abspath(output_file), face="🐼")
            if isinstance(config, dict):
                config["subfile_index"] = index
                config["subfile_total"] = total
                if index + 1 == total:
                    config["num_of_lines"] = self.count % self.max_items
                else:
                    config["num_of_lines"] = self.max_items
                write_config(output_file, config)



