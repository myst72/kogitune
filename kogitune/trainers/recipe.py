from typing import Union, Any, List

import random

from torch.utils.data import Dataset

from .commons import adhoc
import kogitune.datasets.chunks

# from .gpus import *
# from .collators import TextBlockCollator, NumpyCollator, TensorCollator
# from .callbacks import TimeoutStoppingCallback

# recipe
def parse_recipe(recipe: Union[list, str]):
    if isinstance(recipe, str):
        if recipe.endswith(".txt"):
            with open(recipe) as f:
                recipe = [
                    url.strip()
                    for url in f.readlines()
                    if url.strip() != "" and not url.startswith("#")
                ]
        else:
            recipe = recipe.split("|")
    items = []
    for item in recipe:
        if isinstance(item, str):
            item = item.partition('#')[0]
            columns = item.split()
            if len(columns) == 1:
                items.append(dict(path=columns[0]))
            elif len(columns) > 1:
                items.append(dict(path=columns[0], ratio=float(columns[1])))
        elif isinstance(item, dict):
            items.append(item)
    print('@', adhoc.dump(items))
    return items

def prepare_recipe(recipe: Union[str, list], block_size, batch_size):
    recipe = parse_recipe(recipe)
    given_ratio = 0.0
    given_blocks = 0
    ungiven_blocks = 0
    for item in recipe:
        chunk_dataset = adhoc.load("chunk", item["path"], block_size=block_size)
        item["dataset"] = chunk_dataset
        item["num_blocks"] = len(chunk_dataset)
        item["num_tokens"] = len(chunk_dataset) * chunk_dataset.block_size
        if "ratio" in item:
            given_ratio += float(item["ratio"])
            given_blocks += len(chunk_dataset)
        else:
            ungiven_blocks += len(chunk_dataset)
    if ungiven_blocks > 0:
        if given_ratio < 0.95:
            ungiven_ratio = 1.0 - given_ratio
        else:
            ungiven_ratio = ungiven_blocks / (given_blocks + ungiven_blocks)
            ratio = given_blocks / (given_blocks + ungiven_blocks)
            scale_factor = ratio / given_ratio
    else:
        scale_factor = 1.0 / given_ratio
    batch_size_count = 0
    maxitem = recipe[0]
    for item in recipe:
        if "ratio" in item:
            ratio = item["ratio"] * scale_factor
        else:
            ratio = (item["num_blocks"] / ungiven_blocks) * ungiven_ratio
        item["trained_tokens"] = item["num_tokens"] * ratio
        item["batch_size"] = min(round(ratio * batch_size), 1) # 必ず一つは入れる
        if item["batch_size"] > maxitem["batch_size"]:
            maxitem = item
        item["maxstep"] = item["num_blocks"] // item["batch_size"]
        batch_size_count += item["batch_size"]

    if batch_size_count != batch_size:
        maxitem["batch_size"] += batch_size - batch_size_count
        maxitem["maxstep"] = maxitem["num_blocks"] // maxitem["batch_size"]
    print("@", adhoc.dump(recipe))
    return recipe


class DatasetRecipe(Dataset):
    def __init__(self, recipe: Union[list, str], batch_size=1024, block_size=512):
        self.recipe = prepare_recipe(recipe, batch_size=batch_size, block_size=block_size)
        self.batch_size = batch_size
        datasets = [item["dataset"] for item in self.recipe]
        batch_sizes = [item["batch_size"] for item in self.recipe]
        assert sum(batch_sizes) == batch_size
        epoch_step = min([item["maxstep"] for item in self.recipe])
        self.count = 0
        self.total = epoch_step * batch_size
        adhoc.print(f"データセットの混成比率: {batch_sizes} total={self.total}")
        indexers = []
        for dataset, mix in zip(datasets, batch_sizes):
            indexers.extend([dataset] * mix)
        self.indexers = indexers

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        if self.count % self.batch_size == 0:
            random.shuffle(self.indexers)
        self.count += 1
        return self.indexers[self.count % self.batch_size][index]

    def skip(self, count):
        for c in range(count):
            self.indexers[c % self.batch_size].skip()
        self.count = count

    def get_trained_count(self):
        tokens = {}
        for ds in self.indexers:
            key = str(ds)
            if key not in tokens:
                tokens[key] = ds.count
        return tokens

# TEAM
# PROJECT
# RUN

# def get_trained_global_step(path: str):
#     state_file = os.path.join(path, "trainer_state.json")
#     if os.path.exists(state_file):
#         try:
#             with open(state_file) as f:
#                 data = json.load(f)
#                 return data["global_step"]
#         except:
#             pass

#     if not os.path.isdir(path):
#         return 0
#     # 指定されたパス内のすべてのファイルとディレクトリのリストを取得
#     dirs = [
#         os.path.join(path, item)
#         for item in os.listdir(path)
#         if os.path.isdir(os.path.join(path, item))
#     ]
#     if len(dirs) == 0:
#         return 0
#     # 最も新しいディレクトリを見つける
#     newest = max(dirs, key=lambda dir: os.path.getmtime(dir))
#     return get_trained_global_step(newest)


# def create_output_path(run_name):
#     for i in range(1, 1000):
#         output_path = f"output_{run_name}_{i}"
#         if not os.path.exists(output_path):
#             return output_path
#     return f"output_{run_name}"


# def check_composer_args(aargs: None):
#     if "resume_from_checkpoint" in aargs and not adhoc.get(kwargs, "overwrite_output_dir|=True"]:
#         resume_from_checkpoint = safe_dir(str(adhoc.get(kwargs, "resume_from_checkpoint"]))
#         if "output_dir" not in aargs and os.path.isdir(resume_from_checkpoint):
#             adhoc.get(kwargs, "output_dir"] = os.path.dirname(resume_from_checkpoint)

#     if "project" not in aargs:
#         adhoc.get(kwargs, "project"] = f"kogitune-sandbox"

#     if "run_name" not in aargs:
#         adhoc.get(kwargs, "run_name"] = f"run{os.getpid()}"

#     if "output_dir" not in aargs:
#         adhoc.get(kwargs, "output_dir"] = create_output_path(adhoc.get(kwargs, "run_name"])
#         adhoc.print(f"出力先:", adhoc.get(kwargs, "output_dir"])
#     return aargs


#         self.train_dataset = None
#         if collator_fn:
#             self.collator_fn = collator_fn
#         else:
#             self.collator_fn = TextBlockCollator(self.max_length, self.aargs)

#     def with_format(self, type):
#         if type == "tensor" or type == "torch":
#             self.collator_fn = TensorCollator(self.max_length, self.aargs)
#         if type == "numpy":
#             self.collator_fn = NumpyCollator(self.max_length, self.aargs)


#     def get_train_dataset(self):
#         if not self.train_dataset:
#             self.train_dataset = MixierDataset(
#                 self.datasets, self.collator_fn, batch_size
#             )

        # resume_path = self.adhoc.get(kwargs, "resume_from_checkpoint"]
        # if resume_path == True:
        #     resume_path = self.adhoc.get(kwargs, "output_dir"]
        # if isinstance(resume_path, str):
        #     resume_step = get_trained_global_step(resume_path)
        #     if resume_step == 0:
        #         adhoc.print(f"チェックポイント {resume_path} が見つかりません")
        #         self.adhoc.get(kwargs, "resume_from_checkpoint"] = False
        #     if resume_step > 0:
        #         adhoc.print(f"チェックポイント step={resume_step} から再開します。")
        #         self.train_dataset.skip(resume_step * batch_size)
        # return self.train_dataset

    # def report(self):
    #     if self.train_dataset:
    #         global_count = self.train_dataset.count
    #         global_step = global_count // 1024
    #         total_tokens = global_count * self.max_length
    #         adhoc.print(
    #             f"ステップ {global_step:,} イテレーション {global_count:,} トークン数 {adhoc.format_unit(total_tokens)} {total_tokens:,}"
    #         )

    # def __enter__(self):
    #     self.aargs.__enter__()
    #     return self

    # def __exit__(self, exc_type, exc_value, traceback):
    #     self.n_items = 0
    #     self.mixed = None
    #     self.report()
    #     if self.cleanup and os.path.isdir(self.cache_dir):
    #         try:
    #             shutil.rmtree(self.cache_dir)
    #             adhoc.print("Cleaned up", self.cache_dir)
    #         except:
    #             pass
    #     self.aargs.__exit__(exc_type, exc_value, traceback)


DatasetComposer = DatasetRecipe
