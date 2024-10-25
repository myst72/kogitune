import random

from torch.utils.data import Dataset

from ..loads.commons import *
import kogitune.datasets.chunks # load("chunk")

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
    #adhoc.debug_print('@recipe', adhoc.dump(items))
    return items

def prepare_recipe(recipe: Union[str, list], block_size, batch_size):
    recipe = parse_recipe(recipe)
    ratio_sum = 0.0
    given_blocks = 0
    ungiven_blocks = 0
    for item in recipe:
        chunk_dataset = adhoc.load("chunk", item["path"], block_size=block_size)
        item["dataset"] = chunk_dataset
        item["num_blocks"] = len(chunk_dataset)
        item["num_tokens"] = len(chunk_dataset) * chunk_dataset.block_size
        if "ratio" in item:
            ratio_sum += float(item["ratio"])
            given_blocks += len(chunk_dataset)
        else:
            ungiven_blocks += len(chunk_dataset)
    # print('@items', adhoc.dump(recipe))
    # print('given =', given_blocks, 'ungiven =', ungiven_blocks, 'ratio_sum =', ratio_sum)
    if ratio_sum > 1.0:
        if ungiven_blocks > 0:
            ratio_sum += 0.1 # 10%だけ空ける
        ratio_factor = 1 / ratio_sum
        adhoc.verbose_print('比率が100%を超えました', ratio_factor)
    if ungiven_blocks > 0:
        unratio_sum = 1.0 - ratio_sum
    batch_size_count = 0

    maxitem = recipe[0]

    for item in recipe:
        if "ratio" in item:
            ratio = item["ratio"] * ratio_factor
        else:
            ratio = (item["num_blocks"] / ungiven_blocks) * unratio_sum
        # print('@ratio', ratio, round(ratio*batch_size))
        # item["trained_tokens"] = item["num_tokens"] * ratio
        item["batch_size"] = max(round(ratio * batch_size), 1) # 必ず一つは入れる
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


DatasetComposer = DatasetRecipe
