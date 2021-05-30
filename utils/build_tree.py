import json
from pathlib import Path

import numpy as np
from annoy import AnnoyIndex
from loguru import logger

from bot_app.config import ANNOY_INDEX_PATH, TREE_CONFIG_PATH, EMBEDDINGS_PATH, MAPPING_EMBEDDINGS_PATH


def check_tree_or_build(vector_size: int, n_trees: int = 10000):
    annoy_index = AnnoyIndex(vector_size, "angular")

    with open(TREE_CONFIG_PATH) as config_file:
        config = json.load(config_file)

    if config["built"]:
        annoy_index.load(ANNOY_INDEX_PATH)
        logger.info("Index loaded")
        return annoy_index

    mapping = {}

    files = \
        Path(EMBEDDINGS_PATH).glob("*.npz")

    for i, file in enumerate(files):
        mapping[i] = str(file)
        vector = np.loadtxt(str(file))
        annoy_index.add_item(i, vector)

    # n_jobs specifies number of threads to be used
    annoy_index.build(n_trees, n_jobs=-1)

    annoy_index.save(ANNOY_INDEX_PATH)
    with open(MAPPING_EMBEDDINGS_PATH, "w") as fp:
        json.dump(mapping, fp)

    with open(TREE_CONFIG_PATH, "w") as config_file:
        json.dump({"built": True}, config_file)

    logger.info("Index built")
    return annoy_index
