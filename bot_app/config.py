from pathlib import Path

from environs import Env

env = Env()
env.read_env()

BOT_TOKEN = env.str("BOT_TOKEN")
SKIP_UPDATES = env.bool("SKIP_UPDATES", False)
WORK_PATH: Path = Path(__file__).parent.parent

SUPERUSER_IDS = env.list("SUPERUSER_IDS")

BEST_MODEL = env.str("BEST_MODEL")
EMBEDDINGS_PATH = env.str("EMBEDDINGS_PATH")
TREE_CONFIG_PATH = env.str("TREE_CONFIG_PATH")
ANNOY_INDEX_PATH = env.str("ANNOY_INDEX_PATH")
MAPPING_EMBEDDINGS_PATH = env.str("MAPPING_EMBEDDINGS_PATH")
DATASET_PATH = env.str("DATASET_PATH")
