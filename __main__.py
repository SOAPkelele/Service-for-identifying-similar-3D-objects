import os

import torch
from aiogram import Dispatcher, Bot, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.utils import executor

from bot_app import handlers
from bot_app import middlewares
from bot_app import utils, config
from bot_app.config import BEST_MODEL
from nets.MeshNet import MeshNet
from utils.build_tree import check_tree_or_build

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


async def on_startup(dispatcher: Dispatcher):
    # init model
    model = MeshNet(require_fea=True)
    model.cuda()
    model = torch.nn.DataParallel(model)

    # load weights
    model.load_state_dict(
        torch.load(BEST_MODEL))

    # switch to evaluation mode
    model.eval()

    # check index, build if needed
    annoy_index = check_tree_or_build(vector_size=256)

    # setup bot stuff
    handlers.setup(dp)
    middlewares.setup(dp, model=model, index=annoy_index)
    await utils.setup_default_commands(dispatcher)
    await utils.notify_admins(config.SUPERUSER_IDS)


if __name__ == '__main__':
    utils.setup_logger("INFO", ["aiogram.bot.api"])

    bot = Bot(token=config.BOT_TOKEN, parse_mode=types.ParseMode.HTML)
    storage = MemoryStorage()
    dp = Dispatcher(bot=bot, storage=storage)

    executor.start_polling(
        dp, on_startup=on_startup, skip_updates=config.SKIP_UPDATES
    )
