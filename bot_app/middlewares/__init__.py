import logging

from aiogram.dispatcher import Dispatcher

from .files_media import FilesMiddleware
from .objects import ObjectsMiddleware


def setup(dp: Dispatcher, model, index):
    dp.middleware.setup(FilesMiddleware())
    dp.middleware.setup(ObjectsMiddleware(model, index))
    logging.info('Middlewares are successfully configured')
