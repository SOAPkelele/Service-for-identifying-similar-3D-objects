import logging

from aiogram import Dispatcher

from . import private


def setup(dp: Dispatcher):
    private.setup(dp)
    logging.info("Handlers are successfully configured")
