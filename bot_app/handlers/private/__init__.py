from aiogram import Dispatcher
from aiogram import filters
from aiogram import types
from aiogram.dispatcher.filters import Command

from .compare import compare_command_handler, get_models_to_compare_handler, GET_MODELS
from .default import command_start_handler
from .search import search_command_handler, get_model_to_search_handler, GET_MODEL


def setup(dp: Dispatcher):
    dp.register_message_handler(command_start_handler, filters.Command('start', ignore_mention=True))

    # handlers to ask and capture files to compare
    dp.register_message_handler(compare_command_handler, Command("compare"), state=None)
    dp.register_message_handler(get_models_to_compare_handler,
                                state=GET_MODELS,
                                is_media_group=True,
                                content_types=types.ContentType.DOCUMENT)

    # handlers to ask and capture file to search
    dp.register_message_handler(search_command_handler, Command("search"), state=None)
    dp.register_message_handler(get_model_to_search_handler,
                                state=GET_MODEL,
                                content_types=types.ContentType.DOCUMENT)
