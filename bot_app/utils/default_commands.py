import logging

from aiogram import types


async def setup_default_commands(dp):
    await dp.bot.set_my_commands(
        [
            types.BotCommand("compare", "Сравнение двух моделей"),
            types.BotCommand("search", "Поиск похожих моделей по базе")
        ]
    )
    logging.info('Standard commands are successfully configured')
