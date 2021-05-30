import logging
from typing import List, Union

from aiogram import Bot


async def notify_admins(admins: Union[List[int], List[str], int, str]):
    bot = Bot.get_current()
    count = 0
    for admin in admins:
        await bot.send_message(text='The bot is running!', chat_id=admin)
        count += 1
    logging.info(f"{count} admins received messages")
