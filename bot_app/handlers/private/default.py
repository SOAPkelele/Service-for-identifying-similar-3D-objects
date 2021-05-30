from aiogram import types


async def command_start_handler(message: types.Message):
    await message.answer(f"Здравствуйте, <b>{message.from_user.full_name}</b>!\n"
                         f"Для сравнения моделей используйте - /compare\n"
                         f"Для поиска похожих моделей в базе - /search")
