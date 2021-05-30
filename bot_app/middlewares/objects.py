from aiogram.dispatcher.middlewares import BaseMiddleware


class ObjectsMiddleware(BaseMiddleware):
    def __init__(self, model, index, context=None):
        super(ObjectsMiddleware, self).__init__()

        if context is None:
            context = {}
        self.context = context
        self.model = model
        self.index = index

    def update_data(self, data):
        data.update(
            model=self.model,
            index=self.index
        )
        if self.context:
            data.update(self.context)

    async def trigger(self, action, args):
        if 'error' not in action and action.startswith('pre_process_'):
            self.update_data(args[-1])
            return True
