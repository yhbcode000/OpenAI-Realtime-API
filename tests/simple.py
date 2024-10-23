import os
import asyncio

import dotenv

import openai_realtime_api
from openai_realtime_api.shared import *

class MyClient(openai_realtime_api.Client):
    def onError(
        self, event_id: EventID, 
        error: OpenAIError,
    ):
        super().onError(event_id, error, do_warn=True)
        input('Press Enter to continue...')
        # blocks all coroutines. Good!

async def main():
    assert dotenv.load_dotenv('openai_api.env')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    assert openai_api_key is not None
    async with MyClient.Context(openai_api_key) as client:
        del openai_api_key
        ...

if __name__ == '__main__':
    asyncio.run(main())
