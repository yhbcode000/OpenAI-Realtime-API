import os
import asyncio

import dotenv

import openai_realtime_api
from openai_realtime_api.shared import *

class MyClient(openai_realtime_api.Client):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.display = Display()

    def onError(
        self, event_id: EventID, 
        error: OpenAIError,
    ):
        super().onError(event_id, error, do_warn=True)
        input('Press Enter to continue...')
        # blocks all coroutines. Good!

    def onConversationItemInputAudioTranscriptionCompleted(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, transcript: str, 
    ):
        super().onConversationItemInputAudioTranscriptionCompleted(
            event_id, item_id, content_index, transcript,
        )
        self.display.print('Transcribed:', transcript)

class Display:
    def __init__(self):
        self.is_streaming = False
        self.buf: tp.List[str] = []
    
    def print(self, *a, **kw):
        if self.is_streaming:
            print()
            self.is_streaming = False
        print(*a, **kw)
    
    def delta(self, x: str, /):
        self.buf.append(x)
        if self.is_streaming:
            print(x, end='', flush=True)
        else:
            print(*self.buf, sep='', end='', flush=True)
            self.is_streaming = True
    
    def finishStream(self):
        self.buf.clear()
        if self.is_streaming:
            print()
            self.is_streaming = False

async def main():
    assert dotenv.load_dotenv('openai_api.env')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    assert openai_api_key is not None
    async with MyClient.Context(openai_api_key) as client:
        del openai_api_key
        ...

if __name__ == '__main__':
    asyncio.run(main())
