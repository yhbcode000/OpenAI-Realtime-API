import os
import asyncio
import base64

import dotenv
import pyaudio

import openai_realtime_api
from openai_realtime_api.shared import *
from openai_realtime_api import defaults
from openai_realtime_api.tests.my_pyaudio import pyAudio, SmallBuffer, PushToTalk

SR = 24000
PAGE_LEN = 2048
print(f'audio page: {round(PAGE_LEN / SR * 1000)} ms')

class MyClient(openai_realtime_api.Client):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.display = Display()
    
    def logSnapshot(self):
        with open('conversation.md', 'w') as f:
            for i, item_id in enumerate(self.server_conversation):
                item = self.items[item_id]
                print(i, ':', item, file=f)
    
    async def keepLoggingSnapshots(self):
        while True:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break
            self.logSnapshot()

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
    async with MyClient.Context(openai_api_key, SessionConfig(
        ResponseConfig(
            (Modality.AUDIO, Modality.TEXT),
            defaults.INSTRUCTIONS, 
            'alloy', 'pcm16', (), 'auto', 0.8, 'inf', 
        ), 
        'pcm16', 'whisper-1', None, 
    )) as client:
        del openai_api_key

        async def appendAudio(
            data: bytes, # paInt16
            n_samples: int,
        ):
            base64_encoded = base64.b64encode(data).decode()
            await client.interface.inputAudioBufferAppend(
                base64_encoded, client.nextEventId(), 
            )
        
        async def commitAudio():
            await client.interface.inputAudioBufferCommit(client.nextEventId())
        
        with pyAudio() as pa:
            outStream = pa.open(
                rate=SR, channels=1, format=pyaudio.paInt16, 
                output=True, frames_per_buffer=PAGE_LEN, 
                start=False, 
            )
            with SmallBuffer(outStream, PAGE_LEN * 2, SR) as writeAudio:
                inStream = pa.open(
                    rate=SR, channels=1, format=pyaudio.paInt16, 
                    input=True, frames_per_buffer=PAGE_LEN, 
                    start=False, 
                )
                with PushToTalk(
                    inStream, appendAudio, commitAudio, PAGE_LEN, 
                ) as pushToTalk:
                    await asyncio.gather(
                        client.interface.receiveLoop(), 
                        client.keepLoggingSnapshots(), 
                        pushToTalk, 
                    )

if __name__ == '__main__':
    asyncio.run(main())
