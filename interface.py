import typing as tp
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
import warnings
import websockets

from shared import *

MODEL = 'gpt-4o-realtime-preview-2024-10-01'    # The only available model at this time. When Realtime API leaves beta, there will prolly be a way to point to a stable one.

ENDPOINT = 'wss://api.openai.com/v1/realtime'

@dataclass
class ResponseConfig:
    modalities: tp.List[Modality] | OmitType = OMIT
    instructions: str | OmitType = OMIT
    voice: str | OmitType = OMIT
    output_audio_format: str | OmitType = OMIT
    tools: tp.List[Tool] | OmitType = OMIT
    tool_choice: str | OmitType = OMIT
    temperature: float | OmitType = OMIT
    max_output_tokens: int | InfType | OmitType = OMIT

    def asPrimitive(self):
        return withoutOmits({
            'modalities': [str(x) for x in self.modalities],
            'instructions': self.instructions,
            'voice': self.voice,
            'output_audio_format': self.output_audio_format,
            'tools': OMIT if isinstance(self.tools, OmitType) else [
                x.asPrimitive() for x in self.tools
            ],
            'tool_choice': self.tool_choice,
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens,
        })

class Interface:
    def __init__(
        self, ws: websockets.WebSocketClientProtocol, 
    ):
        '''
        Don't use this constructor directly. Use `with Interface.Context(...) as interface:` instead.  
        If you are in a hurry, it's also ok to `Interface.Open(...)`.  
        '''
        self.ws = ws

    @classmethod
    @asynccontextmanager
    async def Context(cls, api_key: str, model: str = MODEL):
        async with cls.__newWebSocket(api_key, model) as ws:
            yield cls(ws)
    
    @classmethod
    async def Open(cls, api_key: str, model: str = MODEL):
        warnings.warn('Why not use Interface.Context() instead?')
        ws = await cls.__newWebSocket(api_key, model)
        return cls(ws)
    
    @staticmethod
    def __newWebSocket(api_key: str, model: str = MODEL):
        return websockets.connect(
            ENDPOINT + '?model=' + model, 
            extra_headers={
                'Authorization': 'Bearer ' + api_key,
                'OpenAI-Beta': 'realtime=v1',
            },
        )

    async def __omitJsonSend(self, event: tp.Dict):
        await self.ws.send(json.dumps(withoutOmits(event)))

    async def sessionUpdate(
        self, 
        responseConfig: ResponseConfig, 
        input_audio_format: str | OmitType = OMIT,
        input_audio_transcription_model: str | None | OmitType = OMIT,
        turn_detection: TurnDetectionConfig | None | OmitType = OMIT,
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{SESSION}.{UPDATE}',
            'session': {
                **responseConfig.asPrimitive(),
                'input_audio_format': input_audio_format,
                'input_audio_transcription_model': input_audio_transcription_model,
                'turn_detection': turn_detection,
            }, 
        }
        await self.__omitJsonSend(event)
    
    async def inputAudioBufferAppend(
        self, audio: str, event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{INPUT_AUDIO_BUFFER}.{APPEND}',
            'audio': audio,
        }
        await self.__omitJsonSend(event)
    
    async def inputAudioBufferCommit(
        self, event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{INPUT_AUDIO_BUFFER}.{COMMIT}',
        }
        await self.__omitJsonSend(event)
    
    async def inputAudioBufferClear(
        self, event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{INPUT_AUDIO_BUFFER}.{CLEAR}',
        }
        await self.__omitJsonSend(event)
    
    async def conversationItemCreate(
        self, 
        item: ConversationItem, 
        previous_item_id: str | OmitType = OMIT,
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{CONVERSATION}.{ITEM}.{CREATE}',
            'previous_item_id': previous_item_id,
            'item': item.asPrimitive(),
        }
        await self.__omitJsonSend(event)
    
    async def conversationItemTruncate(
        self, 
        item_id: str,
        content_index: int | OmitType = OMIT,
        audio_end_ms: int | OmitType = OMIT,
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{CONVERSATION}.{ITEM}.{TRUNCATE}',
            'item_id': item_id,
            'content_index': content_index,
            'audio_end_ms': audio_end_ms,
        }
        await self.__omitJsonSend(event)
    
    async def conversationItemDelete(
        self, 
        item_id: str,
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{CONVERSATION}.{ITEM}.{DELETE}',
            'item_id': item_id,
        }
        await self.__omitJsonSend(event)
    
    async def responseCreate(
        self, 
        responseConfig: ResponseConfig | OmitType = OMIT, 
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{RESPONSE}.{CREATE}',
            'response': responseConfig.asPrimitive(),
        }
        await self.__omitJsonSend(event)
    
    async def responseCancel(
        self, 
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{RESPONSE}.{CANCEL}',
        }
        await self.__omitJsonSend(event)
