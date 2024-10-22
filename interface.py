import typing as tp
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
import warnings
import websockets

from shared import *

MODEL = 'gpt-4o-realtime-preview-2024-10-01'    # The only available model at this time. When Realtime API leaves beta, there will prolly be a way to point to a stable one.

ENDPOINT = 'wss://api.openai.com/v1/realtime'

TAG = 'OpenAI Realtime API'

class Interface:
    '''
    - Inherit this class and override the `onServer*` handler methods.  
      - The handlers are synchronous and should return fast.  
    '''
    def __init__(
        self, ws: websockets.WebSocketClientProtocol, 
    ):
        '''
        Don't use this constructor directly. Use `with Interface.Context(...) as interface:` instead.  
        If you are in a hurry, it's also ok to `Interface.Open(...)`.  
        '''
        self.ws = ws

        self.closed_by_me = False

    @classmethod
    @asynccontextmanager
    async def Context(cls, api_key: str, model: str = MODEL):
        async with cls.__newWebSocket(api_key, model) as ws:
            interface = cls(ws)
            try:
                yield interface
            finally:
                interface.closed_by_me = True
    
    @classmethod
    async def Open(cls, api_key: str, model: str = MODEL):
        warnings.warn('Why not use Interface.Context() instead?')
        ws = await cls.__newWebSocket(api_key, model)
        return cls(ws)
    
    async def close(self):
        self.closed_by_me = True
        await self.ws.close()
    
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
        sessionConfig: SessionConfig, 
        event_id: str | OmitType = OMIT,
    ):
        event = {
            'event_id': event_id,
            'type': f'{SESSION}.{UPDATE}',
            'session': sessionConfig.asPrimitive(), 
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
            'response': OMIT if isinstance(
                responseConfig, OmitType, 
            ) else responseConfig.asPrimitive(),
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
    
    async def receiveLoop(self):
        try:
            async for message in self.ws:
                event: tp.Dict = json.loads(message)
                await self.onServerEvent(event)
        except websockets.ConnectionClosed:
            if not self.closed_by_me:
                raise

    async def onServerEvent(self, event: tp.Dict):
        with MustDrain(event) as (e, mutateE):
            event_type = e.pop('type')
            event_id = e.pop('event_id')
            if event_type == ERROR:
                with MustDrain(e.pop('error')) as (error, _):
                    self.onServerError(
                        event_id, 
                        error.pop('type'), error.pop('code'), 
                        error.pop('message'), error.pop('param'), 
                        error.pop('event_id'),
                    )
            elif event_type in (
                f'{SESSION}.{CREATED}', 
                f'{SESSION}.{UPDATED}', 
            ):
                with MustDrain(e.pop('session')) as (session, mutateSession):
                    session_id = session.pop('id')
                    model = session.pop('model')
                    assert session.pop('object') == f'{REALTIME}.{SESSION}'
                    sessionConfig, r = SessionConfig.fromPrimitive(session)
                    mutateSession(r)
                    if event_type == f'{SESSION}.{CREATED}':
                        self.onServerSessionCreated(
                            event_id, 
                            sessionConfig, 
                            session_id, model, 
                        )
                    else:
                        self.onServerSessionUpdated(
                            event_id, 
                            sessionConfig, 
                            session_id, model, 
                        )
            elif event_type == f'{CONVERSATION}.{CREATED}':
                with MustDrain(e.pop('conversation')) as (conversation, _):
                    assert conversation.pop('object') == f'{REALTIME}.{CONVERSATION}'
                    self.onServerConversationCreated(
                            event_id, conversation.pop('id'), 
                        )
    
    def onServerError(
        self, event_id: str, 
        error_type: str, code: str | None, message: str, 
        param: str | None, caused_by_client_event_id: str, 
    ):
        '''
        Override this. 
        '''
        print(
            '[Error]', TAG, ':', error_type, code, ':', message, 
            param, f'; {caused_by_client_event_id = }', 
        )
    
    def onServerSessionCreated(
        self, event_id: str, 
        sessionConfig: SessionConfig,
        session_id: str, model: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerSessionUpdated(
        self, event_id: str, 
        sessionConfig: SessionConfig,
        session_id: str, model: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerConversationCreated(
        self, event_id: str, 
        conversation_id: str, 
    ):
        '''
        Override this. 
        '''
        pass
