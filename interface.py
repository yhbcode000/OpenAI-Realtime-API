import typing as tp
from dataclasses import dataclass
from contextlib import asynccontextmanager
import json
import warnings
import websockets

from shared import *

MODEL = 'gpt-4o-realtime-preview-2024-10-01'    # The only available model at this time. When Realtime API leaves beta, there will prolly be a way to point to a stable one.

ENDPOINT = 'wss://api.openai.com/v1/realtime'

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
            event_type: str = e.pop('type')
            event_id = e.pop('event_id')
            if event_type == ERROR:
                with MustDrain(e.pop('error')) as (error_primitive, mutateErr):
                    error, r = OpenAIError.fromPrimitive(error_primitive)
                    mutateErr(r)
                self.onServerError(
                    event_id, error, 
                )
            elif event_type in (
                f'{SESSION}.{CREATED}', 
                f'{SESSION}.{UPDATED}', 
            ):
                with MustDrain(e.pop('session')) as (session_primitive, mutateSession):
                    session_id = session_primitive.pop('id')
                    model = session_primitive.pop('model')
                    assert session_primitive.pop('object') == f'{REALTIME}.{SESSION}'
                    sessionConfig, r = SessionConfig.fromPrimitive(session_primitive)
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
            elif event_type == f'{CONVERSATION}.{ITEM}.{CREATED}':
                with MustDrain(e.pop('item')) as (item_primitive, mutateItem):
                    assert item_primitive.pop('object') == f'{REALTIME}.{ITEM}'
                    conversationItem, r = ConversationItem.fromPrimitive(item_primitive)
                    mutateItem(r)
                self.onServerConversationItemCreated(
                    event_id, 
                    e.pop('previous_item_id'), 
                    conversationItem, 
                )
            elif event_type == f'{CONVERSATION}.{ITEM}.{INPUT_AUDIO_TRANSCRIPTION}.{COMPLETED}':
                self.onServerConversationItemInputAudioTranscriptionCompleted(
                    event_id, 
                    e.pop('item_id'), 
                    e.pop('content_index'), 
                    e.pop('transcript'), 
                )
            elif event_type == f'{CONVERSATION}.{ITEM}.{INPUT_AUDIO_TRANSCRIPTION}.{FAILED}':
                with MustDrain(e.pop('error')) as (error_primitive, mutateErr):
                    error, r = OpenAIError.fromPrimitive(error_primitive)
                    mutateErr(r)
                self.onServerConversationItemInputAudioTranscriptionFailed(
                    event_id, 
                    e.pop('item_id'), 
                    e.pop('content_index'), 
                    error, 
                )
            elif event_type == f'{CONVERSATION}.{ITEM}.{TRUNCATED}':
                self.onServerConversationItemTruncated(
                    event_id, 
                    e.pop('item_id'), 
                    e.pop('content_index'), 
                    e.pop('audio_end_ms'), 
                )
            elif event_type == f'{CONVERSATION}.{ITEM}.{DELETED}':
                self.onServerConversationItemDeleted(
                    event_id, 
                    e.pop('item_id'), 
                )
            elif event_type == f'{INPUT_AUDIO_BUFFER}.{COMMITTED}':
                self.onServerInputAudioBufferCommitted(
                    event_id, 
                    e.pop('previous_item_id'), 
                    e.pop('item_id'), 
                )
            elif event_type == f'{INPUT_AUDIO_BUFFER}.{CLEARED}':
                self.onServerInputAudioBufferCleared(
                    event_id, 
                )
            elif event_type == f'{INPUT_AUDIO_BUFFER}.{SPEECH_STARTED}':
                self.onServerInputAudioBufferSpeechStarted(
                    event_id, 
                    e.pop('audio_start_ms'), 
                    e.pop('item_id'), 
                )
            elif event_type == f'{INPUT_AUDIO_BUFFER}.{SPEECH_STOPPED}':
                self.onServerInputAudioBufferSpeechStopped(
                    event_id, 
                    e.pop('audio_end_ms'), 
                    e.pop('item_id'), 
                )
            elif event_type == f'{RESPONSE}.{CREATED}':
                with MustDrain(e.pop('response')) as (response, _):
                    response_id = response.pop('id')
                    assert response.pop('object') == f'{REALTIME}.{RESPONSE}'
                    # In the particular case of response.created, the following fields don't matter.  
                    response.pop('status', None)
                    response.pop('status_details', None)
                    response.pop('output', None)
                    response.pop('usage', None)
                self.onServerResponseCreated(
                    event_id, 
                    response_id, 
                )
            elif event_type == f'{RESPONSE}.{DONE}':
                with MustDrain(e.pop('response')) as (response_primitive, mutateResponse):
                    response, r = Response.fromPrimitive(response_primitive)
                    mutateResponse(r)
                self.onServerResponseDone(
                    event_id, 
                    response,
                )
            elif event_type in (
                f'{RESPONSE}.{OUTPUT_ITEM}.{ADDED}', 
                f'{RESPONSE}.{OUTPUT_ITEM}.{DONE}',
            ):
                with MustDrain(e.pop('item')) as (item_primitive, mutateItem):
                    assert item_primitive.pop('object') == f'{REALTIME}.{ITEM}'
                    conversationItem, r = ConversationItem.fromPrimitive(item_primitive)
                    mutateItem(r)
                response_id = e.pop('response_id')
                output_index = e.pop('output_index')
                if event_type == f'{RESPONSE}.{OUTPUT_ITEM}.{ADDED}':
                    self.onServerResponseOutputItemAdded(
                        event_id, 
                        response_id, output_index, conversationItem, 
                    )
                else:
                    self.onServerResponseOutputItemDone(
                        event_id, 
                        response_id, output_index, conversationItem, 
                    )
            elif event_type in (
                f'{RESPONSE}.{CONTENT_PART}.{ADDED}', 
                f'{RESPONSE}.{CONTENT_PART}.{DONE}',
            ):
                with MustDrain(e.pop('part')) as (part_primitive, mutatePart):
                    assert part_primitive.pop('object') == f'{REALTIME}.{CONTENT_PART}'
                    contentPart, r = ContentPart.fromPrimitive(part_primitive)
                    mutatePart(r)
                response_id = e.pop('response_id')
                item_id = e.pop('item_id')
                output_index = e.pop('output_index')
                content_index = e.pop('content_index')
                if event_type == f'{RESPONSE}.{CONTENT_PART}.{ADDED}':
                    self.onServerResponseContentPartAdded(
                        event_id, 
                        response_id, item_id, output_index, 
                        content_index, contentPart, 
                    )
                else:
                    self.onServerResponseContentPartDone(
                        event_id, 
                        response_id, item_id, output_index, 
                        content_index, contentPart, 
                    )
            elif event_type.startswith(f'{RESPONSE}.'):
                kw = {
                    'event_id': event_id,
                    'response_id':   e.pop('response_id'), 
                    'item_id':       e.pop('item_id'),
                    'output_index':  e.pop('output_index'),
                    'content_index': e.pop('content_index'),
                }
                if event_type == f'{RESPONSE}.{TEXT}.{DELTA}':
                    self.onServerResponseTextDelta(
                        **kw, 
                        delta=e.pop('delta'), 
                    )
                elif event_type == f'{RESPONSE}.{TEXT}.{DONE}':
                    self.onServerResponseTextDone(
                        **kw, 
                        text=e.pop('text'), 
                    )
                elif event_type == f'{RESPONSE}.{AUDIO_TRANSCRIPT}.{DELTA}':
                    self.onServerResponseAudioTranscriptDelta(
                        **kw, 
                        delta=e.pop('delta'), 
                    )
                elif event_type == f'{RESPONSE}.{AUDIO_TRANSCRIPT}.{DONE}':
                    self.onServerResponseAudioTranscriptDone(
                        **kw,
                        transcript=e.pop('transcript'),
                    )
                elif event_type == f'{RESPONSE}.{AUDIO}.{DELTA}':
                    self.onServerResponseAudioDelta(
                        **kw, 
                        delta=e.pop('delta'), 
                    )
                elif event_type == f'{RESPONSE}.{AUDIO}.{DONE}':
                    self.onServerResponseAudioDone(
                        **kw, 
                    )
                elif event_type == f'{RESPONSE}.{FUNCTION_CALL_ARGUMENTS}.{DELTA}':
                    self.onServerResponseFunctionCallArgumentsDelta(
                        **kw, 
                        call_id=e.pop('call_id'), 
                        delta=e.pop('delta'), 
                    )
                elif event_type == f'{RESPONSE}.{FUNCTION_CALL_ARGUMENTS}.{DONE}':
                    self.onServerResponseFunctionCallArgumentsDone(
                        **kw, 
                        call_id=e.pop('call_id'), 
                        arguments=e.pop('arguments'), 
                    )
                else:
                    raise ValueError(event_type)
            elif event_type == f'{RATE_LIMITS}.{UPDATED}':
                rate_limits_primitive = e.pop('rate_limits')
                rateLimits = []
                for rate_limit_primitive in rate_limits_primitive:
                    with MustDrain(rate_limit_primitive) as (rlp, mutateRLP):
                        rateLimit, r = RateLimit.fromPrimitive(rlp)
                        rateLimits.append(rateLimit)
                        mutateRLP(r)
                self.onServerRateLimitsUpdated(
                    event_id, 
                    rateLimits, 
                )
            else:
                raise ValueError(event_type)
    
    def onServerError(
        self, event_id: str, 
        error: OpenAIError,
    ):
        '''
        Override this. 
        '''
        error.warn()
    
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
    
    def onServerConversationItemCreated(
        self, event_id: str, 
        previous_item_id: str, item: ConversationItem, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerConversationItemInputAudioTranscriptionCompleted(   # I blame OpenAI
        self, event_id: str, 
        item_id: str, content_index: int, transcript: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerConversationItemInputAudioTranscriptionFailed(
        self, event_id: str, 
        item_id: str, content_index: int, error: OpenAIError, 
    ):
        '''
        Override this. 
        '''
        # If you overrided `...TranscriptionCompleted()`, you should override this too. Hence the default fatal exception.  
        error.throw()
    
    def onServerConversationItemTruncated(
        self, event_id: str, 
        item_id: str, content_index: int, audio_end_ms: int, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerConversationItemDeleted(
        self, event_id: str, 
        item_id: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerInputAudioBufferCommitted(
        self, event_id: str, 
        previous_item_id: str, item_id: str,
    ):
        '''
        Override this. 
        '''
        pass

    def onServerInputAudioBufferCleared(
        self, event_id: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerInputAudioBufferSpeechStarted(
        self, event_id: str, 
        audio_start_ms: int, item_id: str,
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerInputAudioBufferSpeechStopped(
        self, event_id: str, 
        audio_end_ms: int, item_id: str,
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseCreated(
        self, event_id: str, 
        response_id: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerResponseDone(
        self, event_id: str, 
        response: Response, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseOutputItemAdded(
        self, event_id: str, 
        response_id: str, output_index: int, 
        item: ConversationItem, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseOutputItemDone(
        self, event_id: str, 
        response_id: str, output_index: int, 
        item: ConversationItem, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseContentPartAdded(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerResponseContentPartDone(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        '''
        Override this. 
        '''
        pass
    
    def onServerResponseTextDelta(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseTextDone(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        text: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseAudioTranscriptDelta(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseAudioTranscriptDone(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        transcript: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseAudioDelta(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseAudioDone(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerResponseFunctionCallArgumentsDelta(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        call_id: str, delta: str, 
    ):
        '''
        Ignore this. 
        You'd have to have 0 chill to optimize your function to take streaming arguments.
        '''
        pass

    def onServerResponseFunctionCallArgumentsDone(
        self, event_id: str, 
        response_id: str, item_id: str,
        output_index: int, content_index: int, 
        call_id: str, arguments: str, 
    ):
        '''
        Override this. 
        '''
        pass

    def onServerRateLimitsUpdated(
        self, event_id: str, 
        rateLimits: tp.List[RateLimit], 
    ):
        '''
        Override this. 
        '''
        pass
