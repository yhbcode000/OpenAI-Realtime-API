from contextlib import asynccontextmanager
from itertools import count
import asyncio
import inspect
import time

from .shared import *
from .interface import Interface, BaseHandler, MODEL
from openai_realtime_api import defaults
from .conversation import Conversation

class Client(BaseHandler):
    def __init__(self, interface: Interface):
        '''
        Don't use this constructor directly. Use `with Client.Context(...) as client:` instead.  
        If you are in a hurry, it's also ok to `Client.Open(...)`.  
        '''
        self.interface = interface

        self.lock = asyncio.Lock()

        # A shallow copy of all server events we received.  
        self.server_event_log: tp.Dict[EventID, tp.Dict[str, tp.Any]] = {}

        self.server_sessionConfig: SessionConfig = SessionConfig(
            ResponseConfig(
                OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, 
            ), 
            OMIT, OMIT, OMIT, 
        )
        self.items: tp.Dict[ItemID, ConversationItem] = {}
        self.server_conversation = Conversation()
        self.server_responses: tp.Dict[ResponseID, Response] = {}
        
        self.eventIdCount = count()
    
    @classmethod
    @asynccontextmanager
    async def Context(
        cls, api_key: str, sessionConfig: SessionConfig = defaults.SESSION_CONFIG, 
        model: str = MODEL, 
    ):
        client = cls.__new__(cls)
        try:
            async with Interface.Context(api_key, client, model) as interface:
                client.__init__(interface)
                await interface.sessionUpdate(sessionConfig, client.nextEventId())
                yield client
        finally:
            client.cleanup()
    
    @classmethod
    async def Open(
        cls, api_key: str, sessionConfig: SessionConfig = defaults.SESSION_CONFIG, 
        model: str = MODEL, 
    ):
        warnings.warn('Why not use Client.Context() instead?')
        client = cls.__new__(cls)
        interface = await Interface.Open(api_key, client, model)
        client.__init__(interface)
        await interface.sessionUpdate(sessionConfig, client.nextEventId())
        return client
    
    async def close(self):
        await self.interface.close()
        self.cleanup()
    
    def cleanup(self):
        pass    # no cleanup needed yet

    def nextEventId(self) -> EventID:
        return f'ibc_{next(self.eventIdCount) : 04d}'
        # [i]nitiated [b]y [c]lient
    
    @staticmethod
    def log(f: tp.Callable):
        formal_params = inspect.signature(f).parameters.keys()
        def wrapper(self: __class__, *args, **kw):
            result = f(self, *args, **kw)
            for k, v in zip(formal_params, args):
                assert k not in kw
                kw[k] = v
            event_id = kw['event_id']
            assert event_id not in self.server_event_log
            self.server_event_log[event_id] = kw
            return result
        return wrapper
    
    @log
    def onError(
        self, event_id: EventID, 
        error: OpenAIError, do_warn: bool = True, 
    ):
        if do_warn:
            error.warn()
    
    def updateServerSessionConfig(self, sessionConfig: SessionConfig):
        if sessionConfig != self.server_sessionConfig:
            new = deepWithoutOmits(sessionConfig            .asPrimitive())
            old = deepWithoutOmits(self.server_sessionConfig.asPrimitive())
            updated = deepUpdate(old, new)
            assert isinstance(updated, dict)
            with MustDrain(updated) as (u, mutate):
                self.server_sessionConfig, r = SessionConfig.fromPrimitive(u)
                mutate(r)
    
    @log
    def onSessionCreated(
        self, event_id: EventID, 
        sessionConfig: SessionConfig,
        session_id: str, model: str, 
    ):
        self.updateServerSessionConfig(sessionConfig)

    @log
    def onSessionUpdated(
        self, event_id: EventID, 
        sessionConfig: SessionConfig,
        session_id: str, model: str, 
    ):
        self.updateServerSessionConfig(sessionConfig)
    
    @log
    def onConversationCreated(
        self, event_id: EventID, 
        conversation_id: str, 
    ):
        pass
    
    @log
    def onConversationItemCreated(
        self, event_id: EventID, 
        previous_item_id: ItemID, item: ConversationItem, 
    ):
        assert item.id_ not in self.items
        self.items[item.id_] = item
        self.server_conversation.insertAfter(item.id_, previous_item_id)
    
    @log
    def onConversationItemInputAudioTranscriptionCompleted(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, transcript: str, 
    ):
        old_part = self.items[item_id].content[content_index]
        assert old_part.transcript is None
        new_part = ContentPart(
            old_part.type_, 
            old_part.text, 
            old_part.audio, 
            transcript, 
        )
        self.items[item_id] = self.items[item_id].withUpdatedContentPart(
            content_index, new_part, 
        )
    
    @log
    def onConversationItemInputAudioTranscriptionFailed(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, error: OpenAIError, 
    ):
        super().onConversationItemInputAudioTranscriptionFailed(
            event_id, item_id, content_index, error,
        )
    
    @log
    def onConversationItemTruncated(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, audio_end_ms: int, 
    ):
        self.server_conversation.cells[item_id].audio_truncate = (
            content_index, audio_end_ms, 
        )

    @log
    def onConversationItemDeleted(
        self, event_id: EventID, 
        item_id: ItemID, 
    ):
        self.server_conversation.pop(item_id)
    
    @log
    def onInputAudioBufferCommitted(
        self, event_id: EventID, 
        previous_item_id: ItemID, item_id: ItemID,
    ):
        self.server_conversation.insertAfter(item_id, previous_item_id)

    @log
    def onInputAudioBufferCleared(
        self, event_id: EventID, 
    ):
        pass
    
    @log
    def onInputAudioBufferSpeechStarted(
        self, event_id: EventID, 
        audio_start_ms: int, item_id: ItemID,
    ):
        pass
    
    @log
    def onInputAudioBufferSpeechStopped(
        self, event_id: EventID, 
        audio_end_ms: int, item_id: ItemID,
    ):
        pass
    
    @log
    def onResponseCreated(
        self, event_id: EventID, 
        response_id: ResponseID, 
    ):
        self.server_responses[response_id] = Response(
            response_id, Status.IN_PROGRESS, None, (), None, 
        )
    
    @log
    def onResponseDone(
        self, event_id: EventID, 
        response: Response, items: tp.Tuple[ConversationItem, ...], 
    ):
        self.server_responses[response.id_] = response
        for theirs in items:
            mine = self.items[theirs.id_]
            pM = deepWithout((None, OMIT, NOT_HERE), mine  .asPrimitive())
            pT = deepWithout((None, OMIT, NOT_HERE), theirs.asPrimitive())
            assert deepUpdate(pM, pT) == pM

    @log
    def onResponseOutputItemAdded(
        self, event_id: EventID, 
        response_id: ResponseID, output_index: int, 
        item: ConversationItem, 
    ):
        old_response = self.server_responses[response_id]
        assert len(old_response.output) == output_index
        self.server_responses[response_id] = Response(
            old_response.id_,
            old_response.status,
            old_response.status_details,
            old_response.output + (item.id_, ),
            old_response.usage,
        )
        assert item.id_ not in self.items
        self.items[item.id_] = item

    @log
    def onResponseOutputItemDone(
        self, event_id: EventID, 
        response_id: ResponseID, output_index: int, 
        item: ConversationItem, 
    ):
        assert self.server_responses[response_id].output[output_index] == item.id_
        self.items[item.id_] = item

    def updateContentPart(
        self, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        assert self.server_responses[response_id].output[output_index] == item_id
        self.items[item_id] = self.items[
            item_id
        ].withUpdatedContentPart(content_index, part)

    @log
    def onResponseContentPartAdded(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        self.updateContentPart(
            response_id, item_id, output_index, content_index, part,
        )
    
    @log
    def onResponseContentPartDone(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        self.updateContentPart(
            response_id, item_id, output_index, content_index, part,
        )
    
    @log
    def onResponseTextDelta(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseTextDone(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        text: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseAudioTranscriptDelta(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseAudioTranscriptDone(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        transcript: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseAudioDelta(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        delta: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseAudioDone(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseFunctionCallArgumentsDelta(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        call_id: str, delta: str, 
    ):
        '''
        Ignore this. 
        You'd have to have 0 chill to optimize your function to take streaming arguments.
        '''
        pass

    @log
    def onResponseFunctionCallArgumentsDone(
        self, event_id: EventID, 
        response_id: ResponseID, item_id: ItemID,
        output_index: int, content_index: int, 
        call_id: str, arguments: str, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onRateLimitsUpdated(
        self, event_id: EventID, 
        rateLimits: tp.Tuple[RateLimit, ...], 
    ):
        '''
        Override this. 
        '''
        pass

def deepUpdate(
    old: dict | tuple | list, 
    new: dict | tuple | list, 
/):
    '''
    No circular safegaurd.  
    '''
    if (
        isinstance(old, dict) and 
        isinstance(new, dict)
    ):
        d = {**old}
        for k, v in new.items():
            if k in d and (
                isinstance(v, dict) or 
                isinstance(v, tuple) or 
                isinstance(v, list)
            ):
                d[k] = deepUpdate(d[k], v)
            else:
                d[k] = v
        return d
    if (
        isinstance(old, list) and 
        isinstance(new, list)
    ):
        l = []
        for i, v in enumerate(new):
            if i < len(old):
                if (
                    isinstance(v, dict) or 
                    isinstance(v, tuple) or 
                    isinstance(v, list)
                ):
                    l.append(deepUpdate(old[i], v))
                else:
                    l.append(v)
            else:
                l.append(v)
        return l
    if (
        isinstance(old, tuple) and 
        isinstance(new, tuple)
    ):
        return tuple(deepUpdate(list(old), list(new)))
    raise TypeError(f'{old = }, {new = }')
