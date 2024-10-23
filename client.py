from contextlib import asynccontextmanager
from itertools import count
import asyncio
import inspect

from .shared import *
from .interface import Interface, BaseHandler, MODEL
from openai_realtime_api import defaults

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
        self.client_events_unacknowledged: tp.Dict[EventID, tp.Dict[str, tp.Any]] = {}

        self.server_sessionConfig: SessionConfig = SessionConfig(
            ResponseConfig(
                OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, OMIT, 
            ), 
            OMIT, OMIT, OMIT, 
        )
        self.server_conversation: tp.List[...] = []
        
        self.eventIdCount = count()
    
    @classmethod
    @asynccontextmanager
    async def Context(
        cls, api_key: str, sessionConfig: SessionConfig = defaults.SESSION_CONFIG, 
        model: str = MODEL, 
    ):
        client = cls.__new__(cls)
        async with Interface.Context(api_key, client, model) as interface:
            client.__init__(interface)
            await interface.sessionUpdate(sessionConfig, client.nextEventId())
            yield client
    
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
            # >>> critical section. No need to lock because we are not `await`ing.
            assert event_id not in self.server_event_log
            self.server_event_log[event_id] = kw
            self.client_events_unacknowledged.pop(event_id, None)
            # <<<
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
            new = deepWithoutOmits(sessionConfig.asPrimitive())
            old = deepWithoutOmits(self.server_sessionConfig.asPrimitive())
            with MustDrain(deepUpdate(old, new)) as (updated, mutate):
                self.server_sessionConfig, r = SessionConfig.fromPrimitive(updated)
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
        '''
        Override this. 
        '''
        pass
    
    @log
    def onConversationItemInputAudioTranscriptionCompleted(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, transcript: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onConversationItemInputAudioTranscriptionFailed(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, error: OpenAIError, 
    ):
        '''
        Override this. 
        '''
        # If you overrided `...TranscriptionCompleted()`, you should override this too. Hence the default fatal exception.  
        error.throw()
    
    @log
    def onConversationItemTruncated(
        self, event_id: EventID, 
        item_id: ItemID, content_index: int, audio_end_ms: int, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onConversationItemDeleted(
        self, event_id: EventID, 
        item_id: ItemID, 
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onInputAudioBufferCommitted(
        self, event_id: EventID, 
        previous_item_id: ItemID, item_id: ItemID,
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onInputAudioBufferCleared(
        self, event_id: EventID, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onInputAudioBufferSpeechStarted(
        self, event_id: EventID, 
        audio_start_ms: int, item_id: ItemID,
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onInputAudioBufferSpeechStopped(
        self, event_id: EventID, 
        audio_end_ms: int, item_id: ItemID,
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseCreated(
        self, event_id: EventID, 
        response_id: str, 
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onResponseDone(
        self, event_id: EventID, 
        response: Response, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseOutputItemAdded(
        self, event_id: EventID, 
        response_id: str, output_index: int, 
        item: ConversationItem, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseOutputItemDone(
        self, event_id: EventID, 
        response_id: str, output_index: int, 
        item: ConversationItem, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseContentPartAdded(
        self, event_id: EventID, 
        response_id: str, item_id: ItemID,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onResponseContentPartDone(
        self, event_id: EventID, 
        response_id: str, item_id: ItemID,
        output_index: int, content_index: int, 
        part: ContentPart, 
    ):
        '''
        Override this. 
        '''
        pass
    
    @log
    def onResponseTextDelta(
        self, event_id: EventID, 
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
        output_index: int, content_index: int, 
    ):
        '''
        Override this. 
        '''
        pass

    @log
    def onResponseFunctionCallArgumentsDelta(
        self, event_id: EventID, 
        response_id: str, item_id: ItemID,
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
        response_id: str, item_id: ItemID,
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
        rateLimits: tp.List[RateLimit], 
    ):
        '''
        Override this. 
        '''
        pass

def deepUpdate(old: dict, new: dict, /):
    '''
    No circular dict safegaurd.  
    '''
    c = {**old}
    for k, v in new.items():
        if (
            isinstance(v, dict) and 
            k in c and 
            isinstance(c[k], dict)
        ):
            c[k] = deepUpdate(c[k], v)
        else:
            c[k] = v
    return {k: v for k, v in old.items()}
