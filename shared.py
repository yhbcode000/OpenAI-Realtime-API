'''
`asPrimitive()` returns json-able python primitives but allowing `OMIT`.  
'''

from __future__ import annotations

import typing as tp
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

TAG = 'OpenAI Realtime API'

EventID = str
ItemID = str
ResponseID = str

def identity(x: tp.Any, /):
    return x

class OmitType: 
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

OMIT = OmitType()
# Use this to exclude a JSON key from your request.
# Meaning: use default / status quo.  

class NotHereType: 
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

NOT_HERE = NotHereType()
# Meaning: The info is conceptually existent but not here.  
# e.g. audio that the remote, to save bandwidth, won't give us.  

def withoutOmits(x: tp.Dict, /):
    return {k: v for k, v in x.items() if v is not OMIT}

def deepWithoutOmits(x: tp.Dict, /):
    '''
    No circular dict safegaurd.  
    '''
    return {k: (
        deepWithoutOmits(v) if isinstance(v, tp.Dict) else v
    ) for k, v in x.items() if v is not OMIT}

UPDATE = 'update'
UPDATED = 'updated'
APPEND = 'append'
COMMIT = 'commit'
COMMITTED = 'committed'
CLEAR = 'clear'
CLEARED = 'cleared'
CREATE = 'create'
CREATED = 'created'
TRUNCATE = 'truncate'
TRUNCATED = 'truncated'
DELETE = 'delete'
DELETED = 'deleted'
CANCEL = 'cancel'
COMPLETED = 'completed'
FAILED = 'failed'
SPEECH_STARTED = 'speech_started'
SPEECH_STOPPED = 'speech_stopped'
DONE = 'done'
ADDED = 'added'
DELTA = 'delta'

REALTIME = 'realtime'
SESSION = 'session'
INPUT_AUDIO_BUFFER = 'input_audio_buffer'
CONVERSATION = 'conversation'
ITEM = 'item'
RESPONSE = 'response'
ERROR = 'error'
INPUT_AUDIO_TRANSCRIPTION = 'input_audio_transcription'
OUTPUT_ITEM = 'output_item'
CONTENT_PART = 'content_part'
TEXT = 'text'
AUDIO_TRANSCRIPT = 'audio_transcript'   # think "output_"audio_transcription
AUDIO = 'audio'
FUNCTION_CALL_ARGUMENTS = 'function_call_arguments'
RATE_LIMITS = 'rate_limits'

@contextmanager
def MustDrain(a: tp.Dict, /):
    '''
    Promises to exhaust the dict by the end of the with clause.  
    '''
    remaining = {**a}

    def mutate(new_dict: tp.Dict, /):
        remaining.clear()
        remaining.update(new_dict)
    
    try:
        yield remaining, mutate
    finally:
        assert not remaining, f'Unconsumed items: {remaining}'

class Modality(Enum):
    TEXT = 'text'
    AUDIO = 'audio'

@dataclass(frozen=True)
class TurnDetectionConfig:
    type_: str
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int

    def asPrimitive(self):
        return {
            'type': self.type_,
            'threshold': self.threshold,
            'prefix_padding_ms': self.prefix_padding_ms,
            'silence_duration_ms': self.silence_duration_ms,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        try:
            instance = __class__(
                remaining.pop('type'),
                remaining.pop('threshold'),
                remaining.pop('prefix_padding_ms'),
                remaining.pop('silence_duration_ms'),
            )
        except KeyError:
            assert a['type'] == 'none', a
            assert remaining.pop('prefix_padding_ms', None) is None
            assert remaining.pop('silence_duration_ms', None) is None
            return None, remaining
        return instance, remaining

def isTurnDetectionServerVad(a: TurnDetectionConfig | None, /):
    if a is None:
        return False
    if a.type_ == "server_vad":
        return True
    elif a.type_ == "none":
        return False
    raise ValueError(a.type_)

class ToolType(Enum):
    CODE_INTERPRETER = 'code_interpreter'
    FILE_SEARCH = 'file_search'
    FUNCTION = 'function'

@dataclass(frozen=True)
class Tool:
    type_: ToolType
    function_: Function | None

    def __post_init__(self):
        assert (
            self.type_ == ToolType.FUNCTION
        ) == self.function_ is not None

    def asPrimitive(self):
        p = {
            'type': str(self.type_),
        }
        if self.function_ is not None:
            return {
                **p, 
                **self.function_.asPrimitive(),
            }
        return p
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        type_ = ToolType(remaining.pop('type'))
        if type_ == ToolType.FUNCTION:
            function_, remaining_now = Function.fromPrimitive(remaining)
            del remaining
            return __class__(
                type_, function_, 
            ), remaining_now
        assert remaining.pop('name',        None) is None
        assert remaining.pop('description', None) is None
        assert remaining.pop('parameters',  None) is None
        return __class__(type_, None), remaining

@dataclass(frozen=True)
class Function:
    name: str
    description: str
    parameters: Parameters

    def asPrimitive(self):
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters.asPrimitive(),
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        with MustDrain(remaining.pop('parameters')) as (
            parameters_primitive, mutate, 
        ):
            parameters, r = Parameters.fromPrimitive(
                parameters_primitive, 
            )
            mutate(r)
        instance = __class__(
            remaining.pop('name'),
            remaining.pop('description'),
            parameters,
        )
        return instance, remaining

@dataclass(frozen=True)
class Parameters:
    type_: str
    properties: tp.Dict[str, Property]
    required: tp.List[str]
    additionalProperties: bool

    def asPrimitive(self):
        return {
            'type': self.type_,
            'properties': {k: v.asPrimitive() for k, v in self.properties.items()},
            'required': self.required,
            'additionalProperties': self.additionalProperties,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        properties = {}
        for k, v in remaining.pop('properties').items():
            with MustDrain(v) as (property_primitive, mutate):
                properties[k], r = Property.fromPrimitive(property_primitive)
                mutate(r)
        instance = __class__(
            remaining.pop('type'),
            properties,
            remaining.pop('required'),
            remaining.pop('additionalProperties'),
        )
        return instance, remaining

@dataclass(frozen=True)
class Property:
    type_: str
    description: str

    def asPrimitive(self):
        return {
            'type': self.type_,
            'description': self.description,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        instance = __class__(
            remaining.pop('type'),
            remaining.pop('description'),
        )
        return instance, remaining

@dataclass(frozen=True)
class ConversationItem:
    id_: ItemID
    type_: ConversationItemType
    status: Status
    role: Role
    content: tp.List[ContentPart]
    call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    output: str | None = None

    def __post_init__(self):
        assert (
            self.type_ == ConversationItemType.FUNCTION_CALL
        ) == (
            self.call_id is not None
        ) == (
            self.name is not None
        ) == (
            self.arguments is not None
        )
        assert (
            self.type_ == ConversationItemType.FUNCTION_CALL_OUTPUT
        ) == self.output is not None
    
    def asPrimitive(self):
        return {
            'id': self.id_,
            'type': str(self.type_),
            'status': str(self.status),
            'role': str(self.role),
            'content': [x.asPrimitive() for x in self.content],
            'call_id': self.call_id,
            'name': self.name,
            'arguments': self.arguments,
            'output': self.output,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        content = []
        for x in remaining.pop('content'):
            with MustDrain(x) as (content_primitive, mutate):
                contentPart, r = ContentPart.fromPrimitive(content_primitive)
                content.append(contentPart)
                mutate(r)
        instance = __class__(
            remaining.pop('id'),
            ConversationItemType(remaining.pop('type')),
            Status              (remaining.pop('status')),
            Role                (remaining.pop('role')),
            content,
            remaining.pop('call_id', None),
            remaining.pop('name', None),
            remaining.pop('arguments', None),
            remaining.pop('output', None),
        )
        return instance, remaining

class ConversationItemType(Enum):
    MESSAGE = 'message'
    FUNCTION_CALL = 'function_call'
    FUNCTION_CALL_OUTPUT = 'function_call_output'

class Status(Enum):
    COMPLETED = 'completed'
    IN_PROGRESS = 'in_progress'
    INCOMPLETE = 'incomplete'

class Role(Enum):
    USER = 'user'
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    TOOL = 'tool'

@dataclass(frozen=True)
class ContentPart:
    type_: ContentPartType
    text: str | None = None
    audio: str | NotHereType | None = None
    transcript: str | None = None

    def __post_init__(self):
        assert (
            self.type_ in (ContentPartType.TEXT, ContentPartType.INPUT_TEXT)
        ) == self.text is not None
        assert (
            self.type_ in (ContentPartType.AUDIO, ContentPartType.INPUT_AUDIO)
        ) == self.audio is not None
        if self.transcript is not None:
            assert self.audio is not None
    
    def asPrimitive(self):
        assert self.audio is not NOT_HERE
        return {
            'type': str(self.type_),
            'text': self.text,
            'audio': self.audio,
            'transcript': self.transcript,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        type_ = ContentPartType(remaining.pop('type'))
        if type_ == ContentPartType.AUDIO:
            audio = remaining.pop('audio', None)
            if audio is None:
                audio = NOT_HERE
        instance = __class__(
            type_,
            remaining.pop('text', None),
            audio,
            remaining.pop('transcript', None),
        )
        return instance, remaining

class ContentPartType(Enum):
    INPUT_TEXT = 'input_text'
    INPUT_AUDIO = 'input_audio'
    TEXT = 'text'
    AUDIO = 'audio'

@dataclass(frozen=True)
class ResponseConfig:
    modalities: tp.List[Modality] | OmitType = OMIT
    instructions: str | OmitType = OMIT
    voice: str | OmitType = OMIT
    output_audio_format: str | OmitType = OMIT
    tools: tp.List[Tool] | OmitType = OMIT
    tool_choice: str | OmitType = OMIT
    temperature: float | OmitType = OMIT
    max_output_tokens: int | tp.Literal['inf'] | OmitType = OMIT

    def asPrimitive(self):
        return {
            'modalities': OMIT if isinstance(
                self.modalities, OmitType, 
            ) else [str(x) for x in self.modalities],
            'instructions': self.instructions,
            'voice': self.voice,
            'output_audio_format': self.output_audio_format,
            'tools': OMIT if isinstance(self.tools, OmitType) else [
                x.asPrimitive() for x in self.tools
            ],
            'tool_choice': self.tool_choice,
            'temperature': self.temperature,
            'max_output_tokens': self.max_output_tokens, 
        }

    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        tools = []
        for x in remaining.pop('tools'):
            with MustDrain(x) as (tool_primitive, mutate):
                tool, r = Tool.fromPrimitive(tool_primitive)
                tools.append(tool)
                mutate(r)
        instance = __class__(
            [Modality(x) for x in remaining.pop('modalities')], 
            remaining.pop('instructions'), 
            remaining.pop('voice'), 
            remaining.pop('output_audio_format'), 
            tools,
            remaining.pop('tool_choice'), 
            remaining.pop('temperature'), 
            remaining.pop('max_output_tokens'), 
        )
        return instance, remaining

@dataclass(frozen=True)
class SessionConfig:
    response: ResponseConfig
    input_audio_format: str | OmitType = OMIT
    input_audio_transcription_model: str | None | OmitType = OMIT
    turn_detection: TurnDetectionConfig | None | OmitType = OMIT

    def asPrimitive(self):
        return {
            **self.response.asPrimitive(),
            'input_audio_format': self.input_audio_format,
            'input_audio_transcription_model': self.input_audio_transcription_model,
            'turn_detection': self.turn_detection,
        }
    
    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        responseConfig, remaining_now = ResponseConfig.fromPrimitive(remaining)
        del remaining
        with MustDrain(remaining_now.pop(
            'input_audio_transcription', 
        )) as (input_audio_transcription, _):
            if input_audio_transcription.pop('enabled'):
                input_audio_transcription_model = input_audio_transcription.pop('model')
            else:
                input_audio_transcription_model = None
                assert input_audio_transcription.pop('model', None) is None
        with MustDrain(remaining_now.pop(
            'turn_detection', 
        )) as (turn_detection_primitive, mutate):
            turn_detection, r = TurnDetectionConfig.fromPrimitive(turn_detection_primitive)
            mutate(r)
        instance = __class__(
            responseConfig,
            remaining_now.pop('input_audio_format'),
            input_audio_transcription_model,
            turn_detection,
        )
        return instance, remaining_now

@dataclass(frozen=True)
class OpenAIError:
    type_: str
    code: str | None
    message: str
    param: str | None
    caused_by_client_event_id: EventID | None

    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        instance = __class__(
            remaining.pop('type'),
            remaining.pop('code', None),
            remaining.pop('message'),
            remaining.pop('param', None),
            remaining.pop('event_id', None),
        )
        return instance, remaining
    
    def warn(self):
        warnings.warn(str(self))
    
    def throw(self):
        raise RuntimeError(str(self))
    
    def __repr__(self):
        return f'''\
[Error] {TAG}: {self.type_} {self.code}: \
{self.message} {self.param}; \
{self.caused_by_client_event_id = }\
'''

@dataclass(frozen=True)
class Response:
    id_: ResponseID
    status: Status
    status_details: tp.Dict | None
    output: tp.List[ConversationItem]
    usage: tp.Dict | None
    # known keys of `usage`, undocumented: 'total_tokens', 'input_tokens', 'output_tokens'

    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        output = []
        for x in remaining.pop('output'):
            with MustDrain(x) as (conversation_item_primitive, mutate):
                conversationItem, r = ConversationItem.fromPrimitive(conversation_item_primitive)
                output.append(conversationItem)
                mutate(r)
        instance = __class__(
            remaining.pop('id'),
            Status(remaining.pop('status')),
            remaining.pop('status_details', None),
            output,
            remaining.pop('usage', None),
        )
        return instance, remaining

@dataclass(frozen=True)
class RateLimit:
    name: str
    limit: int
    remaining: int
    reset_seconds: float

    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        remaining = {**a}
        instance = __class__(
            remaining.pop('name'),
            remaining.pop('limit'),
            remaining.pop('remaining'),
            remaining.pop('reset_seconds'),
        )
        return instance, remaining
