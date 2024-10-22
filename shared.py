'''
`asPrimitive()` returns json-able python primitives but allowing `OMIT`.  
'''

from __future__ import annotations

import typing as tp
from enum import Enum
from dataclasses import dataclass

class OmitType: 
    _singleton = None

    def __new__(cls):
        if cls._singleton is None:
            cls._singleton = super().__new__(cls)
        return cls._singleton

OMIT = OmitType()   # Use this to exclude a JSON key from your request.

def withoutOmits(x: tp.Dict, /):
    '''
    Shallow.  
    '''
    return {k: v for k, v in x.items() if v is not OMIT}

UPDATE = 'update'
APPEND = 'append'
COMMIT = 'commit'
CLEAR = 'clear'
CREATE = 'create'
CREATED = 'created'
TRUNCATE = 'truncate'
DELETE = 'delete'
CANCEL = 'cancel'

SESSION = 'session'
INPUT_AUDIO_BUFFER = 'input_audio_buffer'
CONVERSATION = 'conversation'
ITEM = 'item'
RESPONSE = 'response'
ERROR = 'error'

def assertAllConsumed(a: tp.Dict, /):
    assert not a, f'Unconsumed items: {a}'

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
        d = {**a}
        instance = __class__(
            d.pop('type'),
            d.pop('threshold'),
            d.pop('prefix_padding_ms'),
            d.pop('silence_duration_ms'),
        )
        assertAllConsumed(d)
        return instance

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
        d = {**a}
        type_ = ToolType(d.pop('type'))
        if type_ == ToolType.FUNCTION:
            return __class__(
                type_,
                Function.fromPrimitive(d),  # inside it asserts `d` is consumed.
            )
        assert d.pop('name',        None) is None
        assert d.pop('description', None) is None
        assert d.pop('parameters',  None) is None
        assertAllConsumed(d)
        return __class__(type_, None)

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
        d = {**a}
        instance = __class__(
            d.pop('name'),
            d.pop('description'),
            Parameters.fromPrimitive(d.pop('parameters')),
        )
        assertAllConsumed(d)
        return instance

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
        d = {**a}
        properties: tp.Dict = d.pop('properties')
        instance = __class__(
            d.pop('type'),
            {k: Property.fromPrimitive(v) for k, v in properties.items()},
            d.pop('required'),
            d.pop('additionalProperties'),
        )
        assertAllConsumed(d)
        return instance

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
        d = {**a}
        instance = __class__(
            d.pop('type'),
            d.pop('description'),
        )
        assertAllConsumed(d)
        return instance

class InfType(Enum):
    INF = 'inf'

@dataclass(frozen=True)
class ConversationItem:
    id_: str
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
    audio: str | None = None
    transcript: str | None = None

    def __post_init__(self):
        assert (
            self.type_ in (ContentPartType.TEXT, ContentPartType.INPUT_TEXT)
        ) == self.text is not None
        assert (
            self.type_ in (ContentPartType.AUDIO, ContentPartType.INPUT_AUDIO)
        ) == self.audio is not None
    
    def asPrimitive(self):
        return {
            'type': str(self.type_),
            'text': self.text,
            'audio': self.audio,
            'transcript': self.transcript,
        }

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
    max_output_tokens: int | InfType | OmitType = OMIT

    def asPrimitive(self):
        return withoutOmits({
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
        })

    @staticmethod
    def fromPrimitiveRecycle(a: tp.Dict, /):
        d = {**a}
        instance = __class__(
            [Modality(x) for x in d.pop('modalities')], 
            d.pop('instructions'), 
            d.pop('voice'), 
            d.pop('output_audio_format'), 
            [Tool.fromPrimitive(x) for x in d.pop('tools')], 
            d.pop('tool_choice'), 
            d.pop('temperature'), 
            d.pop('max_output_tokens'), 
        )
        return instance, d

    @staticmethod
    def fromPrimitive(a: tp.Dict, /):
        instance, d = __class__.fromPrimitiveRecycle(a)
        assertAllConsumed(d)
        return instance

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
        d = {**a}
        responseConfig, d_ = ResponseConfig.fromPrimitiveRecycle(d)
        del d
        instance = __class__(
            responseConfig,
            d_.pop('input_audio_format'),
            d_.pop('input_audio_transcription_model'),
            TurnDetectionConfig.fromPrimitive(d_.pop('turn_detection')),
        )
        assertAllConsumed(d_)
        return instance
