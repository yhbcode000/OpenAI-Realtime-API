from __future__ import annotations

import typing as tp
from enum import Enum
from dataclasses import dataclass

class OmitType: pass
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
TRUNCATE = 'truncate'
DELETE = 'delete'
CANCEL = 'cancel'

SESSION = 'session'
INPUT_AUDIO_BUFFER = 'input_audio_buffer'
CONVERSATION = 'conversation'
ITEM = 'item'
RESPONSE = 'response'
ERROR = 'error'

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

@dataclass(frozen=True)
class Function:
    name: str
    description: str
    parameters: Parameters

    def asPrimitive(self):
        return {
            'type': self.type_,
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters.asPrimitive(),
        }

@dataclass(frozen=True)
class Parameters:
    type_: str
    properties: tp.Dict[str, Parameter]
    required: tp.List[str]
    additionalProperties: bool

    def asPrimitive(self):
        return {
            'type': self.type_,
            'properties': {k: v.asPrimitive() for k, v in self.properties.items()},
            'required': self.required,
            'additionalProperties': self.additionalProperties,
        }

@dataclass(frozen=True)
class Parameter:
    type_: str
    description: str

    def asPrimitive(self):
        return {
            'type': self.type_,
            'description': self.description,
        }

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
