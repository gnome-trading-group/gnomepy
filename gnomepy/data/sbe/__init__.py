from __future__ import annotations

import enum
import struct
from dataclasses import dataclass, field
from typing import TextIO

import lxml.etree


class PrimitiveType(enum.Enum):
    CHAR = 'char'
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'
    INT8 = 'int8'
    INT16 = 'int16'
    INT32 = 'int32'
    INT64 = 'int64'
    FLOAT = 'float'
    DOUBLE = 'double'


class SetEncodingType(enum.Enum):
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    UINT32 = 'uint32'
    UINT64 = 'uint64'


class EnumEncodingType(enum.Enum):
    UINT8 = 'uint8'
    UINT16 = 'uint16'
    CHAR = 'char'


class Presence(enum.Enum):
    CONSTANT = 'constant'
    REQUIRED = 'required'
    OPTIONAL = 'optional'


FORMAT = dict(zip(PrimitiveType.__members__.values(), "cBHIQbhiqfd"))
FORMAT_TO_TYPE = {v: k for k, v in FORMAT.items()}
FORMAT_SIZES = {k: struct.calcsize(v) for k, v in FORMAT.items()}
PRIMITIVE_TYPES = set(x.value for x in PrimitiveType.__members__.values())
ENUM_ENCODING_TYPES = set(x.value for x in EnumEncodingType.__members__.values())
SET_ENCODING_TYPES = set(x.value for x in SetEncodingType.__members__.values())
PRESENCE_TYPES = {x.value: x for x in Presence.__members__.values()}


class CharacterEncoding(enum.Enum):
    ASCII = 'ASCII'


@dataclass
class UnpackedValue:
    value: dict
    size: int

    __slots__ = ('value', 'size')

    def __repr__(self):
        return self.value.__repr__()


@dataclass
class DecodedMessage:
    message_name: str
    header: dict
    value: dict

    __slots__ = ('message_name', 'header', 'value')


@dataclass(init=False)
class Type:
    __slots__ = (
        'name', 'primitive_type', 'presence', 'semantic_type',
        'description', 'length', 'padding', 'character_encoding', 'null_value')

    name: str
    primitive_type: PrimitiveType
    presence: Presence
    semantic_type: str | None
    description: str | None
    length: int     # For strings
    padding: int    # Preceding this primitive
    character_encoding: CharacterEncoding | None
    null_value: str | int | float | None

    def __init__(self, name: str, primitive_type: PrimitiveType, null_value: str | None):
        super().__init__()
        self.name = name
        self.primitive_type = primitive_type
        self.presence = Presence.REQUIRED
        self.length = 1
        self.padding = 0
        self.character_encoding = None
        if null_value is not None:
            if primitive_type == PrimitiveType.CHAR:
                self.null_value = chr(int(null_value)).encode()
            elif primitive_type in (PrimitiveType.FLOAT, PrimitiveType.DOUBLE):
                self.null_value = float(null_value)
            else:
                self.null_value = int(null_value)
        else:
            self.null_value = None

    def __repr__(self):
        rv = self.name + " ("
        rv += self.primitive_type.value
        if rv == PrimitiveType.CHAR or self.length > 1:
            rv += f"[{self.length}]"
        rv += ")"
        return rv


@dataclass
class EnumValue:
    name: str
    value: str

    __slots__ = ('name', 'value')

    def __repr__(self):
        return (self.name, self.value).__repr__()


@dataclass
class RefType:
    name: str
    type: str

    __slots__ = ('name', 'type')

    def __repr__(self):
        return f"{self.name} (ref:{self.type})"

@dataclass
class SetChoice:
    name: str
    value: int

    __slots__ = ('name', 'value')

    def __repr__(self):
        return (self.name, self.value).__repr__()


@dataclass
class Enum:
    name: str
    encoding_type: EnumEncodingType | Type
    presence = Presence.REQUIRED
    semantic_type: str | None = None
    description: str | None = None
    valid_values: list[EnumValue] = field(default_factory=list)
    padding = 0

    def find_value_by_name(self, name: str | None) -> str:
        if name is None:
            return self.encoding_type.null_value
        val = next(x for x in self.valid_values if x.name == name).value
        if self.encoding_type == EnumEncodingType.CHAR or (isinstance(self.encoding_type, Type) and self.encoding_type.primitive_type == PrimitiveType.CHAR):
            return val.encode()
        return int(val)

    def find_name_by_value(self, val: str) -> str:
        if val not in (x.value for x in self.valid_values):
            return None
        return next(x for x in self.valid_values if x.value == val).name

    def __repr__(self):
        return f"<Enum '{self.name}'>"


@dataclass
class Composite:
    name: str
    types: list[Composite | Type | RefType | Enum] = field(default_factory=list)
    description: str | None = None

    def size(self):
        sz = 0
        for t in self.types:
            if isinstance(t, Type):
                if t.primitive_type == PrimitiveType.CHAR:
                    sz += t.length
                else:
                    sz += FORMAT_SIZES[t.primitive_type]
            else:
                assert isinstance(t, Composite)
                sz += t.size()
        return sz

    def __repr__(self):
        return f"<Composite '{self.name}'>"


@dataclass
class Set:
    name: str
    encoding_type: SetEncodingType | Type
    presence = Presence.REQUIRED
    semantic_type: str | None = None
    description: str | None = None
    choices: list[SetChoice] = field(default_factory=list)

    def decode(self, val: int) -> list[str]:
        # if isinstance(self.encodingType, SetEncodingType):
        #     length = FORMAT_SIZES[PrimitiveType[self.encodingType.name]] * 8
        # else:
        #     length = FORMAT_SIZES[self.encodingType.primitiveType] * 8

        return [c.name for c in self.choices if (1 << c.value) & val]

    def __repr__(self):
        return f"<Set '{self.name}'>"


@dataclass(init=False)
class Field:
    name: str
    id: str
    type: PrimitiveType | Set | Enum
    description: str | None
    since_version: int

    def __init__(self, name: str, id_: str, type_: PrimitiveType | str):
        self.name = name
        self.id = id_
        self.type = type_
        self.description = None
        self.since_version = 0

    __slots__ = ('name', 'id', 'type', 'description', 'since_version')

    def __repr__(self):
        if isinstance(self.type, PrimitiveType):
            return f"<{self.name} ({self.type.value})>"
        return f"<{self.name} ({self.type})>"

@dataclass
class Message:
    name: str
    id: int
    description: str | None = None
    fields: list[Field] = field(default_factory=list, repr=False)

    def fields_by_type(self, type_name: str) -> list[str]:
        return [_field.name for _field in self.fields if _field.type.name == type_name]

    @property
    def null_fields(self) -> dict[str, str | int | float]:
        return {
            _field.name: _field.type.null_value for _field in self.fields
            if isinstance(_field.type, Type) and _field.type.null_value is not None
        }

    @property
    def field_names(self) -> list[str]:
        return [_field.name for _field in self.fields]

    @property
    def formats(self) -> list[str]:
        output = []
        for f in self.fields:
            _unpack_format(f, output)
        return output

    @property
    def body_size(self) -> int:
        return struct.calcsize('<' + ''.join(self.formats))

@dataclass
class Cursor:
    val: int
    __slots__ = ('val',)


@dataclass
class Schema:
    id: int
    version: int
    types: dict[str, Composite | Type | RefType | Enum] = field(default_factory=dict)
    messages: dict[int, Message] = field(default_factory=dict)
    header_type_name: str = 'messageHeader'

    @classmethod
    def parse(cls, f: TextIO) -> Schema:
        rv = _parse_schema(f)
        return rv

    def decode_header(self, buffer: bytes | memoryview) -> dict:
        buffer = memoryview(buffer)
        return _unpack_composite(self, self.types[self.header_type_name], buffer).value

    def decode(self, buffer: bytes | memoryview) -> DecodedMessage:
        buffer = memoryview(buffer)

        header = _unpack_composite(self, self.types[self.header_type_name], buffer)
        body_offset = header.size

        m = self.messages[header.value['templateId']]

        format_str = '<' + ''.join(m.formats)
        rv = {}
        vals = struct.unpack(format_str, buffer[body_offset:body_offset+m.body_size])
        _walk_fields_decode(self, rv, m.fields, vals, Cursor(0))
        return DecodedMessage(m.name, header.value, rv)

def _unpack_format(
        type_: Field | PrimitiveType | Type | RefType | Set | Enum | Composite,
        output: list,
):
    if isinstance(type_, PrimitiveType):
        return output.append(FORMAT[type_])

    if isinstance(type_, Field):
        return _unpack_format(type_.type, output)

    if isinstance(type_, Type):
        if type_.presence == Presence.CONSTANT:
            return
        rv = ''
        if type_.padding > 0:
            rv += str(type_.padding) + 'x'
        if type_.primitive_type == PrimitiveType.CHAR:
            return output.append(rv + f"{type_.length}s")

        return output.append(FORMAT[type_.primitive_type])

    if isinstance(type_, (Set, Enum)):
        if type_.presence == Presence.CONSTANT:
            return
        if isinstance(type_.encoding_type, (PrimitiveType, EnumEncodingType, SetEncodingType)):
            if type_.encoding_type.value in PRIMITIVE_TYPES:
                return output.append(FORMAT[PrimitiveType(type_.encoding_type.value)])
        elif isinstance(type_.encoding_type.primitive_type, PrimitiveType):
            if type_.encoding_type.primitive_type.value in PRIMITIVE_TYPES:
                return output.append(FORMAT[PrimitiveType(type_.encoding_type.primitive_type.value)])

        return _unpack_format(type_.encoding_type, output)

    if isinstance(type_, Composite):
        for t in type_.types:
            _unpack_format(t, output)
        return

    assert False, f"unreachable: {type_}"

def _unpack_composite(schema: Schema, composite: Composite, buffer: memoryview):
    output = []
    _unpack_format(composite, output)
    fmt = '<' + ''.join(output)
    size = struct.calcsize(fmt)
    vals = struct.unpack(fmt, buffer[:size])

    rv = {}
    for t, v in zip(composite.types, vals):
        rv[t.name] = _prettify_type(schema, t, v)

    return UnpackedValue(rv, size)

def _prettify_type(_schema: Schema, t: Type, v):
    if t.primitive_type == PrimitiveType.CHAR and (
            t.character_encoding == CharacterEncoding.ASCII or t.character_encoding is None
    ):
        return v.split(b'\x00', 1)[0].decode('ascii', errors='ignore').strip()
    if t.null_value is not None and v == t.null_value:
        return None

    return v

def _decode_value(
    schema: Schema, rv: dict, name: str, t: Type | Set | Enum | PrimitiveType, vals: list, cursor: Cursor
):
    if isinstance(t, Type):
        rv[name] = _prettify_type(schema, t, vals[cursor.val])
        cursor.val += 1

    elif isinstance(t, Set):
        v = vals[cursor.val]
        cursor.val += 1
        rv[name] = t.decode(v)

    elif isinstance(t, Enum):
        v = vals[cursor.val]
        cursor.val += 1

        if isinstance(v, bytes):
            if v == b'\x00':
                rv[name] = v
            else:
                rv[name] = t.find_name_by_value(v.decode("ascii", errors='ignore'))
        else:
            rv[name] = t.find_name_by_value(str(v))

    elif isinstance(t, PrimitiveType):
        v = vals[cursor.val]
        cursor.val += 1
        rv[name] = v

    else:
        assert 0


def _walk_fields_decode_composite(schema: Schema, rv: dict, composite: Composite, vals: list, cursor: Cursor):
    for t in composite.types:
        if isinstance(t, Composite):
            rv[t.name] = {}
            _walk_fields_decode_composite(schema, rv[t.name], t, vals, cursor)
        else:
            if t.presence != Presence.CONSTANT:
                _decode_value(schema, rv, t.name, t, vals, cursor)


def _walk_fields_decode(schema: Schema, rv: dict, fields: list[Field], vals: list, cursor: Cursor):
    for f in fields:
        if isinstance(f.type, Composite):
            rv[f.name] = {}
            _walk_fields_decode_composite(schema, rv[f.name], f.type, vals, cursor)
        else:
            _decode_value(schema, rv, f.name, f.type, vals, cursor)

def _parse_schema(f: TextIO) -> Schema:
    doc = lxml.etree.parse(f)
    doc.xinclude()
    schema_with_types = _parse_schema_impl(doc, ['messageSchema', 'type'])
    return _parse_schema_impl(doc, None, schema_with_types.types)

def _parse_schema_impl(doc, only_tags: list | None = None, extra_types: dict | None = None) -> Schema:
    stack = []
    for action, elem in lxml.etree.iterwalk(doc, ("start", "end")):
        assert action in ("start", "end")

        tag = elem.tag
        if "}" in tag:
            tag = tag[tag.index("}") + 1:]

        if only_tags is not None and tag not in only_tags:
            continue

        if tag == "messageSchema":
            if action == "start":
                attrs = dict(elem.items())
                x = Schema(attrs['id'], attrs['version'],
                           header_type_name=attrs.get('headerType', 'messageHeader'))
                if extra_types is not None:
                    x.types.update(extra_types)
                stack.append(x)
            elif action == "end":
                pass

        elif tag == "types":
            pass

        elif tag == "composite":
            if action == "start":
                name = next(v for k, v in elem.items() if k == 'name')
                x = Composite(name=name)
                for k, v in elem.items():
                    if k == 'name':
                        pass
                    elif k == 'description':
                        x.description = v

                stack.append(x)

            elif action == 'end':
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].types[x.name] = x

        elif tag == "type":
            if action == "start":
                attrs = dict(elem.items())
                x = Type(name=attrs['name'], primitive_type=PrimitiveType(attrs['primitiveType']),
                         null_value=attrs['nullValue'] if 'nullValue' in attrs else None)

                if x.primitive_type == PrimitiveType.CHAR:
                    if 'length' in attrs:
                        x.length = int(attrs['length'])
                    if 'characterEncoding' in attrs:
                        x.character_encoding = CharacterEncoding(attrs['characterEncoding'])
                if 'semanticType' in attrs:
                    x.semantic_type = attrs['semanticType']
                if 'presence' in attrs:
                    x.presence = PRESENCE_TYPES[attrs['presence']]
                if 'offset' in attrs:
                    if isinstance(stack[-1], Composite):
                        x.padding = int(attrs['offset']) - stack[-1].size()
                    else:
                        x.padding = int(attrs['offset'])

                stack.append(x)

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], (Composite, Schema))
                if isinstance(stack[-1], Composite):
                    stack[-1].types.append(x)
                elif isinstance(stack[-1], Schema):
                    stack[-1].types[x.name] = x

        elif tag == "enum":
            if action == "start":
                attrs = dict(elem.items())

                if attrs['encodingType'].lower() in ENUM_ENCODING_TYPES:
                    encoding_type = EnumEncodingType(attrs['encodingType'].lower())
                else:
                    encoding_type = stack[0].types[attrs['encodingType']]

                stack.append(Enum(name=attrs['name'], encoding_type=encoding_type))

            elif action == "end":
                x = stack.pop()
                if isinstance(stack[-1], Composite):
                    stack[-1].types.append(x)
                elif isinstance(stack[-1], Schema):
                    stack[-1].types[x.name] = x

        elif tag == "validValue":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(EnumValue(name=attrs['name'], value=elem.text.strip()))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Enum)
                stack[-1].valid_values.append(x)

        elif tag == "set":
            if action == "start":
                attrs = dict(elem.items())

                if attrs['encodingType'] in SET_ENCODING_TYPES:
                    encoding_type = SetEncodingType(attrs['encodingType'])
                else:
                    encoding_type = stack[0].types[attrs['encodingType']]

                stack.append(Set(name=attrs['name'], encoding_type=encoding_type))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                x.choices = sorted(x.choices, key=lambda y: int(y.value))
                stack[-1].types[x.name] = x

        elif tag == "choice":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(SetChoice(name=attrs['name'], value=int(elem.text.strip())))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Set)
                stack[-1].choices.append(x)

        elif tag in ('field', 'data'):
            if action == "start":
                assert len(elem) == 0

                attrs = dict(elem.items())
                if attrs['type'] in PRIMITIVE_TYPES:
                    field_type = PrimitiveType(attrs['type'])
                else:
                    field_type = stack[0].types[attrs['type']]

                assert isinstance(stack[-1], Message)
                stack[-1].fields.append(Field(name=attrs['name'], id_=attrs['id'], type_=field_type))

        elif tag == "message":
            if action == "start":
                attrs = dict(elem.items())
                stack.append(Message(name=attrs['name'],
                                     id=int(attrs['id']),
                                     description=attrs.get('description')))

            elif action == "end":
                x = stack.pop()
                assert isinstance(stack[-1], Schema)
                stack[-1].messages[x.id] = x

        elif tag == "ref":
            if action == "start":
                stack.append(RefType(name=elem.attrib['name'], type=elem.attrib['type']))

            elif action == "end":
                x = stack.pop()
                if isinstance(stack[-1], Composite):
                    stack[-1].types.append(x)
                elif isinstance(stack[-1], Schema):
                    stack[-1].types[x.name] = x
                else:
                    assert False, "unreachable"

        elif tag == "group":
            raise NotImplementedError
        else:
            assert 0, f"Unknown tag '{tag}'"

    return stack[0]
