from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from gnomepy.java._classpath import discover_classpath
from gnomepy.java.enums import RecordType


_SBE_NS = "http://fixprotocol.io/2016/sbe"

_cached_schema: SbeSchema | None = None


def _camel_to_snake(name: str) -> str:
    s = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    s = re.sub(r"([a-z])(\d+)$", r"\1_\2", s)
    return s


@dataclass
class SbeType:
    name: str
    null_value: int | None = None


@dataclass
class SbeField:
    name: str
    type: SbeType


@dataclass
class SbeMessage:
    name: str
    id: int
    fields: list[SbeField] = field(default_factory=list)

    @property
    def null_fields(self) -> dict[str, int]:
        return {
            f.name: f.type.null_value
            for f in self.fields
            if f.type.null_value is not None
        }

    def fields_by_type(self, type_name: str) -> list[str]:
        return [f.name for f in self.fields if f.type.name == type_name]


@dataclass
class SbeSchema:
    types: dict[str, SbeType]
    messages: dict[str, SbeMessage]

    @classmethod
    def parse(cls, xml_text: str) -> SbeSchema:
        root = ET.fromstring(xml_text)
        ns = {"sbe": _SBE_NS}

        types: dict[str, SbeType] = {}
        for t in root.findall(".//types/type"):
            name = t.get("name")
            null_val = t.get("nullValue")
            null_int = int(null_val) if null_val is not None else None
            types[name] = SbeType(name=name, null_value=null_int)

        messages: dict[str, SbeMessage] = {}
        for msg_el in root.findall("sbe:message", ns):
            msg_name = msg_el.get("name")
            msg_id = int(msg_el.get("id"))
            fields = []
            for f_el in msg_el.findall("field"):
                f_name = _camel_to_snake(f_el.get("name"))
                type_name = f_el.get("type")
                sbe_type = types.get(type_name, SbeType(name=type_name))
                fields.append(SbeField(name=f_name, type=sbe_type))
            messages[msg_name] = SbeMessage(name=msg_name, id=msg_id, fields=fields)

        return cls(types=types, messages=messages)


_FAMILY_SCHEMAS = ("core.xml", "book.xml")


def load_schema() -> SbeSchema:
    """Extract every family's schema XML from the uber JAR and parse them into one. Result is cached.

    Template ids are unique across families, so their messages share one namespace.
    """
    global _cached_schema
    if _cached_schema is not None:
        return _cached_schema

    jars = discover_classpath("gnome-backtest")
    for jar_path in jars:
        with zipfile.ZipFile(jar_path) as zf:
            names = zf.namelist()
            if all(name in names for name in _FAMILY_SCHEMAS):
                types: dict[str, SbeType] = {}
                messages: dict[str, SbeMessage] = {}
                for name in _FAMILY_SCHEMAS:
                    family = SbeSchema.parse(zf.read(name).decode("utf-8"))
                    types.update(family.types)
                    messages.update(family.messages)
                _cached_schema = SbeSchema(types=types, messages=messages)
                return _cached_schema

    raise FileNotFoundError(f"{_FAMILY_SCHEMAS} not found in any JAR on the classpath")


def get_message(record_type: RecordType) -> SbeMessage:
    """Return the SbeMessage metadata for a given RecordType."""
    msg_name = record_type.name.replace("_", "").capitalize()
    return load_schema().messages[msg_name]
