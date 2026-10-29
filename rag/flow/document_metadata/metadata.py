#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import re
import unicodedata
from collections.abc import Iterable
from pathlib import Path

from rag.flow.document_metadata.model_registry import MODEL_GROUPS

_UNICODE_HYPHENS = str.maketrans({char: "-" for char in "‐‑‒–—―﹘﹣－"})
_XC_MODEL_PATTERN = re.compile(r"(?<![A-Z0-9])XC[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9])", re.IGNORECASE)
_SPECIAL_MODEL_PATTERN = re.compile(r"(?<![A-Z0-9])\d{3,4}\([A-Z]{2,}\)[A-Z0-9]+(?:\.[A-Z0-9]+)+(?![A-Z0-9])", re.IGNORECASE)

_DOCUMENT_TYPES = (
    (re.compile(r"控制系统使用说明书"), "控制系统使用说明书", ("操作",)),
    (re.compile(r"显示屏使用说明书"), "显示屏使用说明书", ("操作",)),
    (re.compile(r"(?:操作与保养|操作保养|操保)手册"), "操作与保养手册", ("操作", "保养")),
    (re.compile(r"(?:维修与保养|维修保养)手册"), "维修保养手册", ("维修", "保养")),
    (re.compile(r"售后技术文件"), "售后技术文件", ()),
    (re.compile(r"维修手册"), "维修手册", ("维修",)),
    (re.compile(r"保养手册"), "保养手册", ("保养",)),
    (re.compile(r"操作手册"), "操作手册", ("操作",)),
    (re.compile(r"使用说明书"), "使用说明书", ("操作",)),
)

_EQUIPMENT_TYPES = (
    ("轮胎式装载机", "轮胎式装载机"),
    ("装载机", "轮胎式装载机"),
    ("液压挖掘机", "液压挖掘机"),
    ("挖掘机", "液压挖掘机"),
    ("车载式混凝土泵车", "车载式混凝土泵车"),
    ("混凝土泵车", "车载式混凝土泵车"),
)

_KNOWLEDGE_TYPE_ORDER = ("操作", "维修", "保养", "拆装", "故障维修")


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _normalize_filename(filename: str) -> str:
    name = Path(str(filename or "")).name
    return unicodedata.normalize("NFKC", name).translate(_UNICODE_HYPHENS)


def _normalize_model_key(model: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(model or "")).translate(_UNICODE_HYPHENS)
    return re.sub(r"[\s-]+", "", normalized).upper()


def _build_model_lookup() -> dict[str, dict]:
    lookup = {}
    for group in MODEL_GROUPS:
        for model in (*group["official_models"], *group["aliases"]):
            key = _normalize_model_key(model)
            previous = lookup.get(key)
            if previous is not None and previous is not group:
                raise ValueError(f"Duplicate model registry key: {model}")
            lookup[key] = group
    return lookup


_MODEL_LOOKUP = _build_model_lookup()


def _extract_model_tokens(filename: str) -> list[str]:
    matches = []
    for pattern in (_XC_MODEL_PATTERN, _SPECIAL_MODEL_PATTERN):
        matches.extend((match.start(), match.group(0).upper()) for match in pattern.finditer(filename))
    matches.sort(key=lambda item: item[0])

    tokens = []
    seen = set()
    for _, token in matches:
        key = _normalize_model_key(token)
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
    return tokens


def _detect_document_type(filename: str) -> tuple[str | None, list[str]]:
    document_type = None
    knowledge_types = []
    for pattern, candidate_type, candidate_knowledge_types in _DOCUMENT_TYPES:
        if pattern.search(filename):
            document_type = candidate_type
            _append_unique(knowledge_types, candidate_knowledge_types)
            break

    detected = set(knowledge_types)
    if "操作" in filename or "操保" in filename or "使用说明书" in filename:
        detected.add("操作")
    if "保养" in filename or "操保" in filename:
        detected.add("保养")
    if "故障维修" in filename or "故障诊断" in filename or "故障排查" in filename:
        detected.add("故障维修")
    elif "维修" in filename:
        detected.add("维修")
    if "拆装" in filename:
        detected.add("拆装")

    return document_type, [item for item in _KNOWLEDGE_TYPE_ORDER if item in detected]


def extract_document_metadata(filename: str) -> dict:
    """Extract conservative document-level retrieval metadata from a filename."""
    normalized_name = _normalize_filename(filename)
    metadata = {"brand": "徐工"}
    models = []
    model_aliases = []
    equipment_types = []

    for token in _extract_model_tokens(normalized_name):
        group = _MODEL_LOOKUP.get(_normalize_model_key(token))
        if group is None:
            _append_unique(models, (token,))
            continue
        _append_unique(models, group["official_models"])
        _append_unique(model_aliases, group["aliases"])
        _append_unique(equipment_types, (group["equipment_type"],))

    for marker, equipment_type in _EQUIPMENT_TYPES:
        if marker in normalized_name:
            equipment_types = [equipment_type]
            break

    document_type, knowledge_types = _detect_document_type(normalized_name)

    if equipment_types:
        metadata["equipment_type"] = equipment_types[0]
    if models:
        metadata["models"] = models
    if model_aliases:
        metadata["model_aliases"] = model_aliases
    if document_type:
        metadata["document_type"] = document_type
    if knowledge_types:
        metadata["knowledge_types"] = knowledge_types
    return metadata


def attach_document_metadata(chunks: list[dict] | None, filename: str) -> list[dict]:
    """Attach one document metadata payload to the first chunk for persistence."""
    output = [dict(chunk) for chunk in (chunks or [])]
    if not output:
        return output

    metadata = extract_document_metadata(filename)
    existing = output[0].get("metadata")
    if isinstance(existing, dict):
        metadata = {**existing, **metadata}
    output[0]["metadata"] = metadata
    return output
