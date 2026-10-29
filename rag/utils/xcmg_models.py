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
from dataclasses import dataclass

UNICODE_HYPHENS = str.maketrans({char: "-" for char in "‐‑‒–—―﹘﹣－"})

# Business-confirmed model groups. Keep unconfirmed look-alike model names out
# of this registry: an incorrect equivalence is worse than a missing alias when
# the resulting metadata is used as a hard retrieval filter.
MODEL_GROUPS = (
    {
        "official_models": ("XC975EV", "XC975EU"),
        "aliases": ("XC975-EV", "XC975-EU", "975(EU)EV.AX01"),
        "equipment_type": "轮胎式装载机",
    },
    {
        "official_models": ("XC938U",),
        "aliases": (),
        "equipment_type": "轮胎式装载机",
    },
    {
        "official_models": ("XC975UEV",),
        "aliases": (),
        "equipment_type": "轮胎式装载机",
    },
)

_EQUIPMENT_MODEL_PATTERN = re.compile(r"(?<![A-Z0-9])(?:XC|XE|LW)[A-Z0-9]+(?:-[A-Z0-9]+)*(?![A-Z0-9])", re.IGNORECASE)
_SPECIAL_MODEL_PATTERN = re.compile(r"(?<![A-Z0-9])\d{3,4}\([A-Z]{2,}\)[A-Z0-9]+(?:\.[A-Z0-9]+)+(?![A-Z0-9])", re.IGNORECASE)


@dataclass(frozen=True)
class ModelMention:
    text: str
    start: int
    end: int
    official_models: tuple[str, ...]
    aliases: tuple[str, ...]
    equipment_type: str | None


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).translate(UNICODE_HYPHENS)


def normalize_model_key(model: str) -> str:
    return re.sub(r"[\s-]+", "", normalize_text(model)).upper()


def _build_model_lookup() -> dict[str, dict]:
    lookup = {}
    for group in MODEL_GROUPS:
        for model in (*group["official_models"], *group["aliases"]):
            key = normalize_model_key(model)
            previous = lookup.get(key)
            if previous is not None and previous is not group:
                raise ValueError(f"Duplicate model registry key: {model}")
            lookup[key] = group
    return lookup


_MODEL_LOOKUP = _build_model_lookup()


def find_model_mentions(text: str) -> list[ModelMention]:
    """Find known and unknown model-like tokens without guessing equivalence."""
    normalized = normalize_text(text)
    raw_matches = []
    for pattern in (_EQUIPMENT_MODEL_PATTERN, _SPECIAL_MODEL_PATTERN):
        raw_matches.extend((match.start(), match.end(), match.group(0).upper()) for match in pattern.finditer(normalized))
    raw_matches.sort(key=lambda item: item[0])

    mentions = []
    seen_spans = set()
    for start, end, token in raw_matches:
        if (start, end) in seen_spans:
            continue
        seen_spans.add((start, end))
        group = _MODEL_LOOKUP.get(normalize_model_key(token))
        if group is None:
            mentions.append(ModelMention(token, start, end, (token,), (), None))
        else:
            mentions.append(
                ModelMention(
                    token,
                    start,
                    end,
                    tuple(group["official_models"]),
                    tuple(group["aliases"]),
                    group["equipment_type"],
                )
            )
    return mentions
