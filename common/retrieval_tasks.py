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

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from rag.utils.xcmg_models import ModelMention, find_model_mentions, normalize_text

_CLAUSE_BOUNDARY = re.compile(r"[，,；;。]+")
_LEADING_CONNECTORS = re.compile(r"^[\s和与及、,，;；:：]+")
_TRAILING_CONNECTORS = re.compile(r"[\s和与及、,，;；:：]+$")
_DUPLICATE_CONJUNCT = re.compile(r"(?P<term>[\u4e00-\u9fffA-Za-z0-9]+)和(?P=term)")
_INTENT_PATTERN = re.compile(r"故障维修|故障诊断|故障排查|如何操作|怎么操作|操作|如何使用|怎么用|维修|维护|保养|拆装|拆卸|安装")
_INTENT_MAP = {
    "故障维修": "故障维修",
    "故障诊断": "故障维修",
    "故障排查": "故障维修",
    "如何操作": "操作",
    "怎么操作": "操作",
    "操作": "操作",
    "如何使用": "操作",
    "怎么用": "操作",
    "维修": "维修",
    "维护": "保养",
    "保养": "保养",
    "拆装": "拆装",
    "拆卸": "拆装",
    "安装": "拆装",
}

_ENGINE_OIL_TERMS = ("发动机机油", "发动机油", "柴油机润滑油")
_HYDRAULIC_TANK_AMOUNT_TERMS = (
    "液压油箱容量",
    "液压油箱加注容量",
    "液压油箱加油量",
    "液压油加注量",
    "液压油用量",
    "液压油定量",
)


@dataclass(frozen=True)
class RetrievalTask:
    query: str
    models: tuple[str, ...] = ()
    model_aliases: tuple[str, ...] = ()
    knowledge_types: tuple[str, ...] = ()


def _append_missing_terms(query: str, terms: tuple[str, ...]) -> str:
    missing = [term for term in terms if term not in query]
    return " ".join([query, *missing]) if missing else query


def expand_retrieval_query(query: str) -> str:
    """Add narrow XCMG-manual synonyms for retrieval without changing the user question."""
    expanded = normalize_text(query).strip()
    if not expanded:
        return expanded

    asks_engine_oil = "发动机机油" in expanded or "柴油机润滑油" in expanded or bool(re.search(r"发动机油(?!压|温)", expanded))
    if asks_engine_oil:
        expanded = _append_missing_terms(expanded, _ENGINE_OIL_TERMS)

    asks_hydraulic_tank_amount = "液压油" in expanded and any(
        marker in expanded for marker in ("容量", "加油量", "加注量", "用量", "定量")
    )
    if asks_hydraulic_tank_amount:
        expanded = _append_missing_terms(expanded, _HYDRAULIC_TANK_AMOUNT_TERMS)

    return expanded


def _unique_model_groups(mentions: list[ModelMention]) -> list[ModelMention]:
    groups = []
    seen = set()
    for mention in mentions:
        key = mention.official_models
        if key in seen:
            continue
        seen.add(key)
        groups.append(mention)
    return groups


def _find_intents(text: str) -> list[str]:
    intents = []
    for match in _INTENT_PATTERN.finditer(text):
        intent = _INTENT_MAP[match.group(0)]
        if intent not in intents:
            intents.append(intent)
    return intents


def _remove_model_mentions(text: str) -> str:
    mentions = find_model_mentions(text)
    if not mentions:
        return text
    parts = []
    cursor = 0
    for mention in mentions:
        parts.append(text[cursor : mention.start])
        cursor = mention.end
    parts.append(text[cursor:])
    return "".join(parts)


def _build_task_query(source: str, model: ModelMention, intent: str | None) -> str:
    residual = _remove_model_mentions(source)
    residual = _INTENT_PATTERN.sub("", residual)
    residual = _DUPLICATE_CONJUNCT.sub(r"\g<term>", residual)
    residual = _LEADING_CONNECTORS.sub("", residual)
    residual = _TRAILING_CONNECTORS.sub("", residual)
    terms = [model.official_models[0]]
    if intent:
        terms.append(intent)
    if residual:
        terms.append(residual)
    return " ".join(terms)


def _make_task(source: str, model: ModelMention, intent: str | None = None) -> RetrievalTask:
    return RetrievalTask(
        query=_build_task_query(source, model, intent),
        models=model.official_models,
        model_aliases=model.aliases,
        knowledge_types=(intent,) if intent else (),
    )


def _paired_clause_tasks(question: str, model_groups: list[ModelMention]) -> list[RetrievalTask]:
    tasks = []
    covered_models = set()
    for clause in _CLAUSE_BOUNDARY.split(question):
        clause = clause.strip()
        if not clause:
            continue
        clause_models = _unique_model_groups(find_model_mentions(clause))
        intents = _find_intents(clause)
        if len(clause_models) != 1 or not intents:
            continue
        model = clause_models[0]
        covered_models.add(model.official_models)
        tasks.extend(_make_task(clause, model, intent) for intent in intents)

    expected_models = {model.official_models for model in model_groups}
    return tasks if len(covered_models) > 1 and covered_models == expected_models else []


def split_retrieval_tasks(question: str) -> list[RetrievalTask]:
    """Split an explicit multi-model or multi-intent question into retrieval tasks."""
    normalized_question = normalize_text(question).strip()
    if not normalized_question:
        return []

    model_groups = _unique_model_groups(find_model_mentions(normalized_question))
    intents = _find_intents(normalized_question)
    if not model_groups:
        return [RetrievalTask(query=normalized_question)]

    if len(model_groups) > 1:
        paired_tasks = _paired_clause_tasks(normalized_question, model_groups)
        if paired_tasks:
            return paired_tasks

    if not intents:
        return [_make_task(normalized_question, model) for model in model_groups]

    return [_make_task(normalized_question, model, intent) for model in model_groups for intent in intents]


def _chunk_key(chunk: dict) -> tuple:
    if chunk.get("chunk_id"):
        return ("chunk_id", str(chunk["chunk_id"]))
    return (
        "content",
        str(chunk.get("doc_id") or ""),
        str(chunk.get("content_with_weight") or ""),
    )


def _task_context(task: RetrievalTask) -> dict:
    return {
        "query": task.query,
        "models": list(task.models),
        "knowledge_types": list(task.knowledge_types),
    }


def _merge_task_contexts(target: dict, source: dict) -> None:
    contexts = target.setdefault("retrieval_tasks", [])
    for context in source.get("retrieval_tasks") or []:
        if context not in contexts:
            contexts.append(context)


def merge_task_results(task_results: list[dict], top_n: int, tasks: list[RetrievalTask] | None = None) -> dict:
    """Merge independently ranked task results while preserving task coverage."""
    if top_n <= 0 or not task_results:
        return {"total": 0, "chunks": [], "doc_aggs": []}

    if tasks is not None and len(tasks) != len(task_results):
        raise ValueError("tasks and task_results must have the same length")

    prepared_results = []
    for task_index, result in enumerate(task_results):
        prepared = dict(result)
        prepared["chunks"] = []
        for raw_chunk in result.get("chunks") or []:
            chunk = dict(raw_chunk)
            if tasks is not None:
                chunk["retrieval_tasks"] = [_task_context(tasks[task_index])]
            prepared["chunks"].append(chunk)
        prepared_results.append(prepared)

    quota = max(1, top_n // len(prepared_results))
    selected = []
    selected_by_key = {}
    selected_keys = set()

    def add_chunk(chunk: dict) -> bool:
        key = _chunk_key(chunk)
        if key in selected_keys:
            _merge_task_contexts(selected_by_key[key], chunk)
            return False
        selected_keys.add(key)
        copied_chunk = dict(chunk)
        selected.append(copied_chunk)
        selected_by_key[key] = copied_chunk
        return True

    for result in prepared_results:
        accepted = 0
        for chunk in result.get("chunks") or []:
            if add_chunk(chunk):
                accepted += 1
            if accepted >= quota or len(selected) >= top_n:
                break

    remaining = []
    for task_index, result in enumerate(prepared_results):
        for local_rank, chunk in enumerate(result.get("chunks") or []):
            if _chunk_key(chunk) in selected_keys:
                continue
            remaining.append((-float(chunk.get("similarity") or 0.0), task_index, local_rank, chunk))
    remaining.sort(key=lambda item: item[:3])

    for _, _, _, chunk in remaining:
        if len(selected) >= top_n:
            break
        add_chunk(chunk)

    doc_aggs = {}
    for chunk in selected:
        doc_name = str(chunk.get("docnm_kwd") or "")
        if doc_name not in doc_aggs:
            doc_aggs[doc_name] = {"doc_id": chunk.get("doc_id") or "", "count": 0}
        doc_aggs[doc_name]["count"] += 1

    return {
        "total": len(selected),
        "chunks": selected,
        "doc_aggs": [
            {"doc_name": name, "doc_id": values["doc_id"], "count": values["count"]}
            for name, values in doc_aggs.items()
        ],
    }


async def retrieve_and_merge_tasks(
    tasks: list[RetrievalTask],
    top_n: int,
    rerank_candidates_count: int,
    retrieve_task: Callable[[RetrievalTask, int], Awaitable[dict]],
) -> dict:
    """Retrieve each task independently, then merge with protected quotas."""
    if not tasks or top_n <= 0:
        return {"total": 0, "chunks": [], "doc_aggs": []}

    per_task_quota = max(1, top_n // len(tasks))
    per_task_top_n = min(rerank_candidates_count, max(4, per_task_quota * 2))
    semaphore = asyncio.Semaphore(min(4, len(tasks)))

    async def bounded_retrieve(task):
        async with semaphore:
            return await retrieve_task(task, per_task_top_n)

    results = await asyncio.gather(*(bounded_retrieve(task) for task in tasks))
    return merge_task_results(results, top_n, tasks)
