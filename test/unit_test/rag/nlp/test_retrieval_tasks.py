import pytest

from common.retrieval_tasks import RetrievalTask, expand_retrieval_query, merge_task_results, retrieve_and_merge_tasks, split_retrieval_tasks


def test_leaves_question_without_model_as_one_task():
    assert split_retrieval_tasks("液压油多久更换一次？") == [RetrievalTask(query="液压油多久更换一次?")]


def test_splits_two_models_asking_for_same_fact():
    tasks = split_retrieval_tasks("LW500HV-GIV发动机机油和XE155D发动机机油的更换周期分别是什么？")

    assert [task.models for task in tasks] == [("LW500HV-GIV",), ("XE155D",)]
    assert all("发动机机油的更换周期分别是什么?" in task.query for task in tasks)
    assert all(task.query.count("发动机机油") == 1 for task in tasks)


def test_splits_one_model_with_operation_and_repair_intents():
    tasks = split_retrieval_tasks("XC938U怎么操作和维修？")

    assert [task.knowledge_types for task in tasks] == [("操作",), ("维修",)]
    assert all(task.models == ("XC938U",) for task in tasks)


def test_preserves_explicit_model_to_intent_pairing_across_clauses():
    tasks = split_retrieval_tasks("XC938U怎么操作，XC975EV怎么维修？")

    assert [(task.models, task.knowledge_types) for task in tasks] == [
        (("XC938U",), ("操作",)),
        (("XC975EV", "XC975EU"), ("维修",)),
    ]


def test_builds_cartesian_tasks_when_both_models_share_both_intents():
    tasks = split_retrieval_tasks("XC938U和XC975EV分别怎么操作和维修？")

    assert [(task.models, task.knowledge_types) for task in tasks] == [
        (("XC938U",), ("操作",)),
        (("XC938U",), ("维修",)),
        (("XC975EV", "XC975EU"), ("操作",)),
        (("XC975EV", "XC975EU"), ("维修",)),
    ]


def test_deduplicates_official_names_for_same_product():
    tasks = split_retrieval_tasks("XC975EV和XC975EU怎么保养？")

    assert len(tasks) == 1
    assert tasks[0].models == ("XC975EV", "XC975EU")
    assert tasks[0].knowledge_types == ("保养",)


def test_keeps_unknown_models_separate_without_alias_guessing():
    tasks = split_retrieval_tasks("XC999D和XC888D怎么维修？")

    assert [task.models for task in tasks] == [("XC999D",), ("XC888D",)]


def test_expands_engine_oil_terms_for_retrieval_only():
    expanded = expand_retrieval_query("XE155D发动机机油多久更换？")

    assert expanded.startswith("XE155D发动机机油多久更换?")
    assert all(term in expanded for term in ("发动机机油", "发动机油", "柴油机润滑油"))


def test_expands_hydraulic_tank_amount_terms_conservatively():
    expanded = expand_retrieval_query("LW500HV-GIV液压油箱容量是多少？")

    assert expanded.startswith("LW500HV-GIV液压油箱容量是多少?")
    assert all(term in expanded for term in ("液压油箱容量", "液压油箱加油量", "液压油用量", "液压油定量"))


def test_does_not_expand_unrelated_capacity_question():
    assert expand_retrieval_query("LW500HV-GIV燃油箱容量是多少？") == "LW500HV-GIV燃油箱容量是多少?"


def test_does_not_expand_engine_oil_pressure_question():
    assert expand_retrieval_query("XE155D发动机油压过低怎么办？") == "XE155D发动机油压过低怎么办?"


def _chunk(chunk_id, similarity, doc_name):
    return {
        "chunk_id": chunk_id,
        "similarity": similarity,
        "doc_id": f"doc-{doc_name}",
        "docnm_kwd": doc_name,
        "content_with_weight": chunk_id,
    }


def test_merge_reserves_quota_for_each_task_before_global_fill():
    first = {"chunks": [_chunk(f"a{i}", 1.0 - i / 100, "manual-a") for i in range(6)]}
    second = {"chunks": [_chunk(f"b{i}", 0.5 - i / 100, "manual-b") for i in range(6)]}

    merged = merge_task_results([first, second], top_n=6)

    assert [chunk["chunk_id"] for chunk in merged["chunks"]] == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert {item["doc_name"]: item["count"] for item in merged["doc_aggs"]} == {"manual-a": 3, "manual-b": 3}


def test_merge_deduplicates_chunks_and_fills_freed_slots():
    shared = _chunk("shared", 0.9, "shared-manual")
    first = {"chunks": [shared, _chunk("a1", 0.8, "manual-a"), _chunk("a2", 0.7, "manual-a")]}
    second = {"chunks": [shared, _chunk("b1", 0.6, "manual-b"), _chunk("b2", 0.5, "manual-b")]}

    merged = merge_task_results([first, second], top_n=4)

    assert [chunk["chunk_id"] for chunk in merged["chunks"]] == ["shared", "a1", "b1", "b2"]
    assert merged["total"] == 4


def test_merge_labels_chunks_with_their_retrieval_task():
    tasks = [
        RetrievalTask(query="LW500HV-GIV 发动机油周期", models=("LW500HV-GIV",)),
        RetrievalTask(query="XE155D 发动机油周期", models=("XE155D",)),
    ]
    first = {"chunks": [_chunk("a0", 0.9, "manual-a")]}
    second = {"chunks": [_chunk("b0", 0.8, "manual-b")]}

    merged = merge_task_results([first, second], top_n=2, tasks=tasks)

    assert merged["chunks"][0]["retrieval_tasks"] == [
        {"query": "LW500HV-GIV 发动机油周期", "models": ["LW500HV-GIV"], "knowledge_types": []}
    ]
    assert merged["chunks"][1]["retrieval_tasks"] == [
        {"query": "XE155D 发动机油周期", "models": ["XE155D"], "knowledge_types": []}
    ]


def test_merge_keeps_all_task_labels_when_a_chunk_is_shared():
    tasks = [
        RetrievalTask(query="XC975EV 操作", models=("XC975EV",), knowledge_types=("操作",)),
        RetrievalTask(query="XC975EV 维修", models=("XC975EV",), knowledge_types=("维修",)),
    ]
    shared = _chunk("shared", 0.9, "shared-manual")

    merged = merge_task_results([{"chunks": [shared]}, {"chunks": [shared]}], top_n=2, tasks=tasks)

    assert merged["total"] == 1
    assert [item["knowledge_types"] for item in merged["chunks"][0]["retrieval_tasks"]] == [["操作"], ["维修"]]


@pytest.mark.asyncio
async def test_retrieves_each_task_independently_before_merging():
    tasks = [
        RetrievalTask(query="XC938U 操作", models=("XC938U",), knowledge_types=("操作",)),
        RetrievalTask(query="XC975EV 维修", models=("XC975EV", "XC975EU"), knowledge_types=("维修",)),
    ]
    calls = []

    async def fake_retrieve(task, task_top_n):
        calls.append((task.query, task_top_n))
        prefix = "a" if task.models == ("XC938U",) else "b"
        return {"chunks": [_chunk(f"{prefix}{index}", 1 - index / 10, f"manual-{prefix}") for index in range(task_top_n)]}

    merged = await retrieve_and_merge_tasks(tasks, top_n=6, rerank_candidates_count=64, retrieve_task=fake_retrieve)

    assert calls == [("XC938U 操作", 6), ("XC975EV 维修", 6)]
    assert [chunk["chunk_id"] for chunk in merged["chunks"]] == ["a0", "a1", "a2", "b0", "b1", "b2"]
    assert all(chunk["retrieval_tasks"] for chunk in merged["chunks"])
