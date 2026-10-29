import pytest

from rag.flow.document_metadata.metadata import attach_document_metadata, extract_document_metadata


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "徐工铲运XC938U装载机控制系统使用说明书（工程师版）.docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC938U"],
                "document_type": "控制系统使用说明书",
                "knowledge_types": ["操作"],
            },
        ),
        (
            "操保手册(中文)-253910058-975(EU)EV.AX01宁德+微特利+湿桥+电液比例+空调+小高卸+20260405田建华（新增驾驶员安全操作规定）.docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC975EV", "XC975EU"],
                "model_aliases": ["XC975-EV", "XC975-EU", "975(EU)EV.AX01"],
                "document_type": "操作与保养手册",
                "knowledge_types": ["操作", "保养"],
            },
        ),
        (
            "XC975-EV（253913260、253913261,253912715,253912722）中文操作与保养手册刘浩202608.docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC975EV", "XC975EU"],
                "model_aliases": ["XC975-EV", "XC975-EU", "975(EU)EV.AX01"],
                "document_type": "操作与保养手册",
                "knowledge_types": ["操作", "保养"],
            },
        ),
        (
            "徐工铲运XC938U装载机控制系统使用说明书（客户版）20260813.docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC938U"],
                "document_type": "控制系统使用说明书",
                "knowledge_types": ["操作"],
            },
        ),
        (
            "XC975EV-欧洲产品售后技术文件（整机拆装、保养、故障维修）湿桥朱亮.docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC975EV", "XC975EU"],
                "model_aliases": ["XC975-EV", "XC975-EU", "975(EU)EV.AX01"],
                "document_type": "售后技术文件",
                "knowledge_types": ["保养", "拆装", "故障维修"],
            },
        ),
        (
            "XC975-EV(253913261宁德423kWh+微特利+干桥)维修保养手册刘浩20260824.doc",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC975EV", "XC975EU"],
                "model_aliases": ["XC975-EV", "XC975-EU", "975(EU)EV.AX01"],
                "document_type": "维修保养手册",
                "knowledge_types": ["维修", "保养"],
            },
        ),
        (
            "徐工铲运XC975UEV装载机显示屏使用说明书（客户版）(1).docx",
            {
                "brand": "徐工",
                "equipment_type": "轮胎式装载机",
                "models": ["XC975UEV"],
                "document_type": "显示屏使用说明书",
                "knowledge_types": ["操作"],
            },
        ),
    ],
)
def test_extracts_metadata_from_current_business_filenames(filename, expected):
    assert extract_document_metadata(filename) == expected


def test_omits_unavailable_fields_and_internal_numbers():
    assert extract_document_metadata("253913261刘浩20260824.docx") == {"brand": "徐工"}


def test_keeps_unknown_model_without_guessing_aliases():
    assert extract_document_metadata("XC999D维修手册.docx") == {
        "brand": "徐工",
        "models": ["XC999D"],
        "document_type": "维修手册",
        "knowledge_types": ["维修"],
    }


def test_attaches_metadata_only_to_first_chunk_without_mutating_input():
    chunks = [{"text": "first"}, {"text": "second"}]

    output = attach_document_metadata(chunks, "XC938U装载机操作手册.docx")

    assert output[0]["metadata"] == {
        "brand": "徐工",
        "equipment_type": "轮胎式装载机",
        "models": ["XC938U"],
        "document_type": "操作手册",
        "knowledge_types": ["操作"],
    }
    assert "metadata" not in output[1]
    assert all("metadata" not in chunk for chunk in chunks)
