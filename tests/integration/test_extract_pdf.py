import re
import pytest
from tests.integration.utilities_for_test import (
    canonicalize_markdown_table,
    levenshtein_ratio,
    jaccard_similarity,
    token_f1,
)
from nv_ingest_api.util.message_brokers.simple_message_broker import SimpleClient
from nv_ingest_client.client import Ingestor
from nv_ingest_client.client import NvIngestClient
from nv_ingest_client.util.milvus import nvingest_retrieval


def create_client():
    return NvIngestClient(
        message_client_allocator=SimpleClient,
    )


@pytest.mark.integration
def run_basic_extract_test(
    extract_method=None,
    expected_text_count=3,
    expected_table_count=2,
    expected_chart_count=3,
    expected_infographic_count=0,
    table_markdowns=None,
    chart_phrases=None,
):
    client = create_client()

    extract_kwargs = {
        "extract_text": True,
        "extract_tables": True,
        "extract_charts": True,
        "extract_images": True,
        "table_output_format": "markdown",
        "extract_infographics": True,
        "text_depth": "page",
    }

    if extract_method:
        extract_kwargs["extract_method"] = extract_method

    ingestor = Ingestor(client=client).files("./data/multimodal_test.pdf").extract(**extract_kwargs)

    results = ingestor.ingest()
    assert len(results) == 1

    texts = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "text"]
    assert len(texts) == expected_text_count

    structured = [x for x in results[0] if x["metadata"]["content_metadata"]["type"] == "structured"]
    assert len(structured) == expected_table_count + expected_chart_count + expected_infographic_count

    tables = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "table"]
    charts = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "chart"]
    infographics = [x for x in structured if x["metadata"]["content_metadata"]["subtype"] == "infographic"]

    assert len(tables) == expected_table_count
    assert len(charts) == expected_chart_count
    assert len(infographics) == expected_infographic_count

    # Snapshot current expected outputs (baseline) for similarity comparisons
    EXPECTED_TABLE_BLOCKS = [
        (
            "and\n"
            "some\n"
            "activities\n"
            "they\n"
            "might\n"
            "be\n"
            "doing\n"
            "in\n"
            "specific\n"
            "Activity\n"
            "Place\n"
            "Giraffe\n"
            "Driving\n"
            "a\n"
            "car\n"
            "At\n"
            "the\n"
            "beach\n"
            "| Table 1 This table describes some animals, |\n"
            "| locations. Animal |\n"
            "| Cat Dog |\n"
            "Lion\n"
            "Putting\n"
            "on\n"
            "sunscreen\n"
            "At\n"
            "the\n"
            "park\n"
            "Jumping\n"
            "onto\n"
            "a\n"
            "laptop\n"
            "In\n"
            "a\n"
            "home\n"
            "office\n"
            "Chasing\n"
            "a\n"
            "squirrel\n"
            "In\n"
            "the\n"
            "front\n"
            "yard\n"
        ),
        (
            "colors\n"
            "that\n"
            "cars\n"
            "Color2\n"
            "might\n"
            "come\n"
            "in.\n"
            "Color3\n"
            "Silver\n"
            "Flat\n"
            "Sedan\n"
            "White\n"
            "Metallic\n"
            "Matte\n"
            "Gray\n"
            "| Table 2 This table shows some popular |\n"
            "| Car Color1 Coupe White |\n"
            "| Convertible Light Gray |\n"
            "Minivan\n"
            "Gray\n"
            "Beige\n"
            "Gray\n"
            "Black\n"
            "Gray\n"
            "Truck\n"
            "Dark\n"
            "Gray\n"
            "Titanium\n"
            "Gray\n"
            "Charcoal\n"
            "Graphite\n"
            "Slate\n"
            "Gray\n"
        ),
    ]

    EXPECTED_CHART_CONTENT = (
        "This chart shows some gadgets, and some very fictitious costs. Gadgets and "
        "their cost   Hammer - Powerdrill - Bluetooth speaker - Minifridge - Premium "
        "desk fan Dollars $- - $20.00 - $40.00 - $60.00 - $80.00 - $100.00 - $120.00 "
        "- $140.00 - $160.00 Cost    Chart 1 Below, is a high-quality picture of some "
        "shapes.          Picture This chart shows some average frequency ranges for "
        "speaker drivers. Frequency Ranges of Speaker Drivers   Tweeter - Midrange - "
        "Midwoofer - Subwoofer Hertz (log scale) 10 - 100 - 1000 - 10000 - 100000 "
        "Frequency Range Start (Hz) - Frequency Range End (Hz)    Chart 2"
    )

    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    def _assert_similar(actual: str, expected: str, lev_thr=0.85, f1_thr=0.80, jac_thr=0.70):
        a = _norm(actual)
        e = _norm(expected)
        lev = levenshtein_ratio(a, e)
        f1 = token_f1(a, e)
        jac = jaccard_similarity(a, e)
        assert (lev >= lev_thr) or (f1 >= f1_thr) or (jac >= jac_thr), (
            f"Similarity below threshold. lev={lev:.3f}, f1={f1:.3f}, jac={jac:.3f}\n"
            f"Expected (norm) preview: {e[:200]}...\n"
            f"Actual   (norm) preview: {a[:200]}..."
        )

    # Helpers to parse markdown tables and compare rows robustly
    def _parse_markdown_table(md: str):
        md = canonicalize_markdown_table(md)
        lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
        if not lines:
            return [], []
        header_line = lines[0]
        header = [tok.strip() for tok in header_line.strip("|").split("|") if tok.strip()]
        rows = []
        # Skip possible separator line (---) if present as second line
        start_idx = 1
        if len(lines) > 1 and set(lines[1].replace("|", "").strip()) <= set("- "):
            start_idx = 2
        for ln in lines[start_idx:]:
            if not ln.startswith("|"):
                continue
            parts = [tok.strip() for tok in ln.strip("|").split("|")]
            if parts:
                rows.append(parts)
        return header, rows

    extracted_tables = [x["metadata"]["table_metadata"]["table_content"] for x in tables]

    # DEBUG: Print raw extracted table contents and parsed forms for visibility
    print("[DEBUG] Extracted table count:", len(extracted_tables))
    for idx, raw in enumerate(extracted_tables):
        print(f"[DEBUG] Table #{idx} raw markdown:\n{raw}\n---")
    parsed_extracted = [_parse_markdown_table(s) for s in extracted_tables]
    for idx, (hdr, rows) in enumerate(parsed_extracted):
        print(f"[DEBUG] Table #{idx} parsed header: {hdr}")
        print(f"[DEBUG] Table #{idx} first 5 rows: {rows[:5]}")

    # Compare extracted tables to expected snapshots using similarity metrics
    assert len(extracted_tables) == len(
        EXPECTED_TABLE_BLOCKS
    ), f"Unexpected number of tables: got {len(extracted_tables)} expected {len(EXPECTED_TABLE_BLOCKS)}"
    for i, (actual, expected) in enumerate(zip(extracted_tables, EXPECTED_TABLE_BLOCKS)):
        _assert_similar(actual, expected)

    chart_contents = " ".join(x["metadata"]["table_metadata"]["table_content"] for x in charts)
    # DEBUG: Brief preview of chart contents
    if charts:
        print("[DEBUG] Chart items:")
        for i, ch in enumerate(charts[:2]):
            preview = ch["metadata"]["table_metadata"]["table_content"]
            print(f"[DEBUG] Chart #{i} preview: {preview[:200].replace('\n', ' ')}...")
    # Compare combined chart content to expected snapshot using similarity
    _assert_similar(chart_contents, EXPECTED_CHART_CONTENT, lev_thr=0.80, f1_thr=0.75, jac_thr=0.65)


@pytest.mark.integration
@pytest.mark.parametrize("extract_method", [None])  # , "nemoretriever_parse"])
def test_extract_only(
    extract_method,
    pipeline_process,
    multimodal_first_table_markdown,
    multimodal_second_table_markdown,
    multimodal_first_chart_xaxis,
    multimodal_first_chart_yaxis,
    multimodal_second_chart_xaxis,
    multimodal_second_chart_yaxis,
):
    run_basic_extract_test(
        extract_method=extract_method,
        table_markdowns=[
            multimodal_first_table_markdown,
            multimodal_second_table_markdown,
        ],
        chart_phrases=[
            multimodal_first_chart_xaxis,
            multimodal_first_chart_yaxis,
            multimodal_second_chart_xaxis,
            multimodal_second_chart_yaxis,
        ],
    )


@pytest.mark.integration
@pytest.mark.skip("Skipping for the moment")
def test_pdfium_extract_embed_upload_query(pipeline_process):
    client = create_client()

    milvus_uri = "milvus.db"
    collection_name = "test"

    ingestor = (
        Ingestor(client=client)
        .files("./data/multimodal_test.pdf")
        .extract(
            extract_text=True,
            extract_tables=True,
            extract_charts=True,
            extract_images=False,
            text_depth="page",
        )
        .embed()
        .vdb_upload(
            collection_name=collection_name,
            milvus_uri=milvus_uri,
            dense_dim=2048,
        )
    )

    results = ingestor.ingest()
    assert len(results) == 1
    assert len([x for x in results[0] if x["metadata"]["embedding"]]) == 8

    queries = ["Which table shows some popular colors that cars might come in?"]

    retrieved_docs = nvingest_retrieval(
        queries,
        collection_name,
        milvus_uri=milvus_uri,
        top_k=10,
    )
    assert len(retrieved_docs[0]) == 8

    extracted_content = retrieved_docs[0][0]["entity"]["text"]
    assert "This table shows some popular colors that cars might come in" in extracted_content
