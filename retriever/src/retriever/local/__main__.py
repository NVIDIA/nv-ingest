from __future__ import annotations

import typer

from .stages import (
    stage1_pdf_extraction,
    stage2_infographic_extraction,
    stage3_table_extractor,
    stage4_chart_extractor,
    stage6_embeddings,
    stage7_vdb_upload,
    stage8_recall,
    report_stage_outputs,
    stage999_post_mortem_analysis,
)

app = typer.Typer(help="Simple non-distributed pipeline for local development, debugging, and research.")
app.add_typer(stage1_pdf_extraction.app, name="stage1")
app.add_typer(stage2_infographic_extraction.app, name="stage2")
app.add_typer(stage3_table_extractor.app, name="stage3")
app.add_typer(stage4_chart_extractor.app, name="stage4")
app.add_typer(stage6_embeddings.app, name="stage6")
app.add_typer(stage7_vdb_upload.app, name="stage7")
app.add_typer(stage8_recall.app, name="stage8")
app.add_typer(report_stage_outputs.app, name="report")
app.add_typer(stage999_post_mortem_analysis.app, name="stage999")


def main():
    app()
