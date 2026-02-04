from __future__ import annotations

import typer

from .stages import (
    stage2_page_elements_v3,
    stage3_graphic_elements_v1,
    stage4_table_structure_v1,
    stage5_nemotron_ocr_v1,
    stage6_embeddings,
    stage7_vdb_upload,
    stage8_recall,
    report_stage_outputs,
    stage999_post_mortem_analysis,
)

app = typer.Typer(help="Simplest pipeline with limited CPU parallelism while using maximum GPU possible")
app.add_typer(stage2_page_elements_v3.app, name="stage2")
app.add_typer(stage3_graphic_elements_v1.app, name="stage3")
app.add_typer(stage4_table_structure_v1.app, name="stage4")
app.add_typer(stage5_nemotron_ocr_v1.app, name="stage5")
app.add_typer(stage6_embeddings.app, name="stage6")
app.add_typer(stage7_vdb_upload.app, name="stage7")
app.add_typer(stage8_recall.app, name="stage8")
app.add_typer(report_stage_outputs.app, name="report")
app.add_typer(stage999_post_mortem_analysis.app, name="stage999")


def main():
    app()
