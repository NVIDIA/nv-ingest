"""
Batch runmode.

Intended for large-scale batch execution over large inputs on multiple workers.
"""

from __future__ import annotations

import datetime as _dt
import glob
import logging
import os
from typing import Any, List, Optional

from typing import Union

import ray
import ray.data as rd
from retriever.chart.chart_detection import ChartDetectionActor
from retriever.infographic.infographic_detection import InfographicDetectionActor
from retriever.page_elements import PageElementDetectionActor
from retriever.pdf.extract import PDFExtractionActor
from retriever.pdf.split import PDFSplitActor
from retriever.table.table_structure import TableStructureActor

from ..ingest import Ingestor


class BatchIngestor(Ingestor):
    RUN_MODE = "batch"

    def __init__(self, documents: Optional[List[str]] = None, **kwargs: Any) -> None:
        super().__init__(documents=documents, **kwargs)

        logging.basicConfig(level=logging.INFO)

        # Initialize Ray locally for distributed execution.
        # TODO: This should be a configuration .... 
        ray.init(address="local", ignore_reinit_error=True) 

        # Builder-style task configuration recorded for later execution.
        # Keep backwards-compatibility with code that inspects `Ingestor._documents`
        # (older examples/tests) by ensuring both names refer to the same list.
        self._input_documents: List[str] = self._documents  # List of original input documents.
        self._rd_dataset: rd.Dataset = None # Ray Data dataset created from input documents.
        self._tasks: List[tuple[str, dict[str, Any]]] = []
        self._intermediate_output_dir: Optional[str] = None

    def files(self, documents: Union[str, List[str]]) -> "BatchIngestor":
        """
        Add local files for batch processing.

        This runmode assumes all inputs are local. Any glob pattern or explicit
        path must resolve to at least one existing file.
        """
        if isinstance(documents, str):
            documents = [documents]

        for pattern in documents:
            if not isinstance(pattern, str) or not pattern:
                raise ValueError(f"Invalid document pattern: {pattern!r}")

            # Expand globs (supports ** when recursive=True).
            matches = glob.glob(pattern, recursive=True)
            if matches:
                files = [os.path.abspath(p) for p in matches if os.path.isfile(p)]
                if not files:
                    raise FileNotFoundError(f"Pattern resolved, but no files found: {pattern!r}")
                self._input_documents.extend(files)
                continue

            # No glob matches: treat as explicit path.
            if os.path.isfile(pattern):
                self._input_documents.append(os.path.abspath(pattern))
                continue

            raise FileNotFoundError(f"No local files found for: {pattern!r}")

        self._rd_dataset = rd.read_binary_files(self._input_documents, include_paths=True)

        # TODO: Capture number of GPUs and calculate number of workers based on that
        # self._rd_dataset.repartition(num_workers=100) # divide by number of workers and GPUs available

        return self

    def extract(self, **kwargs: Any) -> "BatchIngestor":
        """
        Configure extraction for batch processing (builder only).

        This does not run extraction yet; it records configuration so the batch
        executor can build a concrete pipeline later.
        """
        # TODO: Pass in kwargs in a cleaner manner. This is messy and not compliant with static typing

        # Downstream batch stages assume `page_image.image_b64` exists for every page.
        # Ensure PDF extraction emits a page image unless the caller explicitly disables it.
        kwargs.setdefault("extract_page_as_image", True)

        print(f"Type kwargs: {type(kwargs)}")
        print(f"kwargs: {kwargs}")
        self._tasks.append(("extract", dict(kwargs)))

        # Stage-specific kwargs: upstream PDF stages accept many options (dpi, extract_*),
        # but downstream detect_* Ray actors accept only a small set. Passing the whole
        # dict can cause TypeErrors (e.g. unexpected `method=`).
        detect_passthrough_keys = {
            "inference_batch_size",
            "output_column",
            "num_detections_column",
            "counts_by_label_column",
        }
        detect_kwargs = {k: kwargs[k] for k in detect_passthrough_keys if k in kwargs}

        # Splitting pdfs is broken into a separate stage to help amoritize downstream processing if PDFs have vastly different number of pages
        pdf_split_actor = PDFSplitActor(**kwargs)

        # TODO: Jacob figure out how to get best batch size based on number of pages
        # num_pages = pdfium.PdfDocument(self._rd_dataset[0]["bytes"]).page_count

        self._rd_dataset = self._rd_dataset.map_batches(pdf_split_actor, batch_size=1, num_cpus=4, num_gpus=0, batch_format="pandas")


        # TODO: I think we can split out each of these sub-tasks within the extraction stage to more discrete tasks so that Ray Data can parallelize them better
        
        # Pre-split pdfs are now ready for extraction
        extraction_actor = PDFExtractionActor(**kwargs)
        self._rd_dataset = self._rd_dataset.map_batches(extraction_actor,
            batch_size=32, batch_format="pandas", num_cpus=4, num_gpus=0,
            compute=rd.TaskPoolStrategy(size=int(8))
        )

        # Now lets call PageElements with a GPU actor pool
        # For ActorPoolStrategy, Ray Data expects a *callable class* (so it can
        # construct one instance per actor). Passing an already-constructed
        # callable object is treated as a "regular function" and will fail.
        self._rd_dataset = self._rd_dataset.map_batches(PageElementDetectionActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=rd.ActorPoolStrategy(size=int(8)),
            fn_constructor_kwargs=dict(detect_kwargs),
        )

        # Now lets call TableExtraction with a GPU actor pool
        self._rd_dataset = self._rd_dataset.map_batches(TableStructureActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=rd.ActorPoolStrategy(size=int(8)),
            fn_constructor_kwargs=dict(detect_kwargs),
        )

        # Now lets call ChartDetection with a GPU actor pool
        self._rd_dataset = self._rd_dataset.map_batches(ChartDetectionActor,
            batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
            compute=rd.ActorPoolStrategy(size=int(8)),
            fn_constructor_kwargs=dict(detect_kwargs),
        )

        # Now lets call InfographicDetection with a GPU actor pool (optional)
        if kwargs.get("extract_infographics") is True:
            self._rd_dataset = self._rd_dataset.map_batches(InfographicDetectionActor,
                batch_size=16, batch_format="pandas", num_cpus=4, num_gpus=1,
                compute=rd.ActorPoolStrategy(size=int(8)),
                fn_constructor_kwargs=dict(detect_kwargs),
            )

        return self

    def save_intermediate_results(self, output_dir: str) -> "BatchIngestor":
        """
        Persist the current Ray Dataset to disk under `output_dir`.

        Writes Parquet files (Ray Data's native/efficient on-disk format) so downstream
        steps can reload as a Ray Dataset using `ray.data.read_parquet(...)`.
        """
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError(f"output_dir must be a non-empty string, got {output_dir!r}")
        if self._rd_dataset is None:
            raise RuntimeError("No Ray Dataset to write. Call .files(...) (and optionally .extract(...)) first.")

        base_dir = os.path.abspath(output_dir)
        os.makedirs(base_dir, exist_ok=True)

        # Ray's writers typically expect a directory that does not already contain output.
        # To avoid destructive behavior, if the directory is non-empty we write to a timestamped subdir.
        target_dir = base_dir
        try:
            if os.listdir(base_dir):
                ts = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                target_dir = os.path.join(base_dir, f"ray_dataset_{ts}")
                os.makedirs(target_dir, exist_ok=False)
        except FileNotFoundError:
            # Rare race: directory disappeared between abspath and listdir.
            os.makedirs(base_dir, exist_ok=True)

        # Trigger execution and write results.
        # Parquet supports nested list/struct columns used by our stages (e.g. detections payloads).
        self._rd_dataset.write_parquet(target_dir)
        self._intermediate_output_dir = target_dir
        return self

    # Backwards-compatibility: some examples call `write_to_disk(...)`.
    def write_to_disk(self, output_dir: str) -> "BatchIngestor":
        return self.save_intermediate_results(output_dir=output_dir)

    def ingest(self) -> "BatchIngestor":
        """
        Ingest the documents into the vector store.
        """
        # self._rd_dataset = self._rd_dataset.materialize()
        num_pages = self._rd_dataset.count()
        print(f"Number of pages: {num_pages}")
        return num_pages

