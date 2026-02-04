# from cuvs.neighbors import vamana
# import cupy as cp
import numpy as np
from nv_ingest_client.util.vdb.adt_vdb import VDB
import diskannpy
from pathlib import Path
from functools import partial
from nv_ingest_client.util.transport import infer_microservice
from urllib.parse import urlparse
import copy
from typing import List


def make_easy_to_read(search_results: List[List[int]], original_records: List[List[dict]]):
    """
    This function converts search result rankings to the original records

    Parameters:
    ----------
    search_results: List[List[int]]
        List of lists of rankings
    original_records: List[List[dict]]
        List of lists of original records with embeddings
    Returns:
    ----------
    List[List[dict]]
        List of lists of records, search results to records
    """
    full_results = []
    cleaned_records = [
        element for record in original_records for element in record if element["metadata"]["embedding"] is not None
    ]
    for ele_res in search_results:
        query_result = []
        for res in ele_res:
            to_add = copy.deepcopy(cleaned_records[res])
            # make easier to read
            to_add["metadata"].pop("embedding")
            query_result.append(to_add)
        if len(query_result) > 0:
            full_results.append(query_result)
    return full_results


class Vamana(VDB):
    def __init__(
        self,
        metric="l2",
        graph_degree=32,
        visited_size=64,
        vamana_iters=1,
        alpha=1.2,
        max_fraction=0.06,
        batch_base=2.0,
        queue_size=127,
        reverse_batchsize=1000000,
        index_path="./vamana_index/vamana_index",
    ):
        """
        This class is a wrapper for the Vamana index

        Parameters:
        ----------
        metric: str
            Metric to use for the index
        graph_degree: int
            Graph degree for the index
        visited_size: int
            Visited size for the index
        vamana_iters: int
            Vamana iterations for the index
        alpha: float
            Alpha for the index
        max_fraction: float
            Max fraction for the index
        batch_base: float
            Batch base for the index
        queue_size: int
            Queue size for the index
        reverse_batchsize: int
            Reverse batchsize for the index
        index_path: str
            Index path for the index, needs to be a valid path in a folder with no non index files in it.
        """
        self.index = None
        self.metric = metric
        self.graph_degree = graph_degree
        self.visited_size = visited_size
        self.vamana_iters = vamana_iters
        self.alpha = alpha
        self.max_fraction = max_fraction
        self.batch_base = batch_base
        self.queue_size = queue_size
        self.reverse_batchsize = reverse_batchsize
        self.index_path = index_path

    def create_index(self):
        """
        This function creates the index parameters for the Vamana index

        Returns:
        ----------
        vamana.IndexParams
            Index parameters for the Vamana index
        """
        # return vamana.IndexParams(
        #     metric=self.metric,
        #     graph_degree=self.graph_degree,
        #     visited_size=self.visited_size,
        #     vamana_iters=self.vamana_iters,
        #     alpha=self.alpha,
        #     max_fraction=self.max_fraction,
        #     batch_base=self.batch_base,
        #     queue_size=self.queue_size,
        #     reverse_batchsize=self.reverse_batchsize,
        # )
        pass

    def write_to_index(self, records: list, **kwargs):
        """
        This function writes the records to the Vamana index

        Parameters:
        ----------
        records: List[List[dict]]
            List of lists of records
        """
        # strip the records for the embeddings only.
        # embeddings = [
        #     element["metadata"]["embedding"]
        #     for record in records
        #     for element in record
        #     if element["metadata"]["embedding"] is not None
        # ]
        # embeddings = np.array(embeddings, dtype=np.float32)
        # index = vamana.build(self.create_index(), embeddings)
        # vamana.save(self.index_path, index, include_dataset=True)
        # index_path = Path(self.index_path)
        # self.vamana = diskannpy.StaticMemoryIndex(
        #     vector_dtype=np.float32,
        #     index_directory=index_path.parent,
        #     index_prefix=index_path.stem,
        #     initial_search_complexity=135,
        #     num_threads=1,
        #     dimensions=2048,
        #     distance_metric="l2",
        # )

        diskannpy.build_disk_index(
            data=records,
            index_directory=Path(self.index_path).parent,
            index_prefix=Path(self.index_path).stem,
            distance_metric="l2",
            graph_degree=self.graph_degree,
            complexity=75,
            search_memory_maximum=4.0,
            build_memory_maximum=64.0,
            num_threads=0,
        )

    def retrieval(self, queries: list, **kwargs):
        """
        This function retrieves relevant records from the Vamana index based on the queries

        Parameters:
        ----------
        queries: List[str]
            List of queries

        Returns:
        ----------
        List[List[dict]]
            List of lists of records, search results to records
        """
        query_embs = []
        model_name = kwargs.get("model_name", "nvidia/llama-3.2-nv-embedqa-1b-v2")
        embedding_endpoint = kwargs.get("embedding_endpoint", "http://localhost:8012/v1")
        nvidia_api_key = kwargs.get("nvidia_api_key", "nv-ingest-api-key")
        embed_model = partial(
            infer_microservice,
            model_name=model_name,
            embedding_endpoint=embedding_endpoint,
            nvidia_api_key=nvidia_api_key,
            input_type="query",
            output_names=["embeddings"],
            grpc=not (urlparse(embedding_endpoint).scheme == "http"),
        )
        query_embs = embed_model(queries)
        query_embs = np.array(query_embs, dtype=np.float32)
        # comes as a list [[rankings], [scores]], only want the rankings
        query_results = self.index.search(query_embs, k=10, complexity=10)
        results = kwargs.get("results", [])
        return make_easy_to_read(query_results, results)

    def run(self, records):
        self.write_to_index(records)
