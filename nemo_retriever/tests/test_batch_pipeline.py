from nemo_retriever.examples.batch_pipeline import _count_materialized_rows


class _DatasetWithoutLen:
    def count(self) -> int:
        return 42

    def __len__(self) -> int:
        raise AssertionError("__len__ should not be used")


def test_count_materialized_rows_prefers_dataset_count() -> None:
    assert _count_materialized_rows(_DatasetWithoutLen()) == 42
