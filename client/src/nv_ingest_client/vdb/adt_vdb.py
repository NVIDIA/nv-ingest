from abc import ABC, abstractmethod


class VDB(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_index(self, collection_name: str):
        pass

    @abstractmethod
    def write_to_index(self, collection_name: str, schema: dict):
        pass

    @abstractmethod
    def search_index(self, collection_name: str, records: list):
        pass

    @abstractmethod
    def reindex(self, collection_name: str, records: list):
        pass

    @abstractmethod
    def run(self, records):
        pass
