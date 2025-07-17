from abc import ABC, abstractmethod


class VDB(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @abstractmethod
    def create_index(self, **kwargs):
        pass

    @abstractmethod
    def write_to_index(self, records: list, **kwargs):
        pass

    @abstractmethod
    def retrieval(self, queries: list, **kwargs):
        pass

    @abstractmethod
    def run(self, records):
        pass

    def reindex(self, records: list, **kwargs):
        pass
