from typing import Iterable


class Index:
    def __init__(self):
        pass

    def index_documents(self, documets: Iterable[dict[str, str]]):
        raise NotImplementedError()

    def get_document(self, doc_id: str) -> dict[str, str]:
        raise NotImplementedError()

class SearchEngine:
    def __init__(self, index: Index):
        self.index = index

    def search(self, query: str) -> list[tuple[str, float]]:
        raise NotImplementedError()

    def boolean_search(self, query: str) -> set[str]:
        raise NotImplementedError()


