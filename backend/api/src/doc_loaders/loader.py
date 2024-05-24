from abc import ABC, abstractmethod


class Loader(ABC):
    def __init__(self, file_path: str, metadata: dict, config: dict) -> None:
        self.file_path = file_path
        self.metadata = metadata
        self.config = config

    @abstractmethod
    async def load_document(self):
        pass
