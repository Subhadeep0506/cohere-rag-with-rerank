import os

from .loader import Loader
from langchain.schema import Document
from langchain.document_loaders.text import TextLoader
from langchain.text_splitter import CharacterTextSplitter


class TxtLoader(Loader):
    def __init__(self, file_path: str, metadata: dict, config: dict) -> None:
        super().__init__(file_path, metadata, config)

    async def load_document(self):
        self.file_name = os.path.basename(self.file_path)
        loader = TextLoader(self.file_path)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=self.config["PDF_CHARSPLITTER_CHUNKSIZE"],
            chunk_overlap=self.config["PDF_CHARSPLITTER_CHUNK_OVERLAP"],
        )
        pages = await loader.aload()
        total_pages = len(pages)
        chunks = []
        for idx, page in enumerate(pages):
            chunks.append(
                Document(
                    page_content=page.page_content,
                    metadata=dict(
                        {
                            "file_name": self.file_name,
                            "page_no": str(idx + 1),
                            "total_pages": str(total_pages),
                            **self.metadata,
                        }
                    ),
                )
            )

        final_chunks = text_splitter.split_documents(chunks)
        return final_chunks
