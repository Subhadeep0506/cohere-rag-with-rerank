import time
import src.constants as constant
import src.config as cfg

from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


class Ingestion:
    def __init__(self):
        self.text_vectorstore = None

        self.embeddings = CohereEmbeddings(
            model=cfg.COHERE_EMBEDDING_MODEL_NAME,
            cohere_api_key=cfg.API_KEY,
        )

        self.mongo_client = MongoClient(cfg.MONGO_URI)
        self.MONGODB_COLLECTION = self.mongo_client[cfg.VECTORSTORE_DB_NAME][
            cfg.VECTORSTORE_COLLECTION_NAME
        ]

    def create_and_add_embeddings(
        self,
        file_path: str,
    ):
        # self.text_vectorstore = DeepLake(
        #     dataset_path=cfg.DEEPLAKE_VECTORSTORE,
        #     embedding=self.embeddings,
        #     verbose=False,
        #     num_workers=4,
        # )

        self.text_vectorstore = MongoDBAtlasVectorSearch(
            collection=self.MONGODB_COLLECTION, embedding=self.embeddings
        )

        loader = PyPDFLoader(file_path=file_path)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
        )
        pages = loader.load()
        chunks = text_splitter.split_documents(pages)

        _ = self.text_vectorstore.add_documents(documents=chunks)
