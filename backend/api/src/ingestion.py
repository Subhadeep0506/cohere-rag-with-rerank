from api.services.singleton import Singleton
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter


class Ingestion(metaclass=Singleton):
    def __init__(self, config: dict):
        print(f"Initializing Ingestion Id: {self}")
        self.config = config
        self.text_vectorstore = None

        self.embeddings = CohereEmbeddings(
            model=self.config["COHERE_EMBEDDING_MODEL_NAME"],
            cohere_api_key=self.config["API_KEY"],
        )

        self.mongo_client = MongoClient(self.config["MONGO_URI"])
        self.MONGODB_COLLECTION = self.mongo_client[self.config["VECTORSTORE_DB_NAME"]][
            self.config["VECTORSTORE_COLLECTION_NAME"]
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
