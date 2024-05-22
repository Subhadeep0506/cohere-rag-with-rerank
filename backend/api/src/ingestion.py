from io import BytesIO
from api.services.singleton import Singleton
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.document_loaders.pdf import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from pypdf.errors import PdfStreamError


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

    async def create_and_add_embeddings(
        self,
        file: str,
    ):
        try:
            if self.config["DEBUG"]:
                self.text_vectorstore = DeepLake(
                    dataset_path=self.config["DEEPLAKE_VECTORSTORE"],
                    embedding=self.embeddings,
                    verbose=False,
                    num_workers=4,
                )
            else:
                self.text_vectorstore = MongoDBAtlasVectorSearch(
                    collection=self.MONGODB_COLLECTION, embedding=self.embeddings
                )

            loader = PyPDFLoader(file_path=file)

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
            )
            pages = await loader.aload()
            chunks = text_splitter.split_documents(pages)

            return await self.text_vectorstore.aadd_documents(documents=chunks)
        except PdfStreamError as e:
            raise Exception(e)
        except Exception as e:
            raise Exception(e)
