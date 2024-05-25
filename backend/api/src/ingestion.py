import datetime

from api.services.singleton import Singleton
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain_mongodb import MongoDBAtlasVectorSearch
from api.src.doc_loaders import PDFLoader, TxtLoader


class Ingestion(metaclass=Singleton):
    def __init__(self, config: dict):
        self.config = config
        self.text_vectorstore = None

        self.embeddings = CohereEmbeddings(
            model=self.config["COHERE_EMBEDDING_MODEL_NAME"],
            cohere_api_key=self.config["API_KEY"],
        )

        self.mongo_client = MongoClient(self.config["MONGO_URI"])
        self.MONGODB_VECTORSTORE_COLLECTION = self.mongo_client[
            self.config["MONGO_VECTORSTORE_DB_NAME"]
        ][self.config["MONGO_VECTORSTORE_COLLECTION_NAME"]]
        self.MONGODB_FILES_METADATA_COLLECTION = self.mongo_client[
            self.config["MONGO_VECTORSTORE_DB_NAME"]
        ][self.config["MONGO_FILES_METADATA_COLLECTION_NAME"]]

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
                    collection=self.MONGODB_VECTORSTORE_COLLECTION,
                    embedding=self.embeddings,
                )
        except Exception as e:
            raise Exception(e)

    def _add_file_info(self, file_info: dict):
        try:
            file_metadata_database_collection = self.MONGODB_FILES_METADATA_COLLECTION
            result = file_metadata_database_collection.insert_one(file_info)
            return {
                "inserted_id": str(result.inserted_id),
                "acknowledged": result.acknowledged,
            }
        except Exception as e:
            raise Exception(e)

    async def create_and_add_embeddings(
        self, file: str, metadata: dict, file_type: str
    ):
        try:
            if file_type == "application/pdf":
                loader = PDFLoader(
                    file_path=file,
                    metadata=metadata,
                    config=self.config,
                )
            elif file_type == "text/plain":
                loader = TxtLoader(
                    file_path=file,
                    metadata=metadata,
                    config=self.config,
                )
            chunks = await loader.load_document()
            _chunks = chunks[0]
            file_info = {
                "file_name": _chunks.metadata["file_name"],
                "metadata": metadata,
                "date_added": str(datetime.datetime.now()),
            }
            file_info_upload_result = self._add_file_info(file_info=file_info)
            return (
                await self.text_vectorstore.aadd_documents(documents=chunks),
                file_info_upload_result,
            )
        except Exception as e:
            raise Exception(e)

    def delete_from_vectorstore(self, filter: dict):
        try:
            if self.config["DEBUG"]:
                result = self.text_vectorstore.delete(filter={"metadata": {**filter}})
                return {"deleted": result}
            else:
                result = self.text_vectorstore._collection.delete_many(filter=filter)
                return {
                    "deleted_count": result.deleted_count,
                    "acknowledged": result.acknowledged,
                }
        except Exception as e:
            raise Exception(e)

    def list_documents_from_vectorstore(self):
        files_list = []
        try:
            if self.config["DEBUG"]:
                result = self.text_vectorstore.vectorstore.metadata.data()["value"]
                print(result)
                return {"message": "success!"}
            else:
                _cursor = self.MONGODB_FILES_METADATA_COLLECTION.find({})
                for item in _cursor:
                    item["_id"] = str(item["_id"])
                    files_list.append(item)
                return {"files": files_list}
        except Exception as e:
            raise Exception(e)
