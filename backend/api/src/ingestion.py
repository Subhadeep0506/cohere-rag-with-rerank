import datetime
import os

from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain_mongodb import MongoDBAtlasVectorSearch
from ..services.singleton import Singleton
from ..src.doc_loaders import PDFLoader, TxtLoader


class Ingestion(metaclass=Singleton):
    """Document Ingestion pipeline."""

    def __init__(self, config: dict):
        """
        Initialize the Ingestion pipeline.

        This method initializes the Ingestion pipeline by setting up the necessary components,
        including the text vectorstore, MongoDB client, and Cohere embeddings.

        Args:
            config (dict): A dictionary containing configuration settings loaded from a YAML file.

        Raises:
            Exception: An exception is raised if there is an error initializing the text vectorstore.
        """
        try:
            self.config = config
            self.text_vectorstore = None

            self.embeddings = CohereEmbeddings(
                model=self.config["COHERE_EMBEDDING_MODEL_NAME"],
                cohere_api_key=self.config.get("API_KEY", os.environ("API_KEY")),
            )

            self.mongo_client = MongoClient(
                self.config.get("MONGO_URI", os.environ("MONGO_URI"))
            )
            self.MONGODB_VECTORSTORE_COLLECTION = self.mongo_client[
                self.config["MONGO_VECTORSTORE_DB_NAME"]
            ][self.config["MONGO_VECTORSTORE_COLLECTION_NAME"]]
            self.MONGODB_FILES_METADATA_COLLECTION = self.mongo_client[
                self.config["MONGO_VECTORSTORE_DB_NAME"]
            ][self.config["MONGO_FILES_METADATA_COLLECTION_NAME"]]

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
        except KeyError as e:
            raise ValueError(f"Config is missing a required key. ERROR: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QnA system. ERROR: {e}")

    def _add_file_info(self, file_info: dict):
        """
        Adds file metadata to the MongoDB collection.

        This method takes a dictionary containing file metadata and inserts it into the MongoDB
        collection specified in the configuration. It returns a dictionary containing the
        inserted ID and a boolean indicating whether the insertion was acknowledged.

        Args:
            file_info (dict): A dictionary containing file metadata.

        Returns:
            dict: A dictionary containing the inserted ID and a boolean indicating whether the
                insertion was acknowledged.

        Raises:
            Exception: An exception is raised if there is an error inserting the file metadata
                into the MongoDB collection.
        """
        try:
            file_metadata_database_collection = self.MONGODB_FILES_METADATA_COLLECTION
            result = file_metadata_database_collection.insert_one(file_info)
            return {
                "inserted_id": str(result.inserted_id),
                "acknowledged": result.acknowledged,
            }
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            raise Exception(f"ERROR: {e}")

    async def create_and_add_embeddings(
        self, file: str, metadata: dict, file_type: str
    ):
        """
        Creates and adds embeddings for a file to the vectorstore.

        This method loads a file using the appropriate loader (PDF or TXT), creates embeddings for the file,
        and adds them to the vectorstore. It also adds the file metadata to the MongoDB collection.

        Args:
            file (str): The path to the file to be ingested.
            metadata (dict): A dictionary containing metadata about the file.
            file_type (str): The type of the file (e.g. application/pdf or text/plain).

        Returns:
            tuple: A tuple containing the result of adding the embeddings to the vectorstore and the result of adding the file metadata to the MongoDB collection.

        Raises:
            Exception: An exception is raised if there is an error creating or adding the embeddings.
        """
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
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            raise Exception(f"ERROR: {e}")

    def delete_from_vectorstore(self, filter: dict):
        """
        Deletes documents from the vectorstore based on a filter.

        This method deletes documents from the vectorstore that match the provided filter.
        In debug mode, it uses the DeepLake vectorstore, and in production mode, it uses the MongoDB Atlas vectorstore.

        Args:
            filter (dict): A dictionary containing the filter criteria for deleting documents.

        Returns:
            dict: A dictionary containing the result of the deletion operation.

        Raises:
            Exception: An exception is raised if there is an error deleting documents from the vectorstore.
        """
        try:
            if self.config["DEBUG"]:
                result = self.text_vectorstore.delete(filter={"metadata": {**filter}})
                return {"deleted": result}
            else:
                result = self.text_vectorstore._collection.delete_many(filter=filter)
                _ = self.MONGODB_FILES_METADATA_COLLECTION.delete_one(filter=filter)
                return {
                    "deleted_count": result.deleted_count,
                    "acknowledged": result.acknowledged,
                }
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            raise Exception(f"ERROR: {e}")

    def list_documents_from_vectorstore(self):
        """
        Lists all documents in the vectorstore.

        This method retrieves a list of all documents in the vectorstore.
        In debug mode, it uses the DeepLake vectorstore, and in production mode, it uses the MongoDB Atlas vectorstore.

        Returns:
            dict: A dictionary containing a list of documents in the vectorstore.

        Raises:
            Exception: An exception is raised if there is an error retrieving documents from the vectorstore.
        """
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
        except (ValueError, RuntimeError, KeyError, TypeError) as e:
            raise Exception(f"ERROR: {e}")
