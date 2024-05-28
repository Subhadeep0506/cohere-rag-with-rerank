import time
from typing import Tuple, List

from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import ChatCohere
from langchain.vectorstores.deeplake import DeepLake
from langchain.memory.chat_message_histories.sql import SQLChatMessageHistory
from langchain.memory.chat_message_histories.mongodb import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from ..services.singleton import Singleton


class QnA(metaclass=Singleton):
    def __init__(self, config: dict):
        """
        Initialize the QnA class with the given configuration.

        Args:
            config (dict): A dictionary containing the configuration for the QnA system.
        """
        try:
            self.config = config
            self.embeddings = CohereEmbeddings(
                model=self.config["COHERE_EMBEDDING_MODEL_NAME"],
                cohere_api_key=self.config["API_KEY"],
            )
            self.model = ChatCohere(
                model=self.config["COHERE_MODEL_NAME"],
                cohere_api_key=self.config["API_KEY"],
                temperature=self.config["TEMPERATURE"],
            )
            self.cohere_rerank = CohereRerank(
                cohere_api_key=self.config["API_KEY"],
                model=self.config["COHERE_RERANK_MODEL_NAME"],
            )
            self.text_vectorstore = None
            self.text_retriever = None

            self.mongo_client = MongoClient(self.config["MONGO_URI"])
            self.MONGODB_COLLECTION = self.mongo_client[
                self.config["MONGO_VECTORSTORE_DB_NAME"]
            ][self.config["MONGO_VECTORSTORE_COLLECTION_NAME"]]

        except KeyError as e:
            raise ValueError(f"Config is missing a required key. ERROR: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize QnA system. ERROR: {e}")

    def ask_question(
        self,
        query: str,
        filters: dict,
        session_id: str,
        verbose: bool = False,
    ) -> Tuple[str, List[Document]]:
        """
        Ask a question to the QnA system and return the answer.

        Args:
            query (str): The question to ask.
            session_id (str): The ID of the current session.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.

        Returns:
            str: The answer to the question.
        """
        start_time = time.time()
        try:
            self.init_vectorstore(filters=filters)

            memory_key = "chat_history"

            if self.config["DEBUG"]:
                history = SQLChatMessageHistory(
                    session_id=session_id,
                    connection_string="sqlite:///memory.db",
                )
            else:
                history = MongoDBChatMessageHistory(
                    session_id=session_id,
                    connection_string=self.config["MONGO_URI"],
                    database_name=self.config["MONGO_CHAT_HISTORY_DATABASE"],
                    collection_name=self.config["MONGO_CHAT_HISTORY_COLLECTION"],
                )

            PROMPT = PromptTemplate(
                template=self.config["PROMPT_TEMPLATE"],
                input_variables=["chat_history", "context", "question"],
            )
            memory = ConversationBufferWindowMemory(
                memory_key=memory_key,
                input_key="question",
                output_key="answer",
                chat_memory=history,
                k=2,
                return_messages=True,
            )
            chain_type_kwargs = {"prompt": PROMPT}
            qa = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                combine_docs_chain_kwargs=chain_type_kwargs,
                retriever=self.text_retriever,
                verbose=verbose,
                memory=memory,
                return_source_documents=True,
                chain_type="stuff",
            )
            response = qa.invoke({"question": query})
            result, source_documents = response["answer"], response["source_documents"]
            exec_time = time.time() - start_time

            _temp = []
            for doc in source_documents:
                _metadata = {k: v for k, v in doc.metadata.items() if k != "_id"}
                _temp.append(
                    Document(page_content=doc.page_content, metadata=_metadata)
                )
            source_documents = _temp.copy()
            del _temp
            return result, source_documents
        except Exception as e:
            raise RuntimeError(f"Failed to ask question! ERROR: {e}")

    def init_vectorstore(self, filters: dict):
        """
        Initialize the vector store for the QnA system.
        """
        try:
            if self.config["DEBUG"]:
                self.text_vectorstore = DeepLake(
                    dataset_path=self.config["DEEPLAKE_VECTORSTORE"],
                    embedding=self.embeddings,
                    verbose=False,
                    read_only=True,
                    num_workers=4,
                )
            else:
                self.text_vectorstore = MongoDBAtlasVectorSearch(
                    collection=self.MONGODB_COLLECTION,
                    embedding=self.embeddings,
                    index_name="vector_index",
                )

            self.text_retriever = ContextualCompressionRetriever(
                base_compressor=self.cohere_rerank,
                base_retriever=self.text_vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "fetch_k": 25,
                        "k": self.config["TOP_K"],
                        "pre_filter": self.format_filter(filters),
                    },
                ),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vector store. ERROR: {e}")

    def format_filter(self, filter: dict):
        """Format filter dictionary accoding to MongoDB Atlas Vector store format:

        Original:
        ```python
        filter = {
            "category": "category",
            "page": "page"
        }
        ```
        Formatted:
        ```python
        filter = {
            "category": {"$eq": "category"},
            "page": {"$eq": "page"}
        }
        ```

        Args:
            filter (dict): Original input filter.
        """
        _filter = {"$and": [{k: {"$eq": v} for k, v in filter.items()}]}
        print(_filter)
        return _filter

    def clear_chat_history(self):
        try:
            MONGO_CHAT_HISTORY_STORE = self.mongo_client[
                self.config["MONGO_CHAT_HISTORY_DATABASE"]
            ][self.config["MONGO_CHAT_HISTORY_COLLECTION"]]

            MONGO_CHAT_HISTORY_STORE.drop()
        except Exception as e:
            raise Exception(f"Could not delete history. ERROR: {e}")
