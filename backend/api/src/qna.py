import time
from typing import Tuple, List

from api.services.singleton import Singleton
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
                self.config["VECTORSTORE_DB_NAME"]
            ][self.config["VECTORSTORE_COLLECTION_NAME"]]
        except KeyError as e:
            raise ValueError("Config is missing a required key") from e
        except Exception as e:
            raise RuntimeError("Failed to initialize QnA system") from e

    def ask_question(
        self,
        query,
        session_id,
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
            self.init_vectorstore()

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
                    database_name="cohere_chat_history",
                    collection_name="chat_histories",
                )

            PROMPT = PromptTemplate(
                template=self.config["PROMPT_TEMPLATE"],
                input_variables=["chat_history", "context", "question"],
            )
            memory = ConversationBufferWindowMemory(
                memory_key=memory_key,
                input_key="question",
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
            )
            response = qa.invoke({"question": query})
            result = response["answer"]
            source_documents = response["source_documents"]
            exec_time = time.time() - start_time

            return result, source_documents
        except Exception as e:
            raise RuntimeError("Failed to ask question!") from e

    def init_vectorstore(self):
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
                        "fetch_k": 20,
                        "k": self.config["TOP_K"],
                    },
                ),
            )
        except Exception as e:
            raise RuntimeError("Failed to initialize vector store") from e
