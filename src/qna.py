import time
import src.constants as constant
import src.config as cfg

from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores.deeplake import DeepLake
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_cohere import ChatCohere
from langchain.memory.chat_message_histories.sql import SQLChatMessageHistory
from langchain.memory.chat_message_histories.mongodb import MongoDBChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.callbacks import StreamingStdOutCallbackHandler


class QnA:
    def __init__(self):
        self.embeddings = CohereEmbeddings(
            model=cfg.COHERE_EMBEDDING_MODEL_NAME,
            cohere_api_key=cfg.API_KEY,
        )
        self.model = ChatCohere(
            model=cfg.COHERE_MODEL_NAME,
            cohere_api_key=cfg.API_KEY,
            temperature=constant.TEMPERATURE,
            streaming=True,
        )
        self.cohere_rerank = CohereRerank(
            cohere_api_key=cfg.API_KEY,
            model=cfg.COHERE_RERANK_MODEL_NAME,
        )
        self.text_vectorstore = None
        self.text_retriever = None

        self.mongo_client = MongoClient(cfg.MONGO_URI)
        self.MONGODB_COLLECTION = self.mongo_client[cfg.VECTORSTORE_DB_NAME][cfg.VECTORSTORE_COLLECTION_NAME]

    async def ask_question(
        self,
        query,
        session_id,
        verbose: bool = False,
        callbacks=[StreamingStdOutCallbackHandler]
    ):
        start_time = time.time()
        self.init_vectorstore()

        memory_key = "chat_history"
        # history = SQLChatMessageHistory(
        #     session_id=session_id,
        #     connection_string="sqlite:///memory.db",
        # )

        history = MongoDBChatMessageHistory(
            session_id=session_id,
            connection_string=cfg.MONGO_URI,
            database_name="cohere_chat_history",
            collection_name="chat_histories"
        )

        PROMPT = PromptTemplate(
            template=constant.PROMPT_TEMPLATE,
            input_variables=["chat_history", "context", "question"],
        )
        memory = ConversationBufferWindowMemory(
            memory_key=memory_key,
            output_key="answer",
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
            callbacks=callbacks,
            return_source_documents=True,
        )
        response = await qa.ainvoke({"question": query}, callbacks=callbacks)
        exec_time = time.time() - start_time

        return response

    def init_vectorstore(self):
        # self.text_vectorstore = DeepLake(
        #     dataset_path=cfg.DEEPLAKE_VECTORSTORE,
        #     embedding=self.embeddings,
        #     verbose=False,
        #     read_only=True,
        #     num_workers=4,
        # )

        self.text_vectorstore = MongoDBAtlasVectorSearch(
            collection=self.MONGODB_COLLECTION,
            embedding=self.embeddings,
            index_name="vector_index"
        )

        self.text_retriever = ContextualCompressionRetriever(
            base_compressor=self.cohere_rerank,
            base_retriever=self.text_vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "fetch_k": 20,
                    "k": constant.TOP_K,
                },
            ),
        )
