from fastapi import APIRouter, Depends, HTTPException, status
from models.query_model import QueryModel
from ..utils.utils import read_config
from ..utils.logger import logger
from ..src.qna import QnA

router = APIRouter()


@router.post(path="/query")
async def run_query(
    query: QueryModel,
    config=Depends(lambda: read_config()),
):
    try:
        qna = QnA(config=config)
        response, source_documents = qna.ask_question(
            query=query.query,
            session_id=query.session_id,
            filters=query.filters,
            verbose=False,
        )
        return {
            "query": query,
            "response": response,
            "source_documents": source_documents,
        }
    except ValueError as e:
        logger.error(f"Invalid query. ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid query",
        ) from e
    except Exception as e:
        logger.error(f"Invalid query. ERROR: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        ) from e


@router.get("/query/delete_history")
def delete_chat_history(
    config=Depends(lambda: read_config()),
):
    try:
        qna = QnA(config=config)
        qna.clear_chat_history()
        return {"message": "Chat history deleted."}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=e)
