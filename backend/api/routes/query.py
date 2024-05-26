from fastapi import APIRouter, Depends, HTTPException, status
from models.query_model import QueryModel
from api.utils.utils import read_config
from api.src.qna import QnA

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
            session_id="",
            verbose=False,
        )
        return {
            "query": query,
            "response": response,
            "source_documents": source_documents,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid query",
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query",
        ) from e
