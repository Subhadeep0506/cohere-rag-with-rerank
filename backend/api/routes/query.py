from fastapi import APIRouter, Depends
from models.query_model import QueryModel
from api.utils.utils import read_config
from api.src.qna import QnA

router = APIRouter()


@router.post(path="/query")
async def run_query(
    query: QueryModel,
    config=Depends(lambda: read_config()),
):
    qna = QnA(config=config)
    print("Query", query)
    return {"query": query, "response": "response"}
