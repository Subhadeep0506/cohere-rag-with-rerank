from fastapi import APIRouter

router = APIRouter()


@router.get(path="/query")
async def upload_file():
    return {"query": "query", "response": "response"}
