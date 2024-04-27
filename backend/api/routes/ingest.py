from fastapi import APIRouter

router = APIRouter()


@router.get(path="/ingest/doc_id")
async def upload_file():
    return {"doc_id": "doc_id"}
