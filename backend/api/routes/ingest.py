from fastapi import APIRouter
from api.src.ingestion import Ingestion

router = APIRouter()


@router.post(path="/ingest/doc_id")
async def ingest_document(file_bytes):
    ingestion = Ingestion()
    return {"doc_id": "doc_id"}
