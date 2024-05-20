from fastapi import APIRouter, Depends
from api.src.ingestion import Ingestion
from api.utils.utils import read_config
from models.ingestion_model import IngestionModel

router = APIRouter()


@router.post(path="/ingest/doc_id")
async def ingest_document(
    file_bytes: IngestionModel,
    config=Depends(lambda: read_config()),
):
    ingestion = Ingestion(config=config)
    return {"doc_id": "doc_id"}
