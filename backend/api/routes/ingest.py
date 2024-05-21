from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File, Form
from api.src.ingestion import Ingestion
from api.utils.utils import read_config

router = APIRouter()


@router.post(path="/ingest")
async def ingest_document(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
    config=Depends(lambda: read_config()),
):
    ingestion = Ingestion(config=config)
    return {
        "file": f"{file.file}",
        "content": f"{file.file.read()}",
        "metadata": metadata,
    }
