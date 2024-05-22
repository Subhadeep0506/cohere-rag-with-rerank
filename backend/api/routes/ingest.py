import os

from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from api.src.ingestion import Ingestion
from api.utils.utils import read_config
from pypdf.errors import PdfStreamError

router = APIRouter()


@router.post(path="/ingest")
async def ingest_document(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
    config=Depends(lambda: read_config()),
):
    if file.content_type == "application/pdf":
        ingestion = Ingestion(config=config)
        _tempfile = f"/tmp/{file.filename}"

        with open(_tempfile, "wb") as file_bytes:
            file_bytes.write(file.file.read())

        _ = await ingestion.create_and_add_embeddings(file=_tempfile)

        if os.path.exists(_tempfile):
            os.remove(_tempfile)
        return {
            "message": "Ingested successfully!",
            "metadata": metadata,
            "file_type": file.content_type,
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Documents other than PDFs are currently not suppoted yet.",
        )
