import os
import ast

from api.utils.logger import logger
from typing import Annotated
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from api.src.ingestion import Ingestion
from api.utils.utils import read_config

router = APIRouter()


@router.post(path="/ingest")
async def ingest_document(
    file: Annotated[UploadFile, File()],
    metadata: Annotated[str, Form()],
    config=Depends(lambda: read_config()),
):
    if file.content_type in ["application/pdf", "text/plain"]:
        ingestion = Ingestion(config=config)
        _tempfile = f"/tmp/{file.filename}"
        metadata = ast.literal_eval(metadata)
        print(file.content_type)
        try:
            with open(_tempfile, "wb") as file_bytes:
                file_bytes.write(file.file.read())

            try:
                _, file_upload_info = await ingestion.create_and_add_embeddings(
                    file=_tempfile, metadata=metadata, file_type=file.content_type
                )
            except Exception as e:
                logger.error(f"Failed to ingest document. ERROR: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to ingest document.",
                ) from e

            return {
                "message": "Ingested successfully!",
                "metadata": metadata,
                "file_type": file.content_type,
                "file_upload_info": file_upload_info,
            }
        finally:
            if os.path.exists(_tempfile):
                os.remove(_tempfile)
    else:
        logger.error(f"User uploaded document other than PDFs and txt.")
        raise HTTPException(
            status_code=status.HTTP_406_NOT_ACCEPTABLE,
            detail="Documents other than PDFs and txt are currently not supported yet.",
        )
