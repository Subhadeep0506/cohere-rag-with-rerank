import ast

from typing import Annotated
from fastapi import APIRouter, Depends, Form, HTTPException, status
from ..src.ingestion import Ingestion
from ..utils.utils import read_config

router = APIRouter()


@router.post(path="/files/delete")
def delete_file_from_store(
    filter: Annotated[str, Form()],
    config=Depends(lambda: read_config()),
):
    filter = ast.literal_eval(filter)
    if isinstance(filter, dict):
        ingestion = Ingestion(config=config)
        result = ingestion.delete_from_vectorstore(filter=filter)
        return result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filter is invalid."
        )


@router.get(path="/files/list")
def list_vectorstore_files(
    config=Depends(lambda: read_config()),
):
    try:
        ingestion = Ingestion(config=config)
        result = ingestion.list_documents_from_vectorstore()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list files",
        ) from e
