from pydantic import BaseModel
from fastapi import UploadFile


class IngestionModel(BaseModel):
    file: UploadFile
    metadata: dict
