from pydantic import BaseModel


class IngestionModel(BaseModel):
    file_bytes: str
    metadata: dict
