from fastapi import APIRouter

router = APIRouter()


@router.get(path="/files/upload")
async def upload_file():
    return {"file_upload": "Uploaded file."}
