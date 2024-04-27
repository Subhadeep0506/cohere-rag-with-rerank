import uvicorn

from fastapi import FastAPI
from api.utils.utils import read_config
from api.routes import files_router, query_router, ingest_router

config = read_config()

app = FastAPI()
app.include_router(files_router)
app.include_router(query_router)
app.include_router(ingest_router)


@app.get(path="/")
async def home():
    return {"message": f"FastAPI running on PORT {config['PORT']}"}


if __name__ == "__main__":
    uvicorn.run(
        app="app:app",
        host=config["HOST"],
        port=config["PORT"],
        use_colors=True,
        reload=True,
    )
