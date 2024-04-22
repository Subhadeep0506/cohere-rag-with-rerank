from api.routes.files import router as files_router
from api.routes.ingest import router as ingest_router
from api.routes.query import router as query_router

__all__ = [
    "files_router",
    "ingest_router",
    "query_router",
]
