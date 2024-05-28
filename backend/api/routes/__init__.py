from ..routes.files import router as files_router
from ..routes.ingest import router as ingest_router
from ..routes.query import router as query_router

__all__ = [
    "files_router",
    "ingest_router",
    "query_router",
]
