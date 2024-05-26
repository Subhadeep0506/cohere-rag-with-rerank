import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_ingest_document_success():
    # Create a test file
    file = {"file": ("test.pdf", b"Hello, World!", "application/pdf")}
    metadata = {"key": "value"}
    response = client.post("/ingest", files=file, data={"metadata": metadata})
    assert response.status_code == 200
    assert response.json()["message"] == "Ingested successfully!"


def test_ingest_document_invalid_file_type():
    # Create a test file with an invalid file type
    file = {
        "file": (
            "test.docx",
            b"Hello, World!",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    }
    metadata = {"key": "value"}
    response = client.post("/ingest", files=file, data={"metadata": metadata})
    assert response.status_code == 406
    assert (
        response.json()["detail"]
        == "Documents other than PDFs are currently not supported yet."
    )


def test_ingest_document_missing_metadata():
    # Create a test file without metadata
    file = {"file": ("test.pdf", b"Hello, World!", "application/pdf")}
    response = client.post("/ingest", files=file)
    assert response.status_code == 422
    assert response.json()["detail"] == "Metadata is required"


def test_ingest_document_invalid_metadata():
    # Create a test file with invalid metadata
    file = {"file": ("test.pdf", b"Hello, World!", "application/pdf")}
    metadata = "invalid metadata"
    response = client.post("/ingest", files=file, data={"metadata": metadata})
    assert response.status_code == 422
    assert response.json()["detail"] == "Metadata must be a dictionary"
