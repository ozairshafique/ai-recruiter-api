
from unittest.mock import patch, MagicMock
import pytest
from langchain_core.documents import Document

from app.services.embeddings import (
    check_faiss_index,
    load_faiss_index,
    save_faiss_index,
    get_embeddings,
    embed_and_store,
    create_faiss_index
)

from app.services.rag_pipeline import (
    run_ingest_pipeline,
    run_query_pipeline
)

from app.services.ingestion import (
    ingest_file,
    validate_file,
    load_pdf,
    split_documents,
    enrich_metadata
)

from app.services.llm_service import (
    llm_generate,
    initialize_llm,
    generate_answer,
    format_context
)
from app.services.retriever import (
    retrieve,
    format_retrieved_documents,
    similarity_search_with_score,
    similarity_search
)

# ── Ingestion Tests ───────────────────
def test_ingest_file_not_valid():
    """ Test ingest_file with invalid file path """
    with pytest.raises(FileNotFoundError):
        validate_file("non_existent_file.pdf")


def test_ingest_file_wrong_type(tmp_path):
    """  Test ingest_file with wrong file type """
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("This is a test text file, not a PDF.")
    with pytest.raises(ValueError):
        validate_file(str(txt_path))

def test_validate_file_empty(tmp_path):
    """ Test validate_file with empty PDF file """
    empty_pdf_path = tmp_path / "empty.pdf"
    empty_pdf_path.write_bytes(b"") # Write minimal PDF content to create a valid PDF file
    with pytest.raises(ValueError):
        validate_file(str(empty_pdf_path))


def test_validate_file_valid(tmp_path):
    """ Test validate_file with valid PDF file """
    valid_pdf_path = tmp_path / "test.pdf"
    valid_pdf_path.write_bytes(b"%PDF-1.4\n%Valid PDF content for testing\n%%EOF") # Write minimal PDF content to create a valid PDF file
    validate_file(str(valid_pdf_path))

def test_split_documents():
    """ Test split_documents with sample documents """

    docs = [
        Document(
            page_content="This is the content of page" * 100,
            metadata ={
                "page": 1,
            }
        )
    ]

    chunks = split_documents(docs, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 0
    assert all(isinstance(chunk, Document) for chunk in chunks) # Check all chunks are Document instances

def test_enrich_metadata():
    """ Test enrich_metadata with sample documents """

    docs = [
        Document(
            page_content="This is the content of page 1",
            metadata ={
                "page": 1,
            }
        ),
        Document(
            page_content="This is the content of page 2",
            metadata ={
                "page": 2,
            }
        )
    ]

    enriched = enrich_metadata(chunks=docs, document_id="test-doc-1234", file_name="test.pdf")

    assert len(enriched) == 2
    assert enriched[0].metadata["document_id"] == "test-doc-1234"
    assert enriched[0].metadata["file_name"] == "test.pdf"
    assert enriched[0].metadata["chunk_index"] == 0
    assert enriched[0].metadata["total_chunks"] == 2
    assert enriched[1].metadata["chunk_index"] == 1


# ── Embeddings Tests ─────────────────────
def test_check_faiss_index_no_index():
    """ Test check_faiss_index when index does not exist """
    with patch("app.services.embeddings.Path.exists", return_value=False):
        results = check_faiss_index("./fake_index")
        assert results  == False

def test_check_faiss_index_exists():
    """ Test check_faiss_index when index exists """
    with patch("app.services.embeddings.Path.exists", return_value=True):
        results = check_faiss_index("./fake_index")
        assert results  == True

def test_create_faiss_index(mock_faiss_index):
    """ Test create_faiss_index with sample documents """
    docs = [
        Document(
            page_content="This is the content of page 1",
            metadata={
                "page": 1
            }
        )
    ]

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5]]

    with patch("app.services.embeddings.FAISS.from_documents", return_value=mock_embeddings):
        with patch("app.services.embeddings.faiss.normalize_L2"):
            with patch("app.services.embeddings.faiss.IndexFlatIP") as mock_index:
                mock_index.return_value = MagicMock()
                results = create_faiss_index(docs, embeddings=mock_embeddings)
                assert results is not None

# ── Retriever Tests ────────────────────
def test_format_retrieved_documents():
    """ Test format_retrieved_documents with sample results """
    mock_docs = MagicMock()
    mock_docs.page_content = "This is the content of page 1"
    mock_docs.metadata = {
        "document_id": "test-doc-1234",
        "file_name": "tests.pdf",
        "chunk_index": 0,
        "page": 1,
    }
    results = [(mock_docs, 0.25)]
    formatted = format_retrieved_documents(results)

    assert len(formatted) == 1
    assert formatted[0]["content"] == "This is the content of page 1"
    assert formatted[0]["document_id"] == "test-doc-1234"
    assert formatted[0]["file_name"] == "tests.pdf"
    assert formatted[0]["chunk_index"] == 0
    assert formatted[0]["page"] == 1

def test_similarity_search(mock_faiss_index):
    """ Test similarity_search with sample results """
    mock_faiss_index.similarity_search.return_value = [
        Document(
            page_content="This is the content of page 1",
            metadata = {
                "page": 1,
            }
        )
    ]

    results = similarity_search(
        query="What is on page 1?",
        vectorstore=mock_faiss_index,
        top_k=5
    )

    assert len(results) == 1

# ── LLM Service Tests ────────────────

def test_initialize_llm():
    """ Test initialize_llm with sample settings """
    with patch("app.services.llm_service.ChatGroq") as mock_chat_groq:
        mock_chat_groq.return_value = MagicMock()
        llm = initialize_llm()
        assert llm is not None

def test_format_context():
    """ Test format_context with sample documents """
    results = format_context([])
    assert results == "No relevant documents found."

def test_format_context_with_docs():
    """ Test format_context with sample documents """
    docs = [
       {
           "content": "This is the content of page 1",
           "score": 0.25,
           "document_id": "test-doc-1234",
           "file_name": "tests.pdf",
           "page": 1
       }
   ]
    results = format_context(docs)
    assert "This is the content of page 1" in results
    assert "tests.pdf" in results

def test_generate_answer(mock_llm_response):
    """ Test generate_answer with sample question and context """

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "content of page 1 is about programmings"
    mock_llm.invoke.return_value = mock_response
    with patch("app.services.llm_service.initialize_llm", return_value=mock_llm):
        with patch("app.services.llm_service.ChatGroq"):
            mock_chain = MagicMock()
            mock_chain.invoke = MagicMock(return_value=mock_response)
            with patch("app.services.llm_service.ChatPromptTemplate.from_messages", return_value=MagicMock(__or__=MagicMock(return_value=mock_chain))):
                answer = generate_answer(
                    question="What is on page 1?",
                    context="This is the content of page 1",
                    llm=None,
                    system_prompt=None)

        assert isinstance(answer, str)

# ── RAG Tests ──────────────
def test_run_ingest_pipeline(mock_ingest_results, tmp_path):
    """ Test run_ingest_pipeline with sample file path and name """
    pdf_files = tmp_path/"tests.pdf"
    pdf_files.write_bytes(b"%PDF-1.4\n PDF content for testing") # Write minimal PDF content to create a valid PDF file

    mock_chucks = [(
        Document(
            page_content="This is the content of page 1",
            metadata={
                "page": 1,
                "document_id": "tests-1234",
                "file_name": "tests.pdf",
                "chunk_index": 0,
                "total_chunks": 10,
            }
        ))
    ]
    with patch("app.services.rag_pipeline.ingest_file", return_value={
        "document_id": "tests-1234",
        "file_name": "tests.pdf",
        "chunks": 10,
        "pages": 2,
        "latency_ms": 250.5,
        "documents": mock_chucks,
    }),patch("app.services.rag_pipeline.check_faiss_index",return_value=False),patch("app.services.rag_pipeline.embed_and_store", return_value={"vector_store": MagicMock()}):
        results = run_ingest_pipeline(
            file_path=str(pdf_files),
            file_name="tests.pdf",
            document_id="tests-1234"
        )

        assert results["document_id"] == "tests-1234"
        assert results["chunks"] == 10
        assert results["pages"] == 2
        assert isinstance(results["latency_ms"], float)

def test_run_query_pipeline(mock_query_results):
    """ Test run query pipelines return answers"""
    with patch(
        "app.services.rag_pipeline.retrieve", return_value=mock_query_results["sources"]
        ),patch("app.services.rag_pipeline.get_all_document_ids", return_value=["tests-1234"]),patch("app.services.rag_pipeline.llm_generate", return_value=mock_query_results):
        results = run_query_pipeline(
            query="What is on page 1?",
            top_k=5
        )
        assert "answer" in results
        assert "model" in results
        assert "sources" in results
        assert "latency_ms" in results


