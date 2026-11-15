"""Filtered retrieval must find a candidate's chunks even when the index holds many other CVs."""

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.services.retriever import similarity_search_with_score


def test_filtered_search_returns_all_chunks_of_one_cv_in_a_large_index():
    docs = [
        Document(page_content=f"other candidate chunk {i}", metadata={"document_id": f"other-{i % 10}", "chunk_index": i})
        for i in range(60)
    ]
    target = [
        Document(page_content=f"David Kim chunk {i}", metadata={"document_id": "david", "chunk_index": i})
        for i in range(3)
    ]
    store = FAISS.from_documents(docs + target, FakeEmbeddings(size=32))

    results = similarity_search_with_score("junior python developer", store, document_id="david", top_k=8)

    assert len(results) == 3  # every David chunk, not only those inside the global top 20
    assert {doc.metadata["chunk_index"] for doc, _ in results} == {0, 1, 2}


def _cv_store():
    docs = []
    for doc_id, file_name, header in [
        ("sarah", "sarah_chen_cv.pdf", "Sarah Chen\nSan Francisco, CA\nPROFESSIONAL SUMMARY Machine Learning Engineer"),
        ("marco", "marco_bianchi_cv.pdf", "Marco Bianchi\nMilano, Italia\nPROFILO PROFESSIONALE Ingegnere del software"),
        ("david", "david_kim_cv.pdf", "David Kim\nToronto, ON\nSenior Frontend Engineer"),
    ]:
        docs.append(Document(page_content=header, metadata={"document_id": doc_id, "file_name": file_name, "chunk_index": 0}))
        docs.append(Document(page_content="Python experience the details", metadata={"document_id": doc_id, "file_name": file_name, "chunk_index": 1}))
    return FAISS.from_documents(docs, FakeEmbeddings(size=32))


def test_find_documents_by_name_exact_candidate():
    from app.services.retriever import find_documents_by_name

    matches = find_documents_by_name("the candidate named Marco Bianchi", vector_store=_cv_store())
    assert matches[0]["document_id"] == "marco"
    assert matches[0]["matched_words"] == 2
    assert len(matches) == 1  # other CVs do not match any name word


def test_find_document_id_tool_returns_the_named_candidate():
    from unittest.mock import patch

    from app.agents import orchestrator
    from app.services.retriever import find_documents_by_name

    store = _cv_store()
    with patch.object(orchestrator, "find_documents_by_name", lambda name: find_documents_by_name(name, vector_store=store)):
        result = orchestrator.find_document_id.invoke({"candidate_name_or_descriptions": "Marco Bianchi"})
    assert "Found document_id: marco" in result