"""LangChain retriever wrapper for the Multi-Domain Support Triage Agent."""
from __future__ import annotations

from typing import Any, List, Optional
from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import Field


class SupportLangChainRetriever(BaseRetriever):
    """Wraps the internal BM25/keyword/vector retriever as a standard LangChain BaseRetriever."""
    retriever: Any = Field(description="Underlying internal Retriever instance")
    domain: Optional[str] = Field(default=None, description="Domain filter (hackerrank, claude, visa)")
    k: int = Field(default=5, description="Number of documents to return")

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LCDocument]:
        """Retrieve relevant documents converted to LangChain Document instances."""
        hits = self.retriever.search(query, domain=self.domain, k=self.k) if self.domain else self.retriever.search(query, k=self.k)
        docs: List[LCDocument] = []
        for hit in hits:
            docs.append(
                LCDocument(
                    page_content=hit.doc.text,
                    metadata={
                        "domain": hit.doc.domain,
                        "product_area": hit.doc.product_area,
                        "path": hit.doc.path,
                        "title": hit.doc.title,
                        "score": hit.score,
                        "overlap": hit.overlap,
                        "snippet": hit.snippet,
                        "hit_obj": hit,  # Preserve reference to internal Hit
                    },
                )
            )
        return docs
