from __future__ import annotations

from pathlib import Path
from typing import Any

from ailib.agent.vector_db.faiss import StudentRecordVectorDB


class SimpleRAG:
    def __init__(
        self,
        vdb_dir: str | Path,
        topk: int = 3,
    ) -> None:
        self.vector_store = StudentRecordVectorDB(
            vdb_dir=vdb_dir,
        ).load()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": topk})

    def invoke(self, query: str, with_score: bool = False):
        if with_score:
            # Langchain's base retriever doesn't return scores, so we use the vector_store directly for scores
            k = self.retriever.search_kwargs.get("k", 3)
            return self.vector_store.similarity_search_with_score(query, k=k)
        return self.retriever.invoke(query)


class MetadataHeaderRAG(SimpleRAG):
    def invoke(
        self,
        query: str,
        with_score: bool = False,
        metadict: dict[str, Any] | None = None,
    ):
        if metadict:
            query = f"{self._format_metadata_header(metadict)}\n\n{query}"

        return super().invoke(query, with_score=with_score)

    @staticmethod
    def _format_metadata_header(metadict: dict[str, Any]) -> str:
        acl_groups = metadict.get("acl_groups", [])
        if isinstance(acl_groups, list):
            acl_groups_str = ", ".join(acl_groups)
        else:
            acl_groups_str = str(acl_groups)
            
        return "\n".join(
            [
                f"doc_id: {metadict.get('doc_id', '')}",
                f"section_title: {metadict.get('section_title', '')}",
                f"page: {metadict.get('page', '')}",
                f"created_at: {metadict.get('created_at', '')}",
                f"acl_groups: {acl_groups_str}",
            ]
        )


# class HyDERAG:
