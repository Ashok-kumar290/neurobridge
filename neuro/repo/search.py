"""Repository search — hybrid FTS5 + symbol search over indexed repos."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neuro.constants import HDD_REPOS, SSD_INDEXES


@dataclass
class SearchResult:
    """A single search result."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    chunk_type: str
    language: str
    score: float = 0.0  # relevance score


class RepoSearch:
    """Hybrid search over an indexed repository."""

    def __init__(self, repo_name: str) -> None:
        self.repo_name = repo_name

        # Prefer SSD cache for speed, fall back to HDD
        ssd_path = SSD_INDEXES / repo_name / "index.sqlite"
        hdd_path = HDD_REPOS / repo_name / "index.sqlite"

        if ssd_path.exists():
            self.db_path = ssd_path
        elif hdd_path.exists():
            self.db_path = hdd_path
        else:
            raise FileNotFoundError(
                f"No index found for repo '{repo_name}'. Run: neuro index <path>"
            )

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def fts_search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Full-text search over code chunks."""
        conn = self._get_conn()
        try:
            # Sanitize query: keep only alphanumeric and spaces for FTS safety
            clean_query = "".join(c if c.isalnum() else " " for c in query)
            fts_query = " OR ".join(
                f'"{term}"' for term in clean_query.split() if term.strip()
            )
            if not fts_query:
                return []

            rows = conn.execute(
                """SELECT c.file_path, c.start_line, c.end_line,
                          c.content, c.chunk_type, c.language,
                          bm25(chunks_fts) as score
                   FROM chunks_fts
                   JOIN chunks c ON chunks_fts.rowid = c.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY score
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()

            return [
                SearchResult(
                    file_path=r["file_path"],
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                    content=r["content"],
                    chunk_type=r["chunk_type"],
                    language=r["language"],
                    score=abs(r["score"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def symbol_search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search symbols by name."""
        conn = self._get_conn()
        try:
            # Truncate query for LIKE safety
            safe_query = query[:200]
            rows = conn.execute(
                """SELECT name, kind, file_path, start_line, signature
                   FROM symbols
                   WHERE name LIKE ?
                   ORDER BY
                     CASE WHEN name = ? THEN 0
                          WHEN name LIKE ? THEN 1
                          ELSE 2
                     END
                   LIMIT ?""",
                (f"%{safe_query}%", safe_query, f"{safe_query}%", limit),
            ).fetchall()

            return [dict(r) for r in rows]
        finally:
            conn.close()

    def search(self, query: str, limit: int = 20) -> list[SearchResult]:
        """Hybrid search: FTS + symbol matching, merged by relevance."""
        # FTS search on code content
        fts_results = self.fts_search(query, limit=limit)

        # Symbol search
        symbol_hits = self.symbol_search(query, limit=10)

        # If we got symbol hits, boost their chunks
        symbol_files = {s["file_path"] for s in symbol_hits}

        # Merge: symbol-matched files get boosted score
        for result in fts_results:
            if result.file_path in symbol_files:
                result.score *= 1.5  # boost

        # Sort by score descending
        fts_results.sort(key=lambda r: r.score, reverse=True)

        return fts_results[:limit]

    def get_file_content(self, file_path: str) -> str | None:
        """Get full file content from chunks."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                """SELECT content FROM chunks
                   WHERE file_path = ?
                   ORDER BY start_line""",
                (file_path,),
            ).fetchall()

            if not rows:
                return None

            # Reconstruct (chunks may overlap, but for display this is fine)
            return "\n".join(r["content"] for r in rows)
        finally:
            conn.close()

    def list_files(self) -> list[dict[str, Any]]:
        """List all indexed files."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT path, language, size_bytes, line_count FROM files ORDER BY path"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_stats(self) -> dict[str, int]:
        """Get index statistics."""
        conn = self._get_conn()
        try:
            files = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            symbols = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
            return {"files": files, "chunks": chunks, "symbols": symbols}
        finally:
            conn.close()
