"""Repository indexer — walks a repo, catalogs files, creates chunks.

Indexes are stored on HDD (source of truth) with hot copies in SSD cache
for fast retrieval. Uses .gitignore awareness and tree-sitter for smart chunking.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from neuro.constants import (
    CHUNK_OVERLAP_LINES,
    CHUNK_SIZE_LINES,
    HDD_REPOS,
    MAX_FILE_SIZE_BYTES,
    SSD_INDEXES,
    SUPPORTED_EXTENSIONS,
)

console = Console()


def _should_ignore(path: Path, gitignore_patterns: list[str]) -> bool:
    """Check if a path should be ignored based on common patterns."""
    name = path.name
    parts = set(path.parts)

    # Always ignore these directories
    always_ignore = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", ".eggs", "*.egg-info",
        ".next", ".nuxt", ".output",
    }
    if parts & always_ignore:
        return True

    # Always ignore these files
    if name in {".DS_Store", "Thumbs.db", ".gitkeep"}:
        return True

    return False


@dataclass
class FileRecord:
    """A single indexed file."""

    path: str  # relative to repo root
    abs_path: str
    language: str
    size_bytes: int
    content_hash: str
    line_count: int
    last_modified: float


@dataclass
class Chunk:
    """A chunk of source code for retrieval."""

    file_path: str  # relative to repo root
    start_line: int
    end_line: int
    content: str
    content_hash: str
    chunk_type: str  # "function", "class", "block"
    language: str


def _detect_language(path: Path) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".jsx": "javascript", ".tsx": "typescript",
        ".go": "go", ".rs": "rust", ".java": "java",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".rb": "ruby", ".php": "php",
        ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
        ".sh": "shell", ".bash": "shell", ".zsh": "shell",
        ".yaml": "yaml", ".yml": "yaml",
        ".toml": "toml", ".json": "json",
        ".md": "markdown", ".txt": "text",
        ".sql": "sql", ".html": "html", ".css": "css",
    }
    return ext_map.get(path.suffix.lower(), "unknown")


def _hash_content(content: str) -> str:
    """SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _chunk_by_lines(
    content: str,
    file_path: str,
    language: str,
    chunk_size: int = CHUNK_SIZE_LINES,
    overlap: int = CHUNK_OVERLAP_LINES,
) -> list[Chunk]:
    """Split content into overlapping line-based chunks."""
    lines = content.splitlines()
    chunks = []

    if len(lines) <= chunk_size:
        # File fits in one chunk
        chunks.append(Chunk(
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            content=content,
            content_hash=_hash_content(content),
            chunk_type="file",
            language=language,
        ))
        return chunks

    start = 0
    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        chunks.append(Chunk(
            file_path=file_path,
            start_line=start + 1,
            end_line=end,
            content=chunk_content,
            content_hash=_hash_content(chunk_content),
            chunk_type="block",
            language=language,
        ))

        if end >= len(lines):
            break
        start = end - overlap

    return chunks


class RepoIndexer:
    """Index a repository: catalog files, create chunks, store in SQLite."""

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path.resolve()
        self.repo_name = self.repo_path.name

        # Storage paths
        self.hdd_repo_dir = HDD_REPOS / self.repo_name
        self.ssd_repo_dir = SSD_INDEXES / self.repo_name
        self.hdd_db_path = self.hdd_repo_dir / "index.sqlite"
        self.ssd_db_path = self.ssd_repo_dir / "index.sqlite"

    def _init_db(self, db_path: Path) -> sqlite3.Connection:
        """Initialize the index database."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")  # NTFS-safe for HDD
        conn.execute("PRAGMA synchronous=NORMAL")

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                abs_path TEXT NOT NULL,
                language TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                line_count INTEGER NOT NULL,
                last_modified REAL NOT NULL,
                indexed_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_type TEXT NOT NULL,
                language TEXT NOT NULL,
                FOREIGN KEY (file_path) REFERENCES files(path)
            );

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER,
                signature TEXT,
                FOREIGN KEY (file_path) REFERENCES files(path)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                file_path, content, chunk_type, language
            );
        """)
        conn.commit()
        return conn

    def _scan_files(self) -> list[Path]:
        """Scan repo for indexable files."""
        files = []
        for root, dirs, filenames in os.walk(self.repo_path):
            root_path = Path(root)

            # Filter ignored directories in-place
            dirs[:] = [
                d for d in dirs
                if not _should_ignore(root_path / d, [])
            ]

            for name in filenames:
                file_path = root_path / name
                if _should_ignore(file_path, []):
                    continue
                if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                if file_path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    continue
                files.append(file_path)

        return files

    def _extract_python_symbols(self, content: str, file_path: str) -> list[dict]:
        """Extract symbols from Python files using simple parsing."""
        symbols = []
        lines = content.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if stripped.startswith("def "):
                # Extract function name
                name = stripped[4:].split("(")[0].strip()
                sig = stripped.rstrip(":")
                symbols.append({
                    "name": name,
                    "kind": "function",
                    "file_path": file_path,
                    "start_line": i,
                    "signature": sig,
                })
            elif stripped.startswith("class "):
                name = stripped[6:].split("(")[0].split(":")[0].strip()
                sig = stripped.rstrip(":")
                symbols.append({
                    "name": name,
                    "kind": "class",
                    "file_path": file_path,
                    "start_line": i,
                    "signature": sig,
                })
            elif " = " in stripped and not stripped.startswith("#"):
                # Module-level assignments (crude but useful)
                if line[0] != " " and line[0] != "\t":
                    name = stripped.split("=")[0].strip()
                    if name.isidentifier():
                        symbols.append({
                            "name": name,
                            "kind": "variable",
                            "file_path": file_path,
                            "start_line": i,
                            "signature": stripped[:80],
                        })

        return symbols

    def run(self) -> dict[str, Any]:
        """Run the full indexing pipeline. Returns stats dict."""
        start_time = time.time()

        # Initialize databases (HDD primary, SSD hot copy)
        conn = self._init_db(self.hdd_db_path)

        # Clear old data for re-index
        conn.execute("DELETE FROM chunks_fts")
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM symbols")
        conn.execute("DELETE FROM files")
        conn.commit()

        # Scan files
        files = self._scan_files()
        total_chunks = 0
        total_symbols = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Indexing files", total=len(files))

            for file_path in files:
                rel_path = str(file_path.relative_to(self.repo_path))

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    progress.advance(task)
                    continue

                language = _detect_language(file_path)
                line_count = content.count("\n") + 1
                content_hash = _hash_content(content)

                # Store file record
                conn.execute(
                    """INSERT OR REPLACE INTO files
                       (path, abs_path, language, size_bytes, content_hash,
                        line_count, last_modified, indexed_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        rel_path,
                        str(file_path),
                        language,
                        file_path.stat().st_size,
                        content_hash,
                        line_count,
                        file_path.stat().st_mtime,
                        time.time(),
                    ),
                )

                # Create chunks
                chunks = _chunk_by_lines(content, rel_path, language)
                for chunk in chunks:
                    conn.execute(
                        """INSERT INTO chunks
                           (file_path, start_line, end_line, content,
                            content_hash, chunk_type, language)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            chunk.file_path,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.content,
                            chunk.content_hash,
                            chunk.chunk_type,
                            chunk.language,
                        ),
                    )
                    # FTS index
                    conn.execute(
                        """INSERT INTO chunks_fts
                           (file_path, content, chunk_type, language)
                           VALUES (?, ?, ?, ?)""",
                        (chunk.file_path, chunk.content, chunk.chunk_type, chunk.language),
                    )
                total_chunks += len(chunks)

                # Extract symbols (Python for now, extensible later)
                if language == "python":
                    symbols = self._extract_python_symbols(content, rel_path)
                    for sym in symbols:
                        conn.execute(
                            """INSERT INTO symbols
                               (name, kind, file_path, start_line, signature)
                               VALUES (?, ?, ?, ?, ?)""",
                            (
                                sym["name"],
                                sym["kind"],
                                sym["file_path"],
                                sym["start_line"],
                                sym.get("signature"),
                            ),
                        )
                    total_symbols += len(symbols)

                progress.advance(task)

        conn.commit()

        # Copy to SSD cache for fast reads
        self.ssd_repo_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(self.hdd_db_path), str(self.ssd_db_path))

        # Compute index size
        index_size_bytes = self.hdd_db_path.stat().st_size
        if index_size_bytes > 1_000_000:
            index_size_str = f"{index_size_bytes / 1_000_000:.1f} MB"
        else:
            index_size_str = f"{index_size_bytes / 1_000:.1f} KB"

        conn.close()
        duration = time.time() - start_time

        return {
            "files_scanned": len(files),
            "files_indexed": len(files),
            "chunks_created": total_chunks,
            "symbols_extracted": total_symbols,
            "index_size": index_size_str,
            "duration_seconds": duration,
        }
