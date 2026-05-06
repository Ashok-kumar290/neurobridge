"""Centralized constants for NeuroBridge."""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
# HDD brain (source of truth)
# HDD brain (source of truth)
HDD_ROOT = Path(os.getenv("NEUROBRIDGE_ROOT", str(Path.home() / ".neurobridge" / "brain")))
HDD_CONFIG = HDD_ROOT / "config"
HDD_REPOS = HDD_ROOT / "repos"
HDD_MEMORY = HDD_ROOT / "memory"
HDD_TRACES = HDD_ROOT / "traces"
HDD_DATASETS = HDD_ROOT / "datasets"
HDD_ADAPTERS = HDD_ROOT / "adapters"
HDD_CHECKPOINTS = HDD_ROOT / "checkpoints"
HDD_EVALS = HDD_ROOT / "evals"
HDD_LOGS = HDD_ROOT / "logs"

# SSD cache (hot indexes for speed)
SSD_CACHE = Path.home() / ".neurobridge" / "cache"
SSD_INDEXES = SSD_CACHE / "indexes"
SSD_VECTORS = SSD_CACHE / "vectors"
SSD_CONFIG = Path.home() / ".neurobridge" / "config.yaml"

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_GENERATE = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_API_CHAT = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_API_TAGS = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_API_SHOW = f"{OLLAMA_BASE_URL}/api/show"
OLLAMA_API_PULL = f"{OLLAMA_BASE_URL}/api/pull"
OLLAMA_API_EMBEDDINGS = f"{OLLAMA_BASE_URL}/api/embeddings"

# ── Model identifiers ─────────────────────────────────────────────────────────
MODEL_ROUTER = "llama3.2:latest"
MODEL_CODER = "llama3.2:latest"
MODEL_COHERE_PLANNER = "command-a-03-2025"
MODEL_COHERE_REASONING = "command-a-reasoning-08-2025"
MODEL_COHERE_FAST = "command-r7b-12-2024"
MODEL_COHERE_EMBED = "embed-v4.0"
MODEL_COHERE_RERANK = "rerank-v4.0-fast"

# ── Local model context limits ────────────────────────────────────────────────
ROUTER_CONTEXT_WINDOW = 4096
CODER_CONTEXT_WINDOW = 4096

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_MODE = "safe"
DEFAULT_TEMPERATURE_ROUTER = 0.1
DEFAULT_TEMPERATURE_CODER = 0.2
MAX_EXPERT_PACKET_TOKENS = 8000
EXPERT_ESCALATION_AFTER_FAILURES = 2

# ── Safety defaults ───────────────────────────────────────────────────────────
BLOCKED_COMMANDS = [
    "rm -rf /",
    "rm -rf ~",
    "mkfs",
    "dd if=",
    "chmod -R 777",
    "curl",  # blocked when piped to sh
    "wget",  # blocked when piped to sh
]

BLOCKED_PATHS = [
    "~/.ssh",
    ".env",
    "~/.aws",
    "~/.gcloud",
    "~/.config/gcloud",
]

# ── Training defaults ─────────────────────────────────────────────────────────
MIN_SAMPLES_FOR_ADAPTER = 100
DEFAULT_LORA_RANK = 16
MAX_LORA_RANK = 64

# ── Embedding defaults ────────────────────────────────────────────────────────
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # MiniLM output dimension

# ── Indexer defaults ──────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh", ".yaml", ".yml",
    ".toml", ".json", ".md", ".txt", ".sql", ".html", ".css",
}

MAX_FILE_SIZE_BYTES = 1_000_000  # skip files > 1 MB
CHUNK_SIZE_LINES = 60
CHUNK_OVERLAP_LINES = 10
