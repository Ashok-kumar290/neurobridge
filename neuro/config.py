"""Pydantic-based configuration system for NeuroBridge.

Config resolution order:
  1. Hardcoded defaults (this file)
  2. HDD config  (/media/.../neurobridge/config/*.yaml)
  3. Local override (~/.neurobridge/config.yaml)
  4. Environment variables (NEURO_*)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from neuro.constants import (
    CODER_CONTEXT_WINDOW,
    DEFAULT_MODE,
    DEFAULT_TEMPERATURE_CODER,
    DEFAULT_TEMPERATURE_ROUTER,
    EXPERT_ESCALATION_AFTER_FAILURES,
    HDD_ROOT,
    MAX_EXPERT_PACKET_TOKENS,
    MODEL_CODER,
    MODEL_COHERE_EMBED,
    MODEL_COHERE_FAST,
    MODEL_COHERE_PLANNER,
    MODEL_COHERE_REASONING,
    MODEL_COHERE_RERANK,
    MODEL_ROUTER,
    OLLAMA_BASE_URL,
    ROUTER_CONTEXT_WINDOW,
    SSD_CONFIG,
)


# ── Sub-configs ────────────────────────────────────────────────────────────────


class LocalModelConfig(BaseModel):
    """Configuration for a local Ollama model."""

    provider: str = "ollama"
    model: str
    context: int
    temperature: float


class CohereConfig(BaseModel):
    """Cohere API configuration."""

    enabled: bool = False
    api_key: Optional[str] = None
    planner_model: str = MODEL_COHERE_PLANNER
    reasoning_model: str = MODEL_COHERE_REASONING
    fast_model: str = MODEL_COHERE_FAST
    embed_model: str = MODEL_COHERE_EMBED
    rerank_model: str = MODEL_COHERE_RERANK


class ExpertConfig(BaseModel):
    """Expert model bridge configuration."""

    claude_enabled: bool = True
    codex_enabled: bool = True
    claude_mode: str = "terminal_bridge"
    codex_mode: str = "terminal_bridge"


class RoutingConfig(BaseModel):
    """Router behavior configuration."""

    local_first: bool = True
    expert_after_local_failures: int = EXPERT_ESCALATION_AFTER_FAILURES
    use_cohere_rerank: bool = False
    max_expert_packet_tokens: int = MAX_EXPERT_PACKET_TOKENS
    prefer_codex_for_patch_tasks: bool = True
    prefer_claude_for_architecture: bool = True
    prefer_cohere_for_long_context: bool = True


class SafetyConfig(BaseModel):
    """Safety control plane configuration."""

    require_human_approval: bool = True
    block_destructive_commands: bool = True
    scan_secrets: bool = True
    sandbox_required_for_lab: bool = True
    allow_training_on_raw_expert_output: bool = False
    allow_auto_adapter_promotion: bool = False
    allow_sudo: bool = False
    allow_home_directory_write: bool = False


class TrainingConfig(BaseModel):
    """Training and adapter configuration."""

    train_only_verified: bool = True
    require_tests_passed: bool = True
    require_human_approval: bool = True
    min_samples_for_adapter: int = 100
    adapter_growth: bool = True
    default_lora_rank: int = 16
    max_lora_rank: int = 64
    colab_enabled: bool = True


# ── Master config ──────────────────────────────────────────────────────────────


class NeuroBridgeConfig(BaseModel):
    """Master NeuroBridge configuration."""

    project_name: str = "neurobridge"
    storage_root: Path = HDD_ROOT
    default_mode: str = DEFAULT_MODE
    ollama_url: str = OLLAMA_BASE_URL

    router: LocalModelConfig = Field(
        default_factory=lambda: LocalModelConfig(
            model=MODEL_ROUTER,
            context=ROUTER_CONTEXT_WINDOW,
            temperature=DEFAULT_TEMPERATURE_ROUTER,
        )
    )

    coder: LocalModelConfig = Field(
        default_factory=lambda: LocalModelConfig(
            model=MODEL_CODER,
            context=CODER_CONTEXT_WINDOW,
            temperature=DEFAULT_TEMPERATURE_CODER,
        )
    )

    cohere: CohereConfig = Field(default_factory=CohereConfig)
    experts: ExpertConfig = Field(default_factory=ExpertConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


# ── Config loading ─────────────────────────────────────────────────────────────


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    hdd_config_path: Path | None = None,
    local_config_path: Path | None = None,
) -> NeuroBridgeConfig:
    """Load configuration with layered overrides.

    Resolution order:
      1. Hardcoded defaults
      2. HDD config file
      3. Local override file
    """
    config_data: dict = {}

    # Layer 2: HDD config
    hdd_path = hdd_config_path or (HDD_ROOT / "config" / "neurobridge.yaml")
    if hdd_path.exists():
        with open(hdd_path) as f:
            hdd_data = yaml.safe_load(f) or {}
        config_data = _deep_merge(config_data, hdd_data)

    # Layer 3: Local override
    local_path = local_config_path or SSD_CONFIG
    if local_path.exists():
        with open(local_path) as f:
            local_data = yaml.safe_load(f) or {}
        config_data = _deep_merge(config_data, local_data)

    # Build config (Layer 1 defaults are in the Pydantic model)
    return NeuroBridgeConfig(**config_data)


def save_config(config: NeuroBridgeConfig, path: Path) -> None:
    """Save configuration to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def get_config() -> NeuroBridgeConfig:
    """Get the active configuration. Convenience function."""
    return load_config()
