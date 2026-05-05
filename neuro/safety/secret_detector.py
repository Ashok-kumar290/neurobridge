"""Secret detector — catches API keys, passwords, PII in text.

Scans model outputs, traces, and patches before they enter
the memory/training pipeline. If secrets are found, the content
is flagged and blocked from training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SecretMatch:
    """A detected secret."""

    secret_type: str
    pattern_name: str
    matched_text: str       # the actual match (redacted in logs)
    line_number: int | None = None
    confidence: float = 1.0  # 0-1 confidence


@dataclass
class SecretScanResult:
    """Result of scanning text for secrets."""

    clean: bool
    secrets_found: list[SecretMatch] = field(default_factory=list)
    redacted_text: str = ""


# ── Secret patterns ────────────────────────────────────────────────────────────
# (name, regex, type, confidence)

SECRET_PATTERNS: list[tuple[str, str, str, float]] = [
    # API Keys
    ("AWS Access Key", r"AKIA[0-9A-Z]{16}", "api_key", 0.95),
    ("AWS Secret Key", r"(?i)aws.{0,20}(?:secret|key).{0,10}['\"][0-9a-zA-Z/+]{40}['\"]", "api_key", 0.9),
    ("Google API Key", r"AIza[0-9A-Za-z\-_]{35}", "api_key", 0.9),
    ("Slack Token", r"xox[bpsorta]-[0-9a-zA-Z]{10,}", "api_key", 0.95),
    ("Slack Webhook", r"https://hooks\.slack\.com/services/T[A-Z0-9]{8}/B[A-Z0-9]{8}/[a-zA-Z0-9]{24}", "api_key", 0.95),
    ("GitHub Token", r"gh[ps]_[A-Za-z0-9_]{36,}", "api_key", 0.95),
    ("GitHub OAuth", r"gho_[A-Za-z0-9_]{36,}", "api_key", 0.95),
    ("Stripe Key", r"(?:sk|pk)_(?:test|live)_[0-9a-zA-Z]{24,}", "api_key", 0.95),
    ("Twilio API Key", r"SK[0-9a-fA-F]{32}", "api_key", 0.8),
    ("SendGrid Key", r"SG\.[0-9A-Za-z\-_]{22}\.[0-9A-Za-z\-_]{43}", "api_key", 0.95),
    ("Heroku API Key", r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", "api_key", 0.4),
    ("OpenAI Key", r"sk-[A-Za-z0-9]{20,}T3BlbkFJ[A-Za-z0-9]{20,}", "api_key", 0.95),
    ("Anthropic Key", r"sk-ant-[A-Za-z0-9\-_]{40,}", "api_key", 0.95),
    ("Cohere Key", r"[A-Za-z0-9]{40}", "api_key", 0.2),  # low confidence, too generic
    ("Generic API Key", r"(?i)(?:api[_-]?key|apikey|api_secret)\s*[:=]\s*['\"]?[A-Za-z0-9\-_]{20,}['\"]?", "api_key", 0.7),

    # Passwords
    ("Password Assignment", r"(?i)(?:password|passwd|pwd)\s*[:=]\s*['\"][^'\"]{6,}['\"]", "password", 0.85),
    ("DB Connection String", r"(?i)(?:mysql|postgres|mongodb|redis)://[^\s'\"]+:[^\s'\"]+@", "password", 0.9),

    # Private Keys
    ("RSA Private Key", r"-----BEGIN RSA PRIVATE KEY-----", "private_key", 0.99),
    ("EC Private Key", r"-----BEGIN EC PRIVATE KEY-----", "private_key", 0.99),
    ("DSA Private Key", r"-----BEGIN DSA PRIVATE KEY-----", "private_key", 0.99),
    ("SSH Private Key", r"-----BEGIN OPENSSH PRIVATE KEY-----", "private_key", 0.99),
    ("PGP Private Key", r"-----BEGIN PGP PRIVATE KEY BLOCK-----", "private_key", 0.99),
    ("Generic Private Key", r"-----BEGIN PRIVATE KEY-----", "private_key", 0.99),

    # PII
    ("Email Address", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "pii", 0.3),  # very common, low confidence
    ("Phone Number", r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", "pii", 0.3),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", "pii", 0.8),
    ("Credit Card", r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "pii", 0.6),

    # Tokens / Secrets in env
    ("JWT Token", r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}", "token", 0.8),
    ("Bearer Token", r"(?i)bearer\s+[A-Za-z0-9\-_\.]{20,}", "token", 0.7),
]

# Minimum confidence to flag
MIN_CONFIDENCE = 0.5


def scan_text(text: str, min_confidence: float = MIN_CONFIDENCE) -> SecretScanResult:
    """Scan text for secrets and PII.

    Args:
        text: Text to scan (model output, trace, patch, etc.)
        min_confidence: Minimum confidence threshold to report

    Returns:
        SecretScanResult with found secrets and redacted text
    """
    secrets: list[SecretMatch] = []
    lines = text.splitlines()

    for name, pattern, secret_type, confidence in SECRET_PATTERNS:
        if confidence < min_confidence:
            continue

        for i, line in enumerate(lines, 1):
            for match in re.finditer(pattern, line):
                matched = match.group(0)
                secrets.append(SecretMatch(
                    secret_type=secret_type,
                    pattern_name=name,
                    matched_text=matched[:8] + "..." + matched[-4:] if len(matched) > 16 else "***",
                    line_number=i,
                    confidence=confidence,
                ))

    # Deduplicate by (type, line)
    seen = set()
    unique_secrets = []
    for s in secrets:
        key = (s.pattern_name, s.line_number)
        if key not in seen:
            seen.add(key)
            unique_secrets.append(s)

    # Build redacted text
    redacted = text
    for name, pattern, secret_type, confidence in SECRET_PATTERNS:
        if confidence < min_confidence:
            continue
        redacted = re.sub(pattern, f"[REDACTED:{name}]", redacted)

    return SecretScanResult(
        clean=len(unique_secrets) == 0,
        secrets_found=unique_secrets,
        redacted_text=redacted,
    )


def is_clean(text: str) -> bool:
    """Quick check — does this text contain any secrets?"""
    return scan_text(text).clean


def redact(text: str) -> str:
    """Return redacted version of text with secrets replaced."""
    return scan_text(text).redacted_text
