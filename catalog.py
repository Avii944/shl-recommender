"""
Catalog manager — loads CATALOGUE.json, cleans data, and builds a TF-IDF index
for fast retrieval.  Everything is loaded once at startup and kept in memory.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from config import CATALOG_PATH, KEY_TO_TYPE_CODE

logger = logging.getLogger(__name__)

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Assessment:
    entity_id: str
    name: str
    url: str
    description: str
    job_levels: List[str]
    languages: List[str]
    duration: str
    remote: bool
    adaptive: bool
    keys: List[str]          # e.g. ["Knowledge & Skills", "Personality & Behavior"]
    type_codes: List[str]    # e.g. ["K", "P"]
    primary_type_code: str   # First / dominant type code

    # ── Full-text field used for TF-IDF indexing ─────────────────────────────
    search_text: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "url": self.url,
            "description": self.description,
            "job_levels": self.job_levels,
            "languages": self.languages,
            "duration": self.duration,
            "remote": self.remote,
            "adaptive": self.adaptive,
            "keys": self.keys,
            "type_codes": self.type_codes,
            "primary_type_code": self.primary_type_code,
        }

    def short_summary(self) -> str:
        """Compact summary injected into LLM context."""
        parts = [
            f"Name: {self.name}",
            f"URL: {self.url}",
            f"Type: {', '.join(self.keys) or 'N/A'}",
            f"Levels: {', '.join(self.job_levels) or 'All'}",
            f"Duration: {self.duration or 'N/A'}",
            f"Remote: {'Yes' if self.remote else 'No'}",
        ]
        if self.description:
            # Cap at 300 chars to stay within token budget
            parts.append(f"Description: {self.description[:300]}")
        return "\n".join(parts)


# ── Catalog loader ─────────────────────────────────────────────────────────────

def _clean_json_text(raw: str) -> str:
    """Remove stray control characters that break json.loads."""
    # Keep \n, \r, \t — strip everything else in C0 control range
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", raw)


def _derive_type_codes(keys: List[str]) -> List[str]:
    return [KEY_TO_TYPE_CODE[k] for k in keys if k in KEY_TO_TYPE_CODE]


def _build_search_text(item: dict) -> str:
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
        item.get("duration", ""),
    ]
    return " ".join(p for p in parts if p).lower()


def load_catalog(path: Path = CATALOG_PATH) -> List[Assessment]:
    """Parse CATALOGUE.json and return cleaned Assessment objects."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    raw = _clean_json_text(raw)
    data: list[dict] = json.loads(raw, strict=False)

    assessments: List[Assessment] = []
    skipped = 0
    for item in data:
        # Skip items without a usable link
        url = item.get("link", "")
        if not url or "shl.com" not in url:
            skipped += 1
            continue

        name = item.get("name", "").strip()
        if not name:
            skipped += 1
            continue

        keys = item.get("keys", [])
        type_codes = _derive_type_codes(keys)
        primary = type_codes[0] if type_codes else "K"

        a = Assessment(
            entity_id=str(item.get("entity_id", "")),
            name=name,
            url=url,
            description=item.get("description", "").strip(),
            job_levels=item.get("job_levels", []),
            languages=item.get("languages", []),
            duration=item.get("duration", ""),
            remote=item.get("remote", "").lower() == "yes",
            adaptive=item.get("adaptive", "").lower() == "yes",
            keys=keys,
            type_codes=type_codes,
            primary_type_code=primary,
        )
        a.search_text = _build_search_text(item)
        assessments.append(a)

    logger.info("Catalog loaded: %d assessments (%d skipped)", len(assessments), skipped)
    return assessments


# ── Singleton catalog ──────────────────────────────────────────────────────────

class CatalogManager:
    """Singleton that holds the loaded assessments and provides lookup methods."""

    _instance: Optional["CatalogManager"] = None

    def __init__(self) -> None:
        self.assessments: List[Assessment] = load_catalog()
        self._name_index: dict[str, Assessment] = {
            a.name.lower(): a for a in self.assessments
        }
        self._id_index: dict[str, Assessment] = {
            a.entity_id: a for a in self.assessments
        }

    @classmethod
    def get(cls) -> "CatalogManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_by_name(self, name: str) -> Optional[Assessment]:
        """Exact (case-insensitive) name lookup."""
        return self._name_index.get(name.lower())

    def fuzzy_name_lookup(self, name: str) -> Optional[Assessment]:
        """Best-effort fuzzy match on name."""
        needle = name.lower()
        # Exact
        if needle in self._name_index:
            return self._name_index[needle]
        # Prefix / substring
        for stored_name, a in self._name_index.items():
            if needle in stored_name or stored_name in needle:
                return a
        return None

    def all(self) -> List[Assessment]:
        return self.assessments

    def get_search_texts(self) -> List[str]:
        return [a.search_text for a in self.assessments]

    def validate_url(self, url: str) -> bool:
        """Return True only if url belongs to a known catalog item."""
        return any(a.url == url for a in self.assessments)
