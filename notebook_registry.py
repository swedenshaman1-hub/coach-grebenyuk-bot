"""Validated registry of NotebookLM collections.

Notebook UUIDs live only in ``config/notebooks.json``.  Telegram handlers never
select an arbitrary notebook supplied by a user.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid as uuid_lib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("config") / "notebooks.json"


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class NotebookConfig:
    id: str
    uuid: str
    url: str
    role: str


@dataclass(frozen=True)
class CollectionConfig:
    id: str
    title: str
    mode: str
    notebooks: tuple[NotebookConfig, ...]

    @property
    def notebook_set_hash(self) -> str:
        payload = "\n".join(sorted(item.uuid for item in self.notebooks))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class NotebookRegistry:
    def __init__(self, collections: dict[str, CollectionConfig], path: Path):
        self.collections = collections
        self.path = path

    def collection(self, collection_id: str) -> CollectionConfig:
        try:
            return self.collections[collection_id]
        except KeyError as exc:
            raise RegistryError(f"Unknown notebook collection: {collection_id}") from exc


def _validated_notebook(value: dict) -> NotebookConfig:
    notebook_id = str(value.get("id") or "").strip()
    notebook_uuid = str(value.get("uuid") or "").strip()
    notebook_url = str(value.get("url") or "").strip()
    role = str(value.get("role") or "").strip()
    if not notebook_id or not notebook_uuid or not notebook_url:
        raise RegistryError("Every enabled notebook requires id, uuid and url")
    try:
        parsed_uuid = str(uuid_lib.UUID(notebook_uuid))
    except ValueError as exc:
        raise RegistryError(f"Invalid NotebookLM UUID: {notebook_uuid}") from exc
    parsed_url = urlparse(notebook_url)
    if parsed_url.scheme != "https" or parsed_url.hostname not in {
        "notebook.google.com",
        "notebooklm.google.com",
    }:
        raise RegistryError(f"Invalid NotebookLM URL: {notebook_url}")
    if parsed_uuid not in parsed_url.path:
        raise RegistryError(f"Notebook URL does not contain UUID {parsed_uuid}")
    return NotebookConfig(
        id=notebook_id,
        uuid=parsed_uuid,
        url=notebook_url,
        role=role,
    )


def load_registry(path: str | os.PathLike[str] | None = None) -> NotebookRegistry:
    registry_path = Path(
        path or os.getenv("NOTEBOOK_REGISTRY_PATH", "") or DEFAULT_REGISTRY_PATH
    ).resolve()
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"Cannot read notebook registry {registry_path}: {exc}") from exc

    raw_collections = payload.get("collections")
    if not isinstance(raw_collections, dict) or not raw_collections:
        raise RegistryError("Notebook registry must contain collections")

    collections: dict[str, CollectionConfig] = {}
    seen_uuids: set[str] = set()
    for collection_id, value in raw_collections.items():
        if not isinstance(value, dict):
            raise RegistryError(f"Collection {collection_id} must be an object")
        mode = str(value.get("mode") or "").strip()
        if mode != "strict":
            raise RegistryError(f"Collection {collection_id} must use strict mode")
        raw_notebooks = value.get("notebooks") or []
        notebooks = tuple(
            _validated_notebook(item)
            for item in raw_notebooks
            if isinstance(item, dict) and item.get("enabled", True)
        )
        if not notebooks:
            raise RegistryError(f"Collection {collection_id} has no enabled notebooks")
        for notebook in notebooks:
            if notebook.uuid in seen_uuids:
                raise RegistryError(f"Duplicate NotebookLM UUID: {notebook.uuid}")
            seen_uuids.add(notebook.uuid)
        collections[str(collection_id)] = CollectionConfig(
            id=str(collection_id),
            title=str(value.get("title") or collection_id),
            mode=mode,
            notebooks=notebooks,
        )
    return NotebookRegistry(collections, registry_path)
