"""Shared fixtures for agent tests."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from agent.memory import MemoryStore
from agent.state_ledger import StateLedger


@pytest_asyncio.fixture
async def memory_store(tmp_path):
    """MemoryStore backed by a temp SQLite DB + temp ChromaDB dir."""
    store = MemoryStore(
        db_path=tmp_path / "memory.db",
        chroma_dir=tmp_path / "chroma",
    )
    await store.init()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def state_ledger(tmp_path):
    """StateLedger backed by a temp SQLite DB."""
    ledger = StateLedger(db_path=tmp_path / "ledger.db")
    await ledger.init()
    yield ledger
    await ledger.close()
