"""
Lightweight Audit Ledger — Thesis-only module (detachable from Springer paper).

Provides a tamper-evident, hash-chained log of:
  - Record-level dedup decisions  (record_id, sil_level, dedup_decision)
  - Model version fingerprints    (model_version_hash, round, timestamp_utc)

Raw sensor values NEVER appear on-chain.

Architecture
------------
Pure-Python hash-chained JSON ledger.
Each block: { index, timestamp_utc, data, prev_hash, block_hash }
block_hash = SHA-256( index || timestamp_utc || data || prev_hash )

The ledger file is append-only JSON-Lines (.jsonl).  For the thesis,
this can be swapped for a Hyperledger Fabric SDK or web3.py+Ganache call
by replacing _commit_block() — the interface stays identical.

Discharge of DRC Objective 3 (Blockchain)
------------------------------------------
This module completes the "private blockchain using hash-stored audit records"
requirement of Phase 5 of the DRC proposal.  It is intentionally minimal:
no smart-contract logic, no consensus protocol — pure tamper-evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)


@dataclass
class AuditEvent:
    """One dedup-decision audit event (never contains raw sensor data)."""
    record_id:      str              # opaque client-local ID
    client_id:      int
    sil_level:      int              # 0 = non-critical, 1 = SIL-flagged
    dedup_decision: str              # "retained" | "removed"
    model_version:  str              # hex hash of model weights
    fl_round:       int
    epsilon_spent:  float
    rho:            float            # local retention fraction at this round


@dataclass
class Block:
    index:        int
    timestamp_utc: float
    data:         Dict[str, Any]
    prev_hash:    str
    block_hash:   str = ""

    def compute_hash(self) -> str:
        payload = json.dumps(
            {k: v for k, v in asdict(self).items() if k != "block_hash"},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class AuditLedger:
    """
    Append-only hash-chained audit ledger.

    Parameters
    ----------
    path : Path to the .jsonl ledger file.
           Existing file is read to recover chain head (allows resumption).
    """

    GENESIS_HASH = "0" * 64

    def __init__(self, path: Path = Path("audit/ledger.jsonl")):
        self.path        = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chain_len  = 0
        self._last_hash  = self.GENESIS_HASH
        self._load_existing()

    # ── Public API ────────────────────────────────────────────────────────────

    def log_dedup_event(self, event: AuditEvent) -> Block:
        """Append a dedup-decision event and return the committed block."""
        return self._commit_block(asdict(event))

    def log_model_round(
        self,
        fl_round: int,
        model_weights: list,
        epsilon: float,
        dataset: str,
    ) -> Block:
        """Append a model-version fingerprint block after each FL round."""
        model_hash = _hash_weights(model_weights)
        data = {
            "type":        "model_checkpoint",
            "fl_round":    fl_round,
            "model_hash":  model_hash,
            "epsilon":     epsilon,
            "dataset":     dataset,
        }
        return self._commit_block(data)

    def verify(self) -> bool:
        """
        Re-read the ledger file and verify every block's hash.
        Returns True if chain is intact.
        """
        if not self.path.exists():
            return True   # empty chain is trivially valid

        blocks = self._read_all()
        if not blocks:
            return True

        prev_hash = self.GENESIS_HASH
        for blk in blocks:
            expected = Block(
                index         = blk["index"],
                timestamp_utc = blk["timestamp_utc"],
                data          = blk["data"],
                prev_hash     = blk["prev_hash"],
            ).compute_hash()
            if expected != blk["block_hash"]:
                log.error("Block %d: hash mismatch (tampered?)", blk["index"])
                return False
            if blk["prev_hash"] != prev_hash:
                log.error("Block %d: broken chain link", blk["index"])
                return False
            prev_hash = blk["block_hash"]

        log.info("Ledger verified: %d blocks, chain intact.", len(blocks))
        return True

    def summary(self) -> dict:
        blocks = self._read_all()
        events = [b["data"] for b in blocks if b["data"].get("type") != "model_checkpoint"]
        checkpoints = [b["data"] for b in blocks if b["data"].get("type") == "model_checkpoint"]
        return {
            "total_blocks":    len(blocks),
            "dedup_events":    len(events),
            "model_rounds":    len(checkpoints),
            "chain_intact":    self.verify(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _commit_block(self, data: dict) -> Block:
        block = Block(
            index         = self._chain_len,
            timestamp_utc = time.time(),
            data          = data,
            prev_hash     = self._last_hash,
        )
        block.block_hash = block.compute_hash()

        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(block)) + "\n")

        self._chain_len += 1
        self._last_hash  = block.block_hash
        log.debug("Ledger block %d committed: %s", block.index, block.block_hash[:16])
        return block

    def _load_existing(self):
        if not self.path.exists():
            return
        blocks = self._read_all()
        if blocks:
            last = blocks[-1]
            self._chain_len = last["index"] + 1
            self._last_hash = last["block_hash"]
            log.info("Resumed ledger: %d blocks, head=%s…", self._chain_len,
                     self._last_hash[:16])

    def _read_all(self) -> list:
        if not self.path.exists():
            return []
        blocks = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        blocks.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return blocks


def _hash_weights(weights: list) -> str:
    """SHA-256 of concatenated model weight bytes."""
    import numpy as np
    h = hashlib.sha256()
    for w in weights:
        h.update(np.asarray(w).tobytes())
    return h.hexdigest()
