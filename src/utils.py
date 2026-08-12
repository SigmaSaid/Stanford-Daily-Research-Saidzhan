"""
utils.py — Shared utilities for the Stanford Daily research project.
"""

import logging
import time
import random
import json
import hashlib
from pathlib import Path
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    REQUEST_DELAY_SECONDS,
    REQUEST_DELAY_JITTER,
    REQUEST_TIMEOUT_SECONDS,
    MAX_RETRIES,
    RETRY_BACKOFF_FACTOR,
    USER_AGENT,
    COLLECTION_LOG,
)


# ─── Logging setup ────────────────────────────────────────────────────────────

def get_logger(name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Return a logger that writes to both console and an optional file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    file_path = log_file or COLLECTION_LOG
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(file_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─── HTTP session ─────────────────────────────────────────────────────────────

def build_session() -> requests.Session:
    """Return a requests Session with retry logic and a polite User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


def polite_get(
    session: requests.Session,
    url: str,
    params: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> Optional[requests.Response]:
    """
    Make a GET request with rate-limiting delay.
    Returns the Response on success, None on unrecoverable error.
    """
    delay = REQUEST_DELAY_SECONDS + random.uniform(0, REQUEST_DELAY_JITTER)
    time.sleep(delay)

    try:
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code == 200:
            return resp
        if logger:
            logger.warning("HTTP %d for %s", resp.status_code, url)
        return None
    except requests.exceptions.Timeout:
        if logger:
            logger.warning("Timeout for %s", url)
        return None
    except requests.exceptions.RequestException as exc:
        if logger:
            logger.error("Request failed for %s: %s", url, exc)
        return None


# ─── JSONL helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load all records from a JSONL file. Returns empty list if file missing."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record to a JSONL file (creates file if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set[int]:
    """Return the set of article IDs already saved in a JSONL file."""
    return {r["id"] for r in load_jsonl(path) if "id" in r}


# ─── Hashing ─────────────────────────────────────────────────────────────────

def sha256_text(text: str) -> str:
    """Return the SHA-256 hex digest of a string (for duplicate detection)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ─── Date helpers ─────────────────────────────────────────────────────────────

def iso_to_year_quarter(iso_date: str) -> tuple[int, int]:
    """
    Parse an ISO-8601 date string (YYYY-MM-DD or full datetime) and return
    (year, quarter) as integers.
    """
    from datetime import datetime as dt
    if not iso_date:
        raise ValueError(f"Cannot parse date: {iso_date!r}")

    # WordPress returns 'YYYY-MM-DDTHH:MM:SS'; some fallback paths yield
    # 'YYYY-MM-DD'. Both share the same first 10 characters, so parse those.
    try:
        d = dt.strptime(iso_date[:10], "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"Cannot parse date: {iso_date!r}") from exc

    month = d.month
    quarter = (month - 1) // 3 + 1
    return d.year, quarter


# ─── Misc ─────────────────────────────────────────────────────────────────────

def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or default if denominator is zero."""
    return numerator / denominator if denominator else default


def chunks(lst: list, n: int):
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]
