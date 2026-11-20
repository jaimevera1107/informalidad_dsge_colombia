"""Small HTTP client with retries and atomic downloads."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

USER_AGENT = "informalidad-dsge-colombia/1.0 (+https://github.com/jaimevera1107)"


class SourceError(RuntimeError):
    """A source is unavailable or its content does not have the expected structure."""


class HttpClient:
    """GET requests with retries; downloads are written atomically."""

    def __init__(self, retries: int = 4, timeout: float = 120.0, backoff: float = 5.0) -> None:
        if retries < 1:
            raise ValueError("retries must be at least 1")
        self.retries = retries
        self.timeout = timeout
        self.backoff = backoff
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT

    def get(self, url: str, stream: bool = False) -> requests.Response:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout, stream=stream)
                if response.status_code == 200:
                    return response
                last = SourceError(f"HTTP {response.status_code} for {url}")
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    break           # client errors do not improve with retries
            except requests.RequestException as exc:
                last = exc
            if attempt < self.retries:
                time.sleep(self.backoff * attempt)
        raise SourceError(f"Could not download {url}: {last}")

    def text(self, url: str) -> str:
        response = self.get(url)
        response.encoding = response.encoding or "utf-8"
        return response.text

    def json(self, url: str) -> dict:
        return self.get(url).json()

    def download(self, url: str, target: Path, min_bytes: int = 1) -> Path:
        """Download ``url`` to ``target`` (written atomically); reuse an existing file."""
        if target.is_file() and target.stat().st_size >= min_bytes:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        response = self.get(url, stream=True)
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        if temporary.stat().st_size < min_bytes:
            temporary.unlink()
            raise SourceError(f"Download too small ({url}); the source may have changed")
        temporary.replace(target)
        return target


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
