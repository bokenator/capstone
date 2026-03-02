"""
RAG Documentation Builder
=========================

Extracts documentation from library sources and indexes into AWS OpenSearch
for RAG-based documentation grounding.

Sources:
- numpy, pandas, scipy: Sphinx objects.inv + HTML pages (concurrent fetching)
- vectorbt: inspect-based docstring extraction

Usage:
    uv run build-rag                    # Index all libraries
    uv run build-rag --lib numpy        # Single library
    uv run build-rag --dry-run          # Preview without indexing
    uv run build-rag --recreate         # Delete and recreate index
    uv run build-rag --concurrency 20   # Set concurrent download limit
"""

import argparse
import asyncio
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    import sphobjinv as soi
except ImportError:
    soi = None  # type: ignore

try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    from requests_aws4auth import AWS4Auth

    HAS_OPENSEARCH = True
except ImportError:
    HAS_OPENSEARCH = False

# Load env
SCRIPT_DIR = Path(__file__).parent
load_dotenv(SCRIPT_DIR / ".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# Configuration
# ===========================================================================

INDEX_NAME = "library-docs"

INVENTORY_URLS = {
    "numpy": "https://numpy.org/doc/stable/objects.inv",
    "pandas": "https://pandas.pydata.org/docs/objects.inv",
    "scipy": "https://docs.scipy.org/doc/scipy/objects.inv",
}

BASE_URLS = {
    "numpy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "scipy": "https://docs.scipy.org/doc/scipy/",
}

# Roles to include from objects.inv
INCLUDED_ROLES = {"py:function", "py:method", "py:class", "py:module"}

# Chunk size limit (characters)
MAX_CHUNK_SIZE = 2000

# Cache directory
CACHE_DIR = SCRIPT_DIR / ".doc_cache"


# ===========================================================================
# Data model
# ===========================================================================


@dataclass
class DocChunk:
    """A single documentation chunk for indexing."""

    chunk_id: str
    library: str
    module_path: str
    object_name: str
    object_type: str
    title: str
    content: str
    signature: str = ""
    parameters: str = ""
    returns: str = ""
    examples: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ===========================================================================
# Async HTTP client with caching and concurrency
# ===========================================================================


class AsyncCachedHTTPClient:
    """Async HTTP client with disk-based caching and concurrency control."""

    def __init__(self, cache_dir: Path, concurrency: int = 10):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client: Optional[httpx.AsyncClient] = None
        self._fetched = 0
        self._cached = 0

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30, follow_redirects=True)
        return self

    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()

    def _cache_key(self, url: str) -> Path:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"

    async def get(self, url: str) -> Optional[str]:
        """Fetch URL with caching and concurrency limiting."""
        cache_file = self._cache_key(url)
        if cache_file.exists():
            self._cached += 1
            return cache_file.read_text(encoding="utf-8")

        async with self.semaphore:
            try:
                resp = await self.client.get(url)
                self._fetched += 1
                if self._fetched % 100 == 0:
                    log.info("  Fetched %d pages so far...", self._fetched)
                if resp.status_code == 200:
                    text = resp.text
                    cache_file.write_text(text, encoding="utf-8")
                    return text
                else:
                    return None
            except httpx.HTTPError:
                return None

    def stats(self) -> str:
        return f"fetched={self._fetched}, cached={self._cached}"


# ===========================================================================
# Sphinx objects.inv processing
# ===========================================================================


def load_inventory(library: str) -> list[dict[str, str]]:
    """Load and filter Sphinx objects.inv for a library.

    Returns list of dicts with keys: name, type, uri, base_url
    """
    if soi is None:
        log.error("sphobjinv not installed. Run: pip install sphobjinv")
        return []

    url = INVENTORY_URLS.get(library)
    if not url:
        log.warning("No inventory URL for library: %s", library)
        return []

    base_url = BASE_URLS[library]

    log.info("Downloading objects.inv for %s...", library)
    try:
        inv = soi.Inventory(url=url)
    except Exception as e:
        log.error("Failed to load inventory for %s: %s", library, e)
        return []

    entries = []
    for obj in inv.objects:
        role = f"{obj.domain}:{obj.role}"
        if role not in INCLUDED_ROLES:
            continue

        # Build URI - sphobjinv uses $ as placeholder for object name
        uri = obj.uri
        if uri.endswith("$"):
            uri = uri[:-1] + obj.name

        entries.append(
            {
                "name": obj.name,
                "type": role,
                "uri": uri,
                "base_url": base_url,
            }
        )

    log.info("Found %d objects for %s (filtered from inventory)", len(entries), library)
    return entries


# ===========================================================================
# HTML doc parsing
# ===========================================================================


def parse_doc_page(html: str, object_name: str) -> dict[str, str]:
    """Parse an HTML documentation page and extract structured content."""
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, str] = {
        "signature": "",
        "description": "",
        "parameters": "",
        "returns": "",
        "examples": "",
    }

    # Try to find the specific function/class definition block
    target_dl = None
    short_name = object_name.split(".")[-1]
    for dl in soup.find_all("dl", class_=re.compile(r"py\s+")):
        dt = dl.find("dt")
        if dt and short_name in dt.get_text():
            target_dl = dl
            break

    if target_dl is None:
        main = soup.find("div", class_="section") or soup.find("article") or soup.find("main")
        if main:
            result["description"] = main.get_text(separator="\n", strip=True)[:MAX_CHUNK_SIZE]
        return result

    dt = target_dl.find("dt")
    if dt:
        result["signature"] = dt.get_text(separator=" ", strip=True)

    dd = target_dl.find("dd")
    if not dd:
        return result

    # Extract description
    desc_parts = []
    for child in dd.children:
        if hasattr(child, "name"):
            if child.name == "p":
                desc_parts.append(child.get_text(strip=True))
            elif child.name in ("dl", "table", "div"):
                break
        elif isinstance(child, str) and child.strip():
            desc_parts.append(child.strip())
    result["description"] = "\n".join(desc_parts)

    # Extract parameters section
    params_section = dd.find("dl", class_="field-list")
    if params_section:
        params_parts = []
        for dt_param in params_section.find_all("dt"):
            dd_param = dt_param.find_next_sibling("dd")
            if dd_param:
                params_parts.append(
                    f"{dt_param.get_text(strip=True)}: {dd_param.get_text(strip=True)}"
                )
        result["parameters"] = "\n".join(params_parts)

    # Extract returns
    returns_header = dd.find(string=re.compile(r"Returns?", re.IGNORECASE))
    if returns_header:
        parent = returns_header.parent
        if parent:
            next_dd = parent.find_next_sibling("dd") if parent.name == "dt" else None
            if next_dd:
                result["returns"] = next_dd.get_text(strip=True)

    # Extract examples
    examples_header = dd.find(string=re.compile(r"Examples?", re.IGNORECASE))
    if examples_header:
        code_blocks = []
        current = examples_header.parent
        if current:
            for sibling in current.find_all_next(["pre", "div"], limit=5):
                if sibling.name == "pre" or "highlight" in " ".join(sibling.get("class", [])):
                    code_blocks.append(sibling.get_text(strip=True))
        result["examples"] = "\n\n".join(code_blocks)

    return result


# ===========================================================================
# vectorbt docstring extraction
# ===========================================================================


def extract_vectorbt_docs() -> list[DocChunk]:
    """Extract documentation from installed vectorbt package via inspect."""
    chunks: list[DocChunk] = []

    try:
        import vectorbt as vbt
    except ImportError:
        log.error("vectorbt not installed. Skipping vectorbt docs.")
        return chunks

    # Key modules to document
    targets = [
        ("vectorbt.indicators.factory", "vbt.IndicatorFactory"),
        ("vectorbt.portfolio.base", "vbt.Portfolio"),
        ("vectorbt.portfolio.nb", "vbt.portfolio.nb"),
        ("vectorbt.portfolio.enums", "vbt.portfolio.enums"),
    ]

    indicator_names = ["RSI", "MACD", "ATR", "MA", "EMA", "BBANDS", "STOCH", "OBV"]
    for ind_name in indicator_names:
        obj = getattr(vbt, ind_name, None)
        if obj is not None:
            targets.append((f"vectorbt.indicators.{ind_name}", f"vbt.{ind_name}"))

    visited = set()

    for module_path, display_prefix in targets:
        try:
            parts = module_path.split(".")
            mod = __import__(parts[0])
            for part in parts[1:]:
                mod = getattr(mod, part, None)
                if mod is None:
                    break
            if mod is None:
                continue
        except Exception:
            continue

        for name in dir(mod):
            if name.startswith("_"):
                continue

            full_name = f"{display_prefix}.{name}"
            if full_name in visited:
                continue
            visited.add(full_name)

            obj = getattr(mod, name, None)
            if obj is None:
                continue

            doc = inspect.getdoc(obj)
            if not doc:
                continue

            sig_str = ""
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    sig_str = f"{full_name}{sig}"
                except (ValueError, TypeError):
                    sig_str = full_name

            if inspect.isclass(obj):
                obj_type = "py:class"
            elif inspect.isfunction(obj) or inspect.ismethod(obj):
                obj_type = "py:function"
            elif inspect.ismodule(obj):
                obj_type = "py:module"
            else:
                obj_type = "py:attribute"

            chunk_id = hashlib.md5(full_name.encode()).hexdigest()

            parameters = ""
            returns = ""
            examples = ""
            description = doc

            sections = re.split(
                r"\n(?=Parameters|Returns|Examples|Raises|Notes|See Also|References)\s*\n-+",
                doc,
            )
            if len(sections) > 1:
                description = sections[0].strip()
                for section in sections[1:]:
                    if section.strip().startswith("Parameters"):
                        parameters = section.strip()
                    elif section.strip().startswith("Returns"):
                        returns = section.strip()
                    elif section.strip().startswith("Examples"):
                        examples = section.strip()

            chunks.append(
                DocChunk(
                    chunk_id=chunk_id,
                    library="vectorbt",
                    module_path=module_path,
                    object_name=full_name,
                    object_type=obj_type,
                    title=full_name,
                    content=description[:MAX_CHUNK_SIZE],
                    signature=sig_str,
                    parameters=parameters[:MAX_CHUNK_SIZE],
                    returns=returns[:MAX_CHUNK_SIZE],
                    examples=examples[:MAX_CHUNK_SIZE],
                )
            )

    log.info("Extracted %d vectorbt doc chunks", len(chunks))
    return chunks


# ===========================================================================
# Sphinx-based library extraction (async concurrent)
# ===========================================================================


async def extract_sphinx_docs(library: str, http: AsyncCachedHTTPClient) -> list[DocChunk]:
    """Extract documentation for a Sphinx-documented library with concurrent fetching."""
    entries = load_inventory(library)
    if not entries:
        return []

    async def process_entry(entry: dict[str, str]) -> Optional[DocChunk | list[DocChunk]]:
        full_url = urljoin(entry["base_url"], entry["uri"])
        html = await http.get(full_url)
        if html is None:
            return None

        parsed = parse_doc_page(html, entry["name"])
        content = parsed["description"]
        if not content and not parsed["signature"]:
            return None

        chunk_id = hashlib.md5(f"{library}:{entry['name']}".encode()).hexdigest()

        chunk = DocChunk(
            chunk_id=chunk_id,
            library=library,
            module_path=entry["name"].rsplit(".", 1)[0] if "." in entry["name"] else library,
            object_name=entry["name"],
            object_type=entry["type"],
            title=entry["name"],
            content=content[:MAX_CHUNK_SIZE],
            signature=parsed["signature"],
            parameters=parsed["parameters"][:MAX_CHUNK_SIZE],
            returns=parsed["returns"][:MAX_CHUNK_SIZE],
            examples=parsed["examples"][:MAX_CHUNK_SIZE],
        )

        # Split large docs into sub-chunks
        total_len = len(content) + len(parsed["parameters"]) + len(parsed["examples"])
        if total_len > MAX_CHUNK_SIZE:
            result_chunks = []
            chunk.parameters = ""
            chunk.returns = ""
            chunk.examples = ""
            result_chunks.append(chunk)

            if parsed["parameters"]:
                result_chunks.append(
                    DocChunk(
                        chunk_id=f"{chunk_id}_params",
                        library=library,
                        module_path=chunk.module_path,
                        object_name=entry["name"],
                        object_type=entry["type"],
                        title=f"{entry['name']} - Parameters",
                        content=parsed["parameters"][:MAX_CHUNK_SIZE],
                        signature=parsed["signature"],
                    )
                )

            if parsed["examples"]:
                result_chunks.append(
                    DocChunk(
                        chunk_id=f"{chunk_id}_examples",
                        library=library,
                        module_path=chunk.module_path,
                        object_name=entry["name"],
                        object_type=entry["type"],
                        title=f"{entry['name']} - Examples",
                        content=parsed["examples"][:MAX_CHUNK_SIZE],
                        signature=parsed["signature"],
                    )
                )
            return result_chunks
        return chunk

    log.info("Fetching %d doc pages for %s concurrently...", len(entries), library)
    results = await asyncio.gather(*[process_entry(e) for e in entries])

    chunks: list[DocChunk] = []
    for r in results:
        if r is None:
            continue
        if isinstance(r, list):
            chunks.extend(r)
        else:
            chunks.append(r)

    log.info("Extracted %d doc chunks for %s (%s)", len(chunks), library, http.stats())
    return chunks


# ===========================================================================
# OpenSearch client
# ===========================================================================


def get_opensearch_client() -> "OpenSearch":
    """Create OpenSearch client with AWS SigV4 auth.

    Auto-detects OpenSearch Serverless (.aoss.) vs managed (.es.) endpoints
    and uses the correct SigV4 service name accordingly. Region is extracted
    from the endpoint URL if AWS_REGION is not set.
    """
    if not HAS_OPENSEARCH:
        raise ImportError(
            "opensearch-py and requests-aws4auth required. "
            "Install with: pip install opensearch-py requests-aws4auth"
        )

    url = os.getenv("AWS_OPENSEARCH_URL", "")
    if not url:
        raise ValueError("AWS_OPENSEARCH_URL environment variable not set")

    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 443

    # Auto-detect region from hostname (e.g., *.ca-central-1.aoss.amazonaws.com)
    region = os.getenv("AWS_REGION", "")
    if not region and host:
        # Extract region from hostname pattern: <id>.<region>.(aoss|es).amazonaws.com
        parts = host.split(".")
        for i, part in enumerate(parts):
            if part in ("aoss", "es") and i >= 2:
                region = parts[i - 1]
                break
    if not region:
        region = "us-east-1"

    # Detect service name: "aoss" for Serverless, "es" for managed OpenSearch
    service = "aoss" if host and ".aoss." in host else "es"
    log.info("OpenSearch endpoint: %s (service=%s, region=%s)", host, service, region)

    awsauth = AWS4Auth(
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_SECRET_ACCESS_KEY"),
        region,
        service,
    )

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=120,
    )

    return client


INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "library": {"type": "keyword"},
            "module_path": {"type": "keyword"},
            "object_name": {"type": "text", "analyzer": "standard"},
            "object_type": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "boost": 2.0,
            },
            "content": {"type": "text", "analyzer": "standard"},
            "signature": {"type": "text", "analyzer": "standard"},
            "parameters": {"type": "text", "analyzer": "standard"},
            "returns": {"type": "text", "analyzer": "standard"},
            "examples": {"type": "text", "analyzer": "standard"},
        }
    },
}


def create_index(client: "OpenSearch", recreate: bool = False) -> None:
    """Create the OpenSearch index."""
    exists = client.indices.exists(index=INDEX_NAME)

    if exists and recreate:
        log.info("Deleting existing index '%s'...", INDEX_NAME)
        client.indices.delete(index=INDEX_NAME)
        exists = False

    if not exists:
        log.info("Creating index '%s'...", INDEX_NAME)
        client.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        log.info("Index '%s' created.", INDEX_NAME)
    else:
        log.info("Index '%s' already exists.", INDEX_NAME)


def index_chunks(client: "OpenSearch", chunks: list[DocChunk], batch_size: int = 500) -> int:
    """Bulk-index chunks into OpenSearch in batches. Returns number indexed."""
    if not chunks:
        return 0

    from opensearchpy.helpers import bulk

    total_success = 0
    total_errors = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        actions = [
            {
                "_index": INDEX_NAME,
                "_id": chunk.chunk_id,
                "_source": chunk.to_dict(),
            }
            for chunk in batch
        ]

        success, errors = bulk(client, actions, raise_on_error=False)
        total_success += success
        if errors:
            total_errors += len(errors)
            for err in errors[:3]:
                log.warning("  Bulk error: %s", err)

        log.info(
            "  Batch %d-%d: indexed %d (%d total so far)",
            i + 1, min(i + batch_size, len(chunks)), success, total_success,
        )

    if total_errors:
        log.warning("Total bulk indexing errors: %d", total_errors)

    log.info("Indexed %d chunks into '%s'", total_success, INDEX_NAME)
    return total_success


# ===========================================================================
# Main orchestration
# ===========================================================================


async def extract_all_docs_async(libraries: list[str], concurrency: int = 10) -> list[DocChunk]:
    """Extract documentation for all specified libraries."""
    all_chunks: list[DocChunk] = []

    async with AsyncCachedHTTPClient(CACHE_DIR, concurrency=concurrency) as http:
        for lib in libraries:
            if lib == "vectorbt":
                chunks = extract_vectorbt_docs()
            elif lib in INVENTORY_URLS:
                chunks = await extract_sphinx_docs(lib, http)
            else:
                log.warning("Unknown library: %s", lib)
                continue
            all_chunks.extend(chunks)

    log.info("Total chunks extracted: %d", len(all_chunks))
    return all_chunks


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build RAG documentation index in OpenSearch"
    )
    parser.add_argument(
        "--lib",
        type=str,
        choices=["numpy", "pandas", "scipy", "vectorbt"],
        help="Index a single library (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and preview chunks without indexing",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the index before indexing",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent HTTP requests (default: 10)",
    )

    args = parser.parse_args()

    if args.lib:
        libraries = [args.lib]
    else:
        libraries = ["numpy", "pandas", "scipy", "vectorbt"]

    log.info("Processing libraries: %s", ", ".join(libraries))

    chunks = asyncio.run(extract_all_docs_async(libraries, concurrency=args.concurrency))

    if not chunks:
        log.warning("No documentation chunks extracted.")
        return

    if args.dry_run:
        log.info("=== DRY RUN: %d chunks extracted ===", len(chunks))
        by_lib: dict[str, int] = {}
        for c in chunks:
            by_lib[c.library] = by_lib.get(c.library, 0) + 1
        for lib, count in sorted(by_lib.items()):
            log.info("  %s: %d chunks", lib, count)

        log.info("--- Sample chunks ---")
        for chunk in chunks[:3]:
            print(json.dumps(chunk.to_dict(), indent=2, default=str))
            print("---")
        return

    # Index into OpenSearch
    client = get_opensearch_client()
    create_index(client, recreate=args.recreate)
    total_indexed = index_chunks(client, chunks)

    log.info("Done. %d chunks indexed into '%s'.", total_indexed, INDEX_NAME)


if __name__ == "__main__":
    main()
