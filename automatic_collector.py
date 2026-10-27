
"""
Automatic ESG Information Collector

Project:
    Advanced ESG Greenwashing Detection Using Web-Based
    Evidence Extraction and Explainable AI

Features:
    1. Search for company ESG information.
    2. Retrieve accessible webpages.
    3. Extract readable HTML text.
    4. Discover PDF report links.
    5. Download accessible PDF reports.
    6. Extract PDF text using PyMuPDF.
    7. Preserve source URLs and metadata.
    8. Continue when individual sources fail.
    9. Save collection results as JSON.

Important:
    Collected information is not automatically verified
    evidence. Missing evidence does not prove greenwashing.
"""

import json
import os
import re
import ssl
import time

from datetime import datetime, timezone
from http import HTTPStatus
from functools import lru_cache
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser

import requests
import pymupdf
import truststore
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

from bs4 import BeautifulSoup

from online_claim_extractor import identify_source_role
from web_search import search_environmental_information


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "data" / "collected_documents"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_HTML_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_ALLOWED_PDF_BYTES = 100 * 1024 * 1024
PDF_SIZE_LIMIT_ENV = "ESG_MAX_PDF_BYTES"


def _configured_pdf_size_limit():
    configured_limit = os.environ.get(PDF_SIZE_LIMIT_ENV)
    if configured_limit is None:
        return DEFAULT_MAX_PDF_BYTES

    try:
        limit = int(configured_limit)
    except ValueError as error:
        raise ValueError(
            f"{PDF_SIZE_LIMIT_ENV} must be an integer byte count."
        ) from error

    if not 1 <= limit <= MAX_ALLOWED_PDF_BYTES:
        raise ValueError(
            f"{PDF_SIZE_LIMIT_ENV} must be between 1 and "
            f"{MAX_ALLOWED_PDF_BYTES} bytes."
        )
    return limit


MAX_PDF_BYTES = _configured_pdf_size_limit()
MAX_REDIRECTS = 10

MAX_PDF_PAGES = 150
MAX_TEXT_CHARS = 200_000
MAX_PDF_ASSURANCE_CHARS = 75_000
MAX_PDF_TEXT_CHARS = MAX_TEXT_CHARS + MAX_PDF_ASSURANCE_CHARS
ASSURANCE_SECTION_PATTERN = re.compile(
    r"\b(?:INDEPENDENT\s+(?:ASSURANCE|REVIEW)\s+STATEMENT|"
    r"(?:COMPREHENSIVE\s+)?CARBON\s+FOOTPRINT\s+LETTER\s+OF\s+ASSURANCE)\b",
    re.IGNORECASE,
)

REQUEST_TIMEOUT = (5, 15)

USER_AGENT = (
    "ESGResearchCollector/1.0 "
    "(educational research project)"
)

# Short delay between document downloads.
REQUEST_DELAY_SECONDS = 0.25

# Search settings.
SEARCH_RESULTS_PER_QUERY = 3
SEARCH_DELAY_SECONDS = 0.25

# Do not attempt an unlimited number of URLs.
MAX_ATTEMPTS_MULTIPLIER = 3
HTTP_RETRY_TOTAL = 2


class _WindowsTrustStoreAdapter(HTTPAdapter):
    """Use OS-trusted certificate authorities without disabling TLS checks."""

    def __init__(self, max_retries=None):
        self._ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        super().__init__(max_retries=max_retries)

    def init_poolmanager(self, connections, maxsize, block=False, **kwargs):
        kwargs["ssl_context"] = self._ssl_context
        super().init_poolmanager(connections, maxsize, block, **kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


# ---------------------------------------------------------
# 2. HTTP SESSION
# ---------------------------------------------------------

def create_session():
    """Create a reusable HTTP session."""

    session = requests.Session()
    retries = Retry(
        total=HTTP_RETRY_TOTAL,
        connect=HTTP_RETRY_TOTAL,
        read=0,
        status=HTTP_RETRY_TOTAL,
        backoff_factor=0.25,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        respect_retry_after_header=True,
        raise_on_status=False,
    )

    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/pdf;q=0.9,*/*;q=0.5"
        ),
    })
    session.mount("https://", _WindowsTrustStoreAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))

    return session


# ---------------------------------------------------------
# 3. URL VALIDATION
# ---------------------------------------------------------

def validate_url(url):
    """Accept HTTP/HTTPS URLs with a valid hostname."""

    if not isinstance(url, str):
        return False

    try:
        parsed = urlparse(url)

        return (
            parsed.scheme.lower() in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
        )

    except (ValueError, AttributeError):
        return False


@lru_cache(maxsize=128)
def load_robots_parser(robots_url):
    """Fetch and cache a site's robots.txt rules."""

    parser = RobotFileParser(robots_url)
    with create_session() as session:
        with session.get(
            robots_url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        ) as response:
            if response.status_code == 404:
                parser.parse([])
            elif response.status_code in (401, 403):
                parser.parse([
                    "User-agent: *",
                    "Disallow: /",
                ])
            else:
                try:
                    response.raise_for_status()
                except requests.HTTPError as error:
                    error.http_status = response.status_code
                    error.final_url = response.url
                    raise
                parser.parse(response.text.splitlines())

    return parser


def is_allowed_by_robots(url):
    """Check whether the collector is allowed to retrieve a URL."""

    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = load_robots_parser(robots_url)
    return parser.can_fetch(USER_AGENT, url)


# ---------------------------------------------------------
# 4. TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):
    """Normalize extracted text."""

    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------------------------------------------------------
# 5. PUBLICATION DATE EXTRACTION
# ---------------------------------------------------------

def extract_page_date(soup):
    """
    Extract a candidate publication date from HTML metadata.

    The returned date is not independently verified.
    """

    selectors = [
        {"property": "article:published_time"},
        {"property": "og:published_time"},
        {"name": "date"},
        {"name": "pubdate"},
        {"name": "publication_date"},
        {"name": "DC.date"},
    ]

    for selector in selectors:

        tag = soup.find("meta", attrs=selector)

        if tag:
            value = tag.get("content", "").strip()

            if value:
                return value

    time_tag = soup.find("time")

    if time_tag:

        value = (
            time_tag.get("datetime", "")
            or time_tag.get_text(" ", strip=True)
        )

        if value:
            return value

    return None


def extract_page_publisher(soup):
    """Extract a publisher name exposed in common page metadata."""

    selectors = [
        {"property": "og:site_name"},
        {"name": "application-name"},
        {"name": "publisher"},
    ]

    for selector in selectors:
        tag = soup.find("meta", attrs=selector)
        if tag:
            value = tag.get("content", "").strip()
            if value:
                return value

    return None


# ---------------------------------------------------------
# 6. PDF DETECTION
# ---------------------------------------------------------

def is_pdf_response(content_type, url, content):
    """Determine whether retrieved content appears to be PDF."""

    content_type = (content_type or "").lower()
    path = urlparse(url).path.lower()

    return (
        "application/pdf" in content_type
        or path.endswith(".pdf")
        or content[:5] == b"%PDF-"
    )


# ---------------------------------------------------------
# 7. DOWNLOAD WITH SIZE LIMITS
# ---------------------------------------------------------

def download_content(session, url, max_bytes):
    """
    Download a resource with size and redirect checks.

    Returns:
        metadata, content_bytes
    """

    current_url = url
    response = None

    for redirect_number in range(MAX_REDIRECTS + 1):
        if not validate_url(current_url):
            raise ValueError("Invalid or unsupported URL.")

        if not is_allowed_by_robots(current_url):
            raise PermissionError(
                "robots.txt disallows collection of this URL."
            )

        response = session.get(
            current_url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=False,
        )

        if not response.is_redirect:
            break

        redirect_url = response.headers.get("Location")
        response.close()

        if not redirect_url:
            raise ValueError("The redirect response did not include a Location.")

        if redirect_number == MAX_REDIRECTS:
            raise ValueError("The resource exceeded the redirect limit.")

        current_url = urljoin(current_url, redirect_url)

    if response is None:
        raise ValueError("The resource could not be retrieved.")

    with response:

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            error.http_status = response.status_code
            error.final_url = response.url
            raise

        if not validate_url(response.url):
            raise ValueError(
                "Redirect destination is not an accepted web URL."
            )

        content_length = response.headers.get("Content-Length")

        if content_length:

            try:
                declared_size = int(content_length)

            except ValueError:
                declared_size = None

            if declared_size is not None and declared_size > max_bytes:
                error = ValueError(
                    f"Resource exceeds the {max_bytes}-byte download limit."
                )
                error.http_status = response.status_code
                error.final_url = response.url
                raise error

        chunks = []
        total_bytes = 0

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):

            if not chunk:
                continue

            total_bytes += len(chunk)

            if total_bytes > max_bytes:
                error = ValueError(
                    f"Resource exceeds the {max_bytes}-byte download limit."
                )
                error.http_status = response.status_code
                error.final_url = response.url
                raise error

            chunks.append(chunk)

        content = b"".join(chunks)

        metadata = {
            "final_url": response.url,
            "status_code": response.status_code,
            "content_type": response.headers.get(
                "Content-Type", ""
            ),
            "headers": dict(response.headers),
        }

        return metadata, content


def _collection_error_message(error):
    """Return a concise, actionable message for failed source retrieval."""

    status_code = getattr(error, "http_status", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    if status_code is not None:
        try:
            phrase = HTTPStatus(status_code).phrase
        except ValueError:
            phrase = "HTTP request failed"
        if status_code in {401, 403}:
            return (
                f"HTTP {status_code} {phrase}: the source denied this "
                "request; automated collection was not authorized."
            )
        return f"HTTP {status_code} {phrase}: the source could not be retrieved."

    message = str(error).strip() or type(error).__name__
    return f"{type(error).__name__}: {message}"


# ---------------------------------------------------------
# 8. BASE DOCUMENT
# ---------------------------------------------------------

def create_document(url, search_result=None, content_type="HTML"):
    """Create a consistent document metadata dictionary."""

    search_result = search_result or {}

    return {
        "title": search_result.get("title", ""),
        "source_url": url,
        "final_url": None,
        "source_domain": urlparse(url).hostname or "",
        "source_type": search_result.get(
            "source_type", "Unclassified"
        ),
        "snippet": search_result.get("snippet", ""),
        "search_result_date": search_result.get("search_result_date"),
        "search_query": search_result.get("search_query", ""),
        "attribution_status": search_result.get(
            "attribution_status",
            "not_recorded",
        ),
        "attribution_basis": search_result.get("attribution_basis", ""),
        "content_type": content_type,
        "publication_date": None,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "text": "",
        "status": "pending",
        "error": None,
    }


# ---------------------------------------------------------
# 9. HTML EXTRACTION
# ---------------------------------------------------------

def extract_html_document(session, url, search_result=None):
    """Retrieve an HTML page and extract readable text."""

    document = create_document(
        url,
        search_result,
        content_type="HTML",
    )

    document["discovered_pdf_urls"] = []

    try:

        metadata, content = download_content(
            session,
            url,
            MAX_HTML_BYTES,
        )

        document["final_url"] = metadata["final_url"]
        document["http_status"] = metadata["status_code"]

        content_type = metadata["content_type"].lower()

        if (
            content_type
            and "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):

            if is_pdf_response(
                content_type,
                metadata["final_url"],
                content,
            ):

                document["status"] = "skipped"
                document["error"] = (
                    "The URL returned PDF content, not HTML."
                )

            else:

                document["status"] = "skipped"
                document["error"] = (
                    "The resource is not identified as an HTML page."
                )

            return document

        encoding = requests.utils.get_encoding_from_headers(
            metadata["headers"]
        ) or "utf-8"

        html = content.decode(
            encoding,
            errors="replace",
        )

        soup = BeautifulSoup(html, "html.parser")

        if soup.title:

            title = soup.title.get_text(
                " ",
                strip=True,
            )

            if title:
                document["title"] = title

        document["publication_date"] = extract_page_date(soup)
        document["publisher"] = extract_page_publisher(soup)

        # Remove common navigation and non-content elements.
        for element in soup([
            "script",
            "style",
            "noscript",
            "svg",
            "nav",
            "footer",
            "header",
            "form",
        ]):

            element.decompose()

        text = soup.get_text(
            separator="\n",
            strip=True,
        )

        document["text"] = clean_text(text)[:MAX_TEXT_CHARS]

        # Discover potentially relevant PDF reports.
        pdf_links = []

        for link in soup.find_all("a", href=True):

            absolute_url = urljoin(
                metadata["final_url"],
                link["href"].strip(),
            )

            absolute_url, _ = urldefrag(absolute_url)

            if not validate_url(absolute_url):
                continue

            path = urlparse(absolute_url).path.lower()

            link_text = link.get_text(
                " ",
                strip=True,
            ).lower()

            looks_like_report = any(
                keyword in link_text
                for keyword in (
                    "sustainability",
                    "impact report",
                    "environmental report",
                    "esg report",
                    "climate report",
                )
            )

            if (
                path.endswith(".pdf")
                or (
                    looks_like_report
                    and "pdf" in link_text
                )
            ):

                if absolute_url not in pdf_links:
                    pdf_links.append(absolute_url)

        document["discovered_pdf_urls"] = pdf_links[:20]

        if document["text"]:
            document["status"] = "success"

        else:
            document["status"] = "empty"
            document["error"] = (
                "No readable text was extracted."
            )

    except Exception as error:

        document["http_status"] = getattr(error, "http_status", None)
        document["final_url"] = (
            getattr(error, "final_url", None)
            or getattr(getattr(error, "response", None), "url", None)
        )
        document["status"] = "failed"
        document["error"] = _collection_error_message(error)

    return document


# ---------------------------------------------------------
# 10. PDF EXTRACTION
# ---------------------------------------------------------

def extract_pdf_document(session, url, search_result=None):
    """Download an accessible PDF and extract its text."""

    document = create_document(
        url,
        search_result,
        content_type="PDF",
    )

    document["page_count"] = None

    try:

        metadata, content = download_content(
            session,
            url,
            MAX_PDF_BYTES,
        )

        document["final_url"] = metadata["final_url"]
        document["http_status"] = metadata["status_code"]

        if not is_pdf_response(
            metadata["content_type"],
            metadata["final_url"],
            content,
        ):

            document["status"] = "skipped"
            document["error"] = (
                "The retrieved resource does not appear to be a PDF."
            )

            return document

        with pymupdf.open(
            stream=content,
            filetype="pdf",
        ) as pdf:

            document["page_count"] = len(pdf)

            extracted_pages = []
            character_count = 0
            assurance_pages = []
            assurance_character_count = 0
            assurance_pages_remaining = 0

            pages_to_read = min(
                len(pdf),
                MAX_PDF_PAGES,
            )

            for page_index in range(pages_to_read):

                page = pdf.load_page(page_index)

                page_text = page.get_text("text")

                if not page_text:
                    continue

                if ASSURANCE_SECTION_PATTERN.search(page_text):
                    assurance_pages_remaining = 3

                # Retain a bounded window around titled assurance appendices.
                remaining = MAX_TEXT_CHARS - character_count

                if remaining > 0:
                    primary_page_text = page_text[:remaining]
                    extracted_pages.append(primary_page_text)
                    character_count += len(primary_page_text)
                elif assurance_pages_remaining > 0:
                    assurance_remaining = (
                        MAX_PDF_ASSURANCE_CHARS
                        - assurance_character_count
                    )
                    if assurance_remaining > 0:
                        assurance_page_text = page_text[:assurance_remaining]
                        assurance_pages.append(assurance_page_text)
                        assurance_character_count += len(
                            assurance_page_text
                        )

                if assurance_pages_remaining > 0:
                    assurance_pages_remaining -= 1

            document["text"] = clean_text(
                "\n".join([*extracted_pages, *assurance_pages])
            )[:MAX_PDF_TEXT_CHARS]

            if document["text"]:
                document["status"] = "success"

            else:
                document["status"] = "empty"
                document["error"] = (
                    "No extractable PDF text was found. "
                    "The report may require OCR."
                )

    except Exception as error:

        document["http_status"] = getattr(error, "http_status", None)
        document["final_url"] = (
            getattr(error, "final_url", None)
            or getattr(getattr(error, "response", None), "url", None)
        )
        document["status"] = "failed"
        document["error"] = _collection_error_message(error)

    return document


# ---------------------------------------------------------
# 11. COLLECT ONE SEARCH RESULT
# ---------------------------------------------------------

def collect_search_result(session, search_result):
    """Collect a search result as HTML or PDF."""

    url = (search_result.get("url") or "").strip()

    if not validate_url(url):

        document = create_document(
            url,
            search_result,
        )

        document["status"] = "skipped"
        document["error"] = "Invalid or unsupported URL."

        return document

    content_type = (
        search_result.get("content_type", "").upper()
    )

    path = urlparse(url).path.lower()

    if content_type == "PDF" or path.endswith(".pdf"):

        return extract_pdf_document(
            session,
            url,
            search_result,
        )

    document = extract_html_document(
        session,
        url,
        search_result,
    )

    # If a URL without a .pdf extension returns PDF content,
    # retry with the PDF extractor.
    if (
        document.get("status") == "skipped"
        and "returned PDF content" in (
            document.get("error") or ""
        )
    ):

        return extract_pdf_document(
            session,
            url,
            search_result,
        )

    return document


# ---------------------------------------------------------
# 12. SAVE DOCUMENTS
# ---------------------------------------------------------

def _document_diagnostic(document):
    source_url = document.get("source_url", "")
    final_url = document.get("final_url") or source_url
    role_document = {
        **document,
        "source_domain": document.get("source_domain")
        or urlparse(final_url).hostname
        or "",
    }
    status = document.get("status", "unknown")
    return {
        "source_url": source_url,
        "final_url": final_url,
        "search_query": document.get("search_query", ""),
        "attribution_status": document.get(
            "attribution_status",
            "not_recorded",
        ),
        "attribution_basis": document.get("attribution_basis", ""),
        "http_status": document.get("http_status"),
        "source_type": document.get("source_type", ""),
        "source_role": identify_source_role(role_document),
        "download_status": status,
        "extraction_status": (
            "success"
            if status == "success" and str(document.get("text") or "").strip()
            else "empty"
            if status == "empty"
            else "failed"
            if status == "failed"
            else "skipped"
            if status == "skipped"
            else "unavailable"
        ),
        "extracted_characters": len(str(document.get("text") or "")),
        "error": document.get("error"),
    }


def build_collection_diagnostics(documents, search_results_count=0):
    document_records = [
        _document_diagnostic(document)
        for document in documents
    ]
    role_counts = {}
    for record in document_records:
        role = record["source_role"]
        role_counts[role] = role_counts.get(role, 0) + 1
    return {
        "schema_version": 1,
        "search": {
            "status": "not_recorded",
            "result_count": search_results_count,
            "queries": [],
            "search_error_count": 0,
        },
        "downloads": {
            "search_result_count": search_results_count,
            "attempt_count": len(documents),
            "successful_extraction_count": sum(
                record["extraction_status"] == "success"
                for record in document_records
            ),
            "failed_download_count": sum(
                record["download_status"] == "failed"
                for record in document_records
            ),
            "empty_extraction_count": sum(
                record["extraction_status"] == "empty"
                for record in document_records
            ),
            "skipped_document_count": sum(
                record["download_status"] == "skipped"
                for record in document_records
            ),
            "source_role_counts": role_counts,
            "documents": document_records,
        },
    }


def finalize_search_diagnostics(
    diagnostics,
    search_results,
    search_errors,
    search_type,
):
    diagnostics.setdefault("search_type", search_type)
    diagnostics.setdefault(
        "result_count",
        len(search_results),
    )
    diagnostics.setdefault(
        "search_error_count",
        len(search_errors),
    )
    if "status" not in diagnostics:
        diagnostics["status"] = (
            "completed"
            if search_results and not search_errors
            else "partial"
            if search_results
            else "failed"
            if search_errors
            else "no_results"
        )


def attach_result_attempt_diagnostics(
    search_diagnostics,
    search_results,
    documents,
    max_documents,
):
    """Persist accepted search candidates and their collection outcomes."""
    candidate_results = search_diagnostics.get("candidate_results")
    if not isinstance(candidate_results, list) or len(candidate_results) != len(
        search_results
    ):
        candidate_results = [
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "domain": result.get("domain", ""),
                "source_type": result.get("source_type", ""),
                "content_type": result.get("content_type", ""),
                "search_result_date": result.get("search_result_date"),
                "attribution_status": result.get(
                    "attribution_status",
                    "not_recorded",
                ),
                "attribution_basis": result.get("attribution_basis", ""),
                "queries": (
                    [{
                        "query": result.get("search_query", ""),
                        "backend": result.get("search_backend", "auto"),
                    }]
                    if result.get("search_query")
                    else []
                ),
            }
            for result in search_results
        ]
    else:
        candidate_results = [
            dict(candidate)
            for candidate in candidate_results
        ]

    attempted_by_url = {
        str(document.get("source_url") or "").rstrip("/").casefold(): document
        for document in documents
        if document.get("source_url")
    }
    attempt_limit = min(
        len(search_results),
        max_documents * MAX_ATTEMPTS_MULTIPLIER,
    )
    attempted_count = 0

    for index, candidate in enumerate(candidate_results):
        url = str(candidate.get("url") or "")
        document = attempted_by_url.get(url.rstrip("/").casefold())
        if document is not None:
            attempted_count += 1
            candidate["download_status"] = "attempted"
            candidate["document_status"] = document.get("status", "unknown")
            candidate["http_status"] = document.get("http_status")
            candidate["extracted_characters"] = len(
                str(document.get("text") or "")
            )
            if document.get("error"):
                candidate["download_error"] = document["error"]
            continue

        candidate["download_status"] = "not_attempted"
        candidate["not_attempted_reason"] = (
            "attempt_limit_reached"
            if index >= attempt_limit
            else "successful_document_limit_reached"
        )

    search_diagnostics["candidate_results"] = candidate_results
    search_diagnostics["attempted_result_count"] = attempted_count
    search_diagnostics["not_attempted_result_count"] = (
        len(candidate_results) - attempted_count
    )
    search_diagnostics["attempt_limit"] = attempt_limit
    search_diagnostics["successful_document_limit"] = max_documents


def persist_collection_diagnostics(output_file, diagnostics):
    """Merge diagnostics into an existing collection artifact."""
    if not output_file:
        return False
    path = Path(output_file)
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    existing = payload.get("diagnostics")
    if not isinstance(existing, dict):
        existing = {}
    existing.update(diagnostics)
    payload["diagnostics"] = existing
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    return True


def save_documents(documents, search_name):
    """Save collected documents to a timestamped JSON file."""

    safe_name = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        search_name.strip(),
    ).strip("_")

    if not safe_name:
        safe_name = "esg_search"

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_file = OUTPUT_DIR / (
        f"{safe_name}_{timestamp}.json"
    )

    payload = {
        "search_name": search_name,
        "collected_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "document_count": len(documents),
        "diagnostics": build_collection_diagnostics(documents),
        "documents": documents,
    }

    collision = 1
    while True:
        try:
            file = output_file.open("x", encoding="utf-8")
            break
        except FileExistsError:
            output_file = OUTPUT_DIR / (
                f"{safe_name}_{timestamp}_{collision}.json"
            )
            collision += 1

    with file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_file


# ---------------------------------------------------------
# 13. COLLECT SEARCH RESULTS WITH FALLBACK
# ---------------------------------------------------------

def collect_documents(
    search_results,
    search_name,
    max_documents=8,
    progress_callback=None,
):
    """
    Try multiple search results until enough successful
    documents are collected or the attempt limit is reached.

    All attempted results are recorded in the output JSON.
    """

    if (
        not isinstance(max_documents, int)
        or isinstance(max_documents, bool)
        or max_documents < 1
    ):

        raise ValueError(
            "max_documents must be a positive integer."
        )

    documents = []

    if not search_results:

        output_file = save_documents(
            documents,
            search_name,
        )

        return documents, output_file

    # Try more results than the desired success count.
    max_attempts = min(
        len(search_results),
        max_documents * MAX_ATTEMPTS_MULTIPLIER,
    )

    attempted_results = search_results[:max_attempts]

    session = create_session()

    successful_count = 0

    try:

        for index, result in enumerate(
            attempted_results,
            start=1,
        ):

            # Stop when the requested success count is reached.
            if successful_count >= max_documents:
                break

            url = result.get("url", "")

            print(
                f"\nAttempt {index}/{max_attempts}"
            )

            print(f"Title: {result.get('title', '')}")
            print(f"URL: {url}")

            document = collect_search_result(
                session,
                result,
            )
            document["source_role"] = identify_source_role(document)

            documents.append(document)

            status = document.get("status", "unknown")
            text_length = len(document.get("text", ""))

            print(f"Status: {status}")
            print(f"Extracted characters: {text_length}")

            if document.get("error"):
                print(f"Reason: {document['error']}")

            if status == "success" and text_length > 0:

                successful_count += 1

                print(
                    f"Successful documents: "
                    f"{successful_count}/{max_documents}"
                )

            else:

                print(
                    "This result did not produce usable text. "
                    "Trying another source if available."
                )

            if progress_callback is not None:
                progress_callback(index, max_attempts, document)

            if (
                successful_count < max_documents
                and index < max_attempts
                and REQUEST_DELAY_SECONDS > 0
            ):

                time.sleep(REQUEST_DELAY_SECONDS)

    finally:
        session.close()

    output_file = save_documents(
        documents,
        search_name,
    )

    print("\n" + "=" * 65)
    print("DOCUMENT COLLECTION FINISHED")
    print("=" * 65)
    print(f"Search results available: {len(search_results)}")
    print(f"URLs attempted: {len(documents)}")
    print(f"Successful documents: {successful_count}")
    print(f"Requested successful documents: {max_documents}")
    print(f"Output file: {output_file}")

    return documents, output_file


# ---------------------------------------------------------
# 14. COLLECTION SUMMARY
# ---------------------------------------------------------

def display_collection_summary(documents, output_file):

    print("\n" + "=" * 70)
    print("AUTOMATIC COLLECTION SUMMARY")
    print("=" * 70)

    for index, document in enumerate(
        documents,
        start=1,
    ):

        print(
            f"\nDocument {index}: "
            f"{document.get('title', 'Untitled')}"
        )

        print(f"Status: {document.get('status')}")
        print(f"Source: {document.get('source_url')}")

        print(
            "Extracted characters: "
            f"{len(document.get('text', ''))}"
        )

        if document.get("publication_date"):
            print(
                "Publication date candidate: "
                f"{document['publication_date']}"
            )

        if document.get("error"):
            print(f"Error: {document['error']}")

    successful = sum(
        1
        for document in documents
        if (
            document.get("status") == "success"
            and document.get("text", "").strip()
        )
    )

    print("\n" + "=" * 70)
    print(
        f"Documents collected successfully: "
        f"{successful}/{len(documents)}"
    )
    print(f"JSON output: {output_file}")
    print("=" * 70)


# ---------------------------------------------------------
# 15. COMPANY ESG COLLECTION API
# ---------------------------------------------------------

def collect_company_esg_information(
    company_name,
    max_documents=8,
    progress_callback=None,
    error_callback=None,
    stage_callback=None,
):
    """
    Search for company ESG information and collect documents.

    This function is suitable for calling from Streamlit.
    """

    if not isinstance(company_name, str):

        raise ValueError(
            "Company name must be text."
        )

    company_name = company_name.strip()

    if not company_name:

        raise ValueError(
            "Company name cannot be empty."
        )

    if (
        not isinstance(max_documents, int)
        or isinstance(max_documents, bool)
        or max_documents < 1
    ):

        raise ValueError(
            "max_documents must be a positive integer."
        )

    print(
        f"\nSearching ESG information for: {company_name}"
    )

    search_errors = []

    def report_search_error(query, message):
        search_errors.append((query, message))
        if error_callback is not None:
            error_callback(query, message)

    if stage_callback is not None:
        stage_callback("Web Search", "RUNNING", "Searching configured sources.")

    search_diagnostics = {}
    try:
        search_results = search_environmental_information(
            name=company_name,
            search_type="company",
            max_results_per_query=SEARCH_RESULTS_PER_QUERY,
            delay_seconds=SEARCH_DELAY_SECONDS,
            error_callback=report_search_error,
            diagnostics=search_diagnostics,
        )
    except Exception as error:
        search_diagnostics.update({
            "search_type": "company",
            "status": "failed",
            "query_count": 0,
            "queries": [],
            "result_count": 0,
            "search_error_count": 1,
            "error": f"{type(error).__name__}: {error}",
        })
        documents, output_file = collect_documents(
            search_results=[],
            search_name=company_name,
            max_documents=max_documents,
            progress_callback=progress_callback,
        )
        diagnostics = build_collection_diagnostics(documents)
        diagnostics["search"] = search_diagnostics
        persist_collection_diagnostics(output_file, diagnostics)
        if stage_callback is not None:
            stage_callback("Web Search", "FAILED", "Web search did not complete.")
            stage_callback(
                "Web Data Collection",
                "NOT AVAILABLE",
                "No search results were available to collect.",
            )
        raise

    finalize_search_diagnostics(
        search_diagnostics,
        search_results,
        search_errors,
        "company",
    )

    search_status = (
        "COMPLETED"
        if search_results and not search_errors
        else "PARTIAL"
    )
    if stage_callback is not None:
        stage_callback(
            "Web Search",
            search_status,
            (
                "Search completed with results."
                if search_results
                else "Search returned no results."
            ),
        )

    if not search_results:
        documents, output_file = collect_documents(
            search_results=[],
            search_name=company_name,
            max_documents=max_documents,
            progress_callback=progress_callback,
        )
        attach_result_attempt_diagnostics(
            search_diagnostics,
            search_results,
            documents,
            max_documents,
        )
        diagnostics = build_collection_diagnostics(documents)
        diagnostics["search"] = search_diagnostics
        persist_collection_diagnostics(output_file, diagnostics)
        if stage_callback is not None:
            stage_callback(
                "Web Data Collection",
                "PARTIAL",
                "No results were available to collect.",
            )

        return {
            "company": company_name,
            "search_results_count": 0,
            "documents": documents,
            "successful_documents": 0,
            "output_file": str(output_file),
            "search_errors": search_errors,
            "diagnostics": diagnostics,
        }

    if stage_callback is not None:
        stage_callback(
            "Web Data Collection",
            "RUNNING",
            "Collecting and parsing search-result documents.",
        )

    try:
        documents, output_file = collect_documents(
            search_results=search_results,
            search_name=company_name,
            max_documents=max_documents,
            progress_callback=progress_callback,
        )
    except Exception:
        if stage_callback is not None:
            stage_callback(
                "Web Data Collection",
                "FAILED",
                "Document collection did not complete.",
            )
        raise

    attach_result_attempt_diagnostics(
        search_diagnostics,
        search_results,
        documents,
        max_documents,
    )
    diagnostics = build_collection_diagnostics(
        documents,
        search_results_count=len(search_results),
    )
    diagnostics["search"] = search_diagnostics
    persist_collection_diagnostics(output_file, diagnostics)

    successful_documents = sum(
        1
        for document in documents
        if (
            document.get("status") == "success"
            and document.get("text", "").strip()
        )
    )
    failed_documents = sum(
        document.get("status") in {"failed", "empty"}
        for document in documents
    )
    collection_status = (
        "COMPLETED"
        if successful_documents and not failed_documents
        else "PARTIAL"
    )
    if stage_callback is not None:
        stage_callback(
            "Web Data Collection",
            collection_status,
            (
                "Documents collected."
                if collection_status == "COMPLETED"
                else "Some sources were unavailable or no usable documents were collected."
            ),
        )

    return {
        "company": company_name,
        "search_results_count": len(search_results),
        "documents": documents,
        "successful_documents": successful_documents,
        "output_file": str(output_file),
        "search_errors": search_errors,
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------
# 16. INTERACTIVE TEST
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("ADVANCED ESG GREENWASHING DETECTION")
    print("Automatic Information Collector")
    print("=" * 65)

    search_name = input(
        "\nEnter company name: "
    ).strip()

    if not search_name:

        print(
            "Company name cannot be empty."
        )

        return

    try:

        print(
            "\nSearching for environmental information..."
        )

        search_results = search_environmental_information(
            name=search_name,
            search_type="company",
            max_results_per_query=SEARCH_RESULTS_PER_QUERY,
            delay_seconds=SEARCH_DELAY_SECONDS,
        )

        if not search_results:

            print(
                "\nNo search results were available."
            )

            return

        print(
            f"\nSearch results retrieved: "
            f"{len(search_results)}"
        )

        documents, output_file = collect_documents(
            search_results=search_results,
            search_name=search_name,
            max_documents=8,
        )

        display_collection_summary(
            documents,
            output_file,
        )

    except ValueError as error:

        print(f"Input error: {error}")

    except Exception as error:

        print(
            f"Collection error: "
            f"{type(error).__name__}: {error}"
        )


if __name__ == "__main__":
    main()
