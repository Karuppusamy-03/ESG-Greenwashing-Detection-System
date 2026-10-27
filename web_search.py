"""Web Search Module
Advanced ESG Greenwashing Detection

Features:
    1. Company environmental searches
    2. Official company domain identification
    3. Government and regulatory source classification
    4. Independent research discovery
    5. Search-result deduplication
    6. PDF report discovery
    7. Source prioritization
    8. Search error handling
    9. Configurable search timeout

Important:
    Search results are not verified evidence.
    Domain classification does not prove factual accuracy.
"""

from ddgs import DDGS
from urllib.parse import urlparse, urldefrag
import re
import time


# ---------------------------------------------------------
# 1. CONFIGURED GOVERNMENT AND TRUSTED DOMAINS
# ---------------------------------------------------------

GOVERNMENT_DOMAINS = {
    "sec.gov",
    "epa.gov",
    "eia.gov",
    "energy.gov",
    "environment.gov.in",
    "moef.gov.in",
}

TRUSTED_DOMAINS = {
    "unfccc.int",
    "ipcc.ch",
    "iea.org",
}


# ---------------------------------------------------------
# 2. OFFICIAL COMPANY DOMAINS
# ---------------------------------------------------------

OFFICIAL_COMPANY_DOMAINS = {
    "tesla": ["tesla.com"],
    "apple": ["apple.com"],
    "samsung": ["samsung.com", "samsungsemiconductor.com"],
    "samsungelectronics": ["samsung.com", "samsungsemiconductor.com"],
    "microsoft": ["microsoft.com"],
}
# ---------------------------------------------------------
# 3. SEARCH QUERIES
# ---------------------------------------------------------

SEARCH_QUERIES = {
    "company": [
        (
            '"{name}" official sustainability report '
            'environmental impact emissions'
        ),
        (
            '"{name}" independent environmental assessment '
            'lifecycle emissions carbon footprint'
        ),
        (
            '"{name}" environmental reporting investigation '
            'regulatory filing emissions'
        ),
        (
            '"{name}" carbon emissions renewable energy '
            'environmental disclosure'
        ),
        (
            '"{name}" government regulatory filing '
            'environmental emissions'
        ),
        (
            '"{name}" independent environmental research '
            'carbon emissions sustainability'
        ),
        (
            '"{name}" sustainability report limited assurance '
            'emissions verification'
        ),
        (
            '"{name}" environmental performance emissions '
            'independent assessment reporting year'
        ),
        (
            '"{name}" climate disclosure emissions data '
            'independent review'
        ),
    ],
}


# ---------------------------------------------------------
# 4. SEARCH CONFIGURATION
# ---------------------------------------------------------

DEFAULT_SEARCH_TIMEOUT = 3
DEFAULT_RESULTS_PER_QUERY = 3
DEFAULT_SEARCH_DELAY = 0.25
CLAIM_EVIDENCE_SEARCH_BACKENDS = ("google", "brave", "yahoo")
SEARCH_ENVIRONMENTAL_TERMS = (
    "environment",
    "emissions",
    "greenhouse gas",
    "carbon",
    "climate",
    "sustainability",
    "sustainable",
    "renewable",
    "recycled",
    "recycling",
    "lifecycle",
    "life cycle",
    "environmental impact",
    "carbon footprint",
    "energy efficiency",
    "pollution",
    "waste",
    "water consumption",
)


# ---------------------------------------------------------
# 5. NORMALIZE COMPANY NAMES
# ---------------------------------------------------------

def normalize_company_name(name):
    """
    Normalize common company-name variations.

    Examples:
        Tesla Inc. -> tesla
        Microsoft Corporation -> microsoft
    """

    name = (name or "").lower().strip()

    name = re.sub(
        r"\b(incorporated|inc|corporation|corp|"
        r"limited|ltd|llc|plc)\b",
        "",
        name,
    )

    return re.sub(r"[^a-z0-9]+", "", name)


def is_ambiguous_apple_food_result(company_name, *text_parts):
    """Reject food-related Apple mentions that lack company context."""

    if normalize_company_name(company_name) != "apple":
        return False

    text = " ".join(str(part or "") for part in text_parts).casefold()
    tokens = re.findall(r"[a-z0-9]+", text)
    food_context = {
        "activity", "activities", "canned", "canning", "food",
        "fruit", "fruits", "friends", "game", "games", "orchard",
        "preserved", "preserves", "quiz", "quizzes", "student",
        "students", "vegetable", "vegetables", "juice", "produce",
    }
    if not food_context.intersection(tokens) or "apple" not in tokens:
        return False

    company_context = {
        "corporate", "inc", "incorporated", "manufacturer",
        "supplier", "business", "operations", "facilities",
        "cupertino",
    }
    apple_positions = [
        index for index, token in enumerate(tokens)
        if token == "apple"
    ]
    has_nearby_company_context = any(
        company_context.intersection(tokens[max(0, index - 5):index + 6])
        for index in apple_positions
    )
    has_apple_product_context = any(
        token in tokens
        for token in {"iphone", "ipad", "macbook", "mac", "watch", "ios"}
    )
    has_company_name_form = bool(
        re.search(
            r"\bapple\s+(?:incorporated|inc\.?|computer)\b"
            r"|\bapple['’]s\b",
            text,
        )
    )
    return not (
        has_nearby_company_context
        or has_apple_product_context
        or has_company_name_form
    )


LOCAL_BASELINE_YEAR_PATTERN = re.compile(
    r"\b(?:compared\s+(?:to|with)|versus|vs\.?)\s+"
    r"(?:(?:a|the|our|its|their)\s+)?(?P<year_cmp>(?:19|20)\d{2})\b"
    r"|\bbaseline(?:\s+year)?\s*[:=]\s*(?P<year>(?:19|20)\d{2})\b"
    r"|\b(?P<year_before>(?:19|20)\d{2})\s+baseline\b",
    re.IGNORECASE,
)


def claim_reporting_year(text):
    """
    Return the single unambiguous reporting year mentioned in a claim.

    Target years ("by 2030"), baseline or comparison years ("compared with
    2019", "from its 2020 baseline", "2020 baseline") are excluded so they
    cannot be used as the reporting period. If zero or several candidate
    years remain, no year is returned rather than guessing one.
    """

    from timeline_analyzer import (
        BASELINE_YEAR_PATTERN,
        TARGET_YEAR_PATTERN,
        YEAR_PATTERN,
    )

    text = str(text or "")
    excluded_spans = set()
    for match in TARGET_YEAR_PATTERN.finditer(text):
        excluded_spans.add(match.span("year"))
    for match in BASELINE_YEAR_PATTERN.finditer(text):
        group = "year_a" if match.group("year_a") else "year_b"
        excluded_spans.add(match.span(group))
    for match in LOCAL_BASELINE_YEAR_PATTERN.finditer(text):
        for group in ("year_cmp", "year", "year_before"):
            if match.group(group):
                excluded_spans.add(match.span(group))

    candidates = [
        match.group(0)
        for match in YEAR_PATTERN.finditer(text)
        if match.span() not in excluded_spans
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None


def build_claim_evidence_queries(company_name, claims, max_queries=3):
    """Build a small set of company- and claim-specific evidence searches."""

    if not isinstance(company_name, str) or not company_name.strip():
        raise ValueError("company_name must be a non-empty string.")
    if (
        not isinstance(max_queries, int)
        or isinstance(max_queries, bool)
        or max_queries < 1
    ):
        raise ValueError("max_queries must be a positive integer.")

    topic_patterns = (
        (
            r"\b(?:scope\s*[123]|greenhouse gas|ghg|carbon|co2e?|emissions?)\b",
            "carbon emissions",
        ),
        (
            r"\b(?:renewable|clean)\s+electricity\b",
            "renewable electricity",
        ),
        (
            r"\b(?:renewable|clean)\s+(?:energy|power)\b",
            "renewable energy",
        ),
        (
            r"\b(?:recycl(?:ed|ing|able)|circularity|materials?)\b",
            "recycled materials",
        ),
        (r"\bwater\b", "water consumption"),
        (r"\b(?:waste|pollution)\b", "waste pollution"),
    )
    measurement_pattern = re.compile(
        r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|tonnes?|tons?|t\s*co2e?|"
        r"kg\s*co2e?|mwh|gwh|kwh)(?=$|[^a-z0-9])",
        re.IGNORECASE,
    )
    year_pattern = re.compile(r"\b(?:19|20)\d{2}\b")
    candidates = []
    seen = set()

    for claim in claims or []:
        text = (
            str(claim.get("statement") or "")
            if isinstance(claim, dict)
            else str(claim or "")
        )
        if not text.strip():
            continue
        lowered = text.casefold()
        topic = next(
            (
                label
                for pattern, label in topic_patterns
                if re.search(pattern, lowered)
            ),
            "environmental performance",
        )
        measurement = measurement_pattern.search(text)
        year = claim_reporting_year(text)
        parts = [f'"{company_name.strip()}"', f'"{topic}"']
        if measurement:
            parts.append(f'"{measurement.group(0)}"')
        if year:
            parts.append(year)
        parts.extend(("independent", "verification assessment"))
        queries_for_claim = [" ".join(parts)]
        if measurement:
            broader_parts = [
                f'"{company_name.strip()}"',
                f'"{topic}"',
            ]
            if year:
                broader_parts.append(year)
            broader_parts.extend(
                ("independent", "third-party", "assurance", "verification")
            )
            queries_for_claim.append(" ".join(broader_parts))
            if topic == "renewable energy":
                claim_phrase = (
                    "renewable energy transition rate"
                    if re.search(r"\btransition\s+rate\b", lowered)
                    else "renewable energy use"
                )
                phrase_parts = [
                    f'"{company_name.strip()}"',
                    f'"{claim_phrase}"',
                ]
                if year:
                    phrase_parts.append(year)
                phrase_parts.append(
                    "independent assurance verification assessment"
                )
                queries_for_claim.append(" ".join(phrase_parts))

        for query in queries_for_claim:
            key = query.casefold()
            if key in seen:
                continue
            seen.add(key)
            candidates.append((topic, query))

    queries = []
    seen_topics = set()
    for topic, query in candidates:
        if topic not in seen_topics:
            queries.append(query)
            seen_topics.add(topic)
            if len(queries) >= max_queries:
                return queries
    for _, query in candidates:
        if query not in queries:
            queries.append(query)
            if len(queries) >= max_queries:
                break
    return queries


# ---------------------------------------------------------
# 6. DOMAIN EXTRACTION
# ---------------------------------------------------------

def get_domain(url):
    """Return a normalized hostname from a URL."""

    if not isinstance(url, str) or not url.strip():
        return ""

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        hostname = hostname.lower().rstrip(".")

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except (ValueError, AttributeError):
        return ""


# ---------------------------------------------------------
# 7. SAFE DOMAIN MATCHING
# ---------------------------------------------------------

def domain_matches(domain, registered_domain):
    """
    Match a registered domain or its subdomains.

    tesla.com matches www.tesla.com.
    fake-tesla.com does not match tesla.com.
    """

    domain = (domain or "").lower().rstrip(".")

    registered_domain = (
        registered_domain or ""
    ).lower().rstrip(".")

    return (
        domain == registered_domain
        or domain.endswith("." + registered_domain)
    )


def get_registered_domain_label(domain):
    """Return the likely company-bearing label of a host name."""

    labels = (domain or "").lower().strip(".").split(".")

    if len(labels) < 2:
        return ""

    second_level_suffixes = {
        ("ac", "uk"), ("co", "in"), ("co", "jp"), ("co", "uk"),
        ("com", "au"), ("com", "br"), ("com", "cn"), ("com", "in"),
        ("com", "jp"), ("com", "uk"), ("gov", "in"), ("gov", "uk"),
        ("org", "uk"),
    }
    suffix_pair = tuple(labels[-2:])
    label_index = -3 if suffix_pair in second_level_suffixes else -2

    if len(labels) < abs(label_index):
        return ""

    return labels[label_index]


# ---------------------------------------------------------
# 8. SOURCE CLASSIFICATION
# ---------------------------------------------------------

def is_government_host(domain):
    """Return True for configured or .gov-style government hostnames."""

    domain = (domain or "").lower().rstrip(".")
    return (
        domain.endswith(".gov")
        or domain.endswith(".gov.in")
        or any(
            domain_matches(domain, government_domain)
            for government_domain in GOVERNMENT_DOMAINS
        )
    )


# ---------------------------------------------------------
# 8a. DYNAMIC ALIAS, OFFICIAL-DOMAIN AND DISAMBIGUATION HELPERS
# ---------------------------------------------------------

LEGAL_SUFFIX_TOKENS = {
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "limited", "llc", "ltd", "plc", "sa", "ag", "nv", "se", "gmbh",
}

ALIAS_STOP_TOKENS = {"the", "and", "of", "for"}

# Words that may follow a company name without making it a different
# company: corporate descriptors, subsidiary-style labels and ESG headings.
NAME_CONTINUATION_ALLOWED = {
    "group", "holdings", "holding", "electronics", "semiconductor",
    "semiconductors", "energy", "motors", "motor", "global", "international",
    "industries", "technologies", "technology", "systems", "solutions",
    "foods", "pharmaceuticals", "pharma", "bank", "financial", "media",
    "software", "networks", "telecom", "india", "uk", "usa", "us", "europe",
    "asia", "china", "japan", "canada", "australia", "brazil", "deutschland",
    "sustainability", "sustainable", "environmental", "environment", "esg",
    "climate", "carbon", "emissions", "annual", "impact", "responsibility",
    "report", "reports", "reporting", "data", "ghg", "scope", "net", "zero",
    "renewable", "recycling", "water", "waste", "supplier", "suppliers",
    "our", "we", "this", "its", "press", "release", "news", "investor",
    "investors", "relations", "newsroom", "statement", "statements",
    "greenhouse", "gas", "social", "governance", "assurance", "corporate",
    "company", "corporation", "co", "inc", "incorporated", "ltd", "limited",
    "llc", "plc", "sa", "ag", "nv", "se", "gmbh",
    "independent", "review", "assessment", "verification", "audit",
    "lifecycle", "life", "cycle", "footprint", "filing", "filings",
    "disclosure", "disclosures", "update", "updates", "progress",
    "performance", "goals", "targets", "commitments", "plan", "results",
    "inventory", "study", "analysis", "research", "investigation",
    "product", "products", "supply", "chain", "operations", "facilities",
    "manufacturing", "sites", "statement", "overview", "highlights",
}

SUBSIDIARY_CUE_PATTERN = re.compile(
    r"\b(?:subsidiary|subsidiaries|part of|parent company|owned by|"
    r"a unit of|division of|brand of|member of|affiliate of|affiliates|"
    r"a business of|operates as|operating under)\b",
    re.IGNORECASE,
)

# Domains whose hostnames are shared hosting or social/aggregator platforms
# and therefore cannot be a company's own official site.
THIRD_PARTY_HOSTING_DOMAINS = {
    "wikipedia.org", "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "tiktok.com", "reddit.com", "medium.com",
    "substack.com", "wordpress.com", "blogspot.com", "github.com",
    "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "crunchbase.com",
    "glassdoor.com", "indeed.com", "yelp.com", "dictionary.com",
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com", "brave.com",
}


def significant_company_tokens(name):
    """Lower-case name tokens without legal suffixes or a leading 'the'."""

    tokens = re.findall(r"[a-z0-9]+", str(name or "").casefold())
    while tokens and tokens[-1] in LEGAL_SUFFIX_TOKENS:
        tokens.pop()
    while tokens and tokens[0] == "the":
        tokens.pop(0)
    return tokens


def derive_company_aliases(name):
    """
    Derive alias keys from the entered name alone.

    Keys are the compact form ("procter & gamble" -> "proctergamble") and,
    for names with two or more significant words, the initialism
    ("procter & gamble" -> "pg", "international business machines" -> "ibm").
    A key is only used to match a domain label; it is not an ownership claim.
    """

    tokens = significant_company_tokens(name)
    if not tokens:
        return []

    aliases = ["".join(tokens)]
    words = [token for token in tokens if token not in ALIAS_STOP_TOKENS]
    aliases.append("".join(words))
    if len(words) >= 2:
        aliases.append("".join(word[0] for word in words))
    return list(dict.fromkeys(alias for alias in aliases if alias))


def ambiguous_name_continuation(name, *text_parts):
    """
    Return True when the entered name is followed by an unfamiliar
    capitalised term, such as "Delta" in "Delta Air Lines".

    Such text may describe a different company. The name is accepted only
    when the continuation is a corporate descriptor, an ESG heading word,
    lower-case prose, or appears in a sentence with a subsidiary cue.
    """

    wanted = significant_company_tokens(name)
    if not wanted:
        return False

    for part in text_parts:
        text = str(part or "")
        raw_tokens = [
            (match.group(0), match.start())
            for match in re.finditer(r"[A-Za-z0-9]+", text)
        ]
        lowered = [token.casefold() for token, _ in raw_tokens]
        for index in range(len(lowered) - len(wanted) + 1):
            if lowered[index:index + len(wanted)] != wanted:
                continue
            next_index = index + len(wanted)
            if next_index >= len(raw_tokens):
                continue
            next_token, position = raw_tokens[next_index]
            next_lower = next_token.casefold()
            if next_lower in NAME_CONTINUATION_ALLOWED:
                continue
            if not next_token[0].isupper():
                continue
            sentence_start = max(
                text.rfind(".", 0, position),
                text.rfind("!", 0, position),
                text.rfind("?", 0, position),
                text.rfind(";", 0, position),
            ) + 1
            sentence_end_candidates = [
                value for value in (
                    text.find(".", position),
                    text.find("!", position),
                    text.find("?", position),
                    text.find(";", position),
                )
                if value != -1
            ]
            sentence_end = (
                min(sentence_end_candidates)
                if sentence_end_candidates else len(text)
            )
            sentence = text[sentence_start:sentence_end]
            if SUBSIDIARY_CUE_PATTERN.search(sentence):
                continue
            return True
    return False


def discover_official_domains(name, results, max_domains=2):
    """
    Find candidate official domains from the collected search results.

    A host qualifies when its registered label matches a key derived from
    the entered name and at least one result on that host names the company
    in its title or snippet without an ambiguous continuation. Results are
    candidates only; they are not verified ownership.
    """

    alias_keys = set(derive_company_aliases(name))
    if not alias_keys:
        return []

    candidates = {}
    for result in results or []:
        url = result.get("url") or ""
        domain = result.get("domain") or get_domain(url)
        if not domain or is_government_host(domain):
            continue
        if any(
            domain_matches(domain, third_party)
            for third_party in THIRD_PARTY_HOSTING_DOMAINS
        ):
            continue
        label = normalize_company_name(get_registered_domain_label(domain))
        if label not in alias_keys:
            continue
        title = result.get("title") or ""
        snippet = result.get("snippet") or ""
        if not search_result_mentions_entity(name, "company", title, snippet):
            continue
        if ambiguous_name_continuation(name, title, snippet):
            continue
        candidates.setdefault(domain, []).append(url)

    ranked = sorted(
        candidates.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    return [
        {
            "domain": domain,
            "supporting_urls": list(dict.fromkeys(urls)),
            "verification_status": "candidate_unverified",
            "basis": (
                "Registered domain label matches a name-derived alias and "
                "search results on this host name the company."
            ),
        }
        for domain, urls in ranked[:max_domains]
    ]



def classify_source(url, company_name=None, discovered_domains=None):
    """
    Classify sources based on their domain.

    Classification is preliminary and does not prove
    that a source is reliable or its claims are true.
    """

    domain = get_domain(url)

    if not domain:
        return "Unknown"

    # Known official company domains.
    if company_name:

        company_key = normalize_company_name(company_name)

        official_domains = OFFICIAL_COMPANY_DOMAINS.get(
            company_key,
            [],
        )

        for official_domain in official_domains:

            if domain_matches(domain, official_domain):
                return "Official company source"

        for discovered_domain in discovered_domains or []:
            if domain_matches(domain, discovered_domain):
                return "Potential company source (discovered domain)"

        # A government host can share a company's name (e.g. acme.gov),
        # but a company cannot own a government domain, so it must not be
        # labelled a company source. It falls through to the government rules.
        if not is_government_host(domain):
            company_key = normalize_company_name(
                re.sub(
                    r"^\s*the\s+", "", company_name, flags=re.IGNORECASE
                )
            )
            domain_label = get_registered_domain_label(domain)
            if (
                company_key
                and normalize_company_name(domain_label) == company_key
            ):
                return "Potential company source (domain-name match)"

    # Government and regulatory sources.
    if any(
        domain_matches(domain, government_domain)
        for government_domain in GOVERNMENT_DOMAINS
    ):
        return "Government / regulatory source"

    # Configured international environmental sources.
    if any(
        domain_matches(domain, trusted_domain)
        for trusted_domain in TRUSTED_DOMAINS
    ):
        return "Configured environmental source"

    # A government top-level domain alone does not establish a regulatory role.
    if (
        domain.endswith(".gov")
        or domain.endswith(".gov.in")
    ):
        return "Government source / regulatory status unverified"

    return "Other / unverified"


def search_result_mentions_entity(name, search_type, *text_parts):
    """Check search-result text for the requested company name."""

    if not isinstance(name, str) or not name.strip():
        return True

    text = " ".join(str(part or "") for part in text_parts).casefold()
    if not text.strip():
        return False

    available = re.findall(r"[a-z0-9]+", text)
    wanted = re.findall(r"[a-z0-9]+", name.casefold())
    suffixes = {
        "co", "company", "corp", "corporation", "inc",
        "incorporated", "limited", "llc", "ltd", "plc",
    }
    while wanted and wanted[-1] in suffixes:
        wanted.pop()
    while wanted and wanted[0] == "the":
        wanted.pop(0)
    mentions_entity = any(
        available[index:index + len(wanted)] == wanted
        for index in range(len(available) - len(wanted) + 1)
    ) if wanted else False
    if not mentions_entity:
        return False

    if search_type == "company" and ambiguous_name_continuation(
        name,
        *text_parts,
    ):
        return False

    return not (
        search_type == "company"
        and is_ambiguous_apple_food_result(name, text)
    )


def has_environmental_search_context(*text_parts):
    """Check that candidate metadata has an environmental relevance cue."""

    text = " ".join(str(part or "") for part in text_parts).casefold()
    return any(term in text for term in SEARCH_ENVIRONMENTAL_TERMS)


# ---------------------------------------------------------
# 9. URL VALIDATION
# ---------------------------------------------------------

def is_valid_web_url(url):
    """Accept plain HTTP(S) URLs without credentials or malformed ports."""

    if not isinstance(url, str) or not url.strip():
        return False

    try:
        parsed = urlparse(url)
        port = parsed.port

        return (
            parsed.scheme.lower() in ("http", "https")
            and bool(parsed.hostname)
            and not parsed.username
            and not parsed.password
            and (port is None or 1 <= port <= 65535)
        )

    except (ValueError, AttributeError):
        return False


def is_search_provider_redirect(url):
    """Identify search-engine tracking and advertising redirect URLs."""

    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        path = parsed.path.lower()
    except (TypeError, ValueError, AttributeError):
        return False

    return (
        (host == "bing.com" or host.endswith(".bing.com"))
        and path == "/aclick"
    ) or (
        (host == "google.com" or host.endswith(".google.com"))
        and path == "/url"
    ) or (
        (host == "duckduckgo.com" or host.endswith(".duckduckgo.com"))
        and path.startswith("/l/")
    )


# ---------------------------------------------------------
# 10. SINGLE WEB SEARCH
# ---------------------------------------------------------

def search_web(
    query,
    max_results=DEFAULT_RESULTS_PER_QUERY,
    company_name=None,
    search_type="company",
    error_callback=None,
    diagnostics=None,
    backend=None,
):
    """
    Search for one query.

    Result fields:
        title
        url
        snippet
        domain
        source_type
        search_query
        content_type
    """

    results = []
    rejected_results = []
    seen_urls = {}
    raw_result_count = 0

    if search_type != "company":
        raise ValueError("Only company searches are supported.")

    try:
        # Reduce the search client's configured timeout.
        with DDGS(timeout=DEFAULT_SEARCH_TIMEOUT) as ddgs:

            search_options = {"max_results": max_results}
            if backend:
                search_options["backend"] = backend
            search_results = ddgs.text(query, **search_options)

            for item in search_results or []:
                raw_result_count += 1

                if not isinstance(item, dict):
                    rejected_results.append({
                        "query": query,
                        "title": "",
                        "url": "",
                        "reason": "invalid_search_result",
                    })
                    continue

                raw_url = item.get("href")
                url = raw_url.strip() if isinstance(raw_url, str) else ""
                title_value = item.get("title")
                title = (
                    title_value.strip()
                    if isinstance(title_value, str)
                    else ""
                )
                snippet_value = item.get("body")
                snippet = (
                    snippet_value.strip()
                    if isinstance(snippet_value, str)
                    else ""
                )
                date_value = item.get("date")
                search_result_date = (
                    date_value.strip()
                    if isinstance(date_value, str)
                    else ""
                )

                if not any((title, url, snippet)):
                    rejected_results.append({
                        "query": query,
                        "title": title,
                        "url": url,
                        "reason": "provider_empty_result",
                    })
                    continue

                if not is_valid_web_url(url):
                    rejected_results.append({
                        "query": query,
                        "title": title,
                        "url": url,
                        "reason": "invalid_or_unsupported_url",
                    })
                    continue
                if is_search_provider_redirect(url):
                    rejected_results.append({
                        "query": query,
                        "title": title,
                        "url": url,
                        "reason": "search_provider_redirect",
                    })
                    continue
                if not search_result_mentions_entity(
                    company_name,
                    search_type,
                    title,
                    snippet,
                ):
                    ambiguous_company_result = (
                        search_type == "company"
                        and (
                            is_ambiguous_apple_food_result(
                                company_name,
                                f"{title} {snippet}",
                            )
                            or ambiguous_name_continuation(
                                company_name,
                                title,
                                snippet,
                            )
                        )
                    )
                    has_environmental_context = (
                        has_environmental_search_context(
                            title,
                            snippet,
                            url,
                        )
                    )
                    if (
                        company_name
                        and not ambiguous_company_result
                        and has_environmental_context
                        and search_result_mentions_entity(
                            company_name,
                            search_type,
                            query,
                        )
                    ):
                        attribution_status = "pending_page_content"
                        attribution_basis = (
                            "The search query contains the requested "
                            "entity and result metadata has an "
                            "environmental relevance cue; page content "
                            "must confirm entity attribution."
                        )
                    else:
                        rejected_results.append({
                            "query": query,
                            "title": title,
                            "url": url,
                            "reason": (
                                "ambiguous_company_reference"
                                if ambiguous_company_result
                                else (
                                    "no_environmental_relevance_cue"
                                    if not has_environmental_context
                                    else "requested_entity_not_in_search_metadata"
                                )
                            ),
                            "attribution_basis": (
                                "Result title/snippet was insufficient "
                                "for entity attribution."
                            ),
                        })
                        continue
                else:
                    attribution_status = "confirmed_by_search_metadata"
                    attribution_basis = (
                        "Requested entity appears in the search title "
                        "or snippet."
                    )

                # Remove URL fragments.
                clean_url, _ = urldefrag(url)

                normalized_url = clean_url.rstrip("/").lower()

                if normalized_url in seen_urls:
                    existing = results[seen_urls[normalized_url]]
                    if len(title) > len(existing.get("title", "")):
                        existing["title"] = title
                    if len(snippet) > len(existing.get("snippet", "")):
                        existing["snippet"] = snippet
                    if (
                        not existing.get("search_result_date")
                        and search_result_date
                    ):
                        existing["search_result_date"] = search_result_date
                    rejected_results.append({
                        "query": query,
                        "title": title,
                        "url": clean_url,
                        "reason": "duplicate_url_in_query",
                    })
                    continue

                seen_urls[normalized_url] = len(results)

                domain = get_domain(clean_url)
                path = urlparse(clean_url).path.lower()

                content_type = (
                    "PDF"
                    if path.endswith(".pdf")
                    else "Web page or unknown"
                )

                source_type = classify_source(
                    clean_url,
                    company_name=company_name,
                )

                results.append({
                    "title": title,
                    "url": clean_url,
                    "snippet": snippet,
                    "search_result_date": search_result_date or None,
                    "domain": domain,
                    "source_type": source_type,
                    "search_query": query,
                    "search_backend": backend or "auto",
                    "content_type": content_type,
                    "attribution_status": attribution_status,
                    "attribution_basis": attribution_basis,
                })

    except Exception as error:

        message = str(error).strip() or type(error).__name__

        if "no results found" in message.lower():
            print("No results found for this query.")
            if diagnostics is not None:
                diagnostics.update({
                    "query": query,
                    "backend": backend or "auto",
                    "status": (
                        "filtered_out"
                        if raw_result_count
                        else "no_results"
                    ),
                    "raw_result_count": raw_result_count,
                    "accepted_result_count": len(results),
                    "rejected_result_count": len(rejected_results),
                    "rejected_results": rejected_results,
                    "empty_reason": (
                        "all_results_filtered"
                        if raw_result_count
                        else "search_returned_no_results"
                    ),
                    "error": None,
                })
        else:
            if error_callback is not None:
                error_callback(query, message)
            if diagnostics is not None:
                diagnostics.update({
                    "query": query,
                    "backend": backend or "auto",
                    "status": "failed",
                    "raw_result_count": raw_result_count,
                    "accepted_result_count": len(results),
                    "rejected_result_count": len(rejected_results),
                    "rejected_results": rejected_results,
                    "error": message,
                })
            print(
                "Search unavailable for this query: "
                f"{message}"
            )

    if diagnostics is not None and not diagnostics:
        diagnostics.update({
            "query": query,
            "backend": backend or "auto",
            "status": (
                "completed"
                if results
                else "filtered_out"
                if raw_result_count
                else "no_results"
            ),
            "raw_result_count": raw_result_count,
            "accepted_result_count": len(results),
            "rejected_result_count": len(rejected_results),
            "rejected_results": rejected_results,
            "empty_reason": (
                None
                if results
                else "all_results_filtered"
                if raw_result_count
                else "search_returned_no_results"
            ),
            "error": None,
        })

    return results


# ---------------------------------------------------------
# 11. SEARCH ALL ENVIRONMENTAL QUERIES
# ---------------------------------------------------------

def search_environmental_information(
    name,
    search_type="company",
    max_results_per_query=DEFAULT_RESULTS_PER_QUERY,
    delay_seconds=DEFAULT_SEARCH_DELAY,
    error_callback=None,
    diagnostics=None,
):
    """
    Search for environmental information about a company.

    Returns deduplicated results with preliminary
    source classifications.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "Please enter a company name."
        )

    name = name.strip()
    search_type = search_type.lower().strip()

    if search_type != "company":
        raise ValueError(
            "Only company searches are supported."
        )

    if (
        not isinstance(max_results_per_query, int)
        or isinstance(max_results_per_query, bool)
        or not 1 <= max_results_per_query <= 10
    ):
        raise ValueError(
            "max_results_per_query must be an integer "
            "between 1 and 10."
        )

    if (
        not isinstance(delay_seconds, (int, float))
        or isinstance(delay_seconds, bool)
        or delay_seconds < 0
    ):
        raise ValueError(
            "delay_seconds must be a non-negative number."
        )

    # Use normalized names for known-company lookup.
    company_key = normalize_company_name(name)

    known_official_domains = OFFICIAL_COMPANY_DOMAINS.get(company_key, [])

    queries = []

    # Search the known official company website directly.
    for official_domain in known_official_domains:

        queries.append(
            f"site:{official_domain} "
            f"sustainability environmental impact emissions "
            f'"{name}"'
        )

    # Search regulatory and government sources.
    queries.append(
        f'site:sec.gov/edgar "{name}" '
        "annual report environmental climate emissions"
    )

    queries.append(
        f'"{name}" site:epa.gov OR site:eia.gov '
        "environmental emissions energy"
    )

    # Add general environmental queries.
    for query_template in SEARCH_QUERIES[search_type]:

        queries.append(query_template.format(name=name))

    # Remove duplicate queries while preserving their order.
    queries = list(dict.fromkeys(queries))

    results_by_url = {}
    result_queries_by_url = {}
    query_diagnostics = []

    for index, query in enumerate(queries):

        print(f"\nSearching: {query}")

        query_diagnostic = {}
        search_results = search_web(
            query=query,
            max_results=max_results_per_query,
            company_name=(
                name
            ),
            search_type=search_type,
            error_callback=error_callback,
            diagnostics=query_diagnostic,
        )
        query_diagnostics.append(query_diagnostic)

        for result in search_results:

            key = result["url"].rstrip("/").lower()
            query_origin = {
                "query": query,
                "backend": result.get("search_backend", "auto"),
            }
            query_origins = result_queries_by_url.setdefault(key, [])
            if query_origin not in query_origins:
                query_origins.append(query_origin)

            if key not in results_by_url:
                results_by_url[key] = result
            else:
                existing = results_by_url[key]
                if len(result.get("title", "")) > len(existing.get("title", "")):
                    existing["title"] = result["title"]
                if len(result.get("snippet", "")) > len(
                    existing.get("snippet", "")
                ):
                    existing["snippet"] = result["snippet"]
                if (
                    not existing.get("search_result_date")
                    and result.get("search_result_date")
                ):
                    existing["search_result_date"] = result[
                        "search_result_date"
                    ]

        # Keep a short delay between search requests.
        if index < len(queries) - 1 and delay_seconds:
            time.sleep(delay_seconds)

    # -----------------------------------------------------
    # Dynamic official-domain discovery (no hardcoded company table)
    # -----------------------------------------------------

    discovered_domains = discover_official_domains(
        name,
        list(results_by_url.values()),
    )
    discovered_domain_names = [item["domain"] for item in discovered_domains]
    discovery_queries = list(dict.fromkeys(
        f'site:{domain} "{name}" '
        "sustainability environmental emissions report"
        for domain in discovered_domain_names
    ))

    for index, query in enumerate(discovery_queries):

        print(f"\nSearching: {query}")

        query_diagnostic = {}
        search_results = search_web(
            query=query,
            max_results=max_results_per_query,
            company_name=name,
            search_type=search_type,
            error_callback=error_callback,
            diagnostics=query_diagnostic,
        )
        query_diagnostics.append(query_diagnostic)

        for result in search_results:
            key = result["url"].rstrip("/").lower()
            query_origin = {
                "query": query,
                "backend": result.get("search_backend", "auto"),
            }
            query_origins = result_queries_by_url.setdefault(key, [])
            if query_origin not in query_origins:
                query_origins.append(query_origin)
            results_by_url.setdefault(key, result)

        if index < len(discovery_queries) - 1 and delay_seconds:
            time.sleep(delay_seconds)

    for item in results_by_url.values():
        item["source_type"] = classify_source(
            item["url"],
            company_name=name,
            discovered_domains=discovered_domain_names,
        )

    results = list(results_by_url.values())

    # -----------------------------------------------------
    # Prioritize potentially useful ESG documents
    # -----------------------------------------------------

    priority_order = {
        "Official company source": 0,
        "Government / regulatory source": 1,
        "Configured environmental source": 2,
        "Other / unverified": 3,
        "Unknown": 4,
    }

    def calculate_result_priority(item):

        title = item.get("title", "").lower()
        url = item.get("url", "").lower()
        snippet = item.get("snippet", "").lower()

        text = f"{title} {url} {snippet}"

        score = priority_order.get(
            item.get("source_type", "Unknown"),
            5,
        ) * 10

        # Prefer reports and substantive ESG documents.
        if url.endswith(".pdf"):
            score -= 5

        useful_terms = [
            "sustainability report",
            "impact report",
            "environmental report",
            "annual report",
            "carbon emissions",
            "greenhouse gas",
            "climate disclosure",
            "carbon footprint",
            "sustainability",
            "emissions",
        ]

        for term in useful_terms:

            if term in text:
                score -= 2

        # Avoid generic pages when better results exist.
        generic_terms = [
            "contact us",
            "privacy policy",
            "legal notice",
            "additional resources",
            "terms of use",
        ]

        for term in generic_terms:

            if term in title:
                score += 10

        return score

    results.sort(
        key=lambda item: (
            calculate_result_priority(item),
            item.get("title", "").lower(),
        )
    )

    official_results = [
        item
        for item in results
        if item.get("source_type") == "Official company source"
    ]
    external_results = [
        item
        for item in results
        if item.get("source_type") != "Official company source"
    ]
    if official_results and external_results:
        results = [
            official_results[0],
            *external_results,
            *official_results[1:],
        ]

    if diagnostics is not None:
        failed_query_count = sum(
            item.get("status") == "failed"
            for item in query_diagnostics
        )
        diagnostics.update({
            "search_type": search_type,
            "name_derived_aliases": derive_company_aliases(name),
            "official_domain_discovery": {
                "discovered_domains": discovered_domains,
                "discovery_query_count": len(discovery_queries),
                "discovery_queries": discovery_queries,
            },
            "query_count": len(query_diagnostics),
            "queries": query_diagnostics,
            "raw_result_count": sum(
                item.get("raw_result_count", 0)
                for item in query_diagnostics
            ),
            "accepted_result_count": sum(
                item.get("accepted_result_count", 0)
                for item in query_diagnostics
            ),
            "result_count": len(results),
            "candidate_results": [
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "domain": item.get("domain", ""),
                    "source_type": item.get("source_type", ""),
                    "content_type": item.get("content_type", ""),
                    "search_result_date": item.get("search_result_date"),
                    "attribution_status": item.get(
                        "attribution_status",
                        "not_recorded",
                    ),
                    "attribution_basis": item.get("attribution_basis", ""),
                    "queries": result_queries_by_url.get(
                        item.get("url", "").rstrip("/").lower(),
                        [],
                    ),
                }
                for item in results
            ],
            "rejected_result_count": sum(
                item.get("rejected_result_count", 0)
                for item in query_diagnostics
            ),
            "rejected_candidates": [
                rejected
                for item in query_diagnostics
                for rejected in item.get("rejected_results", [])
            ],
            "search_error_count": failed_query_count,
            "search_errors": [
                {
                    "query": item.get("query", ""),
                    "message": item.get("error", ""),
                }
                for item in query_diagnostics
                if item.get("status") == "failed"
            ],
            "status": (
                "completed"
                if results and not failed_query_count
                else "partial"
                if results
                else "failed"
                if failed_query_count == len(query_diagnostics)
                else "partial"
                if failed_query_count
                else "filtered_out"
                if any(
                    item.get("status") == "filtered_out"
                    for item in query_diagnostics
                )
                else "no_results"
            ),
        })

    return results


# ---------------------------------------------------------
# 12. DISPLAY SEARCH RESULTS
# ---------------------------------------------------------

def display_results(results):

    if not results:
        print("\nNo search results were retrieved.")
        print(
            "Check your internet connection or try again later."
        )
        return

    print("\n" + "=" * 75)
    print("ENVIRONMENTAL WEB SEARCH RESULTS")
    print("=" * 75)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 55)

        print(f"Title       : {result['title']}")
        print(f"Domain      : {result['domain']}")
        print(f"Source type : {result['source_type']}")
        print(f"Content type: {result['content_type']}")
        print(f"URL         : {result['url']}")
        print(f"Description : {result['snippet']}")
        print(f"Search query: {result['search_query']}")

    print("\n" + "=" * 75)
    print(f"Total unique results: {len(results)}")
    print("=" * 75)

    pdf_count = sum(
        1
        for result in results
        if result["content_type"] == "PDF"
    )

    print(f"Potential PDF links: {pdf_count}")


# ---------------------------------------------------------
# 13. INTERACTIVE TEST
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("ADVANCED ESG GREENWASHING DETECTION")
    print("Improved Environmental Web Search")
    print("=" * 60)

    name = input("\nEnter company name: ").strip()

    try:

        results = search_environmental_information(
            name=name,
            max_results_per_query=DEFAULT_RESULTS_PER_QUERY,
            delay_seconds=DEFAULT_SEARCH_DELAY,
        )

        display_results(results)

    except ValueError as error:
        print(f"Input error: {error}")


if __name__ == "__main__":
    main()
