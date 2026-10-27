import json
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from web_search import GOVERNMENT_DOMAINS

PROJECT_DIR = Path(__file__).resolve().parent
COLLECTED_DIR = PROJECT_DIR / "data" / "collected_documents"
OUTPUT_DIR = PROJECT_DIR / "output"

ESG_KEYWORDS = [
    "environment", "emissions", "carbon", "greenhouse gas",
    "climate", "sustainability", "sustainable", "renewable energy",
    "renewable electricity", "recycled", "recyclable",
    "energy efficiency", "pollution", "hazardous waste",
    "waste management", "recycling", "water consumption",
    "zero emission", "clean air act", "environmental impact",
    "battery"
]

CLAIM_ACTION_PATTERN = re.compile(
    r"\b(?:aims?|commits?|committed|pledges?|pledged|targets?|targeted|"
    r"is|are|was|were|has|have|had|will|reduces?|reduced|reducing|"
    r"increases?|increased|increasing|fell|falls|declined|decreased|"
    r"improves?|improved|achieves?|achieved|uses?|used|sources?|sourced|"
    r"generates?|generated|emits?|emitted|protects?|protected|"
    r"preserves?|preserved|aligns?|aligned|focuses?|focused|"
    r"remains?|remained|accounts?|accounted|prioritizes?|prioritises?|"
    r"supports?|promotes?|invests?|implements?|provides?|fosters?|"
    r"manages?|maintains?|adopts?|drives?|develops?|delivers?|"
    r"eliminates?|ensures?)\b",
    re.IGNORECASE,
)
MEASUREMENT_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|percent|percentage|"
    r"tonnes?|tons?|kilograms?|kg|megawatts?|mw|gigawatt.hours?|gwh|"
    r"t\s*co2e|kg\s*co2e|million|billion)\b"
    r"|\bnet\s+zero\b",
    re.IGNORECASE,
)
NAVIGATION_PATTERN = re.compile(
    r"\b(?:skip to main content|open navigation menu|change language|"
    r"sign in|sign out|sign up|quick link|go back|investor relations|"
    r"investor overview|financial information|financial calendar|"
    r"shareholder information|investor calendar|contact investor relations|"
    r"learn more|see details|read more|claim this profile|"
    r"see verified emissions data for|to see all disclosed climate targets|"
    r"curious to see the emissions|discover our data-driven methodology|"
    r"upload|english|en)\b",
    re.IGNORECASE,
)
PAGE_CHROME_PATTERN = re.compile(
    r"\b\d+\s+(?:views?|ratings?|pages?)\b|"
    r"\b\d+%\s+found this document useful(?:\s*\(\d+\s+votes?\))?|"
    r"\byou are on page \d+(?:\s+\d+)?\b|"
    r"\bshare on (?:facebook|linkedin),?\s+opens? a new window\b|"
    r"\bmillions of documents at your fingertips(?:,\s*ad-free)?\b|"
    r"\bpdf\b",
    re.IGNORECASE,
)
STANDALONE_NAVIGATION_LABELS = {
    "back", "dark", "en", "english", "light", "menu", "mode", "overview",
    "pdf", "upload", "view",
}
PAGE_NUMBER_SEQUENCE_PATTERN = re.compile(
    r"^(?:\d{1,3}\s+){3,}(?=[A-Za-z])"
)
CONTENTS_HEADING_PATTERN = re.compile(
    r"^(?:table\s+of\s+)?contents$",
    re.IGNORECASE,
)
STANDALONE_PAGE_NUMBER_PATTERN = re.compile(r"\d{1,3}")
CONTENTS_NARRATIVE_START_PATTERN = re.compile(
    r"^(?:at\b|we\b|our\b|since\b|in\s+\d{4}\b|this\b)",
    re.IGNORECASE,
)
ASSURANCE_SECTION_PATTERN = re.compile(
    r"(?im)^(?P<header>INDEPENDENT (?:ASSURANCE|REVIEW) STATEMENT|"
    r"(?:COMPREHENSIVE )?CARBON FOOTPRINT LETTER OF ASSURANCE)\s*$"
)
ACCOUNTANT_REVIEW_REPORT_HEADING_PATTERN = re.compile(
    r"(?im)(?<![\w.])(?P<header>\d+(?:\.\d+)+\s+"
    r"INDEPENDENT ACCOUNTANT[’']S REVIEW REPORT)\b"
)
ASSURANCE_TABLE_HEADER_PATTERN = re.compile(
    r"(?im)^[ \t]*Parameter[ \t]*\n[ \t]*Quantity[ \t]*\n"
    r"[ \t]*Units[ \t]*\n[ \t]*Boundary[ \t]*/[ \t]*Protocol[ \t]*$"
)
ASSURANCE_TABLE_END_PATTERN = re.compile(
    r"(?im)^[ \t]*Our Conclusion[ \t]*$"
)
ASSURANCE_ISSUER_PATTERN = re.compile(
    r"(?im)^(?P<publisher>[A-Z][A-Z0-9&.,'’() \-]{2,100}?)\s+"
    r"(?:\([^)]*\)\s*)?(?:was engaged by|reviewed)\s+(?!by\b)"
    r"(?P<entity>[A-Z][A-Z0-9&.,'’() \-]{0,100})"
)
ASSURANCE_PUBLISHER_PATTERN = re.compile(
    r"\b(?:LLC|L\.L\.C\.|Inc\.?|Ltd\.?|Limited|Corp\.?|Corporation|"
    r"GmbH|plc|IZM|Consultants|Solutions|Institute|University|Agency|"
    r"Foundation)\b",
    re.IGNORECASE,
)
ASSURANCE_PUBLICATION_DATE_PATTERN = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan\.?|Feb\.?|Mar\.?|Apr\.?|"
    r"Jun\.?|Jul\.?|Aug\.?|Sep\.?|Oct\.?|Nov\.?|Dec\.?)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?[,]?\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
ASSURANCE_ATTESTATION_PATTERN = re.compile(
    r"(?im)^\s*Attestation\s*:\s*"
)
ASSURANCE_SIGNED_DATE_MARKER_PATTERN = re.compile(
    r"\bsigned\s+(?:this\s+)?(?:assurance\s+)?"
    r"(?:statement|report)\s+on\b",
    re.IGNORECASE,
)


def _remove_navigation_and_heading_fragments(text, title):
    """Remove obvious page chrome before adjacent text blocks are joined."""

    cleaned_lines = []
    in_contents = False
    normalized_title = str(title or "").strip()
    title_core = normalized_title.split("|", 1)[0].strip()
    title_suffix = normalized_title[len(title_core):].strip()
    remove_title_core = (
        len(title_core) >= 12
        and any(
            term in title_core.casefold()
            for term in ("report", "audit", "strategy", "assessment",
                         "compliance", "profile", "score")
        )
    )

    for line in re.split(r"[\r\n]+", str(text or "")):
        line = line.strip()
        if not line:
            continue

        if CONTENTS_HEADING_PATTERN.fullmatch(line):
            in_contents = True
            continue

        if (
            in_contents
            and STANDALONE_PAGE_NUMBER_PATTERN.fullmatch(line)
        ):
            continue

        line = PAGE_NUMBER_SEQUENCE_PATTERN.sub("", line)
        if not line:
            continue

        if remove_title_core:
            line = re.sub(
                rf"(?<!\w){re.escape(title_core)}(?!\w)",
                " ",
                line,
                flags=re.IGNORECASE,
            )
            if title_suffix.strip():
                line = re.sub(
                    rf"(?<!\w){re.escape(title_suffix.strip())}(?!\w)",
                    " ",
                    line,
                    flags=re.IGNORECASE,
                )
        elif normalized_title:
            line = re.sub(
                rf"(?<!\w){re.escape(normalized_title)}(?!\w)",
                " ",
                line,
                count=1,
                flags=re.IGNORECASE,
            )

        line = NAVIGATION_PATTERN.sub(" ", line)
        line = PAGE_CHROME_PATTERN.sub(" ", line)
        line = re.sub(r"\s+", " ", line).strip(" \t|–—:-,")
        if not line or line.casefold() in STANDALONE_NAVIGATION_LABELS:
            continue

        has_action = bool(CLAIM_ACTION_PATTERN.search(line))
        has_measurement = bool(MEASUREMENT_PATTERN.search(line))
        words = line.split()
        if in_contents:
            narrative_start = (
                len(words) >= 12
                or (
                    len(words) >= 6
                    and CONTENTS_NARRATIVE_START_PATTERN.match(line)
                )
                or (
                    len(words) >= 5
                    and has_action
                    and any(
                        keyword in line.casefold()
                        for keyword in ESG_KEYWORDS
                    )
                )
            )
            if not narrative_start:
                continue
            in_contents = False
        if (
            not has_action
            and not has_measurement
            and (
                "|" in line
                or (len(words) <= 6 and line[0].isupper())
            )
        ):
            continue

        cleaned_lines.append(line)

    return " ".join(cleaned_lines)


def split_into_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return re.split(r"(?<=[.!?])\s+", text)


def _split_assurance_sentences(text):
    protected_text = re.sub(
        r"\b(Inc|Ltd|Corp|Co)\.(?=\s)",
        r"\1__ASSURANCE_PERIOD__",
        text,
        flags=re.IGNORECASE,
    )
    return [
        sentence.replace("__ASSURANCE_PERIOD__", ".")
        for sentence in split_into_sentences(protected_text)
    ]


def _assurance_publication_date(text, fallback=None, publisher=""):
    if fallback:
        return fallback

    date_matches = list(ASSURANCE_PUBLICATION_DATE_PATTERN.finditer(text))
    date_match = None
    for marker in reversed(list(ASSURANCE_ATTESTATION_PATTERN.finditer(text))):
        date_match = ASSURANCE_PUBLICATION_DATE_PATTERN.search(
            text,
            marker.end(),
            marker.end() + 1200,
        )
        if date_match:
            break

    if date_match is None:
        for marker in ASSURANCE_SIGNED_DATE_MARKER_PATTERN.finditer(text):
            date_match = ASSURANCE_PUBLICATION_DATE_PATTERN.search(
                text,
                marker.end(),
                marker.end() + 1200,
            )
            if date_match:
                break

    if date_match is None and publisher:
        publisher_name = re.sub(
            r"\b(?:incorporated|inc\.?|l\.?l\.?c\.?|limited|ltd\.?|"
            r"corporation|corp\.?|plc|gmbh)\b",
            "",
            publisher,
            flags=re.IGNORECASE,
        )
        publisher_name = re.sub(
            r"\s+",
            " ",
            publisher_name,
        ).strip(" ,.")
        for candidate in date_matches:
            preceding_text = text[
                max(0, candidate.start() - 300):candidate.start()
            ]
            if (
                publisher_name
                and publisher_name.casefold() in preceding_text.casefold()
            ):
                date_match = candidate

    if date_match is None and len(date_matches) == 1:
        candidate = date_matches[0]
        if len(text[candidate.end():].strip()) <= 500:
            date_match = candidate

    if date_match is None:
        return None

    month = date_match.group("month").rstrip(".")
    date_text = (
        f"{month} {date_match.group('day')} {date_match.group('year')}"
    )
    for date_format in ("%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(
                date_text,
                date_format,
            ).date().isoformat()
        except ValueError:
            continue

    return None


def _assurance_statement_body(text):
    attestation = ASSURANCE_ATTESTATION_PATTERN.search(text)
    if attestation:
        return text[:attestation.start()].strip()
    return text


def _deloitte_accountant_review_report(text, entity_name):
    entity_name = str(entity_name or "").strip()
    if not entity_name:
        return None

    for heading in ACCOUNTANT_REVIEW_REPORT_HEADING_PATTERN.finditer(text):
        window_end = min(len(text), heading.start() + 12000)
        window = text[heading.start():window_end]
        entity_pattern = rf"\b{re.escape(entity_name)}\b"
        board_addressee = re.search(
            rf"\bTo the Board of Directors of\s+[^\n]*{entity_pattern}",
            window,
            re.IGNORECASE,
        )
        if not board_addressee:
            continue
        issuer = re.search(
            r"\bDeloitte\s*&\s*Touche\s+LLP\b",
            window[:board_addressee.start()],
            re.IGNORECASE,
        )
        if not issuer:
            continue
        if not re.search(
            rf"\bWe have reviewed management of\s+[^\n]*{entity_pattern}",
            window,
            re.IGNORECASE,
        ):
            continue

        conclusion = re.search(
            r"(?is)\bBased on our review,\s+.{0,3000}?\bfairly stated\.",
            window,
        )
        if not issuer or not conclusion:
            continue

        report_end = heading.start() + conclusion.end()
        signature_date = re.search(
            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},\s+\d{4}\b",
            text[report_end:report_end + 250],
            re.IGNORECASE,
        )
        if signature_date:
            report_end += signature_date.end()
        report_text = text[heading.start():report_end]
        report_sentences = _split_assurance_sentences(report_text)
        assurance_scope = next(
            (
                sentence.strip()
                for sentence in report_sentences
                if re.search(
                    r"\bWe have reviewed management of\b",
                    sentence,
                    re.IGNORECASE,
                )
            ),
            "",
        )
        scope_exclusions = [
            sentence.strip()
            for sentence in report_sentences
            if re.search(
                r"\bnot subject to our review\b",
                sentence,
                re.IGNORECASE,
            )
        ]
        covered_period = re.search(
            r"\bfiscal year\s+ended\s+June\s+30,\s*\d{4}\b",
            report_text,
            re.IGNORECASE,
        )
        if not assurance_scope or not scope_exclusions or not covered_period:
            continue

        return {
            "start": heading.start(),
            "end": report_end,
            "publisher": "Deloitte & Touche LLP",
            "text": report_text,
            "assurance_scope": assurance_scope,
            "covered_fiscal_period": re.sub(
                r"\s+",
                " ",
                covered_period.group(0),
            ),
            "signature_date": (
                signature_date.group(0) if signature_date else None
            ),
            "scope_exclusions": scope_exclusions,
        }

    return None


def _attributed_assurance_publisher(section_text, entity_name):
    assurance_body = _assurance_statement_body(section_text)
    if not re.search(
        rf"\b{re.escape(entity_name)}\b",
        assurance_body,
        re.IGNORECASE,
    ):
        return None

    issuer = ASSURANCE_ISSUER_PATTERN.search(assurance_body)
    if (
        not issuer
        or not re.search(
            rf"\b{re.escape(entity_name)}\b",
            issuer.group("entity"),
            re.IGNORECASE,
        )
    ):
        return None

    publisher = re.sub(
        r"\s+",
        " ",
        issuer.group("publisher").strip(" ,."),
    )
    if (
        not publisher
        or publisher.casefold() == entity_name.casefold()
        or not ASSURANCE_PUBLISHER_PATTERN.search(publisher)
    ):
        return None

    return publisher


def remove_attributed_assurance_sections(text, entity_name):
    """Remove explicitly attributed third-party assurance sections."""

    text = str(text or "")
    entity_name = str(entity_name or "").strip()
    if not text or not entity_name:
        return text

    attributed_ranges = []
    sections = list(ASSURANCE_SECTION_PATTERN.finditer(text))
    for index, section in enumerate(sections):
        end = (
            sections[index + 1].start()
            if index + 1 < len(sections)
            else len(text)
        )
        if _attributed_assurance_publisher(
            text[section.start():end],
            entity_name,
        ):
            attributed_ranges.append((section.start(), end))

    accountant_report = _deloitte_accountant_review_report(
        text,
        entity_name,
    )
    if accountant_report:
        attributed_ranges.append((
            accountant_report["start"],
            accountant_report["end"],
        ))

    if not attributed_ranges:
        return text

    retained = []
    cursor = 0
    for start, end in sorted(attributed_ranges):
        retained.append(text[cursor:start])
        cursor = max(cursor, end)
    retained.append(text[cursor:])
    return "".join(retained)


def extract_embedded_assurance_statements(document, entity_name):
    """Extract ESG statements from explicitly attributed assurance sections."""

    text = str(document.get("text") or "")
    entity_name = str(entity_name or "").strip()
    if (
        document.get("status") != "success"
        or not text.strip()
        or not entity_name
    ):
        return []

    source_url = document.get("final_url") or document.get("source_url", "")
    source_domain = str(
        document.get("source_domain")
        or urlparse(source_url).hostname
        or ""
    )
    source_title = str(document.get("title") or "Company report")
    sections = list(ASSURANCE_SECTION_PATTERN.finditer(text))
    results = []
    seen = set()

    for index, section in enumerate(sections):
        end = (
            sections[index + 1].start()
            if index + 1 < len(sections)
            else len(text)
        )
        section_text = text[section.start():end]
        publisher = _attributed_assurance_publisher(
            section_text,
            entity_name,
        )
        if not publisher:
            continue

        assurance_body = _assurance_statement_body(section_text)

        publication_date = _assurance_publication_date(
            section_text,
            document.get("publication_date"),
            publisher,
        )

        for sentence in _split_assurance_sentences(assurance_body):
            sentence = sentence.strip()
            sentence_lower = sentence.casefold()
            matched_keywords = [
                keyword for keyword in ESG_KEYWORDS
                if keyword in sentence_lower
            ]
            dedupe_key = sentence_lower
            if (
                len(sentence) < 35
                or len(sentence) > 1500
                or sentence.endswith("?")
                or not matched_keywords
                or dedupe_key in seen
            ):
                continue

            seen.add(dedupe_key)
            results.append({
                "statement": sentence,
                "matched_keywords": matched_keywords,
                "source_role": "independent_third_party_source",
                "source_title": (
                    f"{publisher} assurance statement embedded in "
                    f"{source_title}"
                ),
                "publisher": publisher,
                "source_domain": source_domain,
                "source_url": source_url,
                "original_source_url": document.get("source_url", ""),
                "source_type": (
                    "Independent assurance statement embedded in a "
                    "company-hosted report"
                ),
                "publication_date": publication_date,
                "attribution_basis": (
                    "Explicit assurance-section heading and named issuer "
                    "in the source text."
                ),
            })

        table_header = ASSURANCE_TABLE_HEADER_PATTERN.search(
            assurance_body
        )
        if table_header:
            table_end = ASSURANCE_TABLE_END_PATTERN.search(
                assurance_body,
                table_header.end(),
            )
            if table_end:
                table_text = assurance_body[
                    table_header.start():table_end.start()
                ].strip()
                if (
                    re.search(r"\b\d[\d,]*(?:\.\d+)?\b", table_text)
                    and any(
                        keyword in table_text.casefold()
                        for keyword in ESG_KEYWORDS
                    )
                ):
                    period_context = next(
                        (
                            sentence.strip()
                            for sentence in _split_assurance_sentences(
                                assurance_body[:table_header.start()]
                            )
                            if re.search(
                                r"\bfiscal year\b",
                                sentence,
                                re.IGNORECASE,
                            )
                            and re.search(
                                r"\breporting period\b",
                                sentence,
                                re.IGNORECASE,
                            )
                        ),
                        "",
                    )
                    table_statement = "\n".join(
                        part
                        for part in (period_context, table_text)
                        if part
                    )
                    dedupe_key = table_statement.casefold()
                    if dedupe_key not in seen:
                        results.append({
                            "statement": table_statement,
                            "matched_keywords": [
                                keyword for keyword in ESG_KEYWORDS
                                if keyword in table_statement.casefold()
                            ],
                            "source_role": (
                                "independent_third_party_source"
                            ),
                            "source_title": (
                                f"{publisher} assurance statement embedded "
                                f"in {source_title}"
                            ),
                            "publisher": publisher,
                            "source_domain": source_domain,
                            "source_url": source_url,
                            "original_source_url": document.get(
                                "source_url",
                                "",
                            ),
                            "source_type": (
                                "Independent assurance measurement table "
                                "embedded in a company-hosted report"
                            ),
                            "publication_date": publication_date,
                            "attribution_basis": (
                                "Explicit assurance-section heading and "
                                "named issuer in the source text."
                            ),
                        })

    accountant_report = _deloitte_accountant_review_report(
        text,
        entity_name,
    )
    if accountant_report:
        publisher = accountant_report["publisher"]
        for sentence in _split_assurance_sentences(
            accountant_report["text"]
        ):
            sentence = sentence.strip()
            sentence_lower = sentence.casefold()
            matched_keywords = [
                keyword for keyword in ESG_KEYWORDS
                if keyword in sentence_lower
            ]
            if (
                len(sentence) < 35
                or len(sentence) > 1500
                or sentence.endswith("?")
                or not matched_keywords
                or sentence_lower in seen
            ):
                continue

            seen.add(sentence_lower)
            results.append({
                "statement": sentence,
                "matched_keywords": matched_keywords,
                "source_role": "independent_third_party_source",
                "source_title": (
                    f"{publisher} assurance report embedded in "
                    f"{source_title}"
                ),
                "publisher": publisher,
                "source_domain": source_domain,
                "source_url": source_url,
                "original_source_url": document.get("source_url", ""),
                "source_type": (
                    "Independent accountant limited review embedded in "
                    "a company-hosted report"
                ),
                "publication_date": (
                    _assurance_publication_date(
                        accountant_report["signature_date"]
                    )
                    if accountant_report["signature_date"]
                    else _assurance_publication_date(
                        accountant_report["text"],
                        document.get("publication_date"),
                        publisher,
                    )
                ),
                "assurance_report": True,
                "assurance_scope": accountant_report["assurance_scope"],
                "covered_fiscal_period": (
                    accountant_report["covered_fiscal_period"]
                ),
                "scope_exclusions": accountant_report["scope_exclusions"],
                "original_text": accountant_report["text"],
                "attribution_basis": (
                    "Numbered independent accountant review heading, named "
                    "Deloitte issuer, company board addressee, and first-"
                    "person review conclusion in the source text."
                ),
            })

    return results


def identify_source_role(document):
    source_type = str(document.get("source_type", "")).strip().casefold()
    source_url = document.get("final_url") or document.get("source_url") or ""
    source_domain = str(document.get("source_domain") or "").casefold()
    url_domain = (urlparse(source_url).hostname or "").casefold()
    normalized_domain = source_domain.removeprefix("www.")
    normalized_url_domain = url_domain.removeprefix("www.")
    title = str(document.get("title") or "").casefold()
    source_domain_matches_url = (
        not normalized_domain
        or not normalized_url_domain
        or normalized_domain == normalized_url_domain
    )

    if (
        source_type == "official company source"
        and normalized_url_domain
        and source_domain_matches_url
    ):
        return "company_source"

    if (
        source_type == "potential company source (domain-name match)"
        and normalized_url_domain
        and source_domain_matches_url
    ):
        return "potential_company_source_unverified"

    source_hosts = {normalized_domain, normalized_url_domain}
    rehosting_platform = any(
        domain == platform or domain.endswith(f".{platform}")
        for domain in source_hosts
        for platform in {"archive.org", "scribd.com"}
        if domain
    )
    is_rehosted_report = (
        rehosting_platform
        and any(term in title for term in ("report", "audit"))
        and any(term in title for term in ("environment", "sustainability"))
    )
    if is_rehosted_report:
        return "rehosted_report_unverified"

    if (
        source_domain_matches_url
        and normalized_url_domain
        and any(
            normalized_url_domain == government_domain
            or normalized_url_domain.endswith(f".{government_domain}")
            for government_domain in GOVERNMENT_DOMAINS
        )
    ):
        return "external_regulatory_evidence"

    if (
        "configured environmental source" in source_type
        and source_domain_matches_url
    ):
        return "configured_environmental_source"

    return "external_source_unverified"


def extract_environmental_statements(document):
    text = document.get("text", "")

    if document.get("status") != "success" or not text.strip():
        return []

    text = _remove_navigation_and_heading_fragments(
        text,
        document.get("title", ""),
    )
    source_url = document.get("final_url") or document.get("source_url", "")
    source_domain = str(
        document.get("source_domain")
        or urlparse(source_url).hostname
        or ""
    )
    publisher = document.get("publisher") or source_domain
    results = []
    seen = set()

    for sentence in split_into_sentences(text):
        sentence = sentence.strip()

        if (
            len(sentence) < 35
            or len(sentence) > 1500
            or sentence.endswith("?")
        ):
            continue

        sentence_lower = sentence.lower()
        matched_keywords = [
            keyword for keyword in ESG_KEYWORDS
            if keyword in sentence_lower
        ]
        dedupe_key = (
            str(document.get("source_url") or document.get("final_url") or "").strip(),
            sentence_lower,
        )

        if not matched_keywords or dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        results.append({
            "statement": sentence,
            "matched_keywords": matched_keywords,
            "source_role": identify_source_role(document),
            "source_title": document.get("title", ""),
            "publisher": publisher,
            "source_domain": source_domain,
            "source_url": source_url,
            "original_source_url": document.get("source_url", ""),
            "source_type": document.get("source_type", ""),
            "publication_date": document.get("publication_date")
        })

    return results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not COLLECTED_DIR.exists():
        print("ERROR: Collected documents folder does not exist.")
        return

    json_files = sorted(
        COLLECTED_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )

    if not json_files:
        print("ERROR: No collected JSON files found.")
        return

    input_path = json_files[0]

    with input_path.open("r", encoding="utf-8") as file:
        collection = json.load(file)

    documents = collection.get("documents", [])
    company_name = str(
        collection.get("search_name")
        or collection.get("company")
        or ""
    ).strip()
    statements = []
    successful_count = 0

    for document in documents:
        if (
            document.get("status") == "success"
            and document.get("text", "").strip()
        ):
            successful_count += 1

        assurance_statements = extract_embedded_assurance_statements(
            document,
            company_name,
        )
        company_document = {
            **document,
            "text": remove_attributed_assurance_sections(
                document.get("text", ""),
                company_name,
            ),
        }
        document_statements = extract_environmental_statements(
            company_document
        )
        statements.extend(
            {
                **statement,
                "entity_context": company_name,
            }
            for statement in [*document_statements, *assurance_statements]
        )

    result = {
        "company": company_name,
        "collection_file": input_path.name,
        "successful_documents": successful_count,
        "candidate_statement_count": len(statements),
        "statements": statements
    }

    output_path = OUTPUT_DIR / f"{input_path.stem}_statements.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("=" * 65)
    print("ONLINE ESG STATEMENT EXTRACTION")
    print("=" * 65)
    print("Company:", result["company"])
    print("Input file:", input_path.name)
    print("Successful documents:", successful_count)
    print("Candidate statements:", len(statements))
    print("Output file:", output_path)

    for number, item in enumerate(statements[:10], start=1):
        print("\n" + "-" * 60)
        print("Statement:", number)
        print("Source role:", item["source_role"])
        print("Text:", item["statement"][:400])
        print("URL:", item["source_url"])

    print("\nExtraction completed.")


if __name__ == "__main__":
    main()
