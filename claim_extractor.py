from report_extractor import extract_text
from pathlib import Path
import re


def extract_section(text, section_name, next_section=None):
    """Extract a named section from the ESG report."""

    start = text.find(section_name)

    if start == -1:
        return []

    start += len(section_name)

    if next_section:
        end = text.find(next_section, start)
        if end != -1:
            section = text[start:end]
        else:
            section = text[start:]
    else:
        section = text[start:]

    # Extract numbered items
    items = re.findall(r"^\s*\d+\.\s*(.+)$", section, re.MULTILINE)

    return [item.strip() for item in items]


def extract_claims_and_evidence(text):
    """Separate environmental claims from supporting evidence."""

    claims = extract_section(
        text,
        "Environmental Claims:",
        "Supporting Evidence:"
    )

    evidence = extract_section(
        text,
        "Supporting Evidence:",
        "Note:"
    )

    return claims, evidence


if __name__ == "__main__":
    report_path = Path("data") / "sample_esg_report.txt"

    text = extract_text(report_path)
    claims, evidence = extract_claims_and_evidence(text)

    print("\nENVIRONMENTAL CLAIMS")
    print("=" * 50)

    for i, claim in enumerate(claims, start=1):
        print(f"{i}. {claim}")

    print("\nSUPPORTING EVIDENCE")
    print("=" * 50)

    for i, item in enumerate(evidence, start=1):
        print(f"{i}. {item}")

    print(f"\nTotal claims: {len(claims)}")
    print(f"Total evidence items: {len(evidence)}")