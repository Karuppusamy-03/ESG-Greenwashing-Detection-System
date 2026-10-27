from urllib.parse import urlparse


# Approved starting points for ESG information collection.
# These are source directories and data services, not guaranteed
# direct links to individual company sustainability reports.

TRUSTED_SOURCES = {
    "SEC EDGAR": {
        "url": "https://www.sec.gov/edgar/search/",
        "category": "Official company filings",
        "region": "United States",
    },
    "US EPA": {
        "url": "https://www.epa.gov/",
        "category": "Government environmental information",
        "region": "United States",
    },
    "US Energy Information Administration": {
        "url": "https://www.eia.gov/",
        "category": "Official energy statistics",
        "region": "United States",
    },
    "Apple Environment": {
        "url": "https://www.apple.com/environment/",
        "category": "Official company sustainability information",
        "region": "Global company reporting",
    },
    "Microsoft Sustainability": {
        "url": "https://www.microsoft.com/en-us/corporate-responsibility/sustainability",
        "category": "Official company sustainability information",
        "region": "Global company reporting",
    },
}


def get_trusted_sources():
    """Return the configured source directory."""
    return TRUSTED_SOURCES


def validate_url(url):
    """Basic URL format validation; this does not prove trustworthiness."""
    try:
        parsed = urlparse(url)
        return (
            parsed.scheme == "https"
            and bool(parsed.hostname)
            and "." in parsed.hostname
        )
    except (ValueError, TypeError):
        return False


def display_sources():
    """Print the configured source list."""
    print("\nAPPROVED ESG SOURCE DIRECTORY")
    print("=" * 65)

    for name, details in TRUSTED_SOURCES.items():
        print(f"\nSource: {name}")
        print(f"Category: {details['category']}")
        print(f"Region: {details['region']}")
        print(f"URL: {details['url']}")
        print(f"HTTPS URL format valid: {validate_url(details['url'])}")


if __name__ == "__main__":
    display_sources()

