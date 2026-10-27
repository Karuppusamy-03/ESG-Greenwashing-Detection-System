import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def extract_website_text(url):
    """
    Extract readable text from an accessible official website.

    Returns:
        Dictionary containing the URL, title, extracted text,
        and source domain.
    """

    if not url.startswith(("http://", "https://")):
        raise ValueError("Enter a valid URL beginning with http:// or https://.")

    parsed_url = urlparse(url)

    if not parsed_url.hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; ESGResearchBot/1.0; "
            "+https://example.org/research)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=20,
        allow_redirects=True,
    )
    response.raise_for_status()

    # Only process HTML pages.
    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type:
        raise ValueError(
            "This URL does not return an HTML page. "
            "For PDF reports, use the PDF extraction module."
        )

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove elements that usually do not contain article content.
    for element in soup(
        ["script", "style", "noscript", "nav", "footer", "header"]
    ):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"

    text = soup.get_text(separator=" ", strip=True)

    # Normalize excessive whitespace.
    text = " ".join(text.split())

    if not text:
        raise ValueError(
            "No readable text was found. The website may require JavaScript "
            "or may restrict automated access."
        )

    return {
        "url": response.url,
        "domain": urlparse(response.url).hostname,
        "title": title,
        "text": text,
    }


if __name__ == "__main__":
    print("ESG Website Extractor")
    print("-" * 50)

    url = input("Enter an official company webpage URL: ").strip()

    try:
        result = extract_website_text(url)

        print("\nWebsite title:", result["title"])
        print("Source domain:", result["domain"])
        print("Final URL:", result["url"])
        print("\nExtracted text preview:\n")
        print(result["text"][:3000])

    except requests.RequestException as error:
        print("\nWebsite retrieval failed:", error)

    except ValueError as error:
        print("\nExtraction error:", error)
