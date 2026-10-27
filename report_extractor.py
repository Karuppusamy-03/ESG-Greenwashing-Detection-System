from pathlib import Path


def extract_text(file_path):
    """Read text from a sample ESG text file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")

    text = path.read_text(encoding="utf-8")
    return text


if __name__ == "__main__":
    report_path = Path("data") / "sample_esg_report.txt"

    try:
        report_text = extract_text(report_path)

        print("ESG report loaded successfully!")
        print("-" * 50)
        print(report_text)

    except FileNotFoundError as error:
        print(error)