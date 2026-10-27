from pathlib import Path
import pandas as pd

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Fictional demonstration company
company_name = "Green Future Industries"
report_year = 2024

# Claims and their corresponding evidence
records = [
    {
        "company": company_name,
        "report_year": report_year,
        "claim_id": 1,
        "claim": (
            "The company claims that it reduced carbon "
            "emissions by 30% in 2024."
        ),
        "evidence": (
            "The sustainability report provides no numerical "
            "data to verify the claimed 30% reduction."
        ),
        "similarity": 0.5752,
        "nli_result": "Neutral",
        "historical_change": 20,
    },
    {
        "company": company_name,
        "report_year": report_year,
        "claim_id": 2,
        "claim": (
            "The company claims that 80% of its electricity "
            "comes from renewable energy."
        ),
        "evidence": (
            "The report states that 45% of electricity comes "
            "from renewable energy."
        ),
        "similarity": 0.8408,
        "nli_result": "Contradiction",
        "historical_change": 20,
    },
    {
        "company": company_name,
        "report_year": report_year,
        "claim_id": 3,
        "claim": (
            "The company claims that it will achieve carbon "
            "neutrality by 2030."
        ),
        "evidence": (
            "The company has announced a carbon neutrality "
            "target for 2030, but no detailed action plan "
            "is provided."
        ),
        "similarity": 0.8016,
        "nli_result": "Entailment",
        "historical_change": 20,
    },
]

df = pd.DataFrame(records)

output_file = OUTPUT_DIR / "company_esg_data.csv"

df.to_csv(output_file, index=False)

print("Company ESG demonstration dataset created successfully.")
print(f"File location: {output_file}")
print(f"Company: {company_name}")
print(f"Report year: {report_year}")
print(f"Number of claims: {len(df)}")

print("\nDataset preview:")
print(df.to_string(index=False))

