import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "output"

PREDICTIONS_FILE = OUTPUT_DIR / "esg_risk_predictions.csv"
FEATURES_FILE = OUTPUT_DIR / "esg_features.csv"
COMPANY_DATA_FILE = OUTPUT_DIR / "company_esg_data.csv"

DATABASE_FILE = BASE_DIR / "data" / "greenwashing_risk.db"
SHAP_FILE = OUTPUT_DIR / "shap_summary.png"

st.set_page_config(
    page_title="Advanced ESG Greenwashing Detection",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Advanced ESG Greenwashing Detection")

st.caption(
    "Transformer-Based Analysis and Explainable AI"
)

st.warning(
    "Academic demonstration only. Predictions use a small, manually "
    "labelled dataset and are not validated real-world risk estimates. "
    "Potential inconsistencies require further investigation."
)


# ==================================================
# 2. LOAD DATA
# ==================================================

@st.cache_data
def load_csv(file_path, modified_time):
    return pd.read_csv(file_path)


def load_data(file_path):
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return load_csv(
            str(file_path),
            file_path.stat().st_mtime,
        )
    except Exception as error:
        st.error(
            f"Could not load {file_path.name}: {error}"
        )
        return pd.DataFrame()


predictions = load_data(PREDICTIONS_FILE)
features = load_data(FEATURES_FILE)
company_data = load_data(COMPANY_DATA_FILE)


# ==================================================
# 3. LOAD DATABASE HISTORY - READ ONLY
# ==================================================

def load_database_history():
    if not DATABASE_FILE.exists():
        return pd.DataFrame(), None

    try:
        # SQLite URI read-only mode prevents accidental writes.
        database_uri = DATABASE_FILE.resolve().as_uri() + "?mode=ro"

        with sqlite3.connect(
            database_uri,
            uri=True,
        ) as connection:

            tables = pd.read_sql_query(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """,
                connection,
            )

            if tables.empty:
                return pd.DataFrame(), None

            history_table = None

            for table_name in tables["name"]:

                columns = pd.read_sql_query(
                    f'PRAGMA table_info("{table_name}")',
                    connection,
                )["name"].tolist()

                if "claim_id" in columns:
                    history_table = table_name
                    break

            if history_table is None:
                return pd.DataFrame(), (
                    "No prediction-history table containing "
                    "a claim_id column was found."
                )

            history = pd.read_sql_query(
                f'SELECT * FROM "{history_table}"',
                connection,
            )

            return history, None

    except Exception as error:
        return pd.DataFrame(), str(error)


history, database_error = load_database_history()


# ==================================================
# 4. HELPER FUNCTIONS
# ==================================================

def normalize_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip()


def show_company_record(record):
    """Display one company's claim, evidence, and analysis."""

    st.markdown(
        f"### Claim {record.get('claim_id', 'N/A')}"
    )

    st.write("**ESG Claim**")

    st.write(
        normalize_text(
            record.get("claim", "")
        ) or "Claim text unavailable."
    )

    st.write("**Matched Evidence**")

    st.write(
        normalize_text(
            record.get("evidence", "")
        ) or "Evidence text unavailable."
    )

    col1, col2, col3 = st.columns(3)

    similarity = pd.to_numeric(
        record.get("similarity"),
        errors="coerce",
    )

    col1.metric(
        "Semantic Similarity",
        f"{similarity:.4f}"
        if pd.notna(similarity)
        else "N/A",
    )

    col2.metric(
        "NLI Classification",
        normalize_text(
            record.get("nli_result", "")
        ) or "Unavailable",
    )

    col3.metric(
        "Report Year",
        str(record.get("report_year", "N/A")),
    )

    st.caption(
        "Similarity indicates textual relatedness. NLI results "
        "are model classifications and do not independently "
        "establish whether a company made a false statement."
    )


def get_company_names(data):
    if data.empty or "company" not in data.columns:
        return []

    return sorted(
        data["company"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values != ""]
        .unique()
        .tolist()
    )


# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard Overview",
        "Company Search",
        "Claim Analysis",
        "Timeline Analysis",
        "Company Comparison",
        "Risk Prediction",
        "Risk Charts",
        "Explainability",
        "Database History",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "The dashboard reads existing CSV files and database records. "
    "Opening or refreshing it does not insert prediction records."
)


# ==================================================
# 6. DASHBOARD OVERVIEW
# ==================================================

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    if predictions.empty:

        st.error(
            "Prediction data was not found. Check "
            "output/esg_risk_predictions.csv."
        )

    else:

        total_claims = len(predictions)

        categories = (
            predictions["demonstration_risk_category"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
            if "demonstration_risk_category" in predictions.columns
            else pd.Series(
                ["UNKNOWN"] * total_claims
            )
        )

        high_count = int((categories == "HIGH").sum())
        medium_count = int((categories == "MEDIUM").sum())
        low_count = int((categories == "LOW").sum())

        contradiction_values = pd.to_numeric(
            predictions.get(
                "nli_contradiction",
                pd.Series(
                    0,
                    index=predictions.index,
                ),
            ),
            errors="coerce",
        ).fillna(0)

        contradictions = int(
            (contradiction_values > 0).sum()
        )

        similarity_values = pd.to_numeric(
            predictions.get(
                "similarity",
                pd.Series(
                    dtype=float,
                ),
            ),
            errors="coerce",
        )

        average_similarity = similarity_values.mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Analyzed Claims",
            total_claims,
        )

        col2.metric(
            "Contradiction Flags",
            contradictions,
        )

        col3.metric(
            "Average Similarity",
            f"{average_similarity:.3f}"
            if pd.notna(average_similarity)
            else "N/A",
        )

        col4.metric(
            "Medium Demonstration Risks",
            medium_count,
        )

        st.subheader(
            "Demonstration Risk Categories"
        )

        category_data = pd.DataFrame(
            {
                "Category": [
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                ],
                "Number of Claims": [
                    high_count,
                    medium_count,
                    low_count,
                ],
            }
        )

        fig = px.bar(
            category_data,
            x="Category",
            y="Number of Claims",
            title="Distribution of Demonstration Risk Categories",
            text="Number of Claims",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.subheader(
            "Latest Prediction Results"
        )

        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "A contradiction flag does not independently prove "
            "greenwashing."
        )


# ==================================================
# 7. COMPANY SEARCH
# ==================================================

elif page == "Company Search":

    st.header("Company ESG Search")

    st.write(
        "Search available company records to examine ESG claims, "
        "matched evidence, and analysis results."
    )

    if company_data.empty:

        st.error(
            "Company data was not found. Run "
            "company_data_builder.py first."
        )

    elif "company" not in company_data.columns:

        st.error(
            "The company dataset does not contain a company column."
        )

    else:

        company_names = get_company_names(company_data)

        search_text = st.text_input(
            "Search company name",
            placeholder="Enter a company name...",
        )

        if search_text.strip():

            matching_names = [
                name
                for name in company_names
                if search_text.strip().casefold()
                in name.casefold()
            ]

        else:

            matching_names = company_names

        if not matching_names:

            st.warning(
                "No matching company was found in the available dataset."
            )

        else:

            selected_company = st.selectbox(
                "Select company",
                matching_names,
            )

            company_records = company_data[
                company_data["company"]
                .astype(str)
                .str.strip()
                == selected_company
            ].copy()

            report_year = (
                company_records["report_year"].iloc[0]
                if "report_year" in company_records.columns
                else "N/A"
            )

            st.subheader(selected_company)

            st.write(
                f"**Report year:** {report_year}"
            )

            st.metric(
                "Available ESG Claims",
                len(company_records),
            )

            st.divider()

            st.subheader(
                "ESG Claims and Supporting Evidence"
            )

            for _, record in company_records.iterrows():

                with st.expander(
                    f"Claim {record.get('claim_id', 'N/A')}",
                    expanded=True,
                ):

                    show_company_record(record)

            st.divider()

            st.subheader(
                "Company Analysis Summary"
            )

            if "nli_result" in company_records.columns:

                nli_counts = (
                    company_records["nli_result"]
                    .fillna("Unknown")
                    .value_counts()
                    .rename_axis("NLI Result")
                    .reset_index(name="Number of Claims")
                )

                fig = px.bar(
                    nli_counts,
                    x="NLI Result",
                    y="Number of Claims",
                    title="NLI Classification Summary",
                    text="Number of Claims",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.info(
                "This company dataset is fictional demonstration data. "
                "The results must not be interpreted as verified "
                "real-world company findings."
            )


# ==================================================
# 8. CLAIM ANALYSIS
# ==================================================

elif page == "Claim Analysis":

    st.header("Claim Analysis")

    if not company_data.empty:

        st.subheader(
            "Claims, Evidence, and NLI Results"
        )

        if "company" in company_data.columns:

            available_companies = get_company_names(
                company_data
            )

            if available_companies:

                selected_company = st.selectbox(
                    "Choose company for claim analysis",
                    available_companies,
                )

                filtered_claims = company_data[
                    company_data["company"]
                    .astype(str)
                    .str.strip()
                    == selected_company
                ]

                st.dataframe(
                    filtered_claims,
                    use_container_width=True,
                    hide_index=True,
                )

                for _, record in filtered_claims.iterrows():

                    with st.expander(
                        f"Claim {record.get('claim_id', 'N/A')} details"
                    ):

                        show_company_record(record)

    else:

        st.warning(
            "Company claim data is unavailable. "
            "Showing the feature dataset instead."
        )

        analysis_data = (
            features
            if not features.empty
            else predictions
        )

        if analysis_data.empty:

            st.error(
                "No claim data is available."
            )

        else:

            st.dataframe(
                analysis_data,
                use_container_width=True,
                hide_index=True,
            )

            if "claim_id" in analysis_data.columns:

                selected_claim = st.selectbox(
                    "Select Claim ID",
                    analysis_data["claim_id"]
                    .dropna()
                    .unique(),
                )

                claim = analysis_data[
                    analysis_data["claim_id"]
                    == selected_claim
                ]

                if not claim.empty:

                    st.subheader(
                        f"Claim {selected_claim} Details"
                    )

                    st.json(
                        claim.iloc[0]
                        .where(
                            pd.notna(claim.iloc[0]),
                            None,
                        )
                        .to_dict()
                    )

    st.info(
        "Similarity and NLI results are analytical signals, "
        "not independent proof of factual accuracy."
    )


# ==================================================
# 9. TIMELINE ANALYSIS
# ==================================================

elif page == "Timeline Analysis":

    st.header("ESG Timeline Analysis")

    if company_data.empty:

        st.warning(
            "Company ESG data is unavailable."
        )

    else:

        company_names = get_company_names(
            company_data
        )

        if not company_names:

            st.warning(
                "No company names are available."
            )

        else:

            selected_company = st.selectbox(
                "Select company for timeline",
                company_names,
            )

            timeline_data = company_data[
                company_data["company"]
                .astype(str)
                .str.strip()
                == selected_company
            ].copy()

            if "report_year" in timeline_data.columns:

                timeline_data["report_year"] = pd.to_numeric(
                    timeline_data["report_year"],
                    errors="coerce",
                )

                timeline_data = timeline_data.dropna(
                    subset=["report_year"]
                )

            if (
                not timeline_data.empty
                and "report_year" in timeline_data.columns
                and "historical_change" in timeline_data.columns
            ):

                st.subheader(
                    "Available Historical Information"
                )

                st.dataframe(
                    timeline_data[
                        [
                            column
                            for column in [
                                "report_year",
                                "claim_id",
                                "historical_change",
                                "claim",
                            ]
                            if column in timeline_data.columns
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.warning(
                    "The current data contains only a 2024 report. "
                    "The historical_change value is copied from the "
                    "existing demonstration features and does not "
                    "establish a year-by-year trend."
                )

                st.info(
                    "To create a genuine timeline, add comparable "
                    "records from multiple report years."
                )

            else:

                st.info(
                    "Sufficient dated historical information is "
                    "not available to construct a timeline."
                )


# ==================================================
# 10. COMPANY COMPARISON
# ==================================================

elif page == "Company Comparison":

    st.header("Compare Two Companies")

    if company_data.empty:

        st.warning(
            "Company records are unavailable."
        )

    else:

        company_names = get_company_names(
            company_data
        )

        if len(company_names) < 2:

            st.info(
                "Only one company is currently available: "
                + (
                    company_names[0]
                    if company_names
                    else "no company"
                )
                + ". Add another company's report and evidence "
                "to enable a meaningful comparison."
            )

        else:

            col1, col2 = st.columns(2)

            with col1:

                company_a = st.selectbox(
                    "First company",
                    company_names,
                    key="company_a",
                )

            with col2:

                company_b = st.selectbox(
                    "Second company",
                    company_names,
                    index=1,
                    key="company_b",
                )

            if company_a == company_b:

                st.warning(
                    "Select two different companies."
                )

            else:

                data_a = company_data[
                    company_data["company"]
                    .astype(str)
                    == company_a
                ]

                data_b = company_data[
                    company_data["company"]
                    .astype(str)
                    == company_b
                ]

                st.subheader(
                    "Company Claim Comparison"
                )

                left, right = st.columns(2)

                with left:

                    st.markdown(
                        f"### {company_a}"
                    )

                    st.dataframe(
                        data_a,
                        use_container_width=True,
                        hide_index=True,
                    )

                with right:

                    st.markdown(
                        f"### {company_b}"
                    )

                    st.dataframe(
                        data_b,
                        use_container_width=True,
                        hide_index=True,
                    )

                st.caption(
                    "Different claim topics, report years, and evidence "
                    "availability can affect comparisons. These records "
                    "do not establish an overall company ranking."
                )


# ==================================================
# 12. RISK PREDICTION
# ==================================================

elif page == "Risk Prediction":

    st.header(
        "Risk Prediction Results"
    )

    if predictions.empty:

        st.error(
            "No prediction data is available."
        )

    else:

        display_columns = [
            "claim_id",
            "predicted_risk_class",
            "demonstration_risk_probability",
            "demonstration_risk_category",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in predictions.columns
        ]

        risk_table = predictions[
            available_columns
        ].copy()

        if "demonstration_risk_probability" in risk_table.columns:

            risk_table[
                "demonstration_risk_probability"
            ] = (
                pd.to_numeric(
                    risk_table[
                        "demonstration_risk_probability"
                    ],
                    errors="coerce",
                ) * 100
            )

            risk_table = risk_table.rename(
                columns={
                    "demonstration_risk_probability":
                    "Demonstration Probability (%)"
                }
            )

        st.dataframe(
            risk_table,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "These are unvalidated demonstration outputs, not "
            "calibrated real-world risk probabilities."
        )


# ==================================================
# 13. RISK CHARTS
# ==================================================

elif page == "Risk Charts":

    st.header("Risk Charts")

    if predictions.empty:

        st.error(
            "No prediction data is available."
        )

    else:

        left, right = st.columns(2)

        with left:

            if "demonstration_risk_category" in predictions.columns:

                risk_counts = (
                    predictions[
                        "demonstration_risk_category"
                    ]
                    .fillna("UNKNOWN")
                    .value_counts()
                    .rename_axis("Risk Category")
                    .reset_index(
                        name="Number of Claims"
                    )
                )

                fig = px.pie(
                    risk_counts,
                    names="Risk Category",
                    values="Number of Claims",
                    title="Risk Category Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with right:

            if {
                "claim_id",
                "similarity",
            }.issubset(predictions.columns):

                similarity_data = predictions.copy()

                similarity_data["similarity"] = pd.to_numeric(
                    similarity_data["similarity"],
                    errors="coerce",
                )

                fig = px.bar(
                    similarity_data,
                    x="claim_id",
                    y="similarity",
                    title="Semantic Similarity by Claim",
                    labels={
                        "claim_id": "Claim ID",
                        "similarity": "Similarity Score",
                    },
                    range_y=[0, 1],
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        if {
            "claim_id",
            "nli_contradiction",
        }.issubset(predictions.columns):

            contradiction_data = predictions.copy()

            contradiction_data[
                "nli_contradiction"
            ] = pd.to_numeric(
                contradiction_data[
                    "nli_contradiction"
                ],
                errors="coerce",
            ).fillna(0)

            contradiction_data["NLI Result"] = (
                contradiction_data[
                    "nli_contradiction"
                ]
                .map({
                    1: "Contradiction",
                    0: "Not classified as contradiction",
                })
                .fillna("Unknown")
            )

            st.subheader(
                "Contradiction Classification"
            )

            fig = px.histogram(
                contradiction_data,
                x="NLI Result",
                title="NLI Contradiction Flags",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader(
            "Historical Performance"
        )

        if "historical_change" in predictions.columns:

            historical = predictions.copy()

            historical[
                "historical_change"
            ] = pd.to_numeric(
                historical[
                    "historical_change"
                ],
                errors="coerce",
            )

            historical = historical.dropna(
                subset=["historical_change"]
            )

            if not historical.empty:

                st.bar_chart(
                    historical.set_index(
                        "claim_id"
                    )[["historical_change"]]
                )

                st.caption(
                    "Historical-change values are shown as stored. "
                    "They do not constitute a dated timeline."
                )

            else:

                st.info(
                    "No historical-change values are available."
                )

        else:

            st.info(
                "Historical-change information is unavailable."
            )


# ==================================================
# 14. EXPLAINABILITY
# ==================================================

elif page == "Explainability":

    st.header(
        "Model Explainability — SHAP"
    )

    if SHAP_FILE.exists():

        st.image(
            str(SHAP_FILE),
            caption="SHAP Summary from the Demonstration Model",
            use_container_width=True,
        )

        st.warning(
            "The existing SHAP output previously showed zero "
            "feature contributions. It should not be treated as "
            "a meaningful explanation until this issue is investigated."
        )

    else:

        st.info(
            "SHAP summary image was not found at "
            "output/shap_summary.png."
        )

    st.subheader(
        "What SHAP Does"
    )

    st.write(
        "SHAP estimates how input features contribute to a model's "
        "output. In this project, it is intended to examine features "
        "such as similarity, contradiction indicators, and historical "
        "changes."
    )

    st.error(
        "The demonstration model and its explanations require "
        "further validation."
    )


# ==================================================
# 15. DATABASE HISTORY
# ==================================================

elif page == "Database History":

    st.header(
        "SQLite Database History"
    )

    st.write(
        f"Database location: `{DATABASE_FILE}`"
    )

    if database_error:

        st.error(
            f"Database reading issue: {database_error}"
        )

    elif history.empty:

        st.info(
            "No compatible prediction history was found."
        )

    else:

        st.metric(
            "Stored Database Records",
            len(history),
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "This section reads existing records only. "
            "It does not insert new prediction records."
        )


# ==================================================
# 16. FOOTER
# ==================================================

st.divider()

st.caption(
    "Advanced ESG Greenwashing Detection | Academic Demonstration Project"
)




# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "output" / "esg_risk_predictions.csv"
FEATURES_FILE = BASE_DIR / "output" / "esg_features.csv"
DATABASE_FILE = BASE_DIR / "data" / "greenwashing_risk.db"
SHAP_FILE = BASE_DIR / "output" / "shap_summary.png"

st.set_page_config(
    page_title="ESG Greenwashing Detection",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 Advanced ESG Greenwashing Detection")
st.caption(
    "Transformer-Based Analysis and Explainable AI"
)

st.warning(
    "Demonstration project: Predictions use a small, manually labelled "
    "dataset and are not validated real-world greenwashing risk estimates. "
    "Potential inconsistencies require further investigation."
)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_csv(file_path, modified_time):
    return pd.read_csv(file_path)


def load_data(file_path):
    if not file_path.exists():
        return pd.DataFrame()

    try:
        return load_csv(
            str(file_path),
            file_path.stat().st_mtime,
        )
    except Exception as error:
        st.error(f"Could not load {file_path.name}: {error}")
        return pd.DataFrame()


predictions = load_data(PREDICTIONS_FILE)
features = load_data(FEATURES_FILE)


# --------------------------------------------------
# 3. LOAD DATABASE HISTORY
# --------------------------------------------------

def load_database_history():
    if not DATABASE_FILE.exists():
        return pd.DataFrame(), None

    try:
        with sqlite3.connect(DATABASE_FILE) as connection:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master "
                "WHERE type='table'",
                connection,
            )

            if tables.empty:
                return pd.DataFrame(), None

            history_table = None

            for table_name in tables["name"]:
                columns = pd.read_sql_query(
                    f'PRAGMA table_info("{table_name}")',
                    connection,
                )["name"].tolist()

                if "claim_id" in columns:
                    history_table = table_name
                    break

            if history_table is None:
                return pd.DataFrame(), (
                    "No prediction-history table with a claim_id "
                    "column was found."
                )

            # Table name comes from SQLite's own table listing.
            history = pd.read_sql_query(
                f'SELECT * FROM "{history_table}"',
                connection,
            )

            return history, None

    except Exception as error:
        return pd.DataFrame(), str(error)


history, database_error = load_database_history()


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard Overview",
        "Claim Analysis",
        "Risk Prediction",
        "Risk Charts",
        "Explainability",
        "Database History",
    ],
)

st.sidebar.divider()

st.sidebar.info(
    "This dashboard reads existing CSV files and the SQLite database. "
    "Opening the dashboard does not insert new database records."
)


# --------------------------------------------------
# 5. DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.header("Dashboard Overview")

    if predictions.empty:
        st.error(
            "Prediction data was not found. "
            "Please check output/esg_risk_predictions.csv."
        )

    else:
        total_claims = len(predictions)

        if "demonstration_risk_category" in predictions.columns:
            categories = (
                predictions["demonstration_risk_category"]
                .fillna("UNKNOWN")
                .astype(str)
                .str.upper()
            )
        else:
            categories = pd.Series(
                ["UNKNOWN"] * total_claims
            )

        high_count = (categories == "HIGH").sum()
        medium_count = (categories == "MEDIUM").sum()
        low_count = (categories == "LOW").sum()

        contradictions = (
            pd.to_numeric(
                predictions.get(
                    "nli_contradiction",
                    pd.Series(0, index=predictions.index),
                ),
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        average_similarity = pd.to_numeric(
            predictions.get(
                "similarity",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Analyzed Claims", total_claims)
        col2.metric("Contradiction Flags", int(contradictions))
        col3.metric(
            "Average Similarity",
            f"{average_similarity:.3f}"
            if pd.notna(average_similarity)
            else "N/A",
        )
        col4.metric("MEDIUM Demonstration Risks", int(medium_count))

        st.subheader("Demonstration Risk Categories")

        category_data = pd.DataFrame({
            "Category": ["HIGH", "MEDIUM", "LOW"],
            "Number of Claims": [
                int(high_count),
                int(medium_count),
                int(low_count),
            ],
        })

        fig = px.bar(
            category_data,
            x="Category",
            y="Number of Claims",
            title="Distribution of Demonstration Risk Categories",
            text="Number of Claims",
        )

        st.plotly_chart(fig, use_container_width=True, key="dashboard_claims_chart")

        st.subheader("Latest Prediction Results")
        st.dataframe(
            predictions,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "A contradiction flag indicates an NLI model classification. "
            "It does not independently prove greenwashing."
        )


# --------------------------------------------------
# 6. CLAIM ANALYSIS
# --------------------------------------------------

elif page == "Claim Analysis":

    st.header("Claim Analysis")

    if features.empty:
        st.warning(
            "The feature CSV is missing or empty. "
            "Showing available prediction data instead."
        )
        analysis_data = predictions
    else:
        analysis_data = features

    if analysis_data.empty:
        st.error("No claim data is available.")

    else:
        st.dataframe(
            analysis_data,
            use_container_width=True,
            hide_index=True,
        )

        if "claim_id" in analysis_data.columns:
            selected_claim = st.selectbox(
                "Select a Claim ID",
                analysis_data["claim_id"].dropna().unique(),
            )

            claim = analysis_data[
                analysis_data["claim_id"] == selected_claim
            ]

            st.subheader(f"Claim {selected_claim} Details")

            st.json(
                claim.iloc[0].where(
                    pd.notna(claim.iloc[0]),
                    None,
                ).to_dict()
            )

        st.info(
            "Similarity measures textual relatedness. "
            "NLI classifications assess the relationship between a claim "
            "and matched evidence; neither guarantees factual correctness."
        )


# --------------------------------------------------
# 7. RISK PREDICTION
# --------------------------------------------------

elif page == "Risk Prediction":

    st.header("Risk Prediction Results")

    if predictions.empty:
        st.error("No prediction data is available.")

    else:
        display_columns = [
            "claim_id",
            "predicted_risk_class",
            "demonstration_risk_probability",
            "demonstration_risk_category",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in predictions.columns
        ]

        risk_table = predictions[available_columns].copy()

        if "demonstration_risk_probability" in risk_table.columns:
            risk_table["demonstration_risk_probability"] = (
                pd.to_numeric(
                    risk_table["demonstration_risk_probability"],
                    errors="coerce",
                ) * 100
            )

            risk_table = risk_table.rename(
                columns={
                    "demonstration_risk_probability":
                    "Demonstration Probability (%)"
                }
            )

        st.dataframe(
            risk_table,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "The probability values are demonstration outputs from the "
            "current unvalidated model. They should not be interpreted "
            "as calibrated real-world probabilities."
        )


# --------------------------------------------------
# 8. RISK CHARTS
# --------------------------------------------------

elif page == "Risk Charts":

    st.header("Risk Charts")

    if predictions.empty:
        st.error("No prediction data is available.")

    else:
        left, right = st.columns(2)

        with left:
            if "demonstration_risk_category" in predictions.columns:
                risk_counts = (
                    predictions["demonstration_risk_category"]
                    .fillna("UNKNOWN")
                    .value_counts()
                    .rename_axis("Risk Category")
                    .reset_index(name="Number of Claims")
                )

                fig = px.pie(
                    risk_counts,
                    names="Risk Category",
                    values="Number of Claims",
                    title="Risk Category Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        with right:
            if {
                "claim_id",
                "similarity",
            }.issubset(predictions.columns):

                similarity_data = predictions.copy()

                similarity_data["similarity"] = pd.to_numeric(
                    similarity_data["similarity"],
                    errors="coerce",
                )

                fig = px.bar(
                    similarity_data,
                    x="claim_id",
                    y="similarity",
                    title="Semantic Similarity by Claim",
                    labels={
                        "claim_id": "Claim ID",
                        "similarity": "Similarity Score",
                    },
                    range_y=[0, 1],
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

        if {
            "claim_id",
            "nli_contradiction",
        }.issubset(predictions.columns):

            contradiction_data = predictions.copy()

            contradiction_data["nli_contradiction"] = pd.to_numeric(
                contradiction_data["nli_contradiction"],
                errors="coerce",
            ).fillna(0)

            contradiction_data["NLI Result"] = (
                contradiction_data["nli_contradiction"]
                .map({
                    1: "Contradiction",
                    0: "Not classified as contradiction",
                })
                .fillna("Unknown")
            )

            st.subheader("Contradiction Classification")

            fig = px.histogram(
                contradiction_data,
                x="NLI Result",
                title="NLI Contradiction Flags",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        st.subheader("Historical Performance")

        if "historical_change" in predictions.columns:
            historical = predictions.copy()

            historical["historical_change"] = pd.to_numeric(
                historical["historical_change"],
                errors="coerce",
            )

            historical = historical.dropna(
                subset=["historical_change"]
            )

            if not historical.empty:
                st.bar_chart(
                    historical.set_index("claim_id")[
                        ["historical_change"]
                    ]
                )

                st.caption(
                    "Historical change is shown as stored in the sample "
                    "feature data. It is not a year-by-year timeline."
                )
            else:
                st.info("No historical-change values are available.")

        else:
            st.info(
                "Historical performance data is not available "
                "in the prediction CSV."
            )


# --------------------------------------------------
# 9. EXPLAINABILITY
# --------------------------------------------------

elif page == "Explainability":

    st.header("Model Explainability — SHAP")

    if SHAP_FILE.exists():
        st.image(
            str(SHAP_FILE),
            caption="SHAP Summary from the Demonstration Model",
            use_container_width=True,
        )

        st.warning(
            "The current SHAP output previously showed zero feature "
            "contributions. Therefore, this image should not yet be "
            "treated as a meaningful explanation of risk predictions."
        )

    else:
        st.info(
            "SHAP summary image was not found at "
            "output/shap_summary.png."
        )

    st.subheader("What SHAP Does")

    st.write(
        "SHAP is a model-explainability method that estimates how input "
        "features contribute to a model's output."
    )

    st.write(
        "For this project, SHAP is intended to help explain how features "
        "such as semantic similarity, contradiction indicators, and "
        "historical changes influence the risk classifier."
    )

    st.error(
        "The current demonstration model and its explanations require "
        "further validation before real-world interpretation."
    )


# --------------------------------------------------
# 10. DATABASE HISTORY
# --------------------------------------------------

elif page == "Database History":

    st.header("SQLite Database History")

    st.write(f"Database location: `{DATABASE_FILE}`")

    if database_error:
        st.error(f"Database reading issue: {database_error}")

    elif history.empty:
        st.info(
            "No prediction history was found, or the database "
            "does not contain a compatible prediction table."
        )

    else:
        st.metric(
            "Stored Database Records",
            len(history),
        )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "This section reads existing database records only. "
            "It does not insert or duplicate prediction records."
        )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Advanced ESG Greenwashing Detection | Academic Demonstration Project"
)
