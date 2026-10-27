import hashlib
import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from automatic_collector import (
    collect_documents,
    persist_collection_diagnostics,
    collect_company_esg_information,
)
from contradiction_detector import detect_online_contradictions
from analysis_serialization import dumps_analysis_json
from database import save_company_predictions, save_timeline_records
from explainability import explain_risk_predictions
from feature_builder import build_features_from_results
from online_claim_extractor import (
    extract_embedded_assurance_statements,
    extract_environmental_statements,
    identify_source_role,
    remove_attributed_assurance_sections,
)
from online_evidence_matcher import match_online_evidence
from risk_model import FEATURE_COLUMNS
from risk_predictor import predict_risk_from_features
from timeline_analyzer import (
    analyze_timeline,
    build_trend_points,
    build_timeline_change_rows,
    find_timeline_review_points,
    get_comparable_metrics,
)
from web_search import (
    CLAIM_EVIDENCE_SEARCH_BACKENDS,
    ambiguous_name_continuation,
    build_claim_evidence_queries,
    is_ambiguous_apple_food_result,
    normalize_company_name,
    search_web,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
ANALYSES_DIR = OUTPUT_DIR / "company_analyses"
DATABASE_FILE = BASE_DIR / "data" / "greenwashing_risk.db"
SHAP_SUMMARY_FILE = OUTPUT_DIR / "shap_summary.png"
WORKFLOW_UI_REFRESH_SECONDS = 1

NAVIGATION = [
    "Dashboard Overview",
    "Company ESG Analysis",
]

FEATURE_DESCRIPTIONS = {
    "similarity": "Text similarity to the strongest retrieved statement; not proof.",
    "evidence_available": "1 means a candidate match passed the retrieval threshold.",
    "nli_entailment": "Count of NLI model outputs classified as entailment; not verification.",
    "nli_contradiction": "Count of NLI model outputs classified as contradiction; an investigation signal only.",
    "nli_neutral": "Count of NLI model outputs classified as neutral.",
    "percentage_difference": "Unavailable: no comparable reported value was extracted.",
    "percentage_difference_missing": "1 means the percentage difference is unavailable.",
    "historical_change": "Unavailable: no comparable historical series was extracted.",
}

PROJECT_STAGES = [
    "Web Search",
    "Web Data Collection",
    "Environmental Claim Extraction",
    "Independent Evidence Search",
    "Evidence Matching",
    "NLI Analysis",
    "Risk Prediction",
    "Risk Explanation",
    "Evidence Timeline Analysis",
    "Database Saving",
]
COMPANY_STAGES = PROJECT_STAGES


@st.cache_resource(show_spinner=False)
def get_workflow_runtime():
    return {
        "executor": ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="esg-workflow",
        ),
        "tasks": {},
        "tasks_by_session": {},
        "tasks_lock": threading.RLock(),
        "database_write_lock": threading.Lock(),
    }


READY_STATUS_MESSAGE = (
    "Ready to start. Enter a company name to begin."
)
IN_PROGRESS_STATUS_MESSAGE = "⏳ Analysis in Progress... Please Wait."
COMPANY_SUCCESS_MESSAGE = "✅ Company ESG Analysis Completed Successfully!"
LIMITATIONS_STATUS_MESSAGE = (
    "⚠️ Analysis Completed with Limitations. Please review the warnings "
    "and missing evidence."
)
FAILED_STATUS_MESSAGE = (
    "❌ Analysis Failed. Please check the error details and try again."
)


def redact_error_details(message):
    message = re.sub(
        r"(?i)(api[_-]?key|token|password|secret|authorization)"
        r"(\s*[:=]\s*)([^\s,;]+)",
        r"\1\2[REDACTED]",
        str(message),
    )
    return re.sub(
        r"(https?://)[^/@\s]+:[^/@\s]+@",
        r"\1[REDACTED]@",
        message,
    )


def create_progress_state(applicable_stages):
    applicable = set(applicable_stages)
    ordered_applicable = [
        stage for stage in PROJECT_STAGES if stage in applicable
    ]
    return {
        "status": "WAITING",
        "finished": False,
        "message": READY_STATUS_MESSAGE,
        "progress": 0.0,
        "applicable_stages": ordered_applicable,
        "stages": {
            stage: {
                "status": "WAITING" if stage in applicable else "NOT AVAILABLE",
                "detail": "",
                "progress": 0.0,
            }
            for stage in PROJECT_STAGES
        },
        "stage_runs": {},
    }


def calculate_progress(state):
    applicable = [
        stage for stage in state["applicable_stages"]
        if state["stages"][stage]["status"] != "NOT AVAILABLE"
    ]
    if not applicable:
        progress = 1.0 if state["finished"] else 0.0
        return min(progress, 0.99) if (
            state["finished"] and state.get("status") == "PARTIAL"
        ) else progress

    completed = 0.0
    for stage in applicable:
        stage_state = state["stages"][stage]
        if stage_state["status"] in {"COMPLETED", "PARTIAL", "FAILED"}:
            completed += 1.0
        elif stage_state["status"] == "RUNNING":
            completed += min(max(stage_state["progress"], 0.0), 0.99)

    progress = completed / len(applicable)
    if not state["finished"]:
        return min(progress, 0.99)
    if state.get("status") == "PARTIAL":
        return min(progress, 0.99)
    return progress


def update_project_stage(
    state_key,
    stage,
    status,
    detail="",
    progress=None,
    subtask_index=0,
    subtask_count=1,
    progress_panel=None,
):
    state = st.session_state[state_key]
    update_progress_state(
        state,
        stage,
        status,
        detail,
        progress,
        subtask_index,
        subtask_count,
    )
    if progress_panel is not None:
        render_project_progress(progress_panel)


def update_progress_state(
    state,
    stage,
    status,
    detail="",
    progress=None,
    subtask_index=0,
    subtask_count=1,
):
    if stage not in state["applicable_stages"]:
        return

    stage_state = state["stages"][stage]
    if subtask_count > 1:
        runs = state["stage_runs"].setdefault(stage, {})
        runs[subtask_index] = {
            "status": status,
            "progress": (
                min(max(float(progress), 0.0), 0.99)
                if status == "RUNNING" and progress is not None
                else 0.0
            ),
        }
        run_values = [
            runs.get(index, {"status": "WAITING", "progress": 0.0})
            for index in range(subtask_count)
        ]
        finished_runs = [
            run for run in run_values
            if run["status"] in {
                "COMPLETED",
                "PARTIAL",
                "FAILED",
                "NOT AVAILABLE",
            }
        ]
        if any(run["status"] == "RUNNING" for run in run_values) or len(
            finished_runs
        ) < subtask_count:
            stage_state["status"] = "RUNNING"
            stage_state["progress"] = sum(
                1.0 if run["status"] in {
                    "COMPLETED", "PARTIAL", "FAILED", "NOT AVAILABLE"
                }
                else run["progress"]
                for run in run_values
            ) / subtask_count
        elif any(run["status"] == "FAILED" for run in run_values):
            stage_state["status"] = "FAILED"
            stage_state["progress"] = 1.0
        elif any(run["status"] == "PARTIAL" for run in run_values):
            stage_state["status"] = "PARTIAL"
            stage_state["progress"] = 1.0
        elif all(
            run["status"] == "NOT AVAILABLE"
            for run in run_values
        ):
            stage_state["status"] = "NOT AVAILABLE"
            stage_state["progress"] = 1.0
        elif any(run["status"] == "NOT AVAILABLE" for run in run_values):
            stage_state["status"] = "PARTIAL"
            stage_state["progress"] = 1.0
        else:
            stage_state["status"] = "COMPLETED"
            stage_state["progress"] = 1.0
    else:
        stage_state["status"] = status
        stage_state["progress"] = (
            min(max(float(progress), 0.0), 0.99)
            if status == "RUNNING" and progress is not None
            else 1.0 if status in {"COMPLETED", "PARTIAL", "FAILED"}
            else 0.0
        )
    stage_state["detail"] = detail
    state["progress"] = calculate_progress(state)


def start_project_workflow(state_key, applicable_stages, progress_panel=None):
    st.session_state[state_key] = create_progress_state(applicable_stages)
    st.session_state.pop(
        state_key.replace("_progress_status", "_progress_debug_error"),
        None,
    )
    state = st.session_state[state_key]
    state["status"] = "RUNNING"
    state["message"] = IN_PROGRESS_STATUS_MESSAGE
    if progress_panel is not None:
        render_project_progress(progress_panel)


def finish_project_workflow(
    state_key,
    outcome,
    message,
    progress_panel=None,
):
    state = st.session_state[state_key]
    finish_progress_state(state, outcome, message)
    if progress_panel is not None:
        render_project_progress(progress_panel)


def finish_progress_state(state, outcome, message):
    has_running_stages = any(
        state["stages"][stage]["status"] == "RUNNING"
        for stage in state["applicable_stages"]
    )
    if outcome == "COMPLETED" and has_running_stages:
        outcome = "PARTIAL"
        message = LIMITATIONS_STATUS_MESSAGE
    state["status"] = outcome
    state["finished"] = True
    state["message"] = message
    for stage in state["applicable_stages"]:
        stage_status = state["stages"][stage]["status"]
        if stage_status in {"RUNNING", "WAITING"}:
            update_progress_state(
                state,
                stage,
                (
                    "PARTIAL"
                    if outcome == "PARTIAL"
                    and stage_status == "RUNNING"
                    else "FAILED"
                    if outcome == "FAILED"
                    and stage_status == "RUNNING"
                    else "NOT AVAILABLE"
                ),
                "The stage did not run before the workflow ended.",
            )
    state["progress"] = calculate_progress(state)


def publish_background_stage(
    task,
    stage,
    status,
    detail="",
    progress=None,
    subtask_index=0,
    subtask_count=1,
):
    with task["lock"]:
        update_progress_state(
            task["state"],
            stage,
            status,
            detail,
            progress,
            subtask_index,
            subtask_count,
        )


def save_background_result(task, result):
    with task["lock"]:
        task["result"] = deepcopy(result)


def schedule_background_workflow(workflow, stages, worker, *args):
    session_id = st.session_state.setdefault(
        "_workflow_session_id",
        uuid.uuid4().hex,
    )
    registry_key = (session_id, workflow)
    with _WORKFLOW_TASKS_LOCK:
        existing_id = _WORKFLOW_TASKS_BY_SESSION.get(registry_key)
        existing = _WORKFLOW_TASKS.get(existing_id)
        if existing is not None:
            with existing["lock"]:
                if existing["state"]["status"] in {"WAITING", "RUNNING"}:
                    if existing.get("args") == args:
                        st.session_state[f"_{workflow}_task_id"] = existing_id
                        return False
            _WORKFLOW_TASKS.pop(existing_id, None)

        task_id = uuid.uuid4().hex
        state = create_progress_state(stages)
        state["message"] = "Waiting for an available analysis worker."
        task = {
            "id": task_id,
            "workflow": workflow,
            "args": args,
            "state": state,
            "result": None,
            "error": None,
            "lock": threading.RLock(),
            "future": None,
        }
        _WORKFLOW_TASKS[task_id] = task
        _WORKFLOW_TASKS_BY_SESSION[registry_key] = task_id
        st.session_state[f"_{workflow}_task_id"] = task_id
        st.session_state["active_analysis"] = None
        st.session_state.pop("company_progress_debug_error", None)
        state_key = "company_progress_status"
        st.session_state[state_key] = deepcopy(state)
        task["future"] = _WORKFLOW_EXECUTOR.submit(
            run_background_workflow,
            task,
            worker,
            *args,
        )
    return True


def run_background_workflow(task, worker, *args):
    try:
        with task["lock"]:
            task["state"]["status"] = "RUNNING"
            task["state"]["message"] = IN_PROGRESS_STATUS_MESSAGE
        worker(task, *args)
    except Exception as error:
        raw_error = f"{type(error).__name__}: {error}"
        with task["lock"]:
            task["error"] = raw_error
            finish_progress_state(
                task["state"],
                "FAILED",
                FAILED_STATUS_MESSAGE,
            )
            task["state"]["debug_error"] = redact_error_details(raw_error)


def sync_background_workflow_state():
    session_id = st.session_state.get("_workflow_session_id")
    if not session_id:
        return
    task_id = st.session_state.get("_company_task_id")
    if not task_id:
        return
    with _WORKFLOW_TASKS_LOCK:
        task = _WORKFLOW_TASKS.get(task_id)
    if task is None or task["workflow"] != "company":
        return
    with task["lock"]:
        state = deepcopy(task["state"])
        result = deepcopy(task["result"])
        error = task["error"]
    st.session_state["company_progress_status"] = state
    if result is not None:
        st.session_state["active_analysis"] = result
    if error:
        st.session_state["company_progress_debug_error"] = (
            redact_error_details(error)
        )


def render_project_progress(progress_panel=None):
    if st.session_state.get("dashboard_navigation") == "Company ESG Analysis":
        state = st.session_state["company_progress_status"]
        progress = calculate_progress(state)
        state["progress"] = progress
        target = (
            progress_panel.container()
            if progress_panel is not None
            else st.container()
        )
        with target:
            st.progress(
                progress,
                text=f"Company analysis: {state['status']} · {progress:.0%}",
            )
        return

    target = (
        progress_panel.container()
        if progress_panel is not None
        else st.container()
    )
    with target:
        st.markdown("### ANALYSIS PROGRESS")
        company_state = st.session_state["company_progress_status"]
        rows = []
        for stage in PROJECT_STAGES:
            stage_state = company_state["stages"][stage]
            rows.append({
                "Workflow": "Company ESG Analysis",
                "Stage": stage,
                "Status": stage_state["status"],
                "Details": stage_state["detail"],
            })
        display_dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

        progress = calculate_progress(company_state)
        company_state["progress"] = progress
        st.progress(
            progress,
            text=f"Company ESG Analysis: {company_state['status']} · {progress:.0%}",
        )
        if company_state["status"] == "RUNNING":
            st.info(IN_PROGRESS_STATUS_MESSAGE)
        elif company_state["status"] == "FAILED":
            st.error(FAILED_STATUS_MESSAGE)
        elif company_state["status"] == "PARTIAL":
            st.warning(LIMITATIONS_STATUS_MESSAGE)
        elif company_state["status"] == "COMPLETED":
            st.success(company_state["message"])

        if company_state["status"] == "WAITING":
            st.info(READY_STATUS_MESSAGE)

st.set_page_config(
    page_title="Advanced ESG Greenwashing Detection",
    page_icon="🌱",
    layout="wide",
)
_WORKFLOW_RUNTIME = get_workflow_runtime()
_WORKFLOW_EXECUTOR = _WORKFLOW_RUNTIME["executor"]
_WORKFLOW_TASKS = _WORKFLOW_RUNTIME["tasks"]
_WORKFLOW_TASKS_BY_SESSION = _WORKFLOW_RUNTIME["tasks_by_session"]
_WORKFLOW_TASKS_LOCK = _WORKFLOW_RUNTIME["tasks_lock"]
_DATABASE_WRITE_LOCK = _WORKFLOW_RUNTIME["database_write_lock"]

st.markdown(
    """
    <style>
    :root {
        --ink: #183047;
        --muted: #71839a;
        --line: #e4ebf3;
        --green: #23a87a;
        --blue: #2785d8;
    }
    .stApp {background: #f4f7fb; color: var(--ink);}
    [data-testid="stHeader"] {display: none;}
    .block-container {padding: 1rem 1.35rem 2rem; max-width: 1680px;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #071a2a 0%, #0e2d4c 100%);
        border-right: 1px solid #193b5a;
    }
    section[data-testid="stSidebar"] {
        width: 290px !important; min-width: 290px !important;
    }
    [data-testid="stSidebar"] * {color: #eaf3ff;}
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        font-size: .88rem; white-space: nowrap;
    }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--line); border-radius: 11px;
        padding: .65rem .8rem; box-shadow: 0 3px 12px rgba(24,48,71,.04);
    }
    [data-testid="stMetricLabel"] {color: var(--muted); font-size: .76rem;}
    [data-testid="stMetricValue"] {color: var(--ink); font-size: 1.55rem;}
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: white; border-color: var(--line) !important; border-radius: 11px;
        box-shadow: 0 3px 14px rgba(24,48,71,.035);
    }
    h1, h2, h3 {color: var(--ink);}
    h1 {font-size: 1.65rem !important; margin-bottom: .1rem !important;}
    h2 {font-size: 1.13rem !important; margin: .25rem 0 .5rem !important;}
    h3 {font-size: .98rem !important;}
    div[data-testid="stDataFrame"] {border: 1px solid var(--line); border-radius: 8px;}
    div.stButton > button[kind="primary"] {
        background: #1474d4; border: 0; border-radius: 7px; font-weight: 650;
    }
    .dashboard-eyebrow {
        color: #228d72; font-size: .68rem; font-weight: 800; letter-spacing: .12em;
        text-transform: uppercase; margin-bottom: .2rem;
    }
    .dashboard-caption {color: var(--muted); font-size: .8rem; margin: 0 0 .7rem;}
    .section-heading {
        display: flex; align-items: center; gap: .45rem; margin: .3rem 0 .6rem;
        color: var(--ink); font-size: 1.06rem; font-weight: 750;
    }
    .pill {
        display: inline-block; padding: .18rem .48rem; border-radius: 999px;
        font-size: .68rem; font-weight: 700; background: #e9f7f1; color: #19835f;
    }
    .brand-lockup {font-size: 1.03rem; font-weight: 750; line-height: 1.2;}
    .brand-subtitle {color: #9db5ca; font-size: .68rem; margin-top: .15rem;}
    section.main .block-container {
        padding: 1.35rem 1.65rem 2.5rem;
        max-width: 1480px;
    }
    section.main [data-testid="stTextInput"] input,
    section.main [data-testid="stNumberInput"] input,
    section.main [data-testid="stTextArea"] textarea {
        border-color: #d7e1ec;
        border-radius: 8px;
        background: #fff;
    }
    section.main [data-testid="stTextInput"] input:focus,
    section.main [data-testid="stNumberInput"] input:focus,
    section.main [data-testid="stTextArea"] textarea:focus {
        border-color: #2785d8;
        box-shadow: 0 0 0 2px rgba(39, 133, 216, .12);
    }
    section.main [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: #d7e1ec;
        border-radius: 8px;
    }
    section.main div.stButton > button {
        min-height: 2.45rem;
        border-radius: 8px;
        font-weight: 650;
        transition: border-color .15s ease, background-color .15s ease;
    }
    section.main div.stButton > button[kind="primary"] {
        background: #176fc1;
        border-color: #176fc1;
    }
    section.main [data-testid="stAlert"] {
        border-radius: 9px;
    }
    section.main [data-testid="stPlotlyChart"],
    section.main div[data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 9px;
        overflow: hidden;
    }
    @media (max-width: 850px) {
        .block-container {padding: .8rem .7rem 1.4rem;}
        section.main .block-container {padding: .8rem .7rem 1.4rem;}
        h1 {font-size: 1.35rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource
def load_demo_risk_model():
    # The active risk path is a transparent heuristic. The old six-example
    # classifier is retained only for compatibility and is not used for scoring.
    return None


def company_tokens(company_name):
    tokens = re.findall(r"[a-z0-9]+", company_name.casefold())
    suffixes = {
        "co", "company", "corp", "corporation", "inc", "incorporated",
        "limited", "llc", "ltd", "plc",
    }

    while tokens and tokens[-1] in suffixes:
        tokens.pop()

    while tokens and tokens[0] == "the":
        tokens.pop(0)

    return tokens


def document_mentions_company(company_name, document):
    wanted = company_tokens(company_name)
    if not wanted:
        return False

    title_and_snippet = " ".join(
        (
            str(document.get("title") or ""),
            str(document.get("snippet") or ""),
        )
    ).casefold()
    body = str(document.get("text") or "")
    text = f"{title_and_snippet} {body}".casefold()

    def contains_company(value):
        available = re.findall(r"[a-z0-9]+", value.casefold())
        if not any(
            available[index:index + len(wanted)] == wanted
            for index in range(len(available) - len(wanted) + 1)
        ):
            return False
        # "Delta" in "Delta Air Lines" describes a different company unless
        # the sentence carries corporate or subsidiary context.
        return not ambiguous_name_continuation(company_name, value)

    body_for_relevance = re.sub(
        r"\b(inc|corp|ltd|llc)\.(?=\s)",
        r"\1",
        body,
        flags=re.IGNORECASE,
    )
    body_confirms_company = any(
        contains_company(sentence)
        and any(
            keyword in sentence.casefold()
            for keyword in (
                "environment",
                "emissions",
                "greenhouse gas",
                "carbon",
                "climate",
                "sustainability",
                "renewable",
                "recycled",
                "pollution",
                "waste",
                "water",
                "energy",
                "footprint",
            )
        )
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            body_for_relevance,
        )
    )
    official_company_source = (
        identify_source_role(document) == "company_source"
    )
    mentions_company = body_confirms_company or (
        official_company_source
        and contains_company(title_and_snippet)
        and any(
            keyword in body.casefold()
            for keyword in (
                "environment",
                "emissions",
                "greenhouse gas",
                "carbon",
                "climate",
                "sustainability",
                "renewable",
                "recycled",
                "pollution",
                "waste",
                "water",
                "energy",
                "footprint",
            )
        )
    )
    if not mentions_company:
        return False

    if is_ambiguous_apple_food_result(company_name, text):
        return False

    return True


def collect_candidate_statements(company_name, documents):
    company_claims = []
    external_evidence = []
    all_statements = []
    statements_by_key = {}

    for document in documents:
        if (
            document.get("status") != "success"
            or not str(document.get("text") or "").strip()
            or not document_mentions_company(company_name, document)
        ):
            continue

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
        document_statements = [
            statement
            for statement in extract_environmental_statements(
                company_document
            )
        ]
        for statement in [*document_statements, *assurance_statements]:
            source_role = statement.get("source_role", "unknown")
            item = {
                **statement,
                "entity_context": company_name,
                "source_url": (
                    statement.get("source_url")
                    or document.get("final_url")
                    or document.get("source_url")
                    or ""
                ),
                "source_title": (
                    statement.get("source_title")
                    or document.get("title")
                    or ""
                ),
                "source_type": (
                    statement.get("source_type")
                    or document.get("source_type", "")
                ),
            }
            normalized_statement = " ".join(
                str(item.get("statement") or "").split()
            ).casefold()
            statement_key = (source_role, normalized_statement)
            existing_item = statements_by_key.get(statement_key)
            if existing_item is not None:
                provenance = {
                    "source_title": item.get("source_title", ""),
                    "source_url": item.get("source_url", ""),
                    "source_domain": item.get("source_domain", ""),
                    "publication_date": item.get("publication_date"),
                }
                additional_sources = existing_item.setdefault(
                    "additional_sources",
                    [],
                )
                if provenance not in additional_sources and (
                    provenance["source_url"]
                    != existing_item.get("source_url")
                ):
                    additional_sources.append(provenance)
                continue
            statements_by_key[statement_key] = item
            if source_role in {
                "company_source",
                "potential_company_source_unverified",
            }:
                item["candidate_status"] = "company_claim_candidate"
                item["candidate_rejection_reason"] = None
            elif source_role in {
                "external_source",
                "external_regulatory_evidence",
                "configured_environmental_source",
                "independent_third_party_source",
            }:
                item["candidate_status"] = "eligible_external_candidate"
                item["candidate_rejection_reason"] = None
            elif source_role == "rehosted_report_unverified":
                item["candidate_status"] = "rejected_external_candidate"
                item["candidate_rejection_reason"] = (
                    "The source is a known report-hosting platform; "
                    "independent provenance was not established."
                )
            else:
                item["candidate_status"] = "rejected_external_candidate"
                item["candidate_rejection_reason"] = (
                    "The source is not classified as regulatory, configured, "
                    "or explicitly attributed independent evidence."
                )
            all_statements.append(item)

            if item.get("source_role") in {
                "company_source",
                "potential_company_source_unverified",
            }:
                if (
                    item.get("source_role")
                    == "potential_company_source_unverified"
                    or "potential company source" in str(
                        item.get("source_type", "")
                    ).casefold()
                ):
                    item["statement_kind"] = (
                        "Possible company-source candidate; not verified"
                    )
                else:
                    item["statement_kind"] = (
                        "Company-source candidate; not verified"
                    )
                company_claims.append(item)
            elif item.get("source_role") in {
                "external_source",
                "external_regulatory_evidence",
                "configured_environmental_source",
                "independent_third_party_source",
            }:
                item["statement_kind"] = "External evidence candidate"
                external_evidence.append(item)
            else:
                item["statement_kind"] = (
                    item["candidate_rejection_reason"]
                )

    return company_claims, external_evidence, all_statements


def collect_claim_specific_evidence(company_name, claims, existing_documents):
    """Retrieve a bounded set of additional sources for extracted claims."""

    queries = build_claim_evidence_queries(company_name, claims)
    existing_urls = {
        str(document.get("final_url") or document.get("source_url") or "")
        .strip()
        .rstrip("/")
        .casefold()
        for document in existing_documents
    }
    search_results = []
    search_diagnostics = []
    seen_urls = set(existing_urls)

    for query in queries:
        diagnostics = {}
        backend = CLAIM_EVIDENCE_SEARCH_BACKENDS[
            len(search_diagnostics) % len(CLAIM_EVIDENCE_SEARCH_BACKENDS)
        ]
        results = search_web(
            query,
            max_results=2,
            company_name=company_name,
            diagnostics=diagnostics,
            backend=backend,
        )
        search_diagnostics.append(diagnostics)
        for result in results:
            url = str(result.get("url") or "").strip().rstrip("/").casefold()
            if url and url not in seen_urls:
                seen_urls.add(url)
                search_results.append(result)

    if not search_results:
        return [], None, search_diagnostics

    source_priorities = {
        "government / regulatory source": 0,
        "configured environmental source": 1,
        "independent third-party source": 1,
        "other / unverified": 2,
        "potential company source (domain-name match)": 3,
        "official company source": 4,
        "government source / regulatory status unverified": 5,
        "unknown": 6,
    }
    search_results = [
        result
        for _, result in sorted(
            enumerate(search_results),
            key=lambda item: (
                source_priorities.get(
                    str(item[1].get("source_type") or "")
                    .strip()
                    .casefold(),
                    6,
                ),
                item[0],
            ),
        )
    ]

    documents, output_file = collect_documents(
        search_results,
        company_name,
        max_documents=min(3, len(search_results)),
    )
    return documents, output_file, search_diagnostics


def build_pipeline_diagnostics(
    collection,
    documents,
    relevant_documents,
    statements,
    evidence_candidates,
    matching_results,
    nli_results,
):
    diagnostics = deepcopy(collection.get("diagnostics") or {})
    relevant_ids = {id(document) for document in relevant_documents}

    def role_counts(items):
        counts = {}
        for item in items:
            role = item.get("source_role") or identify_source_role(item)
            counts[role] = counts.get(role, 0) + 1
        return counts

    matched_pairs = sum(
        len(result.get("matches") or [])
        for result in matching_results or []
        if isinstance(result, dict)
    )
    rejected_candidates = [
        {
            "statement": item.get("statement", ""),
            "source_title": item.get("source_title", ""),
            "source_url": item.get("source_url", ""),
            "source_role": item.get("source_role", "unknown"),
            "reason": item.get("candidate_rejection_reason"),
        }
        for item in statements
        if item.get("candidate_status") == "rejected_external_candidate"
    ]

    rejected_documents = []
    for document in documents:
        if id(document) in relevant_ids:
            continue
        if document.get("status") != "success":
            reason = (
                f"Document collection status is "
                f"{document.get('status', 'unknown')}."
            )
        elif not str(document.get("text") or "").strip():
            reason = "No document text was extracted."
        else:
            reason = (
                "The page content did not confirm the requested entity; "
                "search metadata attribution was pending."
                if document.get("attribution_status")
                == "pending_page_content"
                else (
                    "The document did not pass company entity "
                    "relevance filtering; extracted statements were not "
                    "attributed."
                )
            )
        rejected_documents.append({
            "source_title": document.get("title", ""),
            "search_query": document.get("search_query", ""),
            "source_url": (
                document.get("final_url")
                or document.get("source_url")
                or ""
            ),
            "original_source_url": document.get("source_url", ""),
            "attribution_status": document.get(
                "attribution_status",
                "not_recorded",
            ),
            "attribution_basis": document.get("attribution_basis", ""),
            "source_role": (
                document.get("source_role")
                or identify_source_role(document)
            ),
            "status": document.get("status", "unknown"),
            "reason": reason,
        })
    diagnostics["pipeline"] = {
        "relevance": {
            "document_count": len(documents),
            "relevant_document_count": len(relevant_documents),
            "documents": [
                {
                    "source_url": (
                        document.get("final_url")
                        or document.get("source_url")
                        or ""
                    ),
                    "original_source_url": document.get("source_url", ""),
                    "search_query": document.get("search_query", ""),
                    "attribution_status": document.get(
                        "attribution_status",
                        "not_recorded",
                    ),
                    "attribution_basis": document.get(
                        "attribution_basis",
                        "",
                    ),
                    "status": document.get("status", "unknown"),
                    "source_role": (
                        document.get("source_role")
                        or identify_source_role(document)
                    ),
                    "relevant": id(document) in relevant_ids,
                    "rejection_reason": next(
                        (
                            rejected["reason"]
                            for rejected in rejected_documents
                            if rejected["source_url"] == (
                                document.get("final_url")
                                or document.get("source_url")
                                or ""
                            )
                        ),
                        None,
                    ),
                }
                for document in documents
            ],
            "rejected_documents": rejected_documents,
        },
        "source_role_counts": {
            "downloaded_documents": role_counts(documents),
            "relevant_statements": role_counts(statements),
            "evidence_candidates": role_counts(evidence_candidates),
        },
        "evidence": {
            "statement_count": len(statements),
            "external_candidate_count": len(evidence_candidates),
            "matched_claim_count": sum(
                bool(result.get("matches"))
                for result in matching_results or []
                if isinstance(result, dict)
            ),
            "matched_pair_count": matched_pairs,
            "nli_result_count": len(nli_results or []),
            "rejected_candidate_count": len(rejected_candidates),
            "rejected_candidates": rejected_candidates,
        },
    }
    return diagnostics


def match_claims_to_evidence(company_claims, external_evidence):
    claim_texts = [item["statement"] for item in company_claims]

    if not claim_texts:
        return []

    if not external_evidence:
        return [
            {
                "claim": claim,
                "evidence_status": "no_sufficiently_relevant_match",
                "matches_found": 0,
                "matches": [],
            }
            for claim in claim_texts
        ]

    model = load_embedding_model()
    return match_online_evidence(
        claim_texts,
        external_evidence,
        model,
    )


def attach_timeline_evidence(timeline_records, matching_results, nli_results):
    matches_by_claim = {
        result.get("claim"): result
        for result in matching_results or []
    }
    nli_by_match = {
        (
            item.get("claim"),
            item.get("evidence"),
            item.get("source_url"),
        ): item
        for item in nli_results or []
    }
    for record in timeline_records:
        result = matches_by_claim.get(record["statement"], {})
        matches = result.get("matches") or []
        best_match = matches[0] if matches else {}
        nli = nli_by_match.get(
            (
                record["statement"],
                best_match.get("matched_evidence"),
                best_match.get("source_url"),
            ),
            {},
        )
        record["supporting_evidence"] = best_match.get(
            "matched_evidence",
            "",
        )
        record["supporting_evidence_source"] = best_match.get(
            "source_title",
            "",
        )
        record["supporting_evidence_url"] = best_match.get(
            "source_url",
            "",
        )
        record["supporting_evidence_status"] = (
            best_match.get("match_label")
            or "No matched external evidence"
        )
        record["supporting_evidence_nli"] = (
            nli.get("nli_result") or "Not reviewed"
        )


def build_analysis(
    company_name,
    progress_callback=None,
    stage_callback=None,
):
    search_errors = []

    def report_stage(stage, status, detail="", progress=None):
        if stage_callback is not None:
            stage_callback(stage, status, detail, progress)

    collection = collect_company_esg_information(
        company_name,
        max_documents=8,
        progress_callback=progress_callback,
        error_callback=lambda query, message: search_errors.append({
            "query": query,
            "error": message,
        }),
        stage_callback=report_stage,
    )
    collection_has_run_diagnostics = isinstance(
        collection.get("diagnostics"),
        dict,
    )
    collection["search_errors"] = search_errors
    documents = collection.get("documents", [])
    relevant_documents = [
        document
        for document in documents
        if (
            document.get("status") == "success"
            and str(document.get("text") or "").strip()
            and document_mentions_company(company_name, document)
        )
    ]

    report_stage(
        "Environmental Claim Extraction",
        "RUNNING",
        "Extracting candidate statements from relevant documents.",
    )
    seed_claims, _, _ = (
        collect_candidate_statements(company_name, relevant_documents)
    )
    supplemental_documents = []
    supplemental_output_file = None
    evidence_search_diagnostics = []
    evidence_search_error = None
    if seed_claims:
        report_stage(
            "Independent Evidence Search",
            "RUNNING",
            "Searching for external sources tied to extracted claim topics, "
            "measurements, and years.",
        )
        try:
            (
                supplemental_documents,
                supplemental_output_file,
                evidence_search_diagnostics,
            ) = collect_claim_specific_evidence(
                company_name,
                seed_claims,
                documents,
            )
            documents.extend(supplemental_documents)
            if supplemental_output_file:
                collection["supplemental_output_file"] = (
                    supplemental_output_file
                )
            collection["diagnostics"] = collection.get("diagnostics") or {}
            collection["diagnostics"]["claim_specific_search"] = {
                "queries": evidence_search_diagnostics,
                "documents_collected": len(supplemental_documents),
                "output_file": supplemental_output_file,
            }
        except Exception as error:
            evidence_search_error = (
                f"{type(error).__name__}: {error}"
            )
    else:
        report_stage(
            "Independent Evidence Search",
            "NOT AVAILABLE",
            "No company claims were available to build targeted searches.",
        )

    relevant_documents = [
        document
        for document in documents
        if (
            document.get("status") == "success"
            and str(document.get("text") or "").strip()
            and document_mentions_company(company_name, document)
        )
    ]
    company_claims, external_evidence, all_statements = (
        collect_candidate_statements(company_name, relevant_documents)
    )
    extraction_status = (
        "COMPLETED"
        if company_claims or external_evidence
        else "PARTIAL"
    )
    report_stage(
        "Environmental Claim Extraction",
        extraction_status,
        (
            "Candidate statements were extracted."
            if extraction_status == "COMPLETED"
            else "No environmental candidate statements were extracted."
        ),
    )
    if seed_claims:
        if evidence_search_error:
            evidence_search_status = "PARTIAL"
            evidence_search_detail = (
                "Claim-specific source search failed; existing external "
                "evidence candidates were retained."
                if external_evidence
                else "Claim-specific source search failed."
            )
        elif external_evidence:
            evidence_search_status = "COMPLETED"
            evidence_search_detail = (
                f"Found {len(external_evidence)} external evidence candidates."
            )
        else:
            evidence_search_status = "PARTIAL"
            evidence_search_detail = (
                "No relevant external evidence was retrieved."
            )
        report_stage(
            "Independent Evidence Search",
            evidence_search_status,
            evidence_search_detail,
        )

    report_stage(
        "Evidence Timeline Analysis",
        "RUNNING",
        "Extracting dated environmental statements.",
    )
    timeline_error = None
    try:
        timeline_records = analyze_timeline(relevant_documents)
        report_stage(
            "Evidence Timeline Analysis",
            "COMPLETED" if timeline_records else "PARTIAL",
            (
                "Timeline records were extracted."
                if timeline_records
                else "No timeline records were extracted."
            ),
        )
    except Exception as error:
        timeline_records = []
        timeline_error = f"{type(error).__name__}: {error}"
        report_stage(
            "Evidence Timeline Analysis",
            "FAILED",
            "Timeline analysis did not complete.",
        )

    analysis = {
        "company": company_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "collection": collection,
        "relevant_document_count": len(relevant_documents),
        "company_claims": company_claims,
        "external_evidence": external_evidence,
        "all_statements": all_statements,
        "timeline_records": timeline_records,
        "timeline_error": timeline_error,
        "matching_results": None,
        "nli_results": [],
        "features": [],
        "risk_error": None,
        "risk_explanations": [],
        "explainability_error": None,
        "errors": {},
    }
    if evidence_search_error:
        analysis["errors"]["Independent evidence search"] = (
            evidence_search_error
        )

    def record_analysis_diagnostics():
        diagnostics = build_pipeline_diagnostics(
            collection,
            documents,
            relevant_documents,
            all_statements,
            external_evidence,
            analysis.get("matching_results"),
            analysis.get("nli_results"),
        )
        collection["diagnostics"] = diagnostics
        analysis["diagnostics"] = diagnostics
        if collection_has_run_diagnostics:
            try:
                persist_collection_diagnostics(
                    collection.get("output_file"),
                    {"pipeline": diagnostics.get("pipeline", {})},
                )
            except (OSError, TypeError, ValueError) as error:
                analysis["diagnostics_persistence_error"] = (
                    f"{type(error).__name__}: {error}"
                )

    if not relevant_documents:
        analysis["matching_results"] = []
        record_analysis_diagnostics()
        report_stage(
            "Evidence Matching",
            "NOT AVAILABLE",
            "No relevant documents were available for matching.",
        )
        report_stage(
            "NLI Analysis",
            "NOT AVAILABLE",
            "NLI requires an evidence match.",
        )
        report_stage(
            "Risk Prediction",
            "NOT AVAILABLE",
            "Risk prediction requires an eligible evidence match.",
        )
        report_stage(
            "Risk Explanation",
            "NOT AVAILABLE",
            "SHAP requires a successful risk prediction.",
        )
        return analysis

    report_stage(
        "Evidence Matching",
        "RUNNING",
        "Matching company-source statements to external evidence.",
    )
    try:
        matching_results = match_claims_to_evidence(
            company_claims,
            external_evidence,
        )
        analysis["matching_results"] = matching_results
        fully_matched_claims = sum(
            result.get("evidence_status")
            == "potentially_relevant_evidence_found"
            for result in matching_results
        )
        matching_completed = (
            bool(matching_results)
            and fully_matched_claims == len(matching_results)
        )
        report_stage(
            "Evidence Matching",
            "COMPLETED" if matching_completed else "PARTIAL",
            (
                "All claims had a strong external evidence candidate."
                if matching_completed
                else (
                    f"{fully_matched_claims} of {len(matching_results)} "
                    "claims had a strong external evidence candidate."
                )
            ),
        )
    except Exception as error:
        analysis["errors"]["Evidence matching"] = (
            f"{type(error).__name__}: {error}"
        )
        report_stage(
            "Evidence Matching",
            "FAILED",
            "Evidence matching did not complete.",
        )
        report_stage(
            "NLI Analysis",
            "NOT AVAILABLE",
            "NLI requires an evidence match.",
        )
        report_stage(
            "Risk Prediction",
            "NOT AVAILABLE",
            "Risk prediction requires an eligible evidence match.",
        )
        report_stage(
            "Risk Explanation",
            "NOT AVAILABLE",
            "SHAP requires a successful risk prediction.",
        )
        attach_timeline_evidence(
            analysis["timeline_records"],
            [],
            [],
        )
        record_analysis_diagnostics()
        return analysis

    if any(result.get("matches") for result in analysis["matching_results"]):
        report_stage(
            "NLI Analysis",
            "RUNNING",
            "Classifying matched evidence and company statements.",
        )
        try:
            analysis["nli_results"] = detect_online_contradictions({
                "company": company_name,
                "results": analysis["matching_results"],
            })
            classified_results = [
                result
                for result in analysis["nli_results"]
                if result.get("nli_result") in {
                    "ENTAILMENT",
                    "CONTRADICTION",
                    "NEUTRAL",
                }
                and result.get("scoring_eligible", True)
            ]
            nli_completed = (
                bool(analysis["nli_results"])
                and len(classified_results) == len(analysis["nli_results"])
            )
            report_stage(
                "NLI Analysis",
                "COMPLETED" if nli_completed else "PARTIAL",
                (
                    "All matched evidence pairs received eligible NLI classifications."
                    if nli_completed
                    else "One or more matched evidence pairs were unavailable or ineligible for NLI classification."
                ),
            )
        except Exception as error:
            analysis["errors"]["NLI classification"] = (
                f"{type(error).__name__}: {error}"
            )
            report_stage(
                "NLI Analysis",
                "FAILED",
                "NLI classification did not complete.",
            )
    else:
        report_stage(
            "NLI Analysis",
            "NOT AVAILABLE",
            "NLI requires at least one matched evidence pair.",
        )

    attach_timeline_evidence(
        analysis["timeline_records"],
        analysis["matching_results"],
        analysis["nli_results"],
    )
    features = build_features_from_results(
        analysis["matching_results"],
        analysis["nli_results"],
    )

    if not features.empty:
        features["predicted_risk_class"] = None
        features["demonstration_risk_probability"] = None
        features["demonstration_risk_category"] = "NOT_PREDICTED"
        features["risk_score"] = None
        features["risk_score_status"] = "NOT_PREDICTED"
        features["risk_unavailable_reason"] = ""
        features.loc[
            features["evidence_available"].ne(1),
            "risk_unavailable_reason",
        ] = "No eligible external evidence matched this claim."

        eligible = features["evidence_available"].eq(1)
        if eligible.any():
            report_stage(
                "Risk Prediction",
                "RUNNING",
                "Predicting only claims with retrieved evidence matches.",
            )
            try:
                model = load_demo_risk_model()
                predictions = predict_risk_from_features(
                    features.loc[eligible].copy(),
                    model=model,
                )
                prediction_by_claim = predictions.set_index("claim_id")

                for row_index in features.index[eligible]:
                    claim_id = features.at[row_index, "claim_id"]
                    prediction = prediction_by_claim.loc[claim_id]
                    score = prediction["risk_score"]
                    if pd.isna(score):
                        # Unassessed: evidence exists but no usable NLI result.
                        # Never convert a missing score to a number or LOW.
                        features.at[row_index, "risk_score"] = None
                        features.at[row_index, "predicted_risk_class"] = None
                    else:
                        features.at[row_index, "predicted_risk_class"] = int(
                            prediction["predicted_risk_class"]
                        )
                        features.at[row_index, "risk_score"] = float(score)
                    features.at[row_index, "risk_score_status"] = (
                        prediction["risk_score_status"]
                    )
                    features.at[row_index, "demonstration_risk_category"] = (
                        str(prediction["demonstration_risk_category"])
                    )
                    features.at[row_index, "risk_unavailable_reason"] = (
                        str(prediction.get("risk_unavailable_reason", "") or "")
                    )
                risk_completed = (
                    bool(eligible.all())
                    and len(predictions) == len(features)
                )
                report_stage(
                    "Risk Prediction",
                    "COMPLETED" if risk_completed else "PARTIAL",
                    (
                        "Experimental predictions were produced for all claims."
                        if risk_completed
                        else (
                            f"Experimental predictions were produced for "
                            f"{len(predictions)} of {len(features)} claims; "
                            "claims without eligible evidence were not scored."
                        )
                    ),
                )
            except Exception as error:
                analysis["risk_error"] = (
                    f"{type(error).__name__}: {error}"
                )
                report_stage(
                    "Risk Prediction",
                    "FAILED",
                    "Risk prediction did not complete.",
                )
                features["predicted_risk_class"] = None
                features["demonstration_risk_probability"] = None
                features["demonstration_risk_category"] = "NOT_PREDICTED"
                features["risk_score"] = None
                features["risk_score_status"] = "NOT_PREDICTED"
            else:
                report_stage(
                    "Risk Explanation",
                    "RUNNING",
                    "Explaining the heuristic rule applied to each claim.",
                )
                try:
                    analysis["risk_explanations"] = (
                        explain_risk_predictions(
                            predictions,
                            model,
                            company_name,
                        )
                    )
                    explanation_completed = (
                        risk_completed
                        and bool(analysis["risk_explanations"])
                        and len(analysis["risk_explanations"])
                        == len(predictions)
                    )
                    report_stage(
                        "Risk Explanation",
                        "COMPLETED" if explanation_completed else "PARTIAL",
                        (
                            "Heuristic decision factors were described for all predicted claims."
                            if explanation_completed
                            else "Explanations were unavailable for one or more claims."
                        ),
                    )
                except Exception as error:
                    analysis["explainability_error"] = (
                        f"{type(error).__name__}: {error}"
                    )
                    report_stage(
                        "Risk Explanation",
                        "FAILED",
                        "The risk explanation could not be generated.",
                    )
        else:
            report_stage(
                "Risk Prediction",
                "NOT AVAILABLE",
                "No claims had a matched external evidence item.",
            )
            report_stage(
                "Risk Explanation",
                "NOT AVAILABLE",
                "No predictions were eligible for explanation.",
            )
    else:
        report_stage(
            "Risk Prediction",
            "NOT AVAILABLE",
            "No risk feature rows were available.",
        )
        report_stage(
            "Risk Explanation",
            "NOT AVAILABLE",
            "No predictions were eligible for explanation.",
        )

    analysis["features"] = [
        {
            key: _to_plain(value)
            for key, value in row.items()
        }
        for row in features.to_dict(orient="records")
    ]
    record_analysis_diagnostics()
    return analysis


def _to_plain(value):
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if pd.isna(value):
        return None
    return value


def persist_analysis(analysis):
    identity_data = {
        "company": normalize_company_name(analysis["company"]),
        "documents": [
            {
                "url": document.get("final_url") or document.get("source_url"),
                "status": document.get("status"),
                "publication_date": document.get("publication_date"),
            }
            for document in analysis["collection"].get("documents", [])
        ],
        "claims": analysis["company_claims"],
        "matches": analysis["matching_results"],
        "nli": analysis["nli_results"],
        "features": analysis["features"],
        "timeline_records": analysis.get("timeline_records", []),
    }
    identity_json = json.dumps(
        _to_plain(identity_data),
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    analysis_id = hashlib.sha256(identity_json.encode("utf-8")).hexdigest()
    analysis["analysis_id"] = analysis_id
    try:
        analysis["timeline_database_rows_inserted"] = save_timeline_records(
            normalize_company_name(analysis["company"]),
            analysis_id,
            analysis.get("timeline_records", []),
            DATABASE_FILE,
        )
    except (sqlite3.Error, ValueError, TypeError, OSError) as error:
        analysis["timeline_database_error"] = (
            f"{type(error).__name__}: {error}"
        )

    safe_company = normalize_company_name(analysis["company"]) or "company"
    analysis_path = (
        ANALYSES_DIR / f"{safe_company}_{analysis_id[:16]}.json"
    )
    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)

    def validate_saved_analysis(path):
        with path.open("r", encoding="utf-8") as file:
            saved_analysis = json.load(file)
        if (
            not isinstance(saved_analysis, dict)
            or saved_analysis.get("analysis_id") != analysis_id
        ):
            raise ValueError(
                "The existing analysis file does not match its analysis identifier."
            )

    temporary_path = None
    cleanup_error = None
    try:
        if analysis_path.exists():
            validate_saved_analysis(analysis_path)
        else:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=ANALYSES_DIR,
                prefix=f".{analysis_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temporary_path = Path(file.name)
                file.write(dumps_analysis_json(analysis))
                file.flush()
                os.fsync(file.fileno())

            validate_saved_analysis(temporary_path)

            try:
                os.link(temporary_path, analysis_path)
            except FileExistsError:
                validate_saved_analysis(analysis_path)
    except (OSError, TypeError, ValueError) as error:
        analysis["persistence_error"] = (
            f"Could not save the analysis JSON: "
            f"{type(error).__name__}: {error}"
        )
        return
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            except OSError as error:
                cleanup_error = error
    if cleanup_error is not None:
        analysis["persistence_error"] = (
            f"Could not clean up temporary analysis JSON: "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
        return

    analysis["analysis_path"] = str(analysis_path)
    if not analysis["features"]:
        analysis["database_rows_inserted"] = 0
        return

    try:
        analysis["database_rows_inserted"] = save_company_predictions(
            normalize_company_name(analysis["company"]),
            analysis_id,
            pd.DataFrame(analysis["features"]),
            DATABASE_FILE,
        )
    except (sqlite3.Error, ValueError, TypeError, OSError) as error:
        analysis["database_error"] = (
            f"{type(error).__name__}: {error}"
        )


def load_database_history():
    if not DATABASE_FILE.exists():
        return pd.DataFrame(), "The analysis database does not exist yet."

    database_uri = DATABASE_FILE.resolve().as_uri() + "?mode=ro"
    try:
        with sqlite3.connect(database_uri, uri=True) as connection:
            table_exists = connection.execute(
                """
                SELECT 1 FROM sqlite_master
                WHERE type = 'table' AND name = 'risk_predictions'
                """
            ).fetchone()
            if not table_exists:
                return pd.DataFrame(), (
                    "No risk_predictions table is present in the database."
                )
            history = pd.read_sql_query(
                "SELECT * FROM risk_predictions ORDER BY id DESC",
                connection,
            )
            if "company_name" not in history.columns:
                history["company_name"] = pd.NA
            if "analysis_id" not in history.columns:
                history["analysis_id"] = pd.NA
            return history, None
    except sqlite3.Error as error:
        return pd.DataFrame(), f"{type(error).__name__}: {error}"


def load_saved_analyses():
    if not ANALYSES_DIR.exists():
        return [], []

    analyses = []
    errors = []
    for path in sorted(
        ANALYSES_DIR.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    ):
        try:
            with path.open("r", encoding="utf-8") as file:
                analyses.append(json.load(file))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path.name}: {type(error).__name__}: {error}")
    return analyses, errors


def link_column(label="Source URL"):
    return st.column_config.LinkColumn(
        label,
        display_text="Open source",
    )


def prepare_display_dataframe(dataframe):
    """Normalize mixed object columns on a display-only dataframe copy."""

    prepared = dataframe.copy()
    for column in prepared.columns:
        values = prepared[column]
        if not pd.api.types.is_object_dtype(values.dtype):
            continue

        non_missing = values[values.notna()]
        if non_missing.empty:
            continue

        if all(
            isinstance(value, Number) and not isinstance(value, bool)
            for value in non_missing
        ):
            numeric_values = pd.to_numeric(values, errors="coerce")
            numeric_values_non_missing = numeric_values.dropna()
            is_integral = (
                not numeric_values_non_missing.empty
                and numeric_values_non_missing.map(
                    lambda value: float(value).is_integer()
                ).all()
            )
            prepared[column] = numeric_values.astype(
                "Int64" if is_integral else "Float64"
            )
            continue

        value_types = {type(value) for value in non_missing}
        if len(value_types) > 1:
            prepared[column] = values.map(
                lambda value: (
                    "Unavailable" if pd.isna(value) else str(value)
                )
            )

    return prepared


def display_dataframe(dataframe, **kwargs):
    st.dataframe(prepare_display_dataframe(dataframe), **kwargs)


def show_collection_summary(analysis):
    collection = analysis["collection"]
    documents = collection.get("documents", [])
    relevant_urls = {
        document.get("final_url") or document.get("source_url")
        for document in documents
        if (
            document.get("status") == "success"
            and document_mentions_company(analysis["company"], document)
        )
    }
    rows = []
    for document in documents:
        url = document.get("final_url") or document.get("source_url", "")
        relevant = url in relevant_urls
        rows.append({
            "Title": document.get("title") or "Untitled",
            "Status": document.get("status") or "unknown",
            "Company relevance": (
                "Included" if relevant else "Excluded from evidence"
            ),
            "Publication date": (
                document.get("publication_date") or "Date unavailable"
            ),
            "Source URL": url,
            "Details": document.get("error") or "",
        })

    successful = sum(
        document.get("status") == "success"
        and bool(str(document.get("text") or "").strip())
        for document in documents
    )
    failed = sum(
        document.get("status") in {"failed", "empty"}
        for document in documents
    )
    st.write(
        f"Search results: **{collection.get('search_results_count', 0)}** · "
        f"Accessible documents: **{successful}** · "
        f"Failed or empty: **{failed}** · "
        f"Relevant to this company: **{analysis['relevant_document_count']}**"
    )
    if collection.get("output_file"):
        st.caption(
            f"Raw collection saved to `{Path(collection['output_file']).name}`."
        )
    for search_error in collection.get("search_errors", []):
        st.warning(
            "Search request failed for "
            f"`{redact_error_details(search_error['query'])}`: "
            f"{redact_error_details(search_error['error'])}"
        )
    if rows:
        display_dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={"Source URL": link_column()},
        )
    elif (
        not collection.get("search_results_count")
        and not collection.get("search_errors")
    ):
        st.info("The web search returned no results for this company.")
    elif not collection.get("search_results_count"):
        st.error("No search query returned results; see the request errors above.")

    if not analysis["relevant_document_count"]:
        st.warning(
            "No relevant accessible documents were collected. Search results "
            "that did not identify this company are excluded from the evidence."
        )
def show_statements_table(statements):
    if not statements:
        st.info("No ESG candidate statements are available for this section.")
        return

    rows = [
        {
            "Statement": item.get("statement", ""),
            "Type": item.get("statement_kind", ""),
            "Source title": item.get("source_title") or "Title unavailable",
            "Source role": item.get("source_role") or "Unknown",
            "Publication date": (
                item.get("publication_date") or "Date unavailable"
            ),
            "Source URL": item.get("source_url", ""),
        }
        for item in statements
    ]
    display_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={"Source URL": link_column()},
    )


def get_timeline_rows(analysis):
    rows = []
    for statement in analysis["all_statements"]:
        role = (
            "Claim publication date"
            if statement.get("source_role") == "company_source"
            else "Evidence publication date"
        )
        date_value = statement.get("publication_date")
        parsed_date = pd.to_datetime(
            date_value,
            errors="coerce",
            utc=True,
        )
        rows.append({
            "_sort_date": parsed_date,
            "Date": (
                parsed_date.strftime("%Y-%m-%d")
                if not pd.isna(parsed_date)
                else "Date unavailable"
            ),
            "Date meaning": role,
            "Source title": (
                statement.get("source_title") or "Title unavailable"
            ),
            "Description": statement.get("statement", ""),
            "Source URL": statement.get("source_url", ""),
        })

    rows.sort(
        key=lambda item: (
            pd.isna(item["_sort_date"]),
            item["_sort_date"] if not pd.isna(item["_sort_date"]) else pd.Timestamp.max.tz_localize("UTC"),
        )
    )
    for row in rows:
        row.pop("_sort_date")
    return rows


def show_risk_features(analysis):
    features = analysis.get("features", [])
    if not features:
        st.info("No risk features could be built from this analysis.")
        return

    feature_rows = []
    for record in features:
        for feature in FEATURE_COLUMNS:
            value = record.get(feature)
            if value is None:
                display_value = "Unavailable"
            elif feature == "similarity":
                display_value = f"{float(value):.4f}"
            else:
                display_value = value
            feature_rows.append({
                "Claim ID": record.get("claim_id"),
                "Feature": feature,
                "Value": display_value,
                "What it represents": FEATURE_DESCRIPTIONS[feature],
            })

    display_dataframe(
        pd.DataFrame(feature_rows),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Feature values describe inputs available to the model. They do not "
        "establish that any feature caused greenwashing."
    )


def _show_risk_explanations(explanations):
    if not explanations:
        st.info("No risk explanations were returned for the current analysis.")
        return
    rows = [{
        "Claim": item.get("claim", ""),
        "Feature": item.get("feature", ""),
        "Feature value": item.get("feature_value"),
        "Explanation": item.get("explanation", ""),
    } for item in explanations]
    display_dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Feature value": st.column_config.NumberColumn(format="%.3f"),
            "Explanation": st.column_config.TextColumn(width="large"),
        },
    )


def active_analysis():
    return st.session_state.get("active_analysis")


def dashboard_claim_rows(analysis):
    if not analysis:
        return [], "", False

    rows = []
    match_by_claim = {
        result.get("claim"): result
        for result in (analysis.get("matching_results") or [])
    }
    feature_by_claim = {
        item.get("claim"): item
        for item in analysis.get("features", [])
    }
    for claim in analysis.get("company_claims", []):
        text = claim.get("statement", "")
        result = match_by_claim.get(text, {})
        matches = result.get("matches") or []
        best_match = matches[0] if matches else {}
        nli = next(
            (
                item
                for item in analysis.get("nli_results", [])
                if item.get("claim") == text
                and item.get("source_url") == best_match.get("source_url")
                and item.get("evidence") == best_match.get("matched_evidence")
            ),
            {},
        )
        prediction = feature_by_claim.get(text, {})
        publication_date = best_match.get("publication_date")
        year_match = re.search(r"\b(?:19|20)\d{2}\b", str(publication_date or ""))
        rows.append({
            "claim": text,
            "evidence": best_match.get("matched_evidence", ""),
            "evidence_status": result.get(
                "evidence_status", "no_sufficiently_relevant_match"
            ),
            "similarity": best_match.get("similarity"),
            "nli": nli.get("nli_result") or "UNREVIEWED",
            "source_title": best_match.get("source_title") or "Source unavailable",
            "source_url": best_match.get("source_url") or "",
            "publication_year": year_match.group(0) if year_match else "",
            "risk_category": prediction.get(
                "demonstration_risk_category", "NOT_PREDICTED"
            ),
        })
    return rows, analysis.get("company", "Company"), False


def display_similarity(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric_value if math.isfinite(numeric_value) else None


def chart_theme(figure, height=230):
    figure.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#42566d", size=10),
        title=dict(font=dict(size=12, color="#183047")),
        legend=dict(font=dict(size=9)),
    )
    return figure


def company_analysis_worker(task, company_name):
    def report_collection_progress(index, total, document):
        title = document.get("title") or document.get("source_url")
        publish_background_stage(
            task,
            "Web Data Collection",
            "RUNNING",
            f"Collected document {index}/{total}: {title}.",
            progress=index / total if total else 0.0,
        )

    def report_analysis_stage(
        stage,
        stage_status,
        detail="",
        stage_progress=None,
    ):
        publish_background_stage(
            task,
            stage,
            stage_status,
            detail,
            progress=stage_progress,
        )

    result = build_analysis(
        company_name,
        progress_callback=report_collection_progress,
        stage_callback=report_analysis_stage,
    )
    save_background_result(task, result)

    publish_background_stage(
        task,
        "Database Saving",
        "RUNNING",
        "Saving analysis results and timeline records.",
    )
    with _DATABASE_WRITE_LOCK:
        persist_analysis(result)
    persistence_status = (
        "FAILED"
        if result.get("persistence_error")
        else "PARTIAL"
        if result.get("database_error")
        or result.get("timeline_database_error")
        else "COMPLETED"
    )
    publish_background_stage(
        task,
        "Database Saving",
        persistence_status,
        (
            "Analysis results could not be fully saved."
            if persistence_status != "COMPLETED"
            else "Analysis JSON and applicable database rows were saved."
        ),
    )

    with task["lock"]:
        has_limitations = (
            persistence_status != "COMPLETED"
            or bool(result.get("errors"))
            or bool(result.get("risk_error"))
            or bool(result.get("explainability_error"))
            or bool(result.get("timeline_error"))
            or any(
                task["state"]["stages"][stage]["status"] != "COMPLETED"
                for stage in task["state"]["applicable_stages"]
            )
        )
        outcome = "PARTIAL" if has_limitations else "COMPLETED"
        task["result"] = deepcopy(result)
        finish_progress_state(
            task["state"],
            outcome,
            (
                LIMITATIONS_STATUS_MESSAGE
                if has_limitations
                else COMPANY_SUCCESS_MESSAGE
            ),
        )


def run_company_search(progress_panel=None):
    search_col, button_col = st.columns([4, 1])
    with search_col:
        company_name = st.text_input(
            "Company Name",
            placeholder="Enter company name...",
            key="company_search_input",
        )
    with button_col:
        submitted = st.button(
            "Search & Analyse",
            type="primary",
            key="company_search_submit",
        )

    if not submitted:
        return
    cleaned_name = company_name.strip()
    if not cleaned_name:
        st.error("Enter a company name before starting an analysis.")
        return
    st.session_state.pop("timeline_metric_selection", None)
    schedule_background_workflow(
        "company",
        COMPANY_STAGES,
        company_analysis_worker,
        cleaned_name,
    )


def render_environmental_performance_timeline(analysis):
    if analysis.get("timeline_error"):
        st.error(
            "Timeline extraction failed: "
            f"{redact_error_details(analysis['timeline_error'])}"
        )

    records = analysis.get("timeline_records", [])
    if not records:
        st.info(
            "No explicitly year-dated environmental statements were found "
            "in the successfully collected documents. Publication dates are "
            "not substituted for reporting years, and missing years do not "
            "indicate poor performance."
        )
        return

    timeline_rows = []
    for record in records:
        performance = (
            f"{record['performance_type']}: {record['value']:g} {record['unit']}"
            if record["value"] is not None
            else record["performance_type"]
        )
        timeline_rows.append({
            "Year": record["timeline_year"],
            "Year basis": record["year_basis"],
            "Reporting year": (
                str(record["reporting_year"])
                if record["reporting_year"] is not None
                else "Unavailable for target statement"
            ),
            "Environmental metric": record["metric"],
            "Value": (
                record["value"]
                if record["value"] is not None
                else "Not quantified"
            ),
            "Unit": record["unit"] or "Not stated",
            "Environmental claim or target": record["statement"],
            "Reported performance": performance,
            "Previous reported performance": (
                record["previous_reported_performance"] or "Unavailable"
            ),
            "Change over time": record["change_over_time"],
            "Evidence status": record["evidence_status"],
            "Supporting evidence": (
                record.get("supporting_evidence")
                or "No matched external evidence"
            ),
            "Supporting evidence status": record.get(
                "supporting_evidence_status",
                "No matched external evidence",
            ),
            "NLI review": record.get(
                "supporting_evidence_nli",
                "Not reviewed",
            ),
            "Source name": record["source_name"],
            "Publisher": record.get("source_publisher") or "Unavailable",
            "Publication date": record.get("publication_date") or "Unavailable",
            "Retrieved at": record.get("collected_at") or "Unavailable",
            "Target year": (
                str(record["target_year"])
                if record.get("target_year") is not None
                else "—"
            ),
            "Source URL": record["source_url"],
            "Supporting evidence URL": record.get(
                "supporting_evidence_url",
                "",
            ),
        })
    display_dataframe(
        pd.DataFrame(timeline_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Environmental claim or target": st.column_config.TextColumn(
                width="large"
            ),
            "Source URL": link_column(),
            "Year": st.column_config.NumberColumn(format="%d"),
            "Supporting evidence": st.column_config.TextColumn(width="large"),
            "Supporting evidence URL": link_column(
                "Supporting evidence URL"
            ),
        },
    )

    numeric_results = [
        record
        for record in records
        if (
            record["value"] is not None
            and record["performance_type"] != "Target"
        )
    ]
    if len({record["reporting_year"] for record in numeric_results}) == 1:
        st.info(
            "Only one reporting year with an extracted environmental "
            "measurement is available. The data is shown above, but a trend "
            "cannot be established without additional comparable years."
        )

    review_points = find_timeline_review_points(records)
    if review_points:
        st.markdown("**Items for further investigation**")
        for point in review_points:
            st.warning(point)
        st.caption(
            "These source differences or target comparisons are review leads, "
            "not proof of greenwashing or non-compliance."
        )

    comparable_metrics = get_comparable_metrics(records)
    if not comparable_metrics:
        st.info(
            "More comparable historical data is needed for a trend chart. "
            "Measurements must share the same metric, unit, and stated "
            "reporting boundary across at least two years. Targets are not "
            "plotted as achieved performance."
        )
        return

    metric_labels = [
        f"{item['metric']} ({item['unit']}; {item['boundary']})"
        for item in comparable_metrics
    ]
    metric_by_label = dict(zip(metric_labels, comparable_metrics))
    selected_metric = st.selectbox(
        "Environmental metric",
        metric_labels,
        key="timeline_metric_selection",
    )
    if selected_metric is None:
        st.info("Select an environmental metric to display its historical trend.")
        return
    metric = metric_by_label[selected_metric]
    trend_points = build_trend_points(records, metric)
    if not trend_points:
        st.info(
            "There are not enough unambiguous reported values to plot this "
            "metric."
        )
        return

    trend = pd.DataFrame(trend_points)
    figure = px.line(
        trend,
        x="Year",
        y="Value",
        markers=True,
        hover_data=["Data status", "Source titles", "Source URLs"],
        title=f"{metric['metric']} over reporting years",
    )
    figure.update_traces(connectgaps=False)
    figure.update_layout(
        xaxis_title="Year",
        yaxis_title=f"Reported value ({metric['unit']})",
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(
        chart_theme(figure, height=300),
        width="stretch",
        config={"displayModeBar": False},
    )
    reported_points = trend[trend["Value"].notna()].copy()
    st.markdown("**Year-by-year comparison**")
    bar_figure = px.bar(
        reported_points,
        x="Year",
        y="Value",
        hover_data=["Source titles", "Source URLs"],
        title=f"{metric['metric']} by reporting year",
    )
    bar_figure.update_layout(
        xaxis_title="Year",
        yaxis_title=f"{metric['metric']} ({metric['unit']})",
        xaxis=dict(dtick=1),
    )
    st.plotly_chart(
        chart_theme(bar_figure, height=300),
        width="stretch",
        config={"displayModeBar": False},
    )

    change_rows = build_timeline_change_rows(records, metric)
    st.markdown("**Key changes**")
    if change_rows:
        for change in change_rows:
            percent = (
                f"{change['Percentage change']:+g}%"
                if change["Percentage change"] is not None
                else "not calculable (previous value was zero)"
            )
            st.write(
                f"From {change['Previous year']} to {change['Current year']}, "
                f"the reported value {change['Direction']} from "
                f"{change['Previous value']:g} to {change['Current value']:g} "
                f"{change['Unit']} (absolute change "
                f"{change['Absolute change']:+g} {change['Unit']}; "
                f"percentage change {percent})."
            )
    else:
        st.info(
            "No unambiguous year-to-year changes can be calculated for this "
            "metric."
        )
    st.caption(
        "These are changes in reported measurements only. A decrease by "
        "itself does not establish that a company is environmentally "
        "responsible. Blank years indicate no comparable source observation."
    )


def dashboard_workflow_status_rows():
    def combined_status(state, stage_names):
        statuses = [state["stages"][name]["status"] for name in stage_names]
        for status in ("FAILED", "RUNNING", "WAITING", "PARTIAL"):
            if status in statuses:
                return status
        if all(status == "NOT AVAILABLE" for status in statuses):
            return "NOT AVAILABLE"
        if "NOT AVAILABLE" in statuses:
            return "PARTIAL"
        return "COMPLETED"

    company = st.session_state["company_progress_status"]
    company_stage_groups = [
        ("Web Search", ("Web Search",)),
        ("Web Data Collection", ("Web Data Collection",)),
        ("Environmental Claims", ("Environmental Claim Extraction",)),
        ("Evidence Analysis", ("Evidence Matching", "NLI Analysis")),
        (
            "Risk Assessment & Explanation",
            ("Risk Prediction", "Risk Explanation"),
        ),
        ("Company Environmental Timeline", ("Evidence Timeline Analysis",)),
        ("Source Documents & Database Status", ("Web Data Collection", "Database Saving")),
    ]
    company_rows = [
        {
            "Stage": label,
            "Status": combined_status(company, stage_names),
        }
        for label, stage_names in company_stage_groups
    ]

    return company_rows


def render_dashboard_overview(progress_panel=None):
    st.markdown(
        "<div style='text-align: center;'><h1>🌱 ESG Greenwashing Detection Using Web-Based Evidence Analysis</h1></div>",
        unsafe_allow_html=True,
    )
    company_state = st.session_state["company_progress_status"]
    company_progress = calculate_progress(company_state)
    company_state["progress"] = company_progress

    st.markdown("### COMPANY ESG ANALYSIS PROGRESS")
    st.progress(
        company_progress,
        text=f"Company ESG Analysis progress: {company_progress:.0%}",
    )
    company_rows = dashboard_workflow_status_rows()
    display_dataframe(
        pd.DataFrame(
            company_rows,
            columns=["Stage", "Status"],
        ),
        width="stretch",
        hide_index=True,
    )

def build_risk_distribution_figure(features):
    """Build the experimental risk chart, excluding unpredicted claims."""

    risk_categories = [
        item.get("demonstration_risk_category", "NOT_PREDICTED")
        for item in features or []
        if isinstance(item, dict)
        and item.get("demonstration_risk_category")
        in {"LOW", "MEDIUM", "HIGH"}
    ]
    if not risk_categories:
        return None

    distribution = (
        pd.Series(risk_categories)
        .value_counts()
        .reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        .rename_axis("Experimental review category")
        .reset_index(name="Claims")
    )
    figure = px.bar(
        distribution,
        x="Experimental review category",
        y="Claims",
        color="Experimental review category",
        color_discrete_map={
            "LOW": "#25a879",
            "MEDIUM": "#f2ad48",
            "HIGH": "#ef6868",
        },
    )
    figure.update_layout(
        showlegend=False,
        xaxis_title="",
        yaxis_title="Claims",
    )
    return figure


def render_company_analysis_page(progress_panel=None):
    st.markdown("# COMPANY ESG ANALYSIS")
    company_progress_panel = st.empty()
    render_project_progress(company_progress_panel)
    with st.container(border=True):
        st.caption("Analyse a company's environmental claims and evidence.")
        run_company_search(company_progress_panel)
    if st.session_state.get("company_progress_debug_error"):
        st.error(
            "Company analysis failed: "
            + redact_error_details(
                st.session_state["company_progress_debug_error"]
            )
        )

    analysis = active_analysis()
    rows = []
    company_name = "Company"
    if analysis:
        rows, company_name, _ = dashboard_claim_rows(analysis)
        for stage, message in analysis.get("errors", {}).items():
            st.error(f"{stage} failed: {redact_error_details(message)}")
        if analysis.get("risk_error"):
            st.error(
                "Risk prediction failed: "
                f"{redact_error_details(analysis['risk_error'])}"
            )
        if analysis.get("persistence_error"):
            st.error(
                "Analysis persistence failed: "
                f"{redact_error_details(analysis['persistence_error'])}"
            )
        if analysis.get("database_error"):
            st.error(
                "Risk history could not be saved: "
                f"{redact_error_details(analysis['database_error'])}"
            )
        if analysis.get("timeline_database_error"):
            st.error(
                "Historical timeline could not be saved: "
                f"{redact_error_details(analysis['timeline_database_error'])}"
            )
        evidence_count = sum(
            len(result.get("matches") or [])
            for result in (analysis.get("matching_results") or [])
        )
        risk_level = next(
            (
                feature.get("demonstration_risk_category")
                for feature in (analysis.get("features") or [])
                if feature.get("demonstration_risk_category")
                and feature.get("demonstration_risk_category") != "NOT_PREDICTED"
            ),
            "Unavailable",
        )
        metric_columns = st.columns(3)
        metric_columns[0].metric(
            "Company claim candidates",
            len(analysis.get("company_claims", [])),
        )
        metric_columns[1].metric("Evidence", evidence_count)
        metric_columns[2].metric("Risk Level", risk_level)
        st.caption(
            "Company-source statements are deduplicated candidates, "
            "not independently verified claims. Risk Level is an "
            "experimental heuristic signal, not proof of greenwashing."
        )

    st.markdown(
        '<div class="section-heading">ENVIRONMENTAL CLAIMS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Company statements with source information")
    if not analysis:
        st.info("Search for a company to collect its claims and evidence.")
    elif rows:
        claim_evidence_rows = [
            {
                "Company claim": row["claim"],
                "External evidence candidate": (
                    row["evidence"] or "No relevant match"
                ),
                "Evidence status": row["evidence_status"].replace("_", " ").title(),
                "Evidence source": row["source_title"],
                "Source URL": row["source_url"],
            }
            for row in rows
        ]
        display_dataframe(
            pd.DataFrame(claim_evidence_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Company claim": st.column_config.TextColumn(width="large"),
                "External evidence candidate": st.column_config.TextColumn(
                    width="large"
                ),
                "Evidence source": st.column_config.TextColumn(width="small"),
                "Source URL": link_column(),
            },
        )
    else:
        accessible_company_source = any(
            document.get("status") == "success"
            and str(document.get("text") or "").strip()
            and identify_source_role(document) == "company_source"
            for document in analysis.get("collection", {}).get(
                "documents",
                [],
            )
        )
        if accessible_company_source:
            st.info(
                "No company-source environmental claims were extracted "
                "from the accessible official company documents."
            )
        else:
            st.info(
                "No accessible official company-source documents were "
                "available, so company claims could not be extracted. "
                "External statements remain evidence candidates and are "
                "not relabeled as company claims."
            )

    st.markdown(
        '<div class="section-heading">EVIDENCE ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Supporting evidence | Contradicting evidence")
    if not analysis:
        st.info("Evidence analysis becomes available after a company analysis.")
    else:
        contradiction_count = sum(
            row["nli"].casefold() == "contradiction" for row in rows
        )
        unmatched_count = sum(not row["evidence"] for row in rows)
        categories = [
            item.get("demonstration_risk_category", "NOT_PREDICTED")
            for item in analysis.get("features", [])
        ]
        category_columns = st.columns(3)
        category_columns[0].metric(
            "Experimental high-risk flags",
            categories.count("HIGH"),
        )
        category_columns[1].metric(
            "NLI contradiction flags",
            contradiction_count,
        )
        category_columns[2].metric(
            "Claims without a match",
            unmatched_count,
        )
        st.caption(
            "These automated review signals are experimental and require "
            "human verification. Neither missing evidence nor a worsening metric establishes greenwashing."
        )
        if analysis.get("risk_error"):
            st.error(
                "Risk prediction failed: "
                f"{redact_error_details(analysis['risk_error'])}"
            )

    st.markdown(
        '<div class="section-heading">RISK ASSESSMENT & EXPLANATION</div>',
        unsafe_allow_html=True,
    )
    st.caption("Experimental heuristic review category | Why a rule was applied")
    if not analysis:
        st.info("Risk review becomes available after a company analysis.")
    else:
        if analysis.get("risk_error"):
            st.error(
                "Risk prediction failed: "
                f"{redact_error_details(analysis['risk_error'])}"
            )
        elif analysis.get("features"):
            risk_rows = [
                item.get("demonstration_risk_category")
                for item in analysis.get("features", [])
                if item.get("demonstration_risk_category")
                and item.get("demonstration_risk_category") != "NOT_PREDICTED"
            ]
            if risk_rows:
                st.success(f"Experimental risk prediction: {max(risk_rows, key=lambda value: { 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 }.get(value, 0))}")
            else:
                st.info("Experimental risk prediction: Unavailable")
            if analysis.get("risk_explanations"):
                st.markdown("**Risk explanation**")
                for explanation in analysis["risk_explanations"]:
                    st.write(
                        f"- {explanation.get('claim', 'This statement')}: "
                        f"{explanation.get('explanation', 'No additional explanation.')}"
                    )
            else:
                st.info("No explanation was available for the current prediction.")
        else:
            st.info("Experimental risk prediction: Unavailable")

    st.markdown(
        '<div class="section-heading">COMPANY ENVIRONMENTAL TIMELINE</div>',
        unsafe_allow_html=True,
    )
    st.caption("Historical claims and measurements supported by sources")
    if analysis:
        render_environmental_performance_timeline(analysis)
    else:
        st.info("Run a company analysis to search its collected sources for historical information.")

    st.markdown(
        '<div class="section-heading">SOURCE DOCUMENTS & DATABASE STATUS</div>',
        unsafe_allow_html=True,
    )
    st.caption("Clickable source links | Actual save status")
    if analysis:
        documents = analysis.get("collection", {}).get("documents", [])
        if documents:
            display_dataframe(
                pd.DataFrame([
                    {
                        "Title": document.get("title") or "Untitled",
                        "Status": document.get("status") or "unknown",
                        "Publication date": document.get("publication_date") or "Date unavailable",
                        "Source URL": document.get("final_url") or document.get("source_url") or "",
                    }
                    for document in documents
                ]),
                width="stretch",
                hide_index=True,
                column_config={"Source URL": link_column()},
            )
        else:
            st.info("No source documents are available for the current analysis.")
        db_saved = bool(analysis.get("database_rows_inserted") or analysis.get("timeline_database_rows_inserted") or analysis.get("analysis_path"))
        if db_saved:
            st.success("Analysis results were saved to the SQLite database.")
        else:
            st.info("Analysis results were not saved to the SQLite database.")
    else:
        st.info("Run a company analysis to review source documents and database status.")
    if not analysis:
        st.info(
            "No company analysis has been run in this session. Search for a "
            "company above to display the overall results."
        )
        return

    for stage, message in analysis.get("errors", {}).items():
        st.error(f"{stage} failed: {redact_error_details(message)}")
    if analysis.get("persistence_error"):
        st.error(
            "Analysis persistence failed: "
            f"{redact_error_details(analysis['persistence_error'])}"
        )
    if analysis.get("database_error"):
        st.error(
            "Risk history could not be saved: "
            f"{redact_error_details(analysis['database_error'])}"
        )
    if analysis.get("timeline_database_error"):
        st.error(
            "Historical timeline could not be saved: "
            f"{redact_error_details(analysis['timeline_database_error'])}"
        )

    document_count = analysis.get("relevant_document_count", 0)
    match_count = sum(bool(row["evidence"]) for row in rows)
    metric_columns = st.columns(3)
    for column, (label, value) in zip(
        metric_columns,
        [
            ("Documents reviewed", document_count),
            ("Candidate claims", len(rows)),
            ("Evidence matches", match_count),
        ],
    ):
        column.metric(label, value)

    st.markdown("**Risk distribution**")
    figure = build_risk_distribution_figure(
        analysis.get("features", [])
    )
    if figure is not None:
        st.plotly_chart(
            chart_theme(figure, height=260),
            width="stretch",
            config={"displayModeBar": False},
        )
    else:
        st.info("No risk distribution is available from the current evidence.")

    st.markdown("**Detailed claim analysis**")
    claim_table = [
        {
            "Claim": row["claim"],
            "Evidence candidate": row["evidence"] or "No relevant match",
            "Evidence status": row["evidence_status"].replace("_", " ").title(),
            "Evidence source": row["source_title"],
            "Similarity": display_similarity(row["similarity"]),
            "NLI review": row["nli"].title(),
            "Risk": row["risk_category"].replace("_", " ").title(),
            "Publication year": row["publication_year"] or "Unavailable",
            "Source URL": row["source_url"],
        }
        for row in rows
    ]
    if claim_table:
        claim_table_frame = pd.DataFrame(claim_table)
        claim_table_frame["Similarity"] = pd.array(
            claim_table_frame["Similarity"],
            dtype="Float64",
        )
        display_dataframe(
            claim_table_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Claim": st.column_config.TextColumn(width="large"),
                "Evidence candidate": st.column_config.TextColumn(width="large"),
                "Evidence source": st.column_config.TextColumn(width="small"),
                "Source URL": link_column(),
                "Similarity": st.column_config.NumberColumn(
                    format="%.3f",
                ),
            },
        )
    else:
        st.info("No detailed company claim rows are available.")

    categories = [
        row["nli"].title()
        for row in rows
        if row["nli"].upper() in {"ENTAILMENT", "CONTRADICTION", "NEUTRAL"}
    ]
    if categories:
        st.markdown("**Evidence classification signals**")
        category_counts = (
            pd.Series(categories)
            .value_counts()
            .rename_axis("Review")
            .reset_index(name="Claims")
        )
        figure = px.pie(
            category_counts,
            names="Review",
            values="Claims",
            hole=0.62,
            color="Review",
            color_discrete_map={
                "Entailment": "#25a879",
                "Contradiction": "#ef6868",
                "Neutral": "#f2ad48",
            },
        )
        chart_theme(figure, height=220)
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False},
        )

    similarities = [
        {
            "Claim": row["claim"][:76] + ("…" if len(row["claim"]) > 76 else ""),
            "Similarity": display_similarity(row["similarity"]),
        }
        for row in rows
        if display_similarity(row["similarity"]) is not None
    ]
    if similarities:
        st.markdown("**Claim similarity scores**")
        similarity_frame = pd.DataFrame(similarities).sort_values("Similarity")
        figure = px.bar(
            similarity_frame,
            x="Similarity",
            y="Claim",
            orientation="h",
            color="Similarity",
            color_continuous_scale=["#b9e6d4", "#16895f"],
            range_x=[0, 1],
        )
        chart_theme(figure, height=225)
        figure.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Text similarity (0–1)",
            yaxis_title="",
        )
        st.plotly_chart(
            figure,
            width="stretch",
            config={"displayModeBar": False},
        )

    documents = analysis.get("collection", {}).get("documents", [])
    official_source = next(
        (
            document.get("final_url") or document.get("source_url")
            for document in documents
            if (
                "official company source"
                in str(document.get("source_type", "")).casefold()
                and (document.get("final_url") or document.get("source_url"))
            )
        ),
        None,
    )
    if official_source:
        st.link_button(f"Open {company_name} company source", official_source)
    with st.expander("Collection and source details"):
        show_collection_summary(analysis)


st.session_state.setdefault("active_analysis", None)
st.session_state.setdefault("company_search_input", "")
st.session_state.setdefault(
    "company_progress_status",
    create_progress_state(COMPANY_STAGES),
)
sync_background_workflow_state()

with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:.55rem; margin:.25rem 0 1rem;">
            <span style="font-size:1.4rem">🌿</span>
            <div><div class="brand-lockup">ESG Greenwashing<br>Detection</div>
            <div class="brand-subtitle">Evidence review workspace</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    legacy_aliases = {
        "Company Analysis": "Company ESG Analysis",
        "Dashboard": "Dashboard Overview",
    }
    current_page = st.session_state.get("dashboard_navigation")
    if current_page is None:
        current_page = "Dashboard Overview"
    current_page = legacy_aliases.get(current_page, current_page)
    if current_page not in NAVIGATION:
        current_page = "Dashboard Overview"
    st.session_state["dashboard_navigation"] = current_page
    page = st.radio(
        "Pages",
        NAVIGATION,
        key="dashboard_navigation",
        index=NAVIGATION.index(current_page),
        label_visibility="visible",
    )
    st.divider()

@st.fragment(run_every=WORKFLOW_UI_REFRESH_SECONDS)
def render_selected_page(selected_page):
    sync_background_workflow_state()
    progress_panel = st.empty()

    if selected_page == "Dashboard Overview":
        render_dashboard_overview(progress_panel)
    elif selected_page == "Company ESG Analysis":
        render_company_analysis_page(progress_panel)


render_selected_page(page)
