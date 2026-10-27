"""Extract source-backed historical environmental statements and metrics."""

import re


YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
NUMBER_PATTERN = r"\d[\d,]*(?:\.\d+)?"
EMISSIONS_VALUE_PATTERN = re.compile(
    rf"(?P<value>{NUMBER_PATTERN})\s*"
    r"(?P<scale>million|billion|thousand)?\s*"
    r"(?P<unit>metric\s+tons?|tonnes?|tons?|ton|t|kg|kt|Mt)"
    r"\s*(?:of\s*)?(?P<co2>CO2e?|CO₂e?|carbon dioxide equivalents?)?"
    r"(?![a-z])"
    r"(?P<denominator>\s*(?:/|per\s+)[\w$€£-]+)?",
    re.IGNORECASE,
)
PERCENT_PATTERN = re.compile(
    rf"(?P<value>{NUMBER_PATTERN})\s*(?:%|percent\b)",
    re.IGNORECASE,
)
ENERGY_VALUE_PATTERN = re.compile(
    rf"(?P<value>{NUMBER_PATTERN})\s*(?P<unit>MWh|GWh|kWh)\b",
    re.IGNORECASE,
)
TARGET_PATTERN = re.compile(
    r"\b(targets?|goals?|aims?|pledges?|commit(?:ment|ments|ted|ting|s)?|plans?|"
    r"planned|will achieve|"
    r"intend(?:s|ed)? to|expects? to)\b",
    re.IGNORECASE,
)
CONTRACT_PATTERN = re.compile(
    r"\b(?:PPA|power purchase agreement|supply agreement|"
    r"off[\s-]?take agreement|contractual\s+supply|"
    r"(?:renewable(?:\s+energy)?|clean(?:\s+energy)?|"
    r"electricity|energy|power)\s+supply\s+contract|"
    r"contracted\s+(?:renewable|clean|electricity|energy|power))\b",
    re.IGNORECASE,
)
FORECAST_PATTERN = re.compile(
    r"\b(?:estimate[sd]?|forecast(?:ed|s|ing)?|project(?:ed|s|ing)?|"
    r"expected|expects? to|anticipat(?:ed|es|ing)|"
    r"model(?:led|ed|ing|ling))\b",
    re.IGNORECASE,
)
REPORTED_OUTCOME_PATTERN = re.compile(
    r"\b(?:reported|reports|disclosed|discloses|actual|measured)\b",
    re.IGNORECASE,
)
SCOPE_PATTERN = re.compile(
    r"\b(scope\s*(?:1|2|3)(?:\s*(?:,|and|&)\s*(?:scope\s*)?[123])*)\b",
    re.IGNORECASE,
)
ENVIRONMENT_PATTERN = re.compile(
    r"\b(environment|emissions?|carbon|greenhouse gas|GHG|climate|"
    r"renewable|sustainab|energy|electricity|solar|net[- ]zero|"
    r"pollution|waste|water|"
    r"CO2e?|CO₂e?)\b",
    re.IGNORECASE,
)
EMISSIONS_CONTEXT_PATTERN = re.compile(
    r"\b(emissions?|carbon|greenhouse gas|GHG|CO2e?|CO₂e?)\b",
    re.IGNORECASE,
)
RENEWABLE_CONTEXT_PATTERN = re.compile(
    r"\b(renewable|clean electricity|clean energy)\b",
    re.IGNORECASE,
)
REDUCTION_CONTEXT_PATTERN = re.compile(
    r"\b(reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|"
    r"declin(?:e|ed|ing)|fall(?:s|en|ing)?|fell|cut|"
    r"lower(?:ed|ing)?|drop(?:ped|ping)?)\b",
    re.IGNORECASE,
)
REDUCTION_VALUE_PATTERN = re.compile(
    rf"(?:\b(?:reduc(?:e|ed|tion|ing)|decreas(?:e|ed|ing)|"
    rf"declin(?:e|ed|ing)|fall(?:s|en|ing)?|fell|cut|"
    rf"lower(?:ed|ing)?|drop(?:ped|ping)?)"
    rf"[^.!?;]{{0,35}}?\b(?P<after_verb>{NUMBER_PATTERN})\s*"
    rf"(?:%|percent\b)"
    rf"|\b(?P<before_noun>{NUMBER_PATTERN})\s*(?:%|percent\b)\s*"
    rf"(?:reduction|decrease|decline|lower|less))",
    re.IGNORECASE,
)
TARGET_YEAR_PATTERN = re.compile(
    r"\b(?:by|before|until|through)\s+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
FISCAL_PERIOD_PATTERN = re.compile(
    r"\b(?:fiscal\s+year|FY)\s*['’]?(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
QUARTER_PATTERN = re.compile(
    r"\bQ(?P<q_number>[1-4])\b"
    r"|\b(?P<word>first|second|third|fourth)\s+quarter\b",
    re.IGNORECASE,
)
BASELINE_YEAR_PATTERN = re.compile(
    r"\b(?:from|since|versus|vs\.?|compared with|relative to|"
    r"against)\s+(?:(?:a|the|our|its|their)\s+)?"
    r"(?P<year_a>(?:19|20)\d{2})\b"
    r"|\bbaseline(?: year)?(?: of)?\s+(?P<year_b>(?:19|20)\d{2})\b",
    re.IGNORECASE,
)
ORGANIZATIONAL_BOUNDARY_PATTERN = re.compile(
    r"\b(?P<boundary>(?:global|worldwide|consolidated|u\.?\s*s\.?|"
    r"united states|canada|europe|united kingdom|uk|australia|india|china)"
    r"\s+(?:company-wide\s+)?"
    r"(?:operations|business|facilities|footprint|boundary|"
    r"occupied\s+properties))\b",
    re.IGNORECASE,
)
METHODOLOGY_PATTERNS = {
    "location_based": re.compile(r"\blocation[- ]based\b", re.IGNORECASE),
    "market_based": re.compile(r"\bmarket[- ]based\b", re.IGNORECASE),
    "ghg_protocol": re.compile(r"\bGHG Protocol\b", re.IGNORECASE),
    "iso_14064_1": re.compile(r"\bISO\s*14064(?:-1)?\b", re.IGNORECASE),
}
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
SCALE_MULTIPLIERS = {
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _sentences(text):
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(str(text or ""))
        if sentence.strip()
    ]


def _explicit_reporting_years(sentence):
    year_matches = list(YEAR_PATTERN.finditer(sentence))
    target_years = {
        match.group("year")
        for match in TARGET_YEAR_PATTERN.finditer(sentence)
    }
    baseline_years = {
        match.group("year_a") or match.group("year_b")
        for match in BASELINE_YEAR_PATTERN.finditer(sentence)
    }
    if len(target_years) > 1:
        return None
    explicit_years = [
        match
        for match in year_matches
        if (
            match.group(0) not in target_years
            and match.group(0) not in baseline_years
        )
    ]
    if len(explicit_years) == 1:
        return explicit_years
    if not explicit_years and target_years:
        return []
    return None


def _performance_modality(text):
    """Classify explicit target, contract, and forecast language conservatively."""

    target_match = TARGET_PATTERN.search(text)
    reduction_match = REDUCTION_VALUE_PATTERN.search(text)
    if target_match and reduction_match and not (
        target_match.start() < reduction_match.end()
        and reduction_match.start() < target_match.end()
    ):
        return "mixed_or_ambiguous"

    modes = set()
    if TARGET_PATTERN.search(text):
        modes.add("target_or_commitment")
    if CONTRACT_PATTERN.search(text):
        modes.add("contractual_supply")
    if FORECAST_PATTERN.search(text):
        modes.add("estimate_or_forecast")
    if len(modes) > 1:
        return "mixed_or_ambiguous"
    if modes:
        return next(iter(modes))
    if REPORTED_OUTCOME_PATTERN.search(text):
        return "reported_outcome"
    return "unspecified_outcome"


def _baseline_years(text):
    return {
        match.group("year_a") or match.group("year_b")
        for match in BASELINE_YEAR_PATTERN.finditer(str(text or ""))
    }


def _reporting_quarters(text):
    quarter_numbers = {
        "first": "1",
        "second": "2",
        "third": "3",
        "fourth": "4",
    }
    return {
        match.group("q_number") or quarter_numbers[match.group("word").casefold()]
        for match in QUARTER_PATTERN.finditer(str(text or ""))
    }


def _organizational_boundaries(text):
    return {
        re.sub(r"\s+", " ", match.group("boundary")).casefold().strip()
        for match in ORGANIZATIONAL_BOUNDARY_PATTERN.finditer(
            str(text or "")
        )
    }


def _methodologies(text):
    return {
        name
        for name, pattern in METHODOLOGY_PATTERNS.items()
        if pattern.search(str(text or ""))
    }


def _nearest_year(year_matches, position):
    if not year_matches:
        return None
    return min(
        year_matches,
        key=lambda match: abs(match.start() - position),
    ).group(0)


def _measurement_records(sentence, year_matches):
    # Future commitments are only treated as targets when the sentence does not
    # also contain an explicit achieved reporting year. If a sentence reports a
    # measurement alongside a future target, the measured value remains a
    # reported result rather than a target forecast.
    target_match = TARGET_PATTERN.search(sentence)
    target_year_match = TARGET_YEAR_PATTERN.search(sentence)
    target_year = (
        int(target_year_match.group("year"))
        if target_year_match
        else None
    )
    target_span = None
    if target_match and target_year_match:
        target_span = (
            min(target_match.start(), target_year_match.start()),
            max(target_match.end(), target_year_match.end()),
        )
    elif target_match and not year_matches:
        target_span = (0, len(sentence))

    def performance_type(position):
        if target_span and target_span[0] <= position <= target_span[1]:
            return "Target"
        return "Reported result"

    scope_match = SCOPE_PATTERN.search(sentence)
    boundary = (
        re.sub(r"\s+", " ", scope_match.group(0)).strip().casefold()
        if scope_match
        else "not stated"
    )
    records = []

    if EMISSIONS_CONTEXT_PATTERN.search(sentence):
        for match in EMISSIONS_VALUE_PATTERN.finditer(sentence):
            unit_text = match.group("unit").casefold()
            if unit_text == "kg":
                multiplier = 0.001
            elif unit_text == "kt":
                multiplier = 1_000
            elif unit_text == "mt":
                multiplier = 1_000_000
            else:
                multiplier = 1
            unit = (
                "t CO2e"
                if match.group("co2")
                else "t reported emissions (gas basis unspecified)"
            )
            scale = (match.group("scale") or "").casefold()
            value = float(match.group("value").replace(",", ""))
            value *= SCALE_MULTIPLIERS.get(scale, 1) * multiplier
            denominator = (match.group("denominator") or "").strip()
            intensity = bool(denominator) or bool(
                re.search(r"\bintensity\b", sentence, re.IGNORECASE)
            )
            if intensity and not denominator:
                unit += " per unit (denominator not stated)"
            elif denominator:
                unit += f" {denominator}"
            metric = (
                "Emissions intensity"
                if intensity
                else "Absolute emissions"
            )
            records.append({
                "metric": metric,
                "metric_key": (
                    "emissions_intensity"
                    if intensity
                    else "absolute_emissions"
                ),
                "value": value,
                "unit": unit,
                "boundary": boundary,
                "performance_type": performance_type(match.start("value")),
                "reporting_year": None,
                "target_year": (
                    target_year
                    if performance_type(match.start("value")) == "Target"
                    else None
                ),
                "statement": sentence,
            })

    if RENEWABLE_CONTEXT_PATTERN.search(sentence):
        is_electricity = bool(
            re.search(r"\belectric(?:ity|al)\b", sentence, re.IGNORECASE)
        )
        metric = (
            "Renewable electricity share"
            if is_electricity
            else "Renewable energy share"
        )
        metric_key = (
            "renewable_electricity_percent"
            if is_electricity
            else "renewable_energy_percent"
        )
        denominator_match = re.search(
            r"\b(?:of|share of)\s+(?:(?:our|its|the company['’]s)\s+)?"
            r"(?:total\s+)?(?P<denominator>electricity|energy)"
            r"(?:\s+(?:consumption|use))?\b",
            sentence,
            re.IGNORECASE,
        )
        renewable_boundary = (
            f"{boundary}; denominator: "
            f"{denominator_match.group('denominator').casefold()}"
            if denominator_match
            else f"{boundary}; denominator not stated"
        )
        for match in PERCENT_PATTERN.finditer(sentence):
            records.append({
                "metric": metric,
                "metric_key": metric_key,
                "value": float(match.group("value").replace(",", "")),
                "unit": "%",
                "boundary": renewable_boundary,
                "performance_type": performance_type(match.start("value")),
                "reporting_year": None,
                "target_year": (
                    target_year
                    if performance_type(match.start("value")) == "Target"
                    else None
                ),
                "statement": sentence,
            })
        for match in ENERGY_VALUE_PATTERN.finditer(sentence):
            records.append({
                "metric": (
                    "Renewable electricity consumption"
                    if is_electricity
                    else "Renewable energy consumption"
                ),
                "metric_key": (
                    "renewable_electricity_consumption"
                    if is_electricity
                    else "renewable_energy_consumption"
                ),
                "value": float(match.group("value").replace(",", "")),
                "unit": match.group("unit"),
                "boundary": renewable_boundary,
                "performance_type": performance_type(match.start("value")),
                "reporting_year": None,
                "target_year": (
                    target_year
                    if performance_type(match.start("value")) == "Target"
                    else None
                ),
                "statement": sentence,
            })

    if (
        EMISSIONS_CONTEXT_PATTERN.search(sentence)
        and REDUCTION_CONTEXT_PATTERN.search(sentence)
    ):
        for match in REDUCTION_VALUE_PATTERN.finditer(sentence):
            records.append({
                "metric": "Emissions reduction",
                "metric_key": "emissions_reduction_percent",
                "value": float(
                    (
                        match.group("after_verb")
                        or match.group("before_noun")
                    ).replace(",", "")
                ),
                "unit": "%",
                "boundary": boundary,
                "performance_type": performance_type(
                    match.start("after_verb")
                    if match.group("after_verb")
                    else match.start("before_noun")
                ),
                "reporting_year": None,
                "target_year": (
                    target_year
                    if performance_type(
                        match.start("after_verb")
                        if match.group("after_verb")
                        else match.start("before_noun")
                    ) == "Target"
                    else None
                ),
                "statement": sentence,
            })

    if not records and ENVIRONMENT_PATTERN.search(sentence):
        target = bool(target_match)
        records.append({
            "metric": "Environmental claim or target",
            "metric_key": "environmental_claim",
            "value": None,
            "unit": "",
            "boundary": boundary,
            "performance_type": "Target" if target else "Reported statement",
            "reporting_year": None,
            "target_year": target_year,
            "statement": sentence,
        })

    for record in records:
        record["reporting_year"] = (
            None
            if record["performance_type"] == "Target"
            else _nearest_year(
                year_matches,
                sentence.find(record["statement"]),
            )
        )

    return records


def compare_measurement_compatibility(claim, evidence, evidence_metadata=None):
    """Compare explicit quantitative context without inferring missing data."""

    evidence_metadata = evidence_metadata or {}
    if evidence_metadata.get("assurance_report"):
        exclusions = " ".join(
            str(item)
            for item in evidence_metadata.get("scope_exclusions", [])
        )
        covered_years = YEAR_PATTERN.findall(
            str(evidence_metadata.get("covered_fiscal_period") or "")
        )
        baseline_years = _baseline_years(claim)
        if (
            re.search(
                r"\bperiods?\s+prior\s+to\b.*\bnot subject to our review\b",
                exclusions,
                re.IGNORECASE,
            )
            and covered_years
            and any(
                int(year) < int(covered_years[-1])
                for year in baseline_years
            )
        ):
            excluded_years = sorted(
                year
                for year in baseline_years
                if int(year) < int(covered_years[-1])
            )
            return (
                "INCONCLUSIVE",
                "The assurance excludes baseline year(s) "
                f"{', '.join(excluded_years)}; its covered period is "
                f"{evidence_metadata['covered_fiscal_period']}."
            )

        if (
            re.search(
                r"\b(?:targets?|goals?|progress against goals)\b.*"
                r"\bnot subject to our review\b",
                exclusions,
                re.IGNORECASE,
            )
            and re.search(
                r"\b(?:target|goal|progress)\b",
                claim,
                re.IGNORECASE,
            )
        ):
            return (
                "INCONCLUSIVE",
                "The claim includes targets, goals, or progress that the "
                "assurance explicitly excludes.",
            )

    claim_modality = _performance_modality(claim)
    evidence_modality = _performance_modality(evidence)

    def measurements(text):
        extracted = []
        quantitative_signal = False
        for sentence in _sentences(text):
            if not ENVIRONMENT_PATTERN.search(sentence):
                continue
            year_matches = _explicit_reporting_years(sentence)
            year_spans = {
                (match.start(), match.end())
                for match in YEAR_PATTERN.finditer(sentence)
            }
            for number in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\b", sentence):
                if (number.start(), number.end()) in year_spans:
                    continue
                if re.search(
                    r"\bscope\s*$",
                    sentence[max(0, number.start() - 12):number.start()],
                    re.IGNORECASE,
                ):
                    continue
                quantitative_signal = True
            records = _measurement_records(
                sentence,
                year_matches if year_matches is not None else [],
            )
            reporting_year = (
                int(year_matches[0].group(0))
                if year_matches and len(year_matches) == 1
                else None
            )
            for record in records:
                extracted.append({
                    **record,
                    "reporting_year": reporting_year,
                })
        return extracted, quantitative_signal

    claim_measurements, claim_quantitative = measurements(claim)
    evidence_measurements, evidence_quantitative = measurements(evidence)
    if not claim_quantitative and not evidence_quantitative:
        return "NOT_QUANTITATIVE", "No explicit quantitative measurement was found."
    if not claim_measurements and not evidence_measurements:
        if claim_quantitative or evidence_quantitative:
            return "INCONCLUSIVE", "A quantitative statement is missing parseable measurement metadata."
        return "NOT_QUANTITATIVE", "No comparable environmental measurement was extracted."
    if not claim_measurements or not evidence_measurements:
        return "INCONCLUSIVE", "A quantitative measurement was not extracted from both statements."

    if "mixed_or_ambiguous" in {claim_modality, evidence_modality}:
        return (
            "INCONCLUSIVE",
            "At least one statement mixes target, contract, or forecast "
            "language; performance modality is ambiguous.",
        )
    if claim_modality != evidence_modality and (
        claim_modality != "unspecified_outcome"
        or evidence_modality != "unspecified_outcome"
    ):
        status = (
            "INCOMPATIBLE"
            if claim_modality != "unspecified_outcome"
            and evidence_modality != "unspecified_outcome"
            else "INCONCLUSIVE"
        )
        return (
            status,
            "Performance modalities differ "
            f"({claim_modality.replace('_', ' ')} versus "
            f"{evidence_modality.replace('_', ' ')}); contractual supply, "
            "targets, forecasts, and reported outcomes are not interchangeable.",
        )

    same_metric_pairs = [
        (claim_record, evidence_record)
        for claim_record in claim_measurements
        for evidence_record in evidence_measurements
        if claim_record["metric_key"] == evidence_record["metric_key"]
    ]
    if not same_metric_pairs:
        return "INCOMPATIBLE", (
            "The statements expose different quantitative metrics."
        )

    missing_metadata = set()
    if "unspecified_outcome" in {claim_modality, evidence_modality}:
        missing_metadata.add("reported-versus-estimated status")
    incompatibilities = []
    compatible_measurement = False
    claim_baselines = _baseline_years(claim)
    evidence_baselines = _baseline_years(evidence)
    baseline_required = any(
        record["metric_key"] == "emissions_reduction_percent"
        for record in (*claim_measurements, *evidence_measurements)
    )
    if claim_baselines or evidence_baselines or baseline_required:
        if len(claim_baselines) > 1 or len(evidence_baselines) > 1:
            missing_metadata.add("baseline year")
        elif not claim_baselines or not evidence_baselines:
            missing_metadata.add("baseline year")
        elif claim_baselines != evidence_baselines:
            incompatibilities.append("baseline years differ")

    claim_organization = _organizational_boundaries(claim)
    evidence_organization = _organizational_boundaries(evidence)
    if len(claim_organization) != 1 or len(evidence_organization) != 1:
        missing_metadata.add("organizational boundary")
    elif claim_organization != evidence_organization:
        incompatibilities.append("organizational boundaries differ")

    claim_methodologies = _methodologies(claim)
    evidence_methodologies = _methodologies(evidence)
    if not claim_methodologies or not evidence_methodologies:
        missing_metadata.add("methodology")
    elif claim_methodologies != evidence_methodologies:
        incompatibilities.append("measurement methodologies differ")

    if bool(FISCAL_PERIOD_PATTERN.search(claim)) != bool(
        FISCAL_PERIOD_PATTERN.search(evidence)
    ):
        missing_metadata.add("fiscal-period basis")

    claim_quarters = _reporting_quarters(claim)
    evidence_quarters = _reporting_quarters(evidence)
    if len(claim_quarters) > 1 or len(evidence_quarters) > 1:
        missing_metadata.add("reporting-period detail")
    elif claim_quarters and evidence_quarters:
        if claim_quarters != evidence_quarters:
            incompatibilities.append("reporting subperiods differ")
    elif claim_quarters or evidence_quarters:
        missing_metadata.add("reporting-period detail")

    for claim_record, evidence_record in same_metric_pairs:
        if (
            claim_record["performance_type"] == "Target"
            or evidence_record["performance_type"] == "Target"
        ):
            missing_metadata.add("performance status")
            continue

        claim_year = claim_record["reporting_year"]
        evidence_year = evidence_record["reporting_year"]
        if claim_year is None or evidence_year is None:
            missing_metadata.add("reporting period")
            continue
        if claim_year != evidence_year:
            incompatibilities.append("reporting years differ")
            continue

        if claim_record["unit"] != evidence_record["unit"]:
            incompatibilities.append("measurement units differ")
            continue

        claim_boundary = claim_record["boundary"]
        evidence_boundary = evidence_record["boundary"]
        if (
            "not stated" in claim_boundary
            or "not stated" in evidence_boundary
        ):
            missing_metadata.add("emissions scope")
            continue
        if claim_boundary != evidence_boundary:
            incompatibilities.append("measurement scopes or boundaries differ")
            continue

        compatible_measurement = True

    if incompatibilities:
        return "INCOMPATIBLE", "; ".join(dict.fromkeys(incompatibilities)) + "."
    if missing_metadata:
        missing = ", ".join(sorted(missing_metadata))
        return (
            "INCONCLUSIVE",
            f"Missing or ambiguous measurement metadata: {missing}.",
        )
    if compatible_measurement:
        return "COMPATIBLE", (
            "Metric, reporting year, unit, stated boundary, organizational "
            "boundary, methodology, and applicable baseline are compatible."
        )
    return (
        "INCONCLUSIVE",
        "Measurement periods, units, metrics, or emissions scopes could not "
        "be reconciled.",
    )


def extract_timeline_records(documents):
    """Extract explicitly dated environmental claims and measurements.

    Only successful collected documents with an explicit year in the relevant
    statement are used. Publication dates are retained as source metadata but
    are not substituted for a reporting year.
    """

    records = []
    seen = set()

    for document in documents or []:
        if (
            document.get("status") != "success"
            or not str(document.get("text") or "").strip()
        ):
            continue

        source_url = document.get("final_url") or document.get("source_url") or ""
        source_name = document.get("title") or document.get("source_domain") or source_url

        for sentence in _sentences(document.get("text", "")):
            if not ENVIRONMENT_PATTERN.search(sentence):
                continue
            year_matches = _explicit_reporting_years(sentence)
            target_match = TARGET_YEAR_PATTERN.search(sentence)
            target_year = (
                int(target_match.group("year"))
                if target_match
                else None
            )
            if year_matches is None or (not year_matches and target_year is None):
                continue

            for record in _measurement_records(sentence, year_matches):
                reporting_year = record["reporting_year"]
                if reporting_year is None and record["target_year"] is None:
                    continue
                record["timeline_year"] = (
                    reporting_year
                    if reporting_year is not None
                    else record["target_year"]
                )
                record["year_basis"] = (
                    "Reporting year"
                    if reporting_year is not None
                    else "Target year; reporting year unavailable"
                )
                item = {
                    **record,
                    "reporting_year": (
                        int(reporting_year)
                        if reporting_year is not None
                        else None
                    ),
                    "timeline_year": int(record["timeline_year"]),
                    "source_name": source_name,
                    "source_publisher": (
                        document.get("publisher")
                        or document.get("source_domain")
                        or ""
                    ),
                    "source_url": source_url,
                    "publication_date": document.get("publication_date"),
                    "collected_at": document.get("collected_at"),
                    "evidence_status": (
                        "Source retrieved; statement not independently verified"
                    ),
                }
                identity = (
                    item["timeline_year"],
                    item["metric_key"],
                    item["value"],
                    item["unit"],
                    item["boundary"],
                    item["source_url"],
                    item["statement"],
                )
                if identity not in seen:
                    seen.add(identity)
                    records.append(item)

    records.sort(
        key=lambda item: (
            item["timeline_year"],
            item["year_basis"],
            item["metric"],
            item["source_name"],
            item["statement"],
        )
    )
    return records


def analyze_timeline(documents):
    """Add strictly comparable prior-year values and changes to records."""

    records = extract_timeline_records(documents)
    grouped = {}
    for record in records:
        record["previous_reported_performance"] = None
        record["change_over_time"] = "Insufficient comparable prior data"
        if (
            record["performance_type"] == "Target"
            or record["value"] is None
            or record["reporting_year"] is None
            or "not stated" in record["boundary"]
            or record["metric_key"] == "emissions_reduction_percent"
            or (
                record["metric_key"] in {
                    "absolute_emissions",
                    "emissions_intensity",
                }
                and record["unit"] != "t CO2e"
            )
        ):
            continue

        series_key = (
            record["metric_key"],
            record["unit"],
            record["boundary"],
        )
        grouped.setdefault(series_key, {}).setdefault(
            record["reporting_year"],
            [],
        ).append(record)

    for year_groups in grouped.values():
        previous = None
        for year in sorted(year_groups):
            year_records = year_groups[year]
            values = {record["value"] for record in year_records}
            if len(values) > 1:
                for record in year_records:
                    record["change_over_time"] = (
                        "Conflicting values reported for this year; "
                        "comparison withheld"
                    )
                continue

            current = year_records[0]
            for record in year_records:
                if previous is not None:
                    record["previous_reported_performance"] = (
                        f"{previous['value']:g} {previous['unit']}"
                    )
                    difference = record["value"] - previous["value"]
                    direction = (
                        "increased"
                        if difference > 0
                        else "decreased"
                        if difference < 0
                        else "was unchanged"
                    )
                    record["change_over_time"] = (
                        f"{direction} by {abs(difference):g} "
                        f"{record['unit']} since {previous['reporting_year']}"
                    )
            previous = current

    return records


def get_comparable_metrics(records):
    """Return metrics with two unambiguous years and stated boundaries."""

    values_by_year = {}
    for record in records:
        if (
            record["performance_type"] == "Target"
            or record["value"] is None
            or record["reporting_year"] is None
            or "not stated" in record["boundary"]
            or record["metric_key"] == "emissions_reduction_percent"
            or (
                record["metric_key"] in {
                    "absolute_emissions",
                }
                and record["unit"] != "t CO2e"
            )
            or (
                record["metric_key"] == "emissions_intensity"
                and (
                    record["unit"] == "t CO2e"
                    or "denominator not stated" in record["unit"]
                )
            )
        ):
            continue
        key = (
            record["metric"],
            record["metric_key"],
            record["unit"],
            record["boundary"],
        )
        values_by_year.setdefault(key, {}).setdefault(
            record["reporting_year"],
            set(),
        ).add(record["value"])

    return [
        {
            "metric": metric,
            "metric_key": metric_key,
            "unit": unit,
            "boundary": boundary,
        }
        for (metric, metric_key, unit, boundary), year_values in sorted(
            values_by_year.items()
        )
        if sum(len(values) == 1 for values in year_values.values()) >= 2
    ]


def build_trend_points(records, metric):
    """Return one unambiguous, chronologically ordered series with gaps."""

    selected = [
        record
        for record in records
        if (
            record["metric"] == metric["metric"]
            and record["metric_key"] == metric["metric_key"]
            and record["unit"] == metric["unit"]
            and record["boundary"] == metric["boundary"]
            and record["performance_type"] != "Target"
            and record["value"] is not None
            and record["reporting_year"] is not None
        )
    ]
    by_year = {}
    for record in selected:
        by_year.setdefault(record["reporting_year"], set()).add(record["value"])

    valid_values = {
        year: next(iter(values))
        for year, values in by_year.items()
        if len(values) == 1
    }
    if len(valid_values) < 2:
        return []

    first_year = min(valid_values)
    last_year = max(valid_values)
    return [
        {
            "Year": year,
            "Value": valid_values.get(year),
            "Data status": (
                "Reported value"
                if year in valid_values
                else "No comparable source observation found"
            ),
            "Source titles": "; ".join(sorted({
                record.get("source_name") or "Untitled source"
                for record in selected
                if record["reporting_year"] == year
                and record["value"] == valid_values.get(year)
            })),
            "Source URLs": "; ".join(sorted({
                record.get("source_url", "")
                for record in selected
                if record["reporting_year"] == year
                and record["value"] == valid_values.get(year)
                and record.get("source_url")
            })),
        }
        for year in range(first_year, last_year + 1)
    ]


def build_timeline_change_rows(records, metric):
    """Return source-backed changes between unambiguous comparable years."""

    comparable = get_comparable_metrics(records)
    if metric not in comparable:
        return []

    selected = [
        record
        for record in records
        if (
            record["metric"] == metric["metric"]
            and record["metric_key"] == metric["metric_key"]
            and record["unit"] == metric["unit"]
            and record["boundary"] == metric["boundary"]
            and record["performance_type"] != "Target"
            and record["value"] is not None
            and record["reporting_year"] is not None
        )
    ]
    values_by_year = {}
    for record in selected:
        values_by_year.setdefault(record["reporting_year"], set()).add(
            record["value"]
        )
    valid_values = {
        year: next(iter(values))
        for year, values in values_by_year.items()
        if len(values) == 1
    }
    years = sorted(valid_values)
    rows = []
    for previous_year, current_year in zip(years, years[1:]):
        previous_value = valid_values[previous_year]
        current_value = valid_values[current_year]
        difference = current_value - previous_value
        percentage_change = (
            difference / abs(previous_value) * 100
            if previous_value != 0
            else None
        )
        rows.append({
            "Previous year": previous_year,
            "Current year": current_year,
            "Previous value": previous_value,
            "Current value": current_value,
            "Absolute change": difference,
            "Percentage change": percentage_change,
            "Direction": (
                "increased"
                if difference > 0
                else "decreased"
                if difference < 0
                else "unchanged"
            ),
            "Unit": metric["unit"],
        })
    return rows


def find_timeline_review_points(records):
    """Find source-backed inconsistencies and overdue renewable targets."""

    points = []
    values_by_year = {}
    for record in records:
        if (
            record["value"] is None
            or record["performance_type"] == "Target"
            or record["reporting_year"] is None
        ):
            continue
        key = (
            record["reporting_year"],
            record["metric_key"],
            record["metric"],
            record["unit"],
            record["boundary"],
            record["performance_type"],
        )
        values_by_year.setdefault(key, []).append(record)

    for (
        year,
        metric_key,
        metric,
        unit,
        boundary,
        performance_type,
    ), year_records in values_by_year.items():
        distinct_values = {record["value"] for record in year_records}
        if len(distinct_values) > 1:
            points.append(
                f"Sources report differing {year} values for "
                f"{metric.casefold()} "
                f"({performance_type.casefold()}, {unit}, "
                f"boundary: {boundary}). Check reporting scope, "
                "method, and source context."
            )

    targets = [
        record for record in records
        if (
            record["performance_type"] == "Target"
            and record["target_year"] is not None
            and record["metric_key"] in {
                "renewable_electricity_percent",
                "renewable_energy_percent",
                "absolute_emissions",
                "emissions_intensity",
            }
            and "not stated" not in record["boundary"]
        )
    ]
    for target in targets:
        later_results = [
            record for record in records
            if (
                record["performance_type"] != "Target"
                and record["reporting_year"] is not None
                and record["metric_key"] == target["metric_key"]
                and record["unit"] == target["unit"]
                and record["boundary"] == target["boundary"]
                and record["reporting_year"] >= target["target_year"]
            )
        ]
        if not later_results:
            continue
        earliest_year = min(
            record["reporting_year"] for record in later_results
        )
        year_results = [
            record for record in later_results
            if record["reporting_year"] == earliest_year
        ]
        result_values = {record["value"] for record in year_results}
        if len(result_values) == 1 and next(iter(result_values)) < target["value"]:
            points.append(
                f"A later reported {target['metric'].casefold()} value "
                f"({next(iter(result_values)):g} {target['unit']} in "
                f"{earliest_year}) is below the earlier "
                f"{target['value']:g} {target['unit']} target for "
                f"{target['target_year']}. Review scope and progress context; "
                "this is an investigation lead, not proof of a missed target."
            )

    return points
