"""
Deterministic (regex/rule-based) extraction for LiverTox fields.

Extracts fields that can be reliably identified with pattern matching,
without needing an LLM. Each extract_* function takes a ParsedSections
and returns a value or None.
"""

import re
from typing import Optional

from livertox_extraction.models import DrugExtraction, ParsedSections, VALID_INJURY_PATTERNS


# ---------------------------------------------------------------------------
# Individual field extractors
# ---------------------------------------------------------------------------

def extract_dili_score(sections):
    """Extract DILI likelihood score from hepatotoxicity section.

    Matches the pattern: 'Likelihood score: X (description).'
    where X is one of A, B, C, D, E, E*, X.
    This format is 100% consistent across all LiverTox XMLs.

    Args:
        sections: ParsedSections instance

    Returns:
        str like 'A', 'B', 'D', 'E*', 'X', or None if not found
    """
    if not sections.hepatotoxicity:
        return None

    pattern = r"Likelihood score:\s*([A-E]\*?|X)"
    match = re.search(pattern, sections.hepatotoxicity)
    if match:
        return match.group(1)
    return None


def extract_injury_pattern(sections):
    """Extract injury pattern from case report Key Points tables.

    Looks for the 'pattern' field in Key Points, which contains values like:
        'Hepatocellular (R=25)'
        'Cholestatic (R=1.7)'
        'Mixed (R=3.2)'

    If multiple case reports exist, uses the first one.

    Args:
        sections: ParsedSections instance

    Returns:
        str like 'hepatocellular', 'cholestatic', 'mixed', or None
    """
    if not sections.case_report_key_points:
        return None

    for kp in sections.case_report_key_points:
        pattern_text = kp.get("pattern", "")
        if not pattern_text:
            continue

        pattern_lower = pattern_text.lower()
        for valid_pattern in VALID_INJURY_PATTERNS:
            if valid_pattern in pattern_lower:
                return valid_pattern

    return None


def extract_r_ratio(sections):
    """Extract R-ratio from case report Key Points tables.

    Looks for '(R=N)' or '(R = N)' in the pattern field.
    R-ratio classifies injury: >5 hepatocellular, <2 cholestatic, 2-5 mixed.

    Args:
        sections: ParsedSections instance

    Returns:
        float or None
    """
    if not sections.case_report_key_points:
        return None

    for kp in sections.case_report_key_points:
        pattern_text = kp.get("pattern", "")
        if not pattern_text:
            continue

        match = re.search(r"\(R\s*=\s*(\d+\.?\d*)\)", pattern_text)
        if match:
            return float(match.group(1))

    return None


def extract_peak_alt(sections):
    """Extract peak ALT from hepatotoxicity text or case reports.

    Looks for patterns like:
        'ALT 33 times the upper limit of normal'
        'ALT 33 times ULN'
        'ALT 33x ULN'
        'ALT elevations above 3 times the upper limit'

    Args:
        sections: ParsedSections instance

    Returns:
        float (multiples of ULN) or None
    """
    # First check case report key points for explicit values
    if sections.case_reports_text:
        for text in sections.case_reports_text:
            match = re.search(
                r"ALT\s+(\d+\.?\d*)\s*(?:times|x)\s*(?:the\s+)?(?:upper\s+limit|ULN)",
                text, re.IGNORECASE
            )
            if match:
                return float(match.group(1))

    # Then check hepatotoxicity section
    if sections.hepatotoxicity:
        match = re.search(
            r"ALT\s+(\d+\.?\d*)\s*(?:times|x)\s*(?:the\s+)?(?:upper\s+limit|ULN)",
            sections.hepatotoxicity, re.IGNORECASE
        )
        if match:
            return float(match.group(1))

    return None


def extract_peak_alp(sections):
    """Extract peak ALP from case reports or hepatotoxicity text.

    Looks for patterns like:
        'alkaline phosphatase (1.3 x ULN)'
        'ALP 2.5 times ULN'

    Args:
        sections: ParsedSections instance

    Returns:
        float (multiples of ULN) or None
    """
    texts_to_search = []
    if sections.case_reports_text:
        texts_to_search.extend(sections.case_reports_text)
    if sections.hepatotoxicity:
        texts_to_search.append(sections.hepatotoxicity)

    for text in texts_to_search:
        # Pattern: 'alkaline phosphatase (N x ULN)'
        match = re.search(
            r"(?:alkaline\s+phosphatase|ALP)\s*\(?\s*(\d+\.?\d*)\s*(?:x|times)\s*(?:the\s+)?(?:upper\s+limit|ULN)",
            text, re.IGNORECASE
        )
        if match:
            return float(match.group(1))

    return None


def extract_enzyme_elevation_fraction(sections):
    """Extract fraction of patients with enzyme elevations.

    Looks for percentage mentions near ALT/aminotransferase/enzyme keywords
    in the hepatotoxicity section. Converts percentage to fraction (0-1).

    IMPORTANT: This is the enzyme elevation rate (ALT >3x ULN), NOT the
    clinical DILI rate. These are different fields.

    Common patterns:
        'ALT elevations above 3 times... occurred in 1.9% of patients'
        'aminotransferase elevations... in 1% to 5% of patients'

    Args:
        sections: ParsedSections instance

    Returns:
        float (0.0 to 1.0) or None
    """
    if not sections.hepatotoxicity:
        return None

    text = sections.hepatotoxicity

    # Pattern 1: 'N% of patients' near ALT/enzyme/aminotransferase context
    # Look for the percentage that appears closest to enzyme-related keywords
    patterns = [
        # 'ALT elevations... occurred in N% of patients'
        r"(?:ALT|aminotransferase|enzyme)\s+elevations?.*?(\d+\.?\d*)\s*%\s*of\s*(?:patients|recipients)",
        # 'N% of patients... ALT elevations'
        r"(\d+\.?\d*)\s*%\s*of\s*(?:patients|recipients).*?(?:ALT|aminotransferase|enzyme)\s+elevation",
        # 'elevations in N% of patients' (more general)
        r"elevations?.*?(?:in|occurred in)\s+(\d+\.?\d*)\s*%\s*of\s*(?:patients|recipients)",
        # 'up to N% of patients... elevations'
        r"[Uu]p\s+to\s+(\d+\.?\d*)\s*%\s*of\s*(?:patients|recipients).*?(?:elevation|aminotransferase)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            pct = float(match.group(1))
            # Sanity check: percentages should be between 0 and 100
            if 0 < pct <= 100:
                return pct / 100.0

    return None


def extract_regulatory_status(sections):
    """Extract regulatory status from product information.

    Heuristic: if a DailyMed link is present, the drug is FDA-approved.
    DailyMed only lists drugs with FDA-approved labeling.

    Args:
        sections: ParsedSections instance

    Returns:
        str ('approved') or None if can't determine
    """
    if sections.product_information:
        if "dailymed" in sections.product_information.lower():
            return "approved"

    return None


# ---------------------------------------------------------------------------
# Main deterministic extraction function
# ---------------------------------------------------------------------------

def extract_deterministic(sections):
    """Run all deterministic extractors on parsed sections.

    Args:
        sections: ParsedSections instance from parser.py

    Returns:
        DrugExtraction with deterministically extracted fields filled in.
        Fields that couldn't be extracted are left as None.
    """
    return DrugExtraction(
        drug_name=sections.drug_name,
        dili_likelihood_score=extract_dili_score(sections),
        injury_pattern=extract_injury_pattern(sections),
        r_ratio=extract_r_ratio(sections),
        peak_alt=extract_peak_alt(sections),
        peak_alp=extract_peak_alp(sections),
        fraction_patients_with_enzyme_elevation=extract_enzyme_elevation_fraction(sections),
        regulatory_status=extract_regulatory_status(sections),
    )
