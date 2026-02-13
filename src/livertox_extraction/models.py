"""
Data models for LiverTox extraction pipeline.

Uses dataclasses for data structures and simple functions for validation.
All extraction results are stored as DrugExtraction dataclass instances.
Parsed XML sections are stored as ParsedSections dataclass instances.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# Allowed values (used for validation)
# ---------------------------------------------------------------------------

VALID_DILI_SCORES = ["A", "B", "C", "D", "E", "E*", "X"]

VALID_INJURY_PATTERNS = [
    "hepatocellular", "cholestatic", "mixed",
    "intrinsic", "idiosyncratic", "unclear",
]

VALID_REGULATORY_STATUSES = [
    "approved", "investigational", "withdrawn", "unregulated",
]


# ---------------------------------------------------------------------------
# Parsed XML sections (output of parser.py)
# ---------------------------------------------------------------------------

@dataclass
class ParsedSections:
    """Clean text extracted from each XML section.

    All fields are optional (None) except drug_name, because
    section presence varies by drug. Category E drugs may only
    have Introduction, Background, and Hepatotoxicity.
    """
    drug_name: str
    introduction: Optional[str] = None
    background: Optional[str] = None
    hepatotoxicity: Optional[str] = None
    mechanism: Optional[str] = None
    outcome_and_management: Optional[str] = None
    case_reports_text: Optional[List[str]] = None
    case_report_key_points: Optional[List[Dict]] = None
    product_information: Optional[str] = None
    drug_class: Optional[str] = None


# ---------------------------------------------------------------------------
# Main extraction result
# ---------------------------------------------------------------------------

@dataclass
class DrugExtraction:
    """Complete extraction result for one drug.

    All fields are optional (None) except drug_name,
    matching the spec: 'all fields are optional except drug_name'.
    is_immune_mediated defaults to False per the spec.

    Nested fields use plain dicts:
      - risk_factors: [{"factor": str, "supporting_quote": str}, ...]
      - safe_dose / toxic_dose: {"value": float, "unit": str, "frequency": str or None}
      - onset_time: {"min": int or None, "max": int or None, "typical": int or None, "unit": str}
    """
    drug_name: str

    # Priority 1
    dili_likelihood_score: Optional[str] = None       # One of VALID_DILI_SCORES

    # Priority 2
    injury_pattern: Optional[str] = None              # One of VALID_INJURY_PATTERNS
    fraction_patients_with_enzyme_elevation: Optional[float] = None  # 0.0 to 1.0
    fraction_patients_with_dili: Optional[float] = None              # 0.0 to 1.0
    is_immune_mediated: bool = False

    # Priority 3
    risk_factors: Optional[List[Dict]] = None         # [{"factor": str, "supporting_quote": str}]
    safe_dose: Optional[Dict] = None                  # {"value": float, "unit": str, "frequency": str?}
    toxic_dose: Optional[Dict] = None                 # {"value": float, "unit": str, "frequency": str?}
    onset_time: Optional[Dict] = None                 # {"min": int?, "max": int?, "typical": int?, "unit": str}
    peak_alt: Optional[float] = None                  # Multiples of ULN
    peak_alp: Optional[float] = None                  # Multiples of ULN
    r_ratio: Optional[float] = None                   # >5 hepatocellular, <2 cholestatic
    bilirubin_peak: Optional[float] = None            # mg/dL or multiples of ULN
    regulatory_status: Optional[str] = None           # One of VALID_REGULATORY_STATUSES

    def to_dict(self):
        """Convert to a plain dictionary (for JSON export)."""
        return asdict(self)

    def to_json(self, indent=2):
        """Convert to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_extraction(extraction):
    """Check that a DrugExtraction has valid field values.

    Args:
        extraction: a DrugExtraction instance

    Returns:
        list of error strings. Empty list means all valid.
    """
    errors = []

    # drug_name is required
    if not extraction.drug_name:
        errors.append("drug_name is required and cannot be empty")

    # DILI score must be in the allowed set
    if extraction.dili_likelihood_score is not None:
        if extraction.dili_likelihood_score not in VALID_DILI_SCORES:
            errors.append(
                f"dili_likelihood_score '{extraction.dili_likelihood_score}' "
                f"not in {VALID_DILI_SCORES}"
            )

    # Injury pattern must be in the allowed set
    if extraction.injury_pattern is not None:
        if extraction.injury_pattern not in VALID_INJURY_PATTERNS:
            errors.append(
                f"injury_pattern '{extraction.injury_pattern}' "
                f"not in {VALID_INJURY_PATTERNS}"
            )

    # Regulatory status must be in the allowed set
    if extraction.regulatory_status is not None:
        if extraction.regulatory_status not in VALID_REGULATORY_STATUSES:
            errors.append(
                f"regulatory_status '{extraction.regulatory_status}' "
                f"not in {VALID_REGULATORY_STATUSES}"
            )

    # Fractions must be between 0.0 and 1.0
    for field_name in ["fraction_patients_with_enzyme_elevation", "fraction_patients_with_dili"]:
        value = getattr(extraction, field_name)
        if value is not None:
            if not (0.0 <= value <= 1.0):
                errors.append(f"{field_name} must be between 0.0 and 1.0, got {value}")

    # Numeric fields must be non-negative
    for field_name in ["peak_alt", "peak_alp", "r_ratio", "bilirubin_peak"]:
        value = getattr(extraction, field_name)
        if value is not None:
            if value < 0:
                errors.append(f"{field_name} must be non-negative, got {value}")

    # is_immune_mediated must be a boolean
    if not isinstance(extraction.is_immune_mediated, bool):
        errors.append(f"is_immune_mediated must be a boolean, got {type(extraction.is_immune_mediated)}")

    return errors


# ---------------------------------------------------------------------------
# Helper: create DrugExtraction from a dict (e.g., from LLM JSON output)
# ---------------------------------------------------------------------------

def extraction_from_dict(data):
    """Create a DrugExtraction from a dictionary.

    Handles the fact that JSON keys from the LLM may include
    extra keys or have slightly different types (e.g., int vs float).

    Args:
        data: dictionary with extraction fields

    Returns:
        DrugExtraction instance
    """
    # Only keep keys that are actual DrugExtraction fields
    valid_keys = {f.name for f in DrugExtraction.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in valid_keys}

    # Convert numeric strings to floats where needed
    float_fields = [
        "fraction_patients_with_enzyme_elevation",
        "fraction_patients_with_dili",
        "peak_alt", "peak_alp", "r_ratio", "bilirubin_peak",
    ]
    for f in float_fields:
        if f in filtered and filtered[f] is not None:
            try:
                filtered[f] = float(filtered[f])
            except (ValueError, TypeError):
                filtered[f] = None

    return DrugExtraction(**filtered)
