"""
Evaluation framework for the LiverTox extraction pipeline.

Compares pipeline output against gold standard annotations and
computes per-field metrics. Three evaluation strategies:
  1. Gold standard comparison (accuracy per field)
  2. Completeness analysis (what % of drugs have each field)
  3. Validation pass rates
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from livertox_extraction.models import (
    DrugExtraction, extraction_from_dict, validate_extraction,
    VALID_DILI_SCORES, VALID_INJURY_PATTERNS,
)
from livertox_extraction.pipeline import load_all_results


# ---------------------------------------------------------------------------
# Load gold standard
# ---------------------------------------------------------------------------

def load_gold_standard(filepath):
    """Load gold standard annotations from JSON file.

    Args:
        filepath: path to annotations.json

    Returns:
        dict mapping drug names to DrugExtraction instances
    """
    data = json.loads(Path(filepath).read_text())
    annotations = {}
    for entry in data["annotations"]:
        # Remove the _notes field (not part of extraction)
        entry_clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        drug = extraction_from_dict(entry_clean)
        annotations[drug.drug_name] = drug
    return annotations


# ---------------------------------------------------------------------------
# Per-field comparison
# ---------------------------------------------------------------------------

def compare_field(predicted, gold, field_name):
    """Compare a single field between predicted and gold standard.

    Returns a dict with:
        - correct: bool
        - predicted: the predicted value
        - gold: the gold value
        - error_type: 'match', 'mismatch', 'false_positive', 'false_negative', 'both_null'

    Args:
        predicted: predicted DrugExtraction
        gold: gold standard DrugExtraction
        field_name: name of the field to compare

    Returns:
        dict with comparison results
    """
    pred_val = getattr(predicted, field_name)
    gold_val = getattr(gold, field_name)

    result = {
        "field": field_name,
        "predicted": pred_val,
        "gold": gold_val,
    }

    # Both null = correct (nothing to extract, nothing extracted)
    if pred_val is None and gold_val is None:
        result["correct"] = True
        result["error_type"] = "both_null"
    # Both have values - compare them
    elif pred_val is not None and gold_val is not None:
        if field_name in ["fraction_patients_with_enzyme_elevation", "fraction_patients_with_dili"]:
            # For fractions, allow small tolerance
            result["correct"] = abs(pred_val - gold_val) < 0.01
        elif isinstance(pred_val, float) and isinstance(gold_val, float):
            # For other floats, allow small tolerance
            result["correct"] = abs(pred_val - gold_val) < 0.5
        else:
            result["correct"] = pred_val == gold_val
        result["error_type"] = "match" if result["correct"] else "mismatch"
    # Predicted has value but gold is null = false positive
    elif pred_val is not None and gold_val is None:
        result["correct"] = False
        result["error_type"] = "false_positive"
    # Gold has value but predicted is null = false negative
    else:
        result["correct"] = False
        result["error_type"] = "false_negative"

    return result


# ---------------------------------------------------------------------------
# Gold standard evaluation
# ---------------------------------------------------------------------------

# Fields to evaluate (skip drug_name, skip complex nested fields for automated eval)
SIMPLE_FIELDS = [
    "dili_likelihood_score",
    "injury_pattern",
    "fraction_patients_with_enzyme_elevation",
    "fraction_patients_with_dili",
    "is_immune_mediated",
    "r_ratio",
    "peak_alt",
    "peak_alp",
    "bilirubin_peak",
    "regulatory_status",
]


def evaluate_gold_standard(results, gold_standard):
    """Compare pipeline results against gold standard annotations.

    Args:
        results: dict mapping drug names to DrugExtraction (pipeline output)
        gold_standard: dict mapping drug names to DrugExtraction (gold)

    Returns:
        dict with:
            - per_drug: dict of drug_name -> list of field comparisons
            - per_field: dict of field_name -> {accuracy, total, correct, errors}
            - overall_accuracy: float
    """
    per_drug = {}
    per_field = {f: {"correct": 0, "total": 0, "errors": []} for f in SIMPLE_FIELDS}

    for drug_name, gold in gold_standard.items():
        predicted = results.get(drug_name)
        if predicted is None:
            continue

        drug_comparisons = []
        for field in SIMPLE_FIELDS:
            comparison = compare_field(predicted, gold, field)
            comparison["drug"] = drug_name
            drug_comparisons.append(comparison)

            per_field[field]["total"] += 1
            if comparison["correct"]:
                per_field[field]["correct"] += 1
            else:
                per_field[field]["errors"].append(comparison)

        per_drug[drug_name] = drug_comparisons

    # Compute accuracies
    total_correct = 0
    total_comparisons = 0
    for field, stats in per_field.items():
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"]
        else:
            stats["accuracy"] = None
        total_correct += stats["correct"]
        total_comparisons += stats["total"]

    overall_accuracy = total_correct / total_comparisons if total_comparisons > 0 else 0

    return {
        "per_drug": per_drug,
        "per_field": per_field,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_comparisons": total_comparisons,
    }


# ---------------------------------------------------------------------------
# Completeness analysis
# ---------------------------------------------------------------------------

def evaluate_completeness(results):
    """Analyze field completeness across all drugs.

    For each field, compute what percentage of drugs have a non-null value.
    Also categorizes drugs as fictional (0 fields) vs real.

    Args:
        results: dict mapping drug names to DrugExtraction

    Returns:
        dict with:
            - field_coverage: dict of field_name -> {count, total, percentage}
            - drug_field_matrix: dict of drug_name -> dict of field_name -> bool
            - fictional_drugs: list of drug names with 0 extracted fields
    """
    all_fields = [
        "dili_likelihood_score", "injury_pattern",
        "fraction_patients_with_enzyme_elevation", "fraction_patients_with_dili",
        "is_immune_mediated", "risk_factors", "safe_dose", "toxic_dose",
        "onset_time", "peak_alt", "peak_alp", "r_ratio",
        "bilirubin_peak", "regulatory_status",
    ]

    field_coverage = {f: {"count": 0, "total": 0, "percentage": 0} for f in all_fields}
    drug_field_matrix = {}
    fictional_drugs = []

    for drug_name, extraction in results.items():
        d = extraction.to_dict()
        field_present = {}
        fields_filled = 0

        for field in all_fields:
            value = d.get(field)
            # is_immune_mediated: True counts as "present" (False is the default)
            if field == "is_immune_mediated":
                present = value is True
            else:
                present = value is not None
            field_present[field] = present
            field_coverage[field]["total"] += 1
            if present:
                field_coverage[field]["count"] += 1
                fields_filled += 1

        drug_field_matrix[drug_name] = field_present

        if fields_filled == 0:
            fictional_drugs.append(drug_name)

    # Compute percentages
    for field, stats in field_coverage.items():
        if stats["total"] > 0:
            stats["percentage"] = stats["count"] / stats["total"] * 100

    return {
        "field_coverage": field_coverage,
        "drug_field_matrix": drug_field_matrix,
        "fictional_drugs": fictional_drugs,
    }


# ---------------------------------------------------------------------------
# Validation analysis
# ---------------------------------------------------------------------------

def evaluate_validation(results):
    """Run validation on all results and summarize.

    Args:
        results: dict mapping drug names to DrugExtraction

    Returns:
        dict with:
            - pass_rate: float (0-1)
            - total: int
            - passed: int
            - failures: dict of drug_name -> list of error strings
    """
    passed = 0
    total = 0
    failures = {}

    for drug_name, extraction in results.items():
        total += 1
        errors = validate_extraction(extraction)
        if not errors:
            passed += 1
        else:
            failures[drug_name] = errors

    return {
        "pass_rate": passed / total if total > 0 else 0,
        "total": total,
        "passed": passed,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_evaluation(results, gold_standard_path):
    """Run all evaluation strategies and return combined results.

    Args:
        results: dict mapping drug names to DrugExtraction
        gold_standard_path: path to annotations.json

    Returns:
        dict with keys: gold_standard, completeness, validation
    """
    # Load gold standard
    gold = load_gold_standard(gold_standard_path)

    # Run evaluations
    gold_eval = evaluate_gold_standard(results, gold)
    completeness = evaluate_completeness(results)
    validation = evaluate_validation(results)

    return {
        "gold_standard": gold_eval,
        "completeness": completeness,
        "validation": validation,
    }


def print_evaluation_summary(eval_results):
    """Print a human-readable summary of evaluation results.

    Args:
        eval_results: output from run_evaluation()
    """
    gold = eval_results["gold_standard"]
    comp = eval_results["completeness"]
    val = eval_results["validation"]

    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Validation
    print(f"\nValidation: {val['passed']}/{val['total']} passed ({val['pass_rate']:.0%})")
    if val["failures"]:
        for drug, errors in val["failures"].items():
            print(f"  {drug}: {errors}")

    # Gold standard
    print(f"\nGold Standard Accuracy: {gold['overall_accuracy']:.1%}")
    print(f"  ({gold['total_correct']}/{gold['total_comparisons']} correct)")
    print()
    print(f"  {'Field':<45} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
    print("  " + "-" * 70)
    for field, stats in gold["per_field"].items():
        acc = f"{stats['accuracy']:.0%}" if stats['accuracy'] is not None else "N/A"
        print(f"  {field:<45} {acc:>8} {stats['correct']:>8} {stats['total']:>6}")

    # Errors
    print("\n  Errors:")
    for field, stats in gold["per_field"].items():
        for error in stats["errors"]:
            print(f"    {error['drug']}.{field}: predicted={error['predicted']}, "
                  f"gold={error['gold']} ({error['error_type']})")

    # Completeness
    print(f"\nField Coverage (across {len(comp['drug_field_matrix'])} drugs):")
    for field, stats in comp["field_coverage"].items():
        bar = "#" * int(stats["percentage"] / 5) + "-" * (20 - int(stats["percentage"] / 5))
        print(f"  {field:<45} [{bar}] {stats['percentage']:5.1f}%")

    print(f"\nFictional drugs (0 fields): {comp['fictional_drugs']}")
