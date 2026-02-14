"""
Pipeline orchestrator for LiverTox extraction.

Runs the full extraction pipeline:
  1. Parse XML files into clean text sections
  2. Run deterministic (regex) extraction
  3. Run LLM extraction (optional, can be skipped)
  4. Merge results (deterministic takes priority)
  5. Validate merged results
  6. Save per-drug JSON files

Can be run on all drugs or a subset.
"""

import json
import time
from pathlib import Path
from typing import List, Optional, Dict

from livertox_extraction.models import DrugExtraction, validate_extraction
from livertox_extraction.parser import parse_xml_safe
from livertox_extraction.deterministic import extract_deterministic
from livertox_extraction.llm_extractor import create_client, extract_with_llm


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_extractions(deterministic, llm):
    """Merge deterministic and LLM extractions into a single result.

    Strategy: deterministic results take priority when they have a
    non-None value. LLM results fill in the gaps. This is because
    regex extraction is more reliable for the fields it can handle
    (DILI score, R-ratio, etc.), while the LLM handles nuanced fields
    (risk factors, onset time, etc.).

    Args:
        deterministic: DrugExtraction from deterministic extraction
        llm: DrugExtraction from LLM extraction (can be None)

    Returns:
        merged DrugExtraction
    """
    if llm is None:
        return deterministic

    # Start with deterministic as the base
    merged_dict = deterministic.to_dict()

    # Fill in None fields from LLM
    llm_dict = llm.to_dict()
    for key, value in llm_dict.items():
        # Skip drug_name (always from deterministic)
        if key == "drug_name":
            continue
        # Only use LLM value if deterministic value is None
        if merged_dict.get(key) is None and value is not None:
            merged_dict[key] = value

    # Special case: is_immune_mediated defaults to False in both,
    # so use LLM value if it's True (LLM detected immune features)
    if llm_dict.get("is_immune_mediated") is True:
        merged_dict["is_immune_mediated"] = True

    return DrugExtraction(**merged_dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    xml_dir,
    output_dir,
    keys_file="keys.txt",
    skip_llm=False,
    drug_names=None,
):
    """Run the full extraction pipeline on LiverTox XML files.

    Args:
        xml_dir: path to directory containing XML files
        output_dir: path to directory for output JSON files
        keys_file: path to API keys file
        skip_llm: if True, only run deterministic extraction (no API calls)
        drug_names: optional list of drug names to process (default: all)

    Returns:
        dict mapping drug names to their final DrugExtraction results
    """
    xml_dir = Path(xml_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find XML files
    xml_files = sorted(xml_dir.glob("*.xml"))
    if drug_names:
        drug_names_set = set(drug_names)
        xml_files = [f for f in xml_files if f.stem in drug_names_set]

    print(f"Found {len(xml_files)} XML files to process")
    print(f"LLM extraction: {'DISABLED' if skip_llm else 'ENABLED'}")
    print()

    # Create LLM client if needed
    client = None
    if not skip_llm:
        try:
            client = create_client(keys_file)
            print("Anthropic client created successfully")
        except Exception as e:
            print(f"WARNING: Could not create Anthropic client: {e}")
            print("Falling back to deterministic-only extraction")
            skip_llm = True
        print()

    # Process each drug
    all_results = {}
    parse_failures = []
    llm_failures = []
    validation_warnings = {}
    start_time = time.time()

    for i, xml_file in enumerate(xml_files):
        drug_name = xml_file.stem
        print(f"[{i+1}/{len(xml_files)}] Processing {drug_name}...")

        # Step 1: Parse XML
        sections = parse_xml_safe(xml_file)
        if sections is None:
            parse_failures.append(drug_name)
            # Save a minimal result for failed parses
            result = DrugExtraction(drug_name=drug_name)
            all_results[drug_name] = result
            save_result(result, output_dir)
            continue

        # Step 2: Deterministic extraction
        det_result = extract_deterministic(sections)

        # Step 3: LLM extraction (if enabled)
        llm_result = None
        if not skip_llm and client is not None:
            llm_result = extract_with_llm(sections, client)
            if llm_result is None:
                llm_failures.append(drug_name)

        # Step 4: Merge
        if llm_result is not None:
            merged = merge_extractions(det_result, llm_result)
        else:
            merged = det_result

        # Step 5: Validate
        errors = validate_extraction(merged)
        if errors:
            validation_warnings[drug_name] = errors
            print(f"  Validation warnings: {errors}")

        # Step 6: Save
        all_results[drug_name] = merged
        save_result(merged, output_dir)

        # Brief status
        fields_filled = sum(
            1 for k, v in merged.to_dict().items()
            if k != "drug_name" and v is not None and v is not False
        )
        print(f"  Done: {fields_filled} fields extracted")

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total drugs: {len(xml_files)}")
    print(f"Successfully parsed: {len(xml_files) - len(parse_failures)}")
    print(f"Parse failures: {len(parse_failures)} {parse_failures}")
    if not skip_llm:
        print(f"LLM failures: {len(llm_failures)} {llm_failures}")
    print(f"Validation warnings: {len(validation_warnings)}")
    for drug, errs in validation_warnings.items():
        print(f"  {drug}: {errs}")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    print(f"Results saved to: {output_dir}")
    print()

    return all_results


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def save_result(extraction, output_dir):
    """Save a DrugExtraction as a JSON file.

    Args:
        extraction: DrugExtraction instance
        output_dir: directory to save to
    """
    output_dir = Path(output_dir)
    output_path = output_dir / f"{extraction.drug_name}.json"
    output_path.write_text(extraction.to_json())


def load_result(filepath):
    """Load a DrugExtraction from a JSON file.

    Args:
        filepath: path to the JSON file

    Returns:
        DrugExtraction instance
    """
    from livertox_extraction.models import extraction_from_dict

    data = json.loads(Path(filepath).read_text())
    return extraction_from_dict(data)


def load_all_results(output_dir):
    """Load all DrugExtraction results from a directory.

    Args:
        output_dir: directory containing per-drug JSON files

    Returns:
        dict mapping drug names to DrugExtraction instances
    """
    results = {}
    for json_file in sorted(Path(output_dir).glob("*.json")):
        result = load_result(json_file)
        results[result.drug_name] = result
    return results
