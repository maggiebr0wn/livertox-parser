# LiverTox Extraction Pipeline

A pipeline to extract structured hepatotoxicity information from LiverTox XML documents using a combination of deterministic (regex/XML parsing) and LLM-based extraction.

HTML report here: [report.html](https://github.com/maggiebr0wn/livertox-parser/blob/main/report.html)

## Overview

This pipeline processes 85 LiverTox XML drug files and extracts 16 structured fields per drug, including DILI likelihood score, injury pattern, enzyme elevation rates, dose information, and more. It uses a **two-stage extraction strategy**:

1. **Deterministic extraction** — Regex and XML table parsing for fields with consistent formatting (DILI score, R-ratio, injury pattern, enzyme elevation fraction)
2. **LLM extraction** — Anthropic Claude Sonnet for fields requiring natural language understanding (risk factors, onset time, dose information, immune-mediated classification)

Results are merged (deterministic takes priority), validated, and evaluated against a manually annotated gold standard of 15 drugs.

## Results

- **92.7% accuracy** against gold standard (139/150 field comparisons correct)
- **100% validation pass rate** across all 85 drugs
- **0% hallucination** on 11 fictional/malformed drug entries
- DILI score and injury pattern extraction: **100% accurate**

## Project Structure

```
livertox-parser/
├── src/livertox_extraction/
│   ├── models.py          # Data models (dataclasses) and validation
│   ├── parser.py          # XML parsing → clean text sections
│   ├── deterministic.py   # Regex-based field extraction
│   ├── llm_extractor.py   # Anthropic Claude extraction
│   ├── pipeline.py        # Orchestrator: parse → extract → merge → save
│   ├── evaluate.py        # Gold standard evaluation framework
│   ├── report.py          # Interactive HTML report generator
│   └── cli.py             # Command-line interface
├── data/                  # Input XML files (85 drugs)
├── gold_standard/
│   └── annotations.json   # Manual annotations for 15 drugs
├── notebooks/
│   ├── 01_xml_exploration.ipynb   # XML structure analysis & prototyping
│   └── 02_pipeline_testing.ipynb  # Pipeline testing & evaluation
├── outputs/
│   ├── extractions/       # Per-drug JSON files
│   └── report.html        # Interactive HTML report
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Create conda environment
conda create -n livertox python=3.11 -y
conda activate livertox

# Install the package
pip install -e ".[dev]"
```

Requires an Anthropic API key in `keys.txt`:
```
ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Command Line

```bash
# Full pipeline (deterministic + LLM)
livertox-extract --input data/ --output outputs/extractions --keys keys.txt

# Deterministic only (no API calls)
livertox-extract --input data/ --output outputs/extractions --skip-llm

# With evaluation and HTML report
livertox-extract --input data/ --output outputs/extractions --keys keys.txt \
    --gold gold_standard/annotations.json --report outputs/report.html

# Process specific drugs
livertox-extract --input data/ --output outputs/extractions --drugs Zileuton Itraconazole
```

### Python API

```python
from livertox_extraction.pipeline import run_pipeline

results = run_pipeline(
    xml_dir="data/",
    output_dir="outputs/extractions",
    keys_file="keys.txt",
)
```

## Architecture

### Extraction Fields (16 total)

| Field | Type | Method | Priority |
|-------|------|--------|----------|
| drug_name | str | Deterministic | Required |
| dili_likelihood_score | A/B/C/D/E/E*/X | Deterministic | 1 |
| injury_pattern | enum | Both | 2 |
| fraction_patients_with_enzyme_elevation | float [0,1] | Both | 2 |
| fraction_patients_with_dili | float [0,1] | LLM | 2 |
| is_immune_mediated | bool | LLM | 2 |
| risk_factors | list[dict] | LLM | 3 |
| safe_dose | dict | LLM | 3 |
| toxic_dose | dict | LLM | 3 |
| onset_time | dict | LLM | 3 |
| peak_alt | float (xULN) | Both | 3 |
| peak_alp | float (xULN) | Both | 3 |
| r_ratio | float | Deterministic | 3 |
| bilirubin_peak | float | LLM | 3 |
| regulatory_status | enum | Deterministic | 3 |
| drug_class | str | Deterministic | Bonus |

### Merge Strategy

For fields extracted by both methods, deterministic results take priority when non-null. This is because regex extraction is perfectly reliable for structured patterns (100% accuracy on DILI score, R-ratio), while the LLM handles semantic fields better (risk factors, onset time, dose interpretation).

### Evaluation Framework

Three evaluation strategies:

1. **Gold standard comparison** — 15 manually annotated drugs (stratified: Cat A×3, B×3, C×2, D×2, E×2, Fictional×3). Per-field accuracy with tolerance for numeric fields.
2. **Completeness analysis** — Field coverage across all 85 drugs. Expected pattern: Category E drugs have many nulls, Category A drugs are mostly complete.
3. **Validation** — Schema constraint checking (fractions in [0,1], valid enum values, non-negative numerics).

## Design Decisions

**Dataclasses over Pydantic** — Used Python's built-in dataclasses with separate validation functions for simplicity and readability. Validation is explicit via `validate_extraction()` rather than automatic on construction.

**Deterministic-first extraction** — Maximizes reliability and reduces API cost. The DILI likelihood score regex is 100% accurate across all drugs, making LLM extraction unnecessary for this field.

**Conservative LLM prompting** — The prompt instructs Claude to return null rather than guess. This results in some false negatives but avoids dangerous false positives in clinical data.

**Fictional drug handling** — The dataset contains ~11 fictional drugs with empty or malformed XML. The pipeline detects these (0 extracted fields) without crashing or hallucinating.

## Known Limitations

- **Peak ALT/ALP coverage is low** (~5%) — the deterministic regex is too strict for varied prose formats, and the LLM doesn't consistently extract these from case report text
- **is_immune_mediated has false positives** (80% accuracy) — the LLM sometimes over-detects immune features from ambiguous text
- **Tolmetin enzyme fraction ambiguity** — text mentions both 5% (transient) and <1% (marked) elevations; regex picks the wrong one
- **Enzyme elevation vs DILI fraction confusion** — despite explicit prompting, the LLM occasionally extracts enzyme rates as DILI rates
