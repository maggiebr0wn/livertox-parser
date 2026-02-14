"""
LLM-based extraction for LiverTox fields using the Anthropic API.

Uses Claude Sonnet to extract fields that require natural language
understanding, such as risk factors, dose information, and distinguishing
enzyme elevation rates from clinical DILI rates.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import anthropic

from livertox_extraction.models import DrugExtraction, ParsedSections, extraction_from_dict


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------

def load_api_key(keys_file="keys.txt"):
    """Load the Anthropic API key from keys.txt.

    The file format is: ANTHROPIC_API_KEY="sk-ant-..."

    Args:
        keys_file: path to keys.txt

    Returns:
        API key string
    """
    keys_path = Path(keys_file)
    if not keys_path.exists():
        raise FileNotFoundError(f"Keys file not found: {keys_file}")

    text = keys_path.read_text()
    for line in text.strip().split("\n"):
        if line.startswith("ANTHROPIC_API_KEY"):
            # Extract the key from: ANTHROPIC_API_KEY="sk-ant-..."
            key = line.split("=", 1)[1].strip().strip('"')
            return key

    raise ValueError("ANTHROPIC_API_KEY not found in keys file")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a clinical pharmacology expert extracting structured data from LiverTox drug entries.

You will receive text from a LiverTox XML document about a drug's hepatotoxicity. Extract the following fields and return ONLY valid JSON (no markdown, no explanation, no code blocks).

FIELD DEFINITIONS:

1. "injury_pattern" (string or null): The pattern of liver injury. Must be one of: "hepatocellular", "cholestatic", "mixed", "intrinsic", "idiosyncratic", "unclear". Use null if not mentioned.

2. "fraction_patients_with_enzyme_elevation" (float or null): The fraction (0.0 to 1.0) of patients who had elevated liver enzymes (ALT >3x ULN) when taking this drug. Convert percentages to fractions: 5% = 0.05. Use null if no specific number is given.

3. "fraction_patients_with_dili" (float or null): The fraction (0.0 to 1.0) of patients who developed clinically significant drug-induced liver injury (jaundice, hospitalization, liver failure). This is DIFFERENT from enzyme elevation — it refers to actual clinical illness, not just lab abnormalities. Convert percentages to fractions. Use null if only qualitative terms like "rare" are used without a number.

4. "risk_factors" (list or null): Risk factors for developing liver injury from this drug. Each item should have "factor" (description) and "supporting_quote" (exact quote from text). Use null if none mentioned.

5. "safe_dose" (object or null): The recommended/therapeutic dose. Format: {"value": number, "unit": "mg" or similar, "frequency": "daily" or similar}. Use null if not mentioned.

6. "toxic_dose" (object or null): The dose associated with liver toxicity. Same format as safe_dose. Use null if not mentioned.

7. "onset_time" (object or null): Time from starting the drug to liver injury onset. Format: {"min": number or null, "max": number or null, "typical": number or null, "unit": "days" or "weeks" or "months"}. Use null if not mentioned.

8. "peak_alt" (float or null): Peak ALT level reported, in multiples of the upper limit of normal (ULN). For example, "ALT 33 times ULN" = 33.0. Use null if not mentioned.

9. "peak_alp" (float or null): Peak alkaline phosphatase level, in multiples of ULN. Use null if not mentioned.

10. "bilirubin_peak" (float or null): Peak bilirubin level in mg/dL. Use null if not mentioned or only reported in ULN multiples.

11. "is_immune_mediated" (boolean): Whether the liver injury involves immune-mediated mechanisms. Look for: autoimmune features, hypersensitivity, eosinophilia, rash and fever, autoantibodies. If the text says immune features were "not prominent" or "uncommon", return false. Default to false if unclear.

CRITICAL RULES:
- Only extract information EXPLICITLY stated in the provided text.
- Do NOT use external knowledge about the drug. If a drug is unfamiliar, that is fine — just extract what the text says.
- Return null for any field where the information is not available in the text.
- Do NOT guess or infer values. Return null rather than guessing.
- For fraction fields, convert percentages to decimals (5% → 0.05).
- fraction_patients_with_enzyme_elevation is about LAB ABNORMALITIES (elevated ALT).
- fraction_patients_with_dili is about CLINICAL ILLNESS (jaundice, symptoms, hospitalization).
- These are DIFFERENT measurements. Do not confuse them.

Return ONLY a JSON object with these fields. No other text."""


def build_user_prompt(sections):
    """Build the user message from parsed sections.

    Includes only the sections relevant for extraction,
    skipping bibliography and references to reduce noise.

    Args:
        sections: ParsedSections instance

    Returns:
        string with the drug text for the LLM
    """
    parts = [f"DRUG NAME: {sections.drug_name}\n"]

    if sections.introduction:
        parts.append(f"INTRODUCTION:\n{sections.introduction}\n")

    if sections.background:
        parts.append(f"BACKGROUND:\n{sections.background}\n")

    if sections.hepatotoxicity:
        parts.append(f"HEPATOTOXICITY:\n{sections.hepatotoxicity}\n")

    if sections.mechanism:
        parts.append(f"MECHANISM OF INJURY:\n{sections.mechanism}\n")

    if sections.outcome_and_management:
        parts.append(f"OUTCOME AND MANAGEMENT:\n{sections.outcome_and_management}\n")

    # Include case report key points as a summary (not full prose)
    if sections.case_report_key_points:
        parts.append("CASE REPORT KEY POINTS:")
        for i, kp in enumerate(sections.case_report_key_points):
            parts.append(f"  Case {i+1}:")
            for key, value in kp.items():
                parts.append(f"    {key}: {value}")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def extract_with_llm(sections, client, model="claude-sonnet-4-20250514", max_retries=2):
    """Extract fields from parsed sections using the Anthropic API.

    Sends the relevant text sections to Claude and parses the JSON response
    into a DrugExtraction. Retries on JSON parse failures.

    Args:
        sections: ParsedSections instance
        client: anthropic.Anthropic client instance
        model: model name to use
        max_retries: number of retries on parse failure

    Returns:
        DrugExtraction with LLM-extracted fields, or None if extraction failed
    """
    # Skip drugs with no content (fictional/empty drugs)
    has_content = any([
        sections.introduction,
        sections.background,
        sections.hepatotoxicity,
    ])
    if not has_content:
        return DrugExtraction(drug_name=sections.drug_name)

    user_prompt = build_user_prompt(sections)

    # Try extraction with retries
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0,
                messages=[
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + user_prompt}
                ],
            )

            # Get the text response
            response_text = response.content[0].text.strip()

            # Clean up response: remove markdown code blocks if present
            if response_text.startswith("```"):
                # Remove ```json and ``` wrapping
                lines = response_text.split("\n")
                # Remove first line (```json) and last line (```)
                lines = [l for l in lines if not l.strip().startswith("```")]
                response_text = "\n".join(lines)

            # Parse the JSON
            data = json.loads(response_text)

            # Add drug_name (LLM doesn't return it)
            data["drug_name"] = sections.drug_name

            # Convert to DrugExtraction
            result = extraction_from_dict(data)
            return result

        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    Retry {attempt + 1}: JSON parse error for {sections.drug_name}: {e}")
                time.sleep(1)  # Brief pause before retry
            else:
                print(f"    FAILED: Could not parse JSON for {sections.drug_name} after {max_retries + 1} attempts")
                return DrugExtraction(drug_name=sections.drug_name)

        except anthropic.APIError as e:
            if attempt < max_retries:
                print(f"    Retry {attempt + 1}: API error for {sections.drug_name}: {e}")
                time.sleep(2)  # Longer pause for API errors
            else:
                print(f"    FAILED: API error for {sections.drug_name}: {e}")
                return DrugExtraction(drug_name=sections.drug_name)

    return DrugExtraction(drug_name=sections.drug_name)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def create_client(keys_file="keys.txt"):
    """Create an Anthropic client from the keys file.

    Args:
        keys_file: path to keys.txt

    Returns:
        anthropic.Anthropic client instance
    """
    api_key = load_api_key(keys_file)
    return anthropic.Anthropic(api_key=api_key)
