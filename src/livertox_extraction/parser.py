"""
XML parser for LiverTox drug files.

Reads a LiverTox XML file and extracts clean text from each section,
returning a ParsedSections dataclass. Handles missing sections gracefully
and parses case report Key Points tables into structured dicts.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict

from livertox_extraction.models import ParsedSections


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_text_from_element(element):
    """Extract all text from an XML element, stripping tags.

    Uses ElementTree's method="text" to get just the text content,
    then cleans up extra whitespace.

    Args:
        element: an XML Element

    Returns:
        cleaned text string, or empty string if element is None
    """
    if element is None:
        return ""
    raw = ET.tostring(element, encoding="unicode", method="text")
    # Collapse all whitespace (newlines, tabs, multiple spaces) into single spaces
    cleaned = re.sub(r"\s+", " ", raw).strip()
    return cleaned


def get_paragraph_text(section_element):
    """Extract text from only the direct <p> tags in a section.

    This avoids pulling in text from nested sub-sections.
    For example, the OVERVIEW section contains Hepatotoxicity as a child,
    but we want them separately.

    Args:
        section_element: a <sec> XML element

    Returns:
        combined text from direct <p> children, or None if no text found
    """
    if section_element is None:
        return None

    paragraphs = []
    for child in section_element:
        if child.tag == "p":
            text = get_text_from_element(child)
            if text:
                paragraphs.append(text)

    if not paragraphs:
        return None
    return " ".join(paragraphs)


def find_section_by_title(root, title_keywords):
    """Find a <sec> element whose <title> contains any of the given keywords.

    Searches case-insensitively. Returns the first match.
    This is more robust than matching by section ID, since IDs vary
    (e.g., "Mechanism_of_Injury" vs "Mechanism_of_Liver_Injury").

    Args:
        root: the XML root element
        title_keywords: list of strings to match against section titles

    Returns:
        the matching <sec> element, or None
    """
    for sec in root.iter("sec"):
        title_el = sec.find("title")
        if title_el is not None and title_el.text:
            title_lower = title_el.text.strip().lower()
            for keyword in title_keywords:
                if keyword.lower() in title_lower:
                    return sec
    return None


def parse_key_points_table(table_wrap):
    """Parse a case report Key Points table into a dictionary.

    Key Points tables have rows with <th> (label) and <td> (value) pairs.
    Example labels: Medication, Pattern, Severity, Latency, Recovery.

    Args:
        table_wrap: a <table-wrap> XML element

    Returns:
        dict with keys like "medication", "pattern", "severity", etc.
    """
    key_points = {}

    for tr in table_wrap.iter("tr"):
        # Some XMLs use <th> for labels and <td> for values (Zileuton).
        # Others use two <td> elements per row (Itraconazole).
        # Handle both formats.
        th = tr.find("th")
        tds = tr.findall("td")

        if th is not None and tds:
            # Format 1: <th>Label</th><td>Value</td>
            label = get_text_from_element(th).strip().rstrip(":")
            value = get_text_from_element(tds[0])
        elif len(tds) >= 2:
            # Format 2: <td>Label</td><td>Value</td>
            label = get_text_from_element(tds[0]).strip().rstrip(":")
            value = get_text_from_element(tds[1])
        else:
            continue

        if not label or not value:
            continue

        # Normalize the label to a standard key name
        # Check "other medications" BEFORE "medication" since it contains "medication"
        label_lower = label.lower()
        if "other" in label_lower and "medication" in label_lower:
            key_points["other_medications"] = value
        elif "medication" in label_lower:
            key_points["medication"] = value
        elif "pattern" in label_lower:
            key_points["pattern"] = value
        elif "severity" in label_lower:
            key_points["severity"] = value
        elif "latency" in label_lower:
            key_points["latency"] = value
        elif "recovery" in label_lower:
            key_points["recovery"] = value

    return key_points


def find_case_report_sections(root):
    """Find all case report sections and their Key Points tables.

    Case reports may be in a section titled "CASE REPORT" or "CASE REPORTS".
    Each individual case is a sub-section with its own Key Points table.

    Args:
        root: the XML root element

    Returns:
        tuple of (case_texts, key_points_list):
            - case_texts: list of strings, one per case report
            - key_points_list: list of dicts from Key Points tables
    """
    # Find the main case report(s) section
    case_section = find_section_by_title(root, ["case report"])
    if case_section is None:
        return None, None

    case_texts = []
    key_points_list = []

    # Each case is typically a sub-section within the case report section.
    # Key Points tables can be:
    #   1. In a "Key Points" sub-section inside the case sub-section
    #   2. Directly inside the case sub-section as a <table-wrap>
    # We search all sub-sections and look for both case text and key points.
    for sub_sec in case_section.iter("sec"):
        title_el = sub_sec.find("title")
        if title_el is None or not title_el.text:
            continue
        title = title_el.text.strip()

        # Case sub-sections: extract prose text
        if title.lower().startswith("case"):
            text = get_paragraph_text(sub_sec)
            if text:
                case_texts.append(text)

        # Key Points sub-sections: extract the table
        if "key points" in title.lower():
            for table_wrap in sub_sec.iter("table-wrap"):
                kp = parse_key_points_table(table_wrap)
                if kp:
                    key_points_list.append(kp)

    return (case_texts or None), (key_points_list or None)


def extract_drug_class(root):
    """Extract the drug class from the PRODUCT INFORMATION section.

    The drug class typically appears after a "DRUG CLASS" bold heading
    in the product information section.

    Args:
        root: the XML root element

    Returns:
        drug class string, or None
    """
    prod_section = find_section_by_title(root, ["product information"])
    if prod_section is None:
        return None

    # Look through paragraphs for "DRUG CLASS" followed by the class name
    paragraphs = list(prod_section.iter("p"))
    for i, p in enumerate(paragraphs):
        text = get_text_from_element(p)
        if "DRUG CLASS" in text.upper():
            # The drug class is usually in the next paragraph
            if i + 1 < len(paragraphs):
                drug_class = get_text_from_element(paragraphs[i + 1])
                if drug_class:
                    return drug_class
    return None


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_xml(filepath):
    """Parse a LiverTox XML file into a ParsedSections dataclass.

    Reads the XML, finds each relevant section by title, extracts
    clean text, and returns a structured result. Handles missing
    sections gracefully (they become None in the output).

    Args:
        filepath: path to the XML file (string or Path)

    Returns:
        ParsedSections dataclass instance

    Raises:
        ET.ParseError: if the XML is malformed (caller should catch this)
    """
    filepath = Path(filepath)
    drug_name = filepath.stem  # e.g., "Zileuton" from "Zileuton.xml"

    tree = ET.parse(filepath)
    root = tree.getroot()

    # Find each section by title keywords
    intro_sec = find_section_by_title(root, ["introduction"])
    background_sec = find_section_by_title(root, ["background"])
    hepatotox_sec = find_section_by_title(root, ["hepatotoxicity"])
    mechanism_sec = find_section_by_title(root, ["mechanism"])
    outcome_sec = find_section_by_title(root, ["outcome"])

    # Extract paragraph text from each section
    introduction = get_paragraph_text(intro_sec)
    background = get_paragraph_text(background_sec)
    hepatotoxicity = get_paragraph_text(hepatotox_sec)
    mechanism = get_paragraph_text(mechanism_sec)
    outcome_and_management = get_paragraph_text(outcome_sec)

    # Extract product info section text
    prod_sec = find_section_by_title(root, ["product information"])
    product_information = get_paragraph_text(prod_sec)

    # Extract drug class
    drug_class = extract_drug_class(root)

    # Extract case reports and key points tables
    case_reports_text, case_report_key_points = find_case_report_sections(root)

    return ParsedSections(
        drug_name=drug_name,
        introduction=introduction,
        background=background,
        hepatotoxicity=hepatotoxicity,
        mechanism=mechanism,
        outcome_and_management=outcome_and_management,
        case_reports_text=case_reports_text,
        case_report_key_points=case_report_key_points,
        product_information=product_information,
        drug_class=drug_class,
    )


def parse_xml_safe(filepath):
    """Parse a LiverTox XML file, returning None if parsing fails.

    Wraps parse_xml with error handling for malformed XML files.
    Prints a warning message on failure.

    Args:
        filepath: path to the XML file

    Returns:
        ParsedSections if successful, None if parsing failed
    """
    try:
        return parse_xml(filepath)
    except ET.ParseError as e:
        print(f"  WARNING: Could not parse {Path(filepath).name}: {e}")
        return None
    except Exception as e:
        print(f"  ERROR: Unexpected error parsing {Path(filepath).name}: {e}")
        return None
