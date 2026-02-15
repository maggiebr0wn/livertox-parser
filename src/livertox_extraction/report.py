"""
HTML report generator for LiverTox extraction pipeline.

Generates a self-contained HTML file with:
  - Summary dashboard (DILI score distribution, field coverage)
  - Evaluation metrics (accuracy per field, error analysis)
  - Filterable drug table with all extracted fields
  - CSV download button
"""

import json
from pathlib import Path
from typing import Dict

from livertox_extraction.models import DrugExtraction
from livertox_extraction.evaluate import run_evaluation


def generate_report(results, eval_results, output_path):
    """Generate an interactive HTML report.

    Args:
        results: dict mapping drug names to DrugExtraction
        eval_results: output from run_evaluation()
        output_path: path to write the HTML file
    """
    # Prepare data for charts and tables
    dili_counts = _count_dili_scores(results)
    field_coverage = eval_results["completeness"]["field_coverage"]
    gold_eval = eval_results["gold_standard"]
    validation = eval_results["validation"]
    drug_table_data = _build_table_data(results)

    html = _build_html(
        dili_counts=dili_counts,
        field_coverage=field_coverage,
        gold_eval=gold_eval,
        validation=validation,
        drug_table_data=drug_table_data,
        total_drugs=len(results),
        fictional_drugs=eval_results["completeness"]["fictional_drugs"],
    )

    Path(output_path).write_text(html)
    print(f"Report saved to {output_path}")


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def _count_dili_scores(results):
    """Count DILI score distribution."""
    counts = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "E*": 0, "X": 0, "None": 0}
    for drug_name, extraction in results.items():
        score = extraction.dili_likelihood_score
        if score is None:
            counts["None"] += 1
        else:
            counts[score] = counts.get(score, 0) + 1
    return counts


def _build_table_data(results):
    """Convert results to a list of dicts for the HTML table."""
    rows = []
    for name, extraction in sorted(results.items()):
        d = extraction.to_dict()
        fields_filled = sum(
            1 for k, v in d.items()
            if k != "drug_name" and v is not None and v is not False
        )
        rows.append({
            "drug_name": name,
            "dili_likelihood_score": d["dili_likelihood_score"] or "",
            "injury_pattern": d["injury_pattern"] or "",
            "fraction_enzyme": f"{d['fraction_patients_with_enzyme_elevation']:.4f}" if d["fraction_patients_with_enzyme_elevation"] is not None else "",
            "fraction_dili": f"{d['fraction_patients_with_dili']:.4f}" if d["fraction_patients_with_dili"] is not None else "",
            "r_ratio": f"{d['r_ratio']:.1f}" if d["r_ratio"] is not None else "",
            "peak_alt": f"{d['peak_alt']:.1f}" if d["peak_alt"] is not None else "",
            "is_immune_mediated": "Yes" if d["is_immune_mediated"] else "",
            "safe_dose": _format_dose(d["safe_dose"]),
            "onset_time": _format_onset(d["onset_time"]),
            "regulatory_status": d["regulatory_status"] or "",
            "fields_filled": fields_filled,
        })
    return rows


def _format_dose(dose):
    if dose is None:
        return ""
    parts = [str(dose.get("value", "")), dose.get("unit", "")]
    if dose.get("frequency"):
        parts.append(dose["frequency"])
    return " ".join(parts)


def _format_onset(onset):
    if onset is None:
        return ""
    parts = []
    if onset.get("min") is not None and onset.get("max") is not None:
        parts.append(f"{onset['min']}-{onset['max']}")
    elif onset.get("typical") is not None:
        parts.append(str(onset["typical"]))
    elif onset.get("min") is not None:
        parts.append(f">{onset['min']}")
    elif onset.get("max") is not None:
        parts.append(f"<{onset['max']}")
    if onset.get("unit"):
        parts.append(onset["unit"])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

def _build_html(dili_counts, field_coverage, gold_eval, validation,
                drug_table_data, total_drugs, fictional_drugs):
    """Build the full HTML string."""

    # Prepare Plotly chart data
    dili_labels = list(dili_counts.keys())
    dili_values = list(dili_counts.values())
    dili_colors = {
        "A": "#d32f2f", "B": "#f57c00", "C": "#fbc02d",
        "D": "#7cb342", "E": "#2196f3", "E*": "#9c27b0",
        "X": "#757575", "None": "#bdbdbd"
    }
    dili_color_list = [dili_colors.get(l, "#999") for l in dili_labels]

    # Field accuracy data for bar chart
    field_names = list(gold_eval["per_field"].keys())
    field_accuracies = [
        gold_eval["per_field"][f]["accuracy"] * 100
        if gold_eval["per_field"][f]["accuracy"] is not None else 0
        for f in field_names
    ]

    # Coverage data
    coverage_fields = list(field_coverage.keys())
    coverage_pcts = [field_coverage[f]["percentage"] for f in coverage_fields]

    # Error details
    error_rows = ""
    for field, stats in gold_eval["per_field"].items():
        for error in stats["errors"]:
            pred_str = str(error["predicted"]) if error["predicted"] is not None else "null"
            gold_str = str(error["gold"]) if error["gold"] is not None else "null"
            error_rows += f"""
                <tr>
                    <td>{error['drug']}</td>
                    <td>{error['field']}</td>
                    <td>{pred_str}</td>
                    <td>{gold_str}</td>
                    <td><span class="tag tag-{error['error_type']}">{error['error_type']}</span></td>
                </tr>"""

    # Drug table rows
    table_rows = ""
    for row in drug_table_data:
        dili_class = f"dili-{row['dili_likelihood_score']}" if row['dili_likelihood_score'] else ""
        table_rows += f"""
                <tr class="{dili_class}">
                    <td class="drug-name">{row['drug_name']}</td>
                    <td class="center"><span class="dili-badge {dili_class}">{row['dili_likelihood_score']}</span></td>
                    <td>{row['injury_pattern']}</td>
                    <td class="num">{row['fraction_enzyme']}</td>
                    <td class="num">{row['fraction_dili']}</td>
                    <td class="num">{row['r_ratio']}</td>
                    <td class="num">{row['peak_alt']}</td>
                    <td class="center">{row['is_immune_mediated']}</td>
                    <td>{row['safe_dose']}</td>
                    <td>{row['onset_time']}</td>
                    <td>{row['regulatory_status']}</td>
                    <td class="num">{row['fields_filled']}</td>
                </tr>"""

    # Table data as JSON for CSV export
    table_json = json.dumps(drug_table_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LiverTox Extraction Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        :root {{
            --bg: #fafafa;
            --surface: #ffffff;
            --border: #e0e0e0;
            --text: #212121;
            --text-muted: #757575;
            --primary: #1565c0;
            --primary-light: #e3f2fd;
            --success: #2e7d32;
            --warning: #f57f17;
            --danger: #c62828;
            --radius: 8px;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Source Sans 3', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}

        header {{
            text-align: center;
            padding: 3rem 0 2rem;
            border-bottom: 2px solid var(--border);
            margin-bottom: 2rem;
        }}

        header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            color: var(--text);
            letter-spacing: -0.5px;
        }}

        header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }}

        /* Metric cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .metric-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
            text-align: center;
        }}

        .metric-card .value {{
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--primary);
            font-family: 'JetBrains Mono', monospace;
        }}

        .metric-card .value.success {{ color: var(--success); }}
        .metric-card .value.warning {{ color: var(--warning); }}

        .metric-card .label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}

        /* Section headings */
        .section {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 2rem;
            margin-bottom: 2rem;
        }}

        .section h2 {{
            font-size: 1.4rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }}

        /* Charts grid */
        .charts-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-bottom: 0.5rem;
        }}

        .chart-container {{
            min-height: 350px;
        }}

        /* Error table */
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 0.6rem 0.8rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            font-weight: 600;
            background: var(--bg);
            position: sticky;
            top: 0;
            z-index: 1;
        }}

        td.num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            text-align: right;
        }}

        td.center, th.center {{ text-align: center; }}

        tr:hover {{ background: var(--primary-light); }}

        .drug-name {{ font-weight: 600; }}

        /* Tags */
        .tag {{
            display: inline-block;
            padding: 0.15rem 0.5rem;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .tag-mismatch {{ background: #ffebee; color: var(--danger); }}
        .tag-false_positive {{ background: #fff3e0; color: var(--warning); }}
        .tag-false_negative {{ background: #e3f2fd; color: var(--primary); }}

        /* DILI badges */
        .dili-badge {{
            display: inline-block;
            width: 28px;
            height: 28px;
            line-height: 28px;
            text-align: center;
            border-radius: 50%;
            font-weight: 700;
            font-size: 0.8rem;
            color: white;
        }}

        .dili-badge.dili-A {{ background: #d32f2f; }}
        .dili-badge.dili-B {{ background: #f57c00; }}
        .dili-badge.dili-C {{ background: #fbc02d; color: #333; }}
        .dili-badge.dili-D {{ background: #7cb342; }}
        .dili-badge.dili-E {{ background: #2196f3; }}

        /* Search and controls */
        .table-controls {{
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            align-items: center;
        }}

        .search-input {{
            padding: 0.5rem 1rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            font-size: 0.9rem;
            font-family: inherit;
            flex: 1;
            max-width: 300px;
        }}

        .search-input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-light);
        }}

        .btn {{
            padding: 0.5rem 1.25rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            background: var(--surface);
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9rem;
            font-weight: 600;
            transition: all 0.15s;
        }}

        .btn:hover {{
            background: var(--primary);
            color: white;
            border-color: var(--primary);
        }}

        .drug-table-wrapper {{
            overflow-x: auto;
            max-height: 600px;
            overflow-y: auto;
        }}

        /* Responsive */
        @media (max-width: 900px) {{
            .charts-grid {{ grid-template-columns: 1fr; }}
            .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LiverTox Extraction Pipeline Report</h1>
            <p>Structured data extraction from {total_drugs} LiverTox drug XML documents</p>
        </header>

        <!-- Metric cards -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="value">{total_drugs}</div>
                <div class="label">Total Drugs</div>
            </div>
            <div class="metric-card">
                <div class="value success">{gold_eval['overall_accuracy']:.0%}</div>
                <div class="label">Gold Standard Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="value success">{validation['pass_rate']:.0%}</div>
                <div class="label">Validation Pass Rate</div>
            </div>
            <div class="metric-card">
                <div class="value">{len(fictional_drugs)}</div>
                <div class="label">Fictional Drugs Detected</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="section">
            <h2>Extraction Overview</h2>
            <div class="charts-grid">
                <div id="dili-chart" class="chart-container"></div>
                <div id="coverage-chart" class="chart-container"></div>
            </div>
        </div>

        <!-- Accuracy -->
        <div class="section">
            <h2>Per-Field Accuracy (Gold Standard, n={gold_eval['total_comparisons'] // len(gold_eval['per_field'])})</h2>
            <div id="accuracy-chart" class="chart-container"></div>
        </div>

        <!-- Errors -->
        <div class="section">
            <h2>Error Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Drug</th>
                        <th>Field</th>
                        <th>Predicted</th>
                        <th>Gold Standard</th>
                        <th>Error Type</th>
                    </tr>
                </thead>
                <tbody>{error_rows}
                </tbody>
            </table>
        </div>

        <!-- Drug table -->
        <div class="section">
            <h2>All Drug Extractions</h2>
            <div class="table-controls">
                <input type="text" id="search" class="search-input" placeholder="Search drugs..." onkeyup="filterTable()">
                <button class="btn" onclick="downloadCSV()">Download CSV</button>
            </div>
            <div class="drug-table-wrapper">
                <table id="drug-table">
                    <thead>
                        <tr>
                            <th>Drug</th>
                            <th class="center">DILI</th>
                            <th>Pattern</th>
                            <th>Enzyme Frac</th>
                            <th>DILI Frac</th>
                            <th>R-ratio</th>
                            <th>Peak ALT</th>
                            <th class="center">Immune</th>
                            <th>Safe Dose</th>
                            <th>Onset</th>
                            <th>Reg. Status</th>
                            <th>Fields</th>
                        </tr>
                    </thead>
                    <tbody>{table_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        // DILI score distribution chart
        Plotly.newPlot('dili-chart', [{{
            x: {json.dumps(dili_labels)},
            y: {json.dumps(dili_values)},
            type: 'bar',
            marker: {{ color: {json.dumps(dili_color_list)} }},
        }}], {{
            title: 'DILI Likelihood Score Distribution',
            xaxis: {{ title: 'Score' }},
            yaxis: {{ title: 'Count' }},
            margin: {{ t: 40, b: 40, l: 50, r: 20 }},
            height: 350,
        }}, {{ responsive: true }});

        // Field coverage chart
        Plotly.newPlot('coverage-chart', [{{
            y: {json.dumps(coverage_fields)},
            x: {json.dumps(coverage_pcts)},
            type: 'bar',
            orientation: 'h',
            marker: {{ color: '#1565c0' }},
        }}], {{
            title: 'Field Coverage (%)',
            xaxis: {{ title: '% of drugs with value', range: [0, 100] }},
            margin: {{ t: 40, b: 40, l: 200, r: 20 }},
            height: 350,
        }}, {{ responsive: true }});

        // Per-field accuracy chart
        Plotly.newPlot('accuracy-chart', [{{
            x: {json.dumps(field_names)},
            y: {json.dumps(field_accuracies)},
            type: 'bar',
            marker: {{
                color: {json.dumps(field_accuracies)},
                colorscale: [[0, '#c62828'], [0.7, '#f57f17'], [0.9, '#7cb342'], [1, '#2e7d32']],
                cmin: 50,
                cmax: 100,
            }},
        }}], {{
            title: 'Accuracy by Field (%)',
            yaxis: {{ title: 'Accuracy %', range: [0, 105] }},
            xaxis: {{ tickangle: -35 }},
            margin: {{ t: 40, b: 120, l: 50, r: 20 }},
            height: 380,
        }}, {{ responsive: true }});

        // Table search
        function filterTable() {{
            const query = document.getElementById('search').value.toLowerCase();
            const rows = document.querySelectorAll('#drug-table tbody tr');
            rows.forEach(row => {{
                const drug = row.cells[0].textContent.toLowerCase();
                row.style.display = drug.includes(query) ? '' : 'none';
            }});
        }}

        // CSV download
        const tableData = {table_json};
        function downloadCSV() {{
            const headers = Object.keys(tableData[0]);
            const csv = [headers.join(',')];
            tableData.forEach(row => {{
                csv.push(headers.map(h => {{
                    let val = row[h];
                    if (typeof val === 'string' && val.includes(',')) val = '"' + val + '"';
                    return val;
                }}).join(','));
            }});
            const blob = new Blob([csv.join('\\n')], {{ type: 'text/csv' }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'livertox_extractions.csv';
            a.click();
        }}
    </script>
</body>
</html>"""
