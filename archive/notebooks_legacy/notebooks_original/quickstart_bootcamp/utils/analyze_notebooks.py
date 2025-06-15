#!/usr/bin/env python3
"""
Notebook Analysis and Modularization Planning Tool
Analyzes current notebook structure and creates modularization plan
"""

import json
import os
import sys
from pathlib import Path


def analyze_notebook(filepath):
    """Analyze a notebook file and return structure information"""
    try:
        with open(filepath, "r") as f:
            nb = json.load(f)
    except Exception as e:
        return None, f"Error reading {filepath}: {e}"

    cells = nb["cells"]
    total_cells = len(cells)
    total_lines = sum(len(cell.get("source", [])) for cell in cells)

    # Find sections
    sections = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "markdown":
            source = "".join(cell.get("source", []))
            if "Section" in source and any(str(j) in source for j in range(1, 6)):
                section_title = source.split("\n")[0].strip()
                sections.append(
                    {
                        "cell_index": i,
                        "title": section_title,
                        "content_preview": section_title[:80],
                    }
                )

    # Count cell types
    markdown_cells = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
    code_cells = sum(1 for cell in cells if cell.get("cell_type") == "code")

    return {
        "total_cells": total_cells,
        "total_lines": total_lines,
        "markdown_cells": markdown_cells,
        "code_cells": code_cells,
        "sections": sections,
        "filename": os.path.basename(filepath),
    }, None


def create_modularization_plan(analysis):
    """Create a plan for breaking down dense notebooks"""
    plan = {
        "current_size": analysis["total_lines"],
        "complexity_level": "normal",
        "modules_needed": 1,
        "split_strategy": "no_split",
    }

    if analysis["total_lines"] > 4000:
        plan["complexity_level"] = "very_high"
        plan["modules_needed"] = 3
        plan["split_strategy"] = "three_modules"
    elif analysis["total_lines"] > 3000:
        plan["complexity_level"] = "high"
        plan["modules_needed"] = 2
        plan["split_strategy"] = "two_modules"
    elif analysis["total_lines"] > 2500:
        plan["complexity_level"] = "moderate"
        plan["modules_needed"] = 2
        plan["split_strategy"] = "optional_split"

    return plan


def main():
    """Main analysis function"""
    bootcamp_dir = Path(
        "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp"
    )

    print("🔍 NOTEBOOK STRUCTURE ANALYSIS")
    print("=" * 60)

    notebook_files = sorted(bootcamp_dir.glob("day_*.ipynb"))
    analysis_results = {}

    for notebook_file in notebook_files:
        analysis, error = analyze_notebook(notebook_file)
        if error:
            print(f"❌ {notebook_file.name}: {error}")
            continue

        analysis_results[notebook_file.name] = analysis
        plan = create_modularization_plan(analysis)

        print(f"\n📓 {analysis['filename']}")
        print(f"   📊 Total lines: {analysis['total_lines']:,}")
        print(f"   📱 Total cells: {analysis['total_cells']}")
        print(
            f"   📝 Markdown: {analysis['markdown_cells']} | 💻 Code: {analysis['code_cells']}"
        )
        print(f"   📖 Sections: {len(analysis['sections'])}")
        print(f"   🎯 Complexity: {plan['complexity_level']}")
        print(f"   🔄 Modules needed: {plan['modules_needed']}")

        if plan["modules_needed"] > 1:
            print(f"   ⚠️  NEEDS MODULARIZATION: {plan['split_strategy']}")

        # Show sections
        if analysis["sections"]:
            print("   📋 Sections found:")
            for section in analysis["sections"]:
                print(
                    f"      - Cell {section['cell_index']}: {section['content_preview']}"
                )

    print(f"\n📈 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)

    total_notebooks = len(analysis_results)
    dense_notebooks = sum(
        1 for a in analysis_results.values() if a["total_lines"] > 2500
    )
    very_dense_notebooks = sum(
        1 for a in analysis_results.values() if a["total_lines"] > 4000
    )

    print(f"📊 Total notebooks analyzed: {total_notebooks}")
    print(f"⚠️  Notebooks needing optimization: {dense_notebooks}")
    print(f"🚨 Very dense notebooks (>4k lines): {very_dense_notebooks}")

    if dense_notebooks > 0:
        print(f"\n🔧 MODULARIZATION STRATEGY:")
        print(f"   1. Create Core + Advanced modules for dense notebooks")
        print(f"   2. Split content at natural section boundaries")
        print(f"   3. Maintain educational flow between modules")
        print(f"   4. Add clear navigation between modules")

    return analysis_results


if __name__ == "__main__":
    main()
