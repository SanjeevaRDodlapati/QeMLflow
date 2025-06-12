#!/usr/bin/env python3
"""
Fix the dtype mismatch issue in PropertyOptimizer.decode_latent() method
"""

import json
import re


def fix_notebook_dtype_issue(notebook_path):
    """Fix the dtype issue in the PropertyOptimizer decode_latent method"""

    print(f"🔧 Fixing dtype issue in {notebook_path}")

    # Read the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    fixed_cells = 0

    # Iterate through cells to find the PropertyOptimizer class
    for cell_idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            # Join all source lines
            source_lines = cell["source"]
            source_text = "".join(source_lines)

            # Check if this cell contains the PropertyOptimizer decode_latent method
            if (
                "class PropertyOptimizer" in source_text
                and "def decode_latent" in source_text
            ):
                print(f"📍 Found PropertyOptimizer in cell {cell_idx}")

                # Apply the fix
                old_pattern = r"z_tensor = torch\.tensor\(\[z\]\)\.to\(device\)"
                new_replacement = (
                    "z_tensor = torch.tensor([z], dtype=torch.float32).to(device)"
                )

                # Fix the source text
                if re.search(old_pattern, source_text):
                    print("🎯 Found target line, applying fix...")
                    fixed_source = re.sub(old_pattern, new_replacement, source_text)

                    # Split back into lines while preserving original format
                    fixed_lines = []
                    for line in fixed_source.split("\n"):
                        if line or fixed_lines:  # Keep empty lines except at the end
                            fixed_lines.append(line + "\n")

                    # Remove trailing newline from last line if it was added
                    if fixed_lines and fixed_lines[-1].endswith("\n\n"):
                        fixed_lines[-1] = fixed_lines[-1][:-1]

                    # Update the cell
                    cell["source"] = fixed_lines
                    fixed_cells += 1
                    print("✅ Applied dtype fix to PropertyOptimizer.decode_latent()")
                else:
                    print("⚠️ Target pattern not found in exact form")
                    # Try alternative patterns
                    alt_patterns = [
                        r"torch\.tensor\(\[z\]\)",
                        r"z_tensor\s*=\s*torch\.tensor\(\[z\]\)",
                    ]

                    for pattern in alt_patterns:
                        if re.search(pattern, source_text):
                            print(f"🎯 Found alternative pattern: {pattern}")
                            fixed_source = re.sub(
                                pattern,
                                "torch.tensor([z], dtype=torch.float32)",
                                source_text,
                            )

                            # Split back into lines
                            fixed_lines = []
                            for line in fixed_source.split("\n"):
                                if line or fixed_lines:
                                    fixed_lines.append(line + "\n")

                            if fixed_lines and fixed_lines[-1].endswith("\n\n"):
                                fixed_lines[-1] = fixed_lines[-1][:-1]

                            cell["source"] = fixed_lines
                            fixed_cells += 1
                            print("✅ Applied alternative dtype fix")
                            break

    if fixed_cells > 0:
        # Write the fixed notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        print(f"✅ Fixed {fixed_cells} cells in {notebook_path}")
        print("🚀 PropertyOptimizer dtype mismatch should now be resolved!")
        return True
    else:
        print("❌ No cells were fixed - pattern not found")
        return False


if __name__ == "__main__":
    notebook_path = "/Users/sanjeevadodlapati/Downloads/Repos/ChemML/notebooks/quickstart_bootcamp/days/day_02/day_02_deep_learning_molecules_project.ipynb"

    print("🔧 PropertyOptimizer Dtype Fix Script")
    print("=" * 40)

    success = fix_notebook_dtype_issue(notebook_path)

    if success:
        print("\n" + "=" * 50)
        print("🎉 PROPERTY OPTIMIZER DTYPE FIX COMPLETE!")
        print("✅ The 'mat1 and mat2 must have the same dtype' error should be resolved")
        print("🚀 You can now run PropertyOptimizer.optimize_property() successfully")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("⚠️ DTYPE FIX NOT APPLIED")
        print("🔧 Manual intervention may be required")
        print("=" * 50)
