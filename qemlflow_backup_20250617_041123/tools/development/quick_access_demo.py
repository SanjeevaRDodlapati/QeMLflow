#!/usr/bin/env python3
"""
ChemML Bootcamp Quick Access Demo
=====================================

This script demonstrates how to access and run the organized ChemML bootcamp scripts.
It provides a simple interface to navigate and execute the Day 1-7 production-ready scripts.
"""

import os
import subprocess
import sys
from pathlib import Path

# ANSI color codes for better output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
ENDC = "\033[0m"  # End color


def print_header(text):
    """Print a styled header"""
    print(f"\n{BLUE}{BOLD}{'='*60}{ENDC}")
    print(f"{BLUE}{BOLD}{text:^60}{ENDC}")
    print(f"{BLUE}{BOLD}{'='*60}{ENDC}\n")


def print_info(text):
    """Print info text"""
    print(f"{GREEN}ℹ️  {text}{ENDC}")


def print_warning(text):
    """Print warning text"""
    print(f"{YELLOW}⚠️  {text}{ENDC}")


def print_error(text):
    """Print error text"""
    print(f"{RED}❌ {text}{ENDC}")


def get_script_directory():
    """Get the path to the organized scripts directory"""
    current_dir = Path(__file__).parent
    scripts_dir = current_dir / "notebooks" / "quickstart_bootcamp" / "days"
    return scripts_dir


def list_available_days():
    """List all available day directories"""
    scripts_dir = get_script_directory()

    if not scripts_dir.exists():
        print_error(f"Scripts directory not found: {scripts_dir}")
        return []

    days = []
    for day_dir in sorted(scripts_dir.iterdir()):
        if day_dir.is_dir() and day_dir.name.startswith("day_"):
            days.append(day_dir)

    return days


def list_scripts_in_day(day_dir):
    """List all Python scripts in a day directory"""
    scripts = []
    for script_file in day_dir.iterdir():
        if script_file.is_file() and script_file.suffix == ".py":
            scripts.append(script_file)
    return sorted(scripts)


def display_script_info(script_path):
    """Display information about a script"""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Look for docstring or description
        description = "No description available"
        for i, line in enumerate(lines[:20]):  # Check first 20 lines
            if '"""' in line or "'''" in line:
                # Found start of docstring, collect until end
                desc_lines = []
                quote_type = '"""' if '"""' in line else "'''"
                start_line = line.split(quote_type)[1] if quote_type in line else ""
                if start_line.strip():
                    desc_lines.append(start_line.strip())

                for j in range(i + 1, min(i + 10, len(lines))):
                    if quote_type in lines[j]:
                        end_part = lines[j].split(quote_type)[0]
                        if end_part.strip():
                            desc_lines.append(end_part.strip())
                        break
                    desc_lines.append(lines[j].strip())

                description = " ".join(desc_lines).strip()
                break

        size_kb = script_path.stat().st_size / 1024
        line_count = len(lines)

        print(f"    📄 {script_path.name}")
        print(f"       Size: {size_kb:.1f} KB, Lines: {line_count}")
        print(
            f"       Description: {description[:100]}{'...' if len(description) > 100 else ''}"
        )

    except Exception as e:
        print(f"    📄 {script_path.name} (Error reading: {e})")


def run_script(script_path):
    """Run a selected script"""
    print_info(f"Running script: {script_path.name}")
    print_info(f"Working directory: {script_path.parent}")

    try:
        # Change to script directory
        os.chdir(script_path.parent)

        # Run the script
        result = subprocess.run(
            [sys.executable, script_path.name], capture_output=False, text=True
        )

        if result.returncode == 0:
            print_info("Script completed successfully!")
        else:
            print_error(f"Script failed with return code: {result.returncode}")

    except Exception as e:
        print_error(f"Error running script: {e}")


def main():
    """Main function"""
    print_header("ChemML Bootcamp - Quick Access Demo")

    print_info(
        "This tool helps you navigate and run the organized ChemML bootcamp scripts."
    )
    print_info(
        "All scripts are production-ready and can run without interactive input."
    )

    # Check if scripts directory exists
    scripts_dir = get_script_directory()
    if not scripts_dir.exists():
        print_error("Organized scripts directory not found!")
        print_error(f"Expected location: {scripts_dir}")
        print_info("Please ensure you're running this from the ChemML root directory.")
        return

    # List available days
    days = list_available_days()
    if not days:
        print_error("No day directories found!")
        return

    print_header("Available Bootcamp Days")

    for i, day_dir in enumerate(days, 1):
        print(f"{BOLD}{i}. {day_dir.name.title().replace('_', ' ')}{ENDC}")
        scripts = list_scripts_in_day(day_dir)

        if scripts:
            for script in scripts:
                display_script_info(script)
        else:
            print("    No Python scripts found")
        print()

    # Interactive selection
    while True:
        try:
            print_header("Select a Day to Explore")
            print("Enter day number (1-7), 'demo' for framework demo, or 'q' to quit:")

            choice = input(f"{YELLOW}Your choice: {ENDC}").strip().lower()

            if choice == "q":
                print_info("Goodbye!")
                break
            elif choice == "demo":
                demo_script = Path(__file__).parent / "framework_demo.py"
                if demo_script.exists():
                    run_script(demo_script)
                else:
                    print_error("Framework demo not found!")
                continue

            try:
                day_index = int(choice) - 1
                if 0 <= day_index < len(days):
                    selected_day = days[day_index]
                    scripts = list_scripts_in_day(selected_day)

                    if not scripts:
                        print_warning("No scripts found in this day directory")
                        continue

                    print_header(
                        f"{selected_day.name.title().replace('_', ' ')} Scripts"
                    )

                    for i, script in enumerate(scripts, 1):
                        print(f"{i}. {script.name}")

                    script_choice = input(
                        f"{YELLOW}Select script number (1-{len(scripts)}) or 'b' for back: {ENDC}"
                    ).strip()

                    if script_choice.lower() == "b":
                        continue

                    script_index = int(script_choice) - 1
                    if 0 <= script_index < len(scripts):
                        selected_script = scripts[script_index]

                        confirm = (
                            input(f"{YELLOW}Run {selected_script.name}? (y/n): {ENDC}")
                            .strip()
                            .lower()
                        )
                        if confirm == "y":
                            run_script(selected_script)
                    else:
                        print_error("Invalid script number!")
                else:
                    print_error("Invalid day number!")
            except ValueError:
                print_error("Please enter a valid number!")

        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except Exception as e:
            print_error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
