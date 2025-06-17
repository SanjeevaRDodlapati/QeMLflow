#!/usr/bin/env python3
"""
QeMLflow Bootcamp Quick Access Demo
=====================================

This script demonstrates how to access and run the organized QeMLflow bootcamp scripts.
It provides a simple interface to navigate and execute the Day 1-7 production-ready scripts.
"""

import os
import subprocess
import sys
from pathlib import Path

import wandb


# Wandb experiment tracking setup
def setup_wandb_tracking(experiment_name, config=None):
    """Setup wandb experiment tracking."""
    try:
        wandb.login(key="b4f102d87161194b68baa7395d5862aa3f93b2b7", relogin=True)
        run = wandb.init(
            project="qemlflow-experiments",
            name=experiment_name,
            config=config or {},
            tags=["qemlflow"],
        )
        print(f"✅ Wandb tracking started: {run.url}")
        return run
    except Exception as e:
        print(f"⚠️ Wandb setup failed: {e}")
        return None


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

    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        print(f"    📄 {script_path.name} (Error reading: {e})")


def run_script(script_path):
    """Run a selected script"""
    print_info(f"Running script: {script_path.name}")

    # Determine the correct working directory and Python path
    bootcamp_root = Path(__file__).parent / "notebooks" / "quickstart_bootcamp"
    script_dir = script_path.parent

    # For day scripts, we need to run from bootcamp root so qemlflow_common is accessible
    if "day_" in script_path.name and script_dir.name.startswith("day_"):
        working_dir = bootcamp_root
        # Create relative path from bootcamp root to script
        relative_script_path = script_dir.relative_to(bootcamp_root) / script_path.name
        print_info(f"Working directory: {working_dir}")
        print_info(f"Script path: {relative_script_path}")
    else:
        working_dir = script_dir
        relative_script_path = script_path.name
        print_info(f"Working directory: {working_dir}")

    try:
        # Store original directory
        original_dir = os.getcwd()

        # Change to the appropriate working directory
        os.chdir(working_dir)

        # Run the script
        result = subprocess.run(
            [sys.executable, str(relative_script_path)],
            capture_output=False,
            text=True,
            check=False,
        )

        # Return to original directory
        os.chdir(original_dir)

        if result.returncode == 0:
            print_info("Script completed successfully!")
        else:
            print_error(f"Script failed with return code: {result.returncode}")

    except (FileNotFoundError, PermissionError) as e:
        print_error(f"Error running script: {e}")
    finally:
        # Ensure we return to original directory
        try:
            os.chdir(original_dir)
        except NameError:
            pass


def get_user_input(prompt):
    """Get user input with proper EOF handling"""
    try:
        return input(prompt).strip()
    except EOFError:
        print_info("\nReceived EOF signal. Exiting...")
        return "q"
    except KeyboardInterrupt:
        print_info("\nReceived interrupt signal. Exiting...")
        return "q"


def main():
    """Main function"""
    print_header("QeMLflow Bootcamp - Quick Access Demo")

    print_info(
        "This tool helps you navigate and run the organized QeMLflow bootcamp scripts."
    )
    print_info(
        "All scripts are production-ready and can run without interactive input."
    )

    # Check if scripts directory exists
    scripts_dir = get_script_directory()
    if not scripts_dir.exists():
        print_error("Organized scripts directory not found!")
        print_error(f"Expected location: {scripts_dir}")
        print_info("Please ensure you're running this from the QeMLflow root directory.")
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
            print("Options:")
            print("  • Enter day number (1-7) to explore that day's scripts")
            print("  • Enter 'demo' to run framework demonstration")
            print("  • Enter 'analyze' to run code analysis")
            print("  • Enter 'q' to quit")
            print()

            choice = get_user_input(f"{YELLOW}Your choice: {ENDC}").lower()
            print()  # Add spacing after input

            # Handle special exit cases
            if choice in ["q", "quit", "exit"]:
                print_info("Goodbye!")
                break
            elif choice == "":  # Empty input
                continue
            elif choice == "demo":
                demo_script = Path(__file__).parent / "framework_demo.py"
                if demo_script.exists():
                    run_script(demo_script)
                else:
                    print_error("Framework demo not found!")
                get_user_input(f"{YELLOW}Press Enter to continue...{ENDC}")
                continue
            elif choice == "analyze":
                analyze_script = (
                    Path(__file__).parent
                    / "tools"
                    / "analysis"
                    / "analyze_improvements.py"
                )
                if analyze_script.exists():
                    run_script(analyze_script)
                else:
                    print_error("Analysis tool not found!")
                get_user_input(f"{YELLOW}Press Enter to continue...{ENDC}")
                continue

            # Try to parse as day number
            try:
                day_index = int(choice) - 1
                if 0 <= day_index < len(days):
                    selected_day = days[day_index]
                    scripts = list_scripts_in_day(selected_day)

                    if not scripts:
                        print_warning("No scripts found in this day directory")
                        get_user_input(f"{YELLOW}Press Enter to continue...{ENDC}")
                        continue

                    # Script selection loop
                    while True:
                        print_header(
                            f"{selected_day.name.title().replace('_', ' ')} Scripts"
                        )

                        for i, script in enumerate(scripts, 1):
                            print(f"{i}. {script.name}")
                        print()

                        script_choice = get_user_input(
                            f"{YELLOW}Select script number (1-{len(scripts)}) or 'b' for back: {ENDC}"
                        ).lower()
                        print()  # Add spacing after input

                        if script_choice in ["b", "back", "q", "quit"]:
                            break  # Break out of script selection loop

                        try:
                            script_index = int(script_choice) - 1
                            if 0 <= script_index < len(scripts):
                                selected_script = scripts[script_index]

                                confirm = get_user_input(
                                    f"{YELLOW}Run {selected_script.name}? (y/n): {ENDC}"
                                ).lower()
                                print()  # Add spacing after input
                                if confirm in ["y", "yes"]:
                                    run_script(selected_script)
                                    get_user_input(
                                        f"{YELLOW}Press Enter to continue...{ENDC}"
                                    )
                                break  # Break out of script selection loop after running
                            else:
                                print_error("Invalid script number!")
                        except ValueError:
                            print_error("Please enter a valid script number!")
                else:
                    print_error("Invalid day number! Please enter 1-7.")
            except ValueError:
                print_error(
                    "Please enter a valid day number (1-7), 'demo', 'analyze', or 'q'!"
                )

        except KeyboardInterrupt:
            print_info("\nGoodbye!")
            break
        except EOFError:
            print_info("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
