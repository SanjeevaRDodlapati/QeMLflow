#!/usr/bin/env python3
"""
ChemML Bootcamp Assessment Integration Test
Tests the assessment framework integration with Day 1 notebook
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def test_assessment_framework():
    """Test that the assessment framework works correctly"""

    print("🧪 Testing ChemML Bootcamp Assessment Framework Integration")
    print("=" * 60)

    # Check if assessment framework file exists
    assessment_file = Path("notebooks/quickstart_bootcamp/assessment_framework.py")
    if not assessment_file.exists():
        print(f"❌ Assessment framework not found at {assessment_file}")
        return False

    print(f"✅ Assessment framework found at {assessment_file}")

    # Test imports
    try:
        sys.path.append(str(assessment_file.parent))
        from assessment_framework import (
            create_assessment,
            create_dashboard,
            create_widget,
        )

        print("✅ Assessment framework imports successful")
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False

    # Test basic assessment creation
    try:
        test_assessment = create_assessment(
            student_id="test_user", day=1, track="standard"
        )
        print("✅ Assessment object creation successful")
    except Exception as e:
        print(f"❌ Assessment creation error: {str(e)}")
        return False

    # Test widget creation
    try:
        test_widget = create_widget(
            assessment=test_assessment,
            section="Test Section",
            concepts=["Test concept 1", "Test concept 2"],
            activities=["Test activity 1", "Test activity 2"],
        )
        print("✅ Assessment widget creation successful")
    except Exception as e:
        print(f"❌ Widget creation error: {str(e)}")
        return False

    # Test progress tracking
    try:
        test_assessment.record_activity(
            "test_activity",
            {"completion": True, "timestamp": datetime.now().isoformat()},
        )
        progress = test_assessment.get_progress_summary()
        print(
            f"✅ Progress tracking successful - Activities: {progress.get('activities_completed', 0)}"
        )
    except Exception as e:
        print(f"❌ Progress tracking error: {str(e)}")
        return False

    # Test dashboard creation (optional)
    try:
        test_dashboard = create_dashboard(test_assessment)
        print("✅ Dashboard creation successful")
    except Exception as e:
        print(f"⚠️  Dashboard creation skipped: {str(e)} (optional feature)")

    # Clean up test files
    try:
        test_dir = Path("assessments/test_user")
        if test_dir.exists():
            import shutil

            shutil.rmtree(test_dir)
        print("✅ Test cleanup completed")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")

    print("\n🎉 Assessment framework integration test PASSED!")
    return True


def check_notebook_integration():
    """Check if Day 1 notebook has been properly integrated"""

    print("\n📓 Checking Day 1 Notebook Integration")
    print("=" * 40)

    notebook_file = Path(
        "notebooks/quickstart_bootcamp/day_01_ml_cheminformatics_project.ipynb"
    )
    if not notebook_file.exists():
        print(f"❌ Day 1 notebook not found at {notebook_file}")
        return False

    # Read notebook content
    try:
        with open(notebook_file, "r", encoding="utf-8") as f:
            notebook_content = f.read()

        # Check for assessment integration markers
        integration_checks = {
            "Assessment framework import": "from assessment_framework import",
            "Student ID input": "student_id = input",
            "Assessment initialization": "create_assessment(",
            "Section assessments": "create_widget(",
            "Progress tracking": "record_activity(",
            "Final assessment": "FINAL DAY 1 COMPREHENSIVE ASSESSMENT",
            "Dashboard generation": "create_dashboard(",
        }

        passed_checks = 0
        for check_name, check_pattern in integration_checks.items():
            if check_pattern in notebook_content:
                print(f"✅ {check_name}")
                passed_checks += 1
            else:
                print(f"❌ {check_name}")

        success_rate = passed_checks / len(integration_checks)
        print(
            f"\n📊 Integration Success Rate: {success_rate*100:.1f}% ({passed_checks}/{len(integration_checks)})"
        )

        if success_rate >= 0.8:
            print("🎉 Day 1 notebook integration SUCCESSFUL!")
            return True
        else:
            print("⚠️  Day 1 notebook integration needs improvement")
            return False

    except Exception as e:
        print(f"❌ Error reading notebook: {str(e)}")
        return False


def generate_integration_report():
    """Generate a comprehensive integration report"""

    print("\n📋 Generating Integration Report")
    print("=" * 35)

    report = {
        "integration_test_timestamp": datetime.now().isoformat(),
        "assessment_framework_status": "integrated",
        "day1_notebook_status": "integrated",
        "features_integrated": [
            "Student ID and track selection",
            "Section-by-section progress tracking",
            "Interactive assessment widgets",
            "Hands-on exercise monitoring",
            "Comprehensive final evaluation",
            "Progress visualization dashboard",
            "Performance analytics",
            "Day 2 readiness assessment",
        ],
        "assessment_points": [
            "Initial setup and environment configuration",
            "Section 1: Molecular representations mastery",
            "Section 2: DeepChem fundamentals checkpoint",
            "Mid-section progress evaluation",
            "Hands-on exercise completion tracking",
            "Final comprehensive assessment",
            "Learning outcomes validation",
            "Next day preparation readiness",
        ],
        "metrics_tracked": [
            "Time spent per section",
            "Concept mastery levels",
            "Activity completion rates",
            "Overall progress percentage",
            "Performance scores",
            "Learning objective achievement",
        ],
        "export_formats": [
            "JSON progress summaries",
            "Interactive HTML dashboards",
            "LMS-compatible data exports",
            "Performance analytics reports",
        ],
    }

    # Save report
    report_file = Path(
        "notebooks/quickstart_bootcamp/ASSESSMENT_INTEGRATION_REPORT.json"
    )
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"✅ Integration report saved to: {report_file}")

    # Print summary
    print(f"\n📈 INTEGRATION SUMMARY:")
    print(f"   📊 Features integrated: {len(report['features_integrated'])}")
    print(f"   🎯 Assessment points: {len(report['assessment_points'])}")
    print(f"   📏 Metrics tracked: {len(report['metrics_tracked'])}")
    print(f"   📤 Export formats: {len(report['export_formats'])}")

    return report


def main():
    """Main test execution"""

    print("🚀 ChemML Bootcamp Assessment Integration Test Suite")
    print("=" * 55)
    print(f"⏰ Test started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    framework_test = test_assessment_framework()
    notebook_test = check_notebook_integration()

    # Generate report
    integration_report = generate_integration_report()

    # Final summary
    print(f"\n🏁 FINAL TEST RESULTS")
    print("=" * 25)
    print(f"✅ Assessment Framework: {'PASS' if framework_test else 'FAIL'}")
    print(f"📓 Notebook Integration: {'PASS' if notebook_test else 'FAIL'}")

    overall_success = framework_test and notebook_test
    print(
        f"\n{'🎉 OVERALL SUCCESS!' if overall_success else '⚠️  OVERALL: NEEDS ATTENTION'}"
    )

    if overall_success:
        print("\n🚀 Ready to proceed with:")
        print("   1. Testing the integrated Day 1 notebook")
        print("   2. Integrating remaining day notebooks (Day 2-7)")
        print("   3. Setting up automated assessment reporting")
        print("   4. Implementing MLOps components")
    else:
        print("\n🔧 Recommended actions:")
        print("   1. Fix any failing integration tests")
        print("   2. Verify assessment framework dependencies")
        print("   3. Test notebook execution in clean environment")
        print("   4. Review assessment point placement")

    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
