"""
Base Runner Classes for ChemML Scripts
=====================================

Provides abstract base classes for creating modular, testable ChemML scripts
that follow the Single Responsibility Principle and are easy to maintain.
"""

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from ..config.environment import ChemMLConfig


@dataclass
class SectionResult:
    """
    Result of executing a single section of a ChemML script.

    Attributes:
        section_name: Name of the executed section
        success: Whether execution was successful
        execution_time: Time taken to execute (seconds)
        outputs: Dictionary of outputs produced
        errors: List of any errors encountered
        warnings: List of any warnings generated
        metadata: Additional metadata about execution
    """

    section_name: str
    success: bool = True
    execution_time: float = 0.0
    outputs: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize empty collections if None."""
        if self.outputs is None:
            self.outputs = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionSummary:
    """
    Summary of an entire script execution.

    Attributes:
        script_name: Name of the executed script
        total_sections: Total number of sections
        successful_sections: Number of successfully executed sections
        total_execution_time: Total time for all sections
        section_results: List of individual section results
        overall_success: Whether the entire execution was successful
    """

    script_name: str
    total_sections: int = 0
    successful_sections: int = 0
    total_execution_time: float = 0.0
    section_results: List[SectionResult] = None
    overall_success: bool = True

    def __post_init__(self):
        """Initialize section results if None."""
        if self.section_results is None:
            self.section_results = []


class SectionRunner(ABC):
    """
    Abstract base class for individual section runners.

    Each section of a ChemML script should inherit from this class,
    implementing the execute method with specific logic.
    """

    def __init__(self, config: ChemMLConfig, section_name: str):
        """
        Initialize section runner.

        Args:
            config: ChemML configuration
            section_name: Name of this section
        """
        self.config = config
        self.section_name = section_name
        self.logger = logging.getLogger(f"{config.script_name}.{section_name}")

    @abstractmethod
    def execute(self) -> SectionResult:
        """
        Execute this section's logic.

        Returns:
            SectionResult containing execution details and outputs
        """
        pass

    def _create_result(
        self,
        success: bool = True,
        outputs: Dict[str, Any] = None,
        errors: List[str] = None,
        warnings: List[str] = None,
        **metadata,
    ) -> SectionResult:
        """
        Helper method to create a SectionResult.

        Args:
            success: Whether execution was successful
            outputs: Dictionary of outputs
            errors: List of error messages
            warnings: List of warning messages
            **metadata: Additional metadata

        Returns:
            Configured SectionResult
        """
        return SectionResult(
            section_name=self.section_name,
            success=success,
            outputs=outputs or {},
            errors=errors or [],
            warnings=warnings or [],
            metadata=metadata,
        )


class BaseRunner(ABC):
    """
    Abstract base class for ChemML script runners.

    Provides common infrastructure for executing multiple sections,
    handling errors, and generating reports.
    """

    def __init__(self, config: ChemMLConfig):
        """
        Initialize base runner.

        Args:
            config: ChemML configuration
        """
        self.config = config
        self.logger = logging.getLogger(config.script_name)
        self.section_runners: Dict[str, SectionRunner] = {}
        self.execution_summary = ExecutionSummary(script_name=config.script_name)

    def register_section(self, section_runner: SectionRunner) -> None:
        """
        Register a section runner.

        Args:
            section_runner: Section runner to register
        """
        self.section_runners[section_runner.section_name] = section_runner
        self.logger.info(f"Registered section: {section_runner.section_name}")

    def execute_section(self, section_name: str) -> SectionResult:
        """
        Execute a specific section.

        Args:
            section_name: Name of section to execute

        Returns:
            SectionResult with execution details
        """
        if section_name not in self.section_runners:
            error_msg = f"Section '{section_name}' not registered"
            self.logger.error(error_msg)
            return SectionResult(
                section_name=section_name, success=False, errors=[error_msg]
            )

        runner = self.section_runners[section_name]

        self.logger.info(f"🚀 Starting section: {section_name}")
        start_time = time.time()

        try:
            result = runner.execute()
            result.execution_time = time.time() - start_time

            if result.success:
                self.logger.info(
                    f"✅ Section '{section_name}' completed successfully in {result.execution_time:.2f}s"
                )
            else:
                self.logger.warning(
                    f"⚠️ Section '{section_name}' completed with issues in {result.execution_time:.2f}s"
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Section '{section_name}' failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())

            result = SectionResult(
                section_name=section_name,
                success=False,
                execution_time=execution_time,
                errors=[error_msg],
            )

        # Update execution summary
        self.execution_summary.section_results.append(result)
        self.execution_summary.total_sections += 1
        self.execution_summary.total_execution_time += result.execution_time

        if result.success:
            self.execution_summary.successful_sections += 1
        else:
            self.execution_summary.overall_success = False

            # Check if we should continue on error
            if not self.config.force_continue:
                self.logger.error(
                    f"Stopping execution due to error in section '{section_name}'"
                )
                self.logger.error(
                    "Set CHEMML_FORCE_CONTINUE=true to continue on errors"
                )
                raise RuntimeError(
                    f"Section '{section_name}' failed and force_continue is disabled"
                )

        return result

    def execute_all_sections(self) -> ExecutionSummary:
        """
        Execute all registered sections in order.

        Returns:
            ExecutionSummary with overall execution details
        """
        self.logger.info(
            f"🎯 Starting execution of {len(self.section_runners)} sections"
        )

        for section_name in self.section_runners.keys():
            self.execute_section(section_name)

        # Generate final report
        self.generate_report()

        return self.execution_summary

    def generate_report(self) -> None:
        """Generate and save execution report."""
        summary = self.execution_summary

        # Console report
        print("\n" + "=" * 80)
        print(f"📊 Execution Summary: {summary.script_name}")
        print("=" * 80)
        print(f"Total Sections: {summary.total_sections}")
        print(f"Successful: {summary.successful_sections}")
        print(f"Failed: {summary.total_sections - summary.successful_sections}")
        print(f"Total Time: {summary.total_execution_time:.2f}s")
        print(f"Overall Success: {'✅' if summary.overall_success else '❌'}")

        print("\n📋 Section Details:")
        for result in summary.section_results:
            status = "✅" if result.success else "❌"
            print(f"  {status} {result.section_name}: {result.execution_time:.2f}s")

            if result.warnings:
                for warning in result.warnings:
                    print(f"    ⚠️ {warning}")

            if result.errors:
                for error in result.errors:
                    print(f"    ❌ {error}")

        print("=" * 80)

        # Save detailed report to file
        report_file = self.config.get_output_path(
            f"{self.config.script_name}_report.txt"
        )
        with open(report_file, "w") as f:
            f.write(f"ChemML Execution Report: {summary.script_name}\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Configuration:\n")
            for key, value in self.config.to_dict().items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nExecution Summary:\n")
            f.write(f"  Total Sections: {summary.total_sections}\n")
            f.write(f"  Successful: {summary.successful_sections}\n")
            f.write(
                f"  Failed: {summary.total_sections - summary.successful_sections}\n"
            )
            f.write(f"  Total Time: {summary.total_execution_time:.2f}s\n")
            f.write(f"  Overall Success: {summary.overall_success}\n\n")

            f.write("Section Details:\n")
            for result in summary.section_results:
                f.write(f"\n  Section: {result.section_name}\n")
                f.write(f"    Success: {result.success}\n")
                f.write(f"    Execution Time: {result.execution_time:.2f}s\n")
                f.write(f"    Outputs: {len(result.outputs)} items\n")
                f.write(f"    Warnings: {len(result.warnings)}\n")
                f.write(f"    Errors: {len(result.errors)}\n")

                if result.warnings:
                    f.write("    Warning Details:\n")
                    for warning in result.warnings:
                        f.write(f"      - {warning}\n")

                if result.errors:
                    f.write("    Error Details:\n")
                    for error in result.errors:
                        f.write(f"      - {error}\n")

        self.logger.info(f"📄 Detailed report saved to: {report_file}")

    @abstractmethod
    def setup_sections(self) -> None:
        """
        Setup and register all sections for this script.

        This method should create and register all SectionRunner instances
        that this script needs to execute.
        """
        pass
