"""
Environment Configuration Management for ChemML Scripts
======================================================

Unified configuration management system that eliminates code duplication
across all ChemML scripts and provides type-safe configuration handling.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class TrackType(Enum):
    """Available learning tracks for ChemML bootcamp."""

    FAST = "fast"
    COMPLETE = "complete"
    FLEXIBLE = "flexible"


class LogLevel(Enum):
    """Supported logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class ChemMLConfig:
    """
    Unified configuration for all ChemML scripts.

    Attributes:
        student_id: Unique student identifier
        track: Learning track (fast/complete/flexible)
        force_continue: Continue execution on non-critical errors
        output_dir: Directory for output files
        log_level: Logging verbosity level
        day: Which day's script is being executed (1-7)
        script_name: Name of the executing script
        extra_params: Additional script-specific parameters
    """

    student_id: str = "student_001"
    track: TrackType = TrackType.COMPLETE
    force_continue: bool = False
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    log_level: LogLevel = LogLevel.INFO
    day: int = 1
    script_name: str = "unknown"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the current script."""
        log_file = self.output_dir / f"day_{self.day:02d}_{self.script_name}.log"

        # Clear any existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Setup new logging configuration
        logging.basicConfig(
            level=getattr(logging, self.log_level.value),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured for {self.script_name}")
        logger.info(f"Log file: {log_file}")

    def get_output_path(self, filename: str) -> Path:
        """Get full path for an output file."""
        return self.output_dir / filename

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "student_id": self.student_id,
            "track": self.track.value,
            "force_continue": self.force_continue,
            "output_dir": str(self.output_dir),
            "log_level": self.log_level.value,
            "day": self.day,
            "script_name": self.script_name,
            "extra_params": self.extra_params,
        }


def get_config(day: int, script_name: str, **kwargs) -> ChemMLConfig:
    """
    Create ChemMLConfig from environment variables and parameters.

    Args:
        day: Day number (1-7)
        script_name: Name of the executing script
        **kwargs: Additional parameters to override defaults

    Returns:
        Configured ChemMLConfig instance

    Environment Variables:
        CHEMML_STUDENT_ID: Student identifier
        CHEMML_TRACK: Learning track (fast/complete/flexible)
        CHEMML_FORCE_CONTINUE: Continue on errors (true/false)
        CHEMML_OUTPUT_DIR: Output directory path
        CHEMML_LOG_LEVEL: Logging level (DEBUG/INFO/WARNING/ERROR)
    """
    # Parse environment variables with defaults
    student_id = os.getenv("CHEMML_STUDENT_ID", f"student_{day:03d}")

    track_str = os.getenv("CHEMML_TRACK", "complete").lower()
    try:
        track = TrackType(track_str)
    except ValueError:
        track = TrackType.COMPLETE

    force_continue = os.getenv("CHEMML_FORCE_CONTINUE", "false").lower() == "true"

    output_dir = Path(os.getenv("CHEMML_OUTPUT_DIR", f"./day_{day:02d}_outputs"))

    log_level_str = os.getenv("CHEMML_LOG_LEVEL", "INFO").upper()
    try:
        log_level = LogLevel(log_level_str)
    except ValueError:
        log_level = LogLevel.INFO

    # Merge with provided kwargs
    config_params = {
        "student_id": student_id,
        "track": track,
        "force_continue": force_continue,
        "output_dir": output_dir,
        "log_level": log_level,
        "day": day,
        "script_name": script_name,
        **kwargs,
    }

    return ChemMLConfig(**config_params)


def print_banner(config: ChemMLConfig, description: str = "") -> None:
    """
    Print a standardized banner for ChemML scripts.

    Args:
        config: ChemML configuration
        description: Optional description of the script
    """
    banner_width = 80

    print("=" * banner_width)
    print(f"🧪 ChemML Bootcamp - Day {config.day}: {config.script_name}")
    if description:
        print(f"📝 {description}")
    print("=" * banner_width)
    print(f"👤 Student ID: {config.student_id}")
    print(f"🎯 Track: {config.track.value.title()}")
    print(f"📁 Output Directory: {config.output_dir}")
    print(f"📊 Log Level: {config.log_level.value}")
    if config.force_continue:
        print("⚠️  Force Continue: Enabled")
    print("=" * banner_width)
