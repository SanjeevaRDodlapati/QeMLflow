from typing import Dict\nfrom typing import List\nfrom typing import Optional\n"""
QeMLflow Tutorial Framework - Assessment Module
=============================================

Provides learning assessment, progress tracking, and educational evaluation tools.
This module standardizes how learning progress is measured and tracked across all tutorials.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LearningActivity:
    """Represents a single learning activity."""

    activity_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    result: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    understanding_score: Optional[float] = None
    completion_status: str = "in_progress"

    @property
    def duration(self) -> Optional[timedelta]:
        """Get the duration of the activity."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None


@dataclass
class ConceptCheckpoint:
    """Represents a concept understanding checkpoint."""

    concept_name: str
    timestamp: datetime
    understanding_level: float  # 0.0 to 1.0
    confidence_level: float  # 0.0 to 1.0
    notes: str = ""
    practice_needed: bool = False


class LearningAssessment:
    """
    Main assessment class for tracking student learning progress.

    This class provides a standardized way to track learning activities,
    assess understanding, and generate progress reports.
    """

    def __init__(
        self,
        student_id: str,
        section: str = "general",
        track: str = "standard",
        save_dir: Optional[str] = None,
    ):
        """
        Initialize learning assessment.

        Args:
            student_id: Unique identifier for the student
            section: Section or topic being assessed
            track: Learning track (quick, standard, intensive, extended)
            save_dir: Directory to save assessment data
        """
        self.student_id = student_id
        self.section = section
        self.track = track
        self.start_time = datetime.now()

        # Initialize tracking data
        self.activities: List[LearningActivity] = []
        self.checkpoints: List[ConceptCheckpoint] = []
        self.current_activity: Optional[LearningActivity] = None

        # Configuration
        self.track_configs = {
            "quick": {"target_hours": 3, "min_completion": 0.7},
            "standard": {"target_hours": 4.5, "min_completion": 0.8},
            "intensive": {"target_hours": 6, "min_completion": 0.9},
            "extended": {"target_hours": 8, "min_completion": 0.95},
        }

        # Setup save directory
        self.save_dir = Path(save_dir) if save_dir else Path("tutorial_progress")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Learning assessment initialized for {student_id} - {section} ({track})"
        )

    def start_section(self, section_name: str) -> None:
        """Start a new learning section."""
        self.end_current_activity()
        logger.info(f"Starting section: {section_name}")

        activity = LearningActivity(
            activity_name=f"Section: {section_name}", start_time=datetime.now()
        )
        self.current_activity = activity
        self.activities.append(activity)

    def end_section(
        self, section_name: str, understanding_score: Optional[float] = None
    ) -> None:
        """End the current learning section."""
        if self.current_activity:
            self.current_activity.end_time = datetime.now()
            self.current_activity.completion_status = "completed"
            if understanding_score is not None:
                self.current_activity.understanding_score = understanding_score

            logger.info(f"Completed section: {section_name}")

        self.current_activity = None

    def record_activity(
        self,
        activity: str,
        result: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        understanding_score: Optional[float] = None,
    ) -> None:
        """
        Record a learning activity.

        Args:
            activity: Name of the activity
            result: Result of the activity (success, partial, failure)
            metadata: Additional metadata about the activity
            understanding_score: Self-assessed understanding (0.0 to 1.0)
        """
        activity_record = LearningActivity(
            activity_name=activity,
            start_time=datetime.now(),
            end_time=datetime.now(),
            result=result,
            metadata=metadata or {},
            understanding_score=understanding_score,
            completion_status="completed",
        )

        self.activities.append(activity_record)
        logger.debug(f"Recorded activity: {activity} - {result}")

    def add_concept_checkpoint(
        self,
        concept: str,
        understanding: float,
        confidence: float,
        notes: str = "",
        practice_needed: bool = False,
    ) -> None:
        """
        Add a concept understanding checkpoint.

        Args:
            concept: Name of the concept
            understanding: Understanding level (0.0 to 1.0)
            confidence: Confidence level (0.0 to 1.0)
            notes: Additional notes
            practice_needed: Whether additional practice is needed
        """
        checkpoint = ConceptCheckpoint(
            concept_name=concept,
            timestamp=datetime.now(),
            understanding_level=understanding,
            confidence_level=confidence,
            notes=notes,
            practice_needed=practice_needed,
        )

        self.checkpoints.append(checkpoint)
        logger.debug(f"Added concept checkpoint: {concept} - {understanding:.2f}")

    def end_current_activity(self) -> None:
        """End the current activity if one is in progress."""
        if self.current_activity:
            self.current_activity.end_time = datetime.now()
            self.current_activity.completion_status = "completed"

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of learning progress."""
        total_time = sum(
            (a.duration.total_seconds() / 3600) for a in self.activities if a.duration
        )

        completed_activities = [
            a for a in self.activities if a.completion_status == "completed"
        ]
        understanding_scores = [
            a.understanding_score
            for a in completed_activities
            if a.understanding_score is not None
        ]

        return {
            "student_id": self.student_id,
            "section": self.section,
            "track": self.track,
            "total_time_hours": total_time,
            "activities_completed": len(completed_activities),
            "total_activities": len(self.activities),
            "overall_score": (
                np.mean(understanding_scores) if understanding_scores else None
            ),
            "checkpoints_recorded": len(self.checkpoints),
            "start_time": self.start_time.isoformat(),
            "last_activity": datetime.now().isoformat(),
        }

    def save_progress(self, filename: Optional[str] = None) -> Path:
        """Save progress data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.student_id}_{self.section}_{timestamp}.json"

        save_path = self.save_dir / filename

        # Prepare data for serialization
        data = {
            "assessment_info": self.get_progress_summary(),
            "activities": [
                {
                    "activity_name": a.activity_name,
                    "start_time": a.start_time.isoformat(),
                    "end_time": a.end_time.isoformat() if a.end_time else None,
                    "result": a.result,
                    "metadata": a.metadata,
                    "understanding_score": a.understanding_score,
                    "completion_status": a.completion_status,
                }
                for a in self.activities
            ],
            "checkpoints": [
                {
                    "concept_name": c.concept_name,
                    "timestamp": c.timestamp.isoformat(),
                    "understanding_level": c.understanding_level,
                    "confidence_level": c.confidence_level,
                    "notes": c.notes,
                    "practice_needed": c.practice_needed,
                }
                for c in self.checkpoints
            ],
        }

        with open(save_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Progress saved to: {save_path}")
        return save_path


class ProgressTracker:
    """
    Utility class for tracking progress within a learning session.

    This class provides simple methods for logging progress and generating
    visual feedback during tutorial execution.
    """

    def __init__(self, assessment: Optional[LearningAssessment] = None):
        """
        Initialize progress tracker.

        Args:
            assessment: Optional LearningAssessment instance to integrate with
        """
        self.assessment = assessment
        self.session_start = datetime.now()
        self.activities_log: List[Dict[str, Any]] = []

    def log_progress(
        self,
        step: str,
        status: str = "completed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a progress step.

        Args:
            step: Description of the step
            status: Status of the step (completed, failed, in_progress)
            details: Additional details about the step
        """
        timestamp = datetime.now()

        log_entry = {
            "step": step,
            "status": status,
            "timestamp": timestamp.isoformat(),
            "details": details or {},
        }

        self.activities_log.append(log_entry)

        # Log to assessment if available
        if self.assessment:
            self.assessment.record_activity(
                activity=step, result=status, metadata=details
            )

        # Print progress
        status_symbol = {
            "completed": "✅",
            "failed": "❌",
            "in_progress": "⏳",
            "warning": "⚠️",
        }.get(status, "📝")

        print(f"{status_symbol} {step}")

        if details:
            for key, value in details.items():
                print(f"   └─ {key}: {value}")

    def get_session_summary(self) -> Dict[str, Any]:
        """Get a summary of the current session."""
        elapsed_time = datetime.now() - self.session_start

        status_counts = {}
        for entry in self.activities_log:
            status = entry["status"]
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "session_start": self.session_start.isoformat(),
            "elapsed_time_minutes": elapsed_time.total_seconds() / 60,
            "total_steps": len(self.activities_log),
            "status_counts": status_counts,
            "latest_activity": self.activities_log[-1] if self.activities_log else None,
        }

    def print_summary(self) -> None:
        """Print a session summary."""
        summary = self.get_session_summary()

        print("\n" + "=" * 50)
        print("📊 SESSION SUMMARY")
        print("=" * 50)
        print(f"⏱️  Session time: {summary['elapsed_time_minutes']:.1f} minutes")
        print(f"📝 Total steps: {summary['total_steps']}")

        for status, count in summary["status_counts"].items():
            symbol = {
                "completed": "✅",
                "failed": "❌",
                "in_progress": "⏳",
                "warning": "⚠️",
            }.get(status, "📝")
            print(f"{symbol} {status.title()}: {count}")

        print("=" * 50)


# Backwards compatibility aliases
AssessmentFramework = LearningAssessment
create_assessment = LearningAssessment
