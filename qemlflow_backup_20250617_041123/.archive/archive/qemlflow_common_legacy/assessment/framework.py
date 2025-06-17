"""
Unified Assessment Framework for ChemML Scripts
==============================================

Provides a standardized assessment and progress tracking system
for all ChemML bootcamp scripts.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AssessmentQuestion:
    """A single assessment question."""

    question_id: str
    question_text: str
    question_type: str  # 'multiple_choice', 'numeric', 'text', 'code'
    options: List[str] = field(default_factory=list)
    correct_answer: Any = None
    points: int = 1
    explanation: str = ""


@dataclass
class AssessmentResult:
    """Result of an assessment question."""

    question_id: str
    student_answer: Any
    correct_answer: Any
    is_correct: bool
    points_earned: int
    points_possible: int
    time_taken: float = 0.0


@dataclass
class SectionAssessment:
    """Assessment for a complete section."""

    section_name: str
    questions: List[AssessmentQuestion] = field(default_factory=list)
    results: List[AssessmentResult] = field(default_factory=list)
    total_points: int = 0
    earned_points: int = 0
    completion_time: float = 0.0

    def add_question(self, question: AssessmentQuestion) -> None:
        """Add a question to this assessment."""
        self.questions.append(question)
        self.total_points += question.points

    def calculate_score(self) -> float:
        """Calculate the percentage score."""
        if self.total_points == 0:
            return 0.0
        return (self.earned_points / self.total_points) * 100


class AssessmentFramework:
    """
    Unified assessment framework for ChemML scripts.

    Provides standardized assessment capabilities including:
    - Progress tracking
    - Knowledge checks
    - Performance benchmarking
    - Report generation
    """

    def __init__(self, script_name: str, student_id: str, output_dir: Path):
        """
        Initialize assessment framework.

        Args:
            script_name: Name of the script being assessed
            student_id: Student identifier
            output_dir: Directory for output files
        """
        self.script_name = script_name
        self.student_id = student_id
        self.output_dir = output_dir
        self.logger = logging.getLogger(f"Assessment.{script_name}")

        # Assessment tracking
        self.section_assessments: Dict[str, SectionAssessment] = {}
        self.start_time = time.time()
        self.current_section: Optional[str] = None

        # Progress tracking
        self.progress: Dict[str, Any] = {
            "script_name": script_name,
            "student_id": student_id,
            "start_time": self.start_time,
            "sections_completed": [],
            "current_section": None,
            "overall_score": 0.0,
        }

    def start_section(self, section_name: str) -> None:
        """Start assessment for a new section."""
        self.current_section = section_name
        self.progress["current_section"] = section_name

        if section_name not in self.section_assessments:
            self.section_assessments[section_name] = SectionAssessment(
                section_name=section_name
            )

        self.logger.info("Started assessment for section: %s", section_name)

    def add_question(
        self,
        question_id: str,
        question_text: str,
        question_type: str = "multiple_choice",
        options: List[str] = None,
        correct_answer: Any = None,
        points: int = 1,
        explanation: str = "",
    ) -> None:
        """
        Add a question to the current section.

        Args:
            question_id: Unique identifier for the question
            question_text: The question text
            question_type: Type of question (multiple_choice, numeric, text, code)
            options: List of options for multiple choice questions
            correct_answer: The correct answer
            points: Points awarded for correct answer
            explanation: Explanation of the correct answer
        """
        if not self.current_section:
            raise ValueError("No active section. Call start_section() first.")

        question = AssessmentQuestion(
            question_id=question_id,
            question_text=question_text,
            question_type=question_type,
            options=options or [],
            correct_answer=correct_answer,
            points=points,
            explanation=explanation,
        )

        self.section_assessments[self.current_section].add_question(question)
        self.logger.info(
            "Added question %s to section %s", question_id, self.current_section
        )

    def record_answer(
        self, question_id: str, student_answer: Any, time_taken: float = 0.0
    ) -> bool:
        """
        Record a student's answer to a question.

        Args:
            question_id: ID of the question being answered
            student_answer: The student's answer
            time_taken: Time taken to answer (seconds)

        Returns:
            True if answer is correct
        """
        if not self.current_section:
            raise ValueError("No active section.")

        section_assessment = self.section_assessments[self.current_section]

        # Find the question
        question = None
        for q in section_assessment.questions:
            if q.question_id == question_id:
                question = q
                break

        if not question:
            raise ValueError(f"Question {question_id} not found in current section")

        # Evaluate answer
        is_correct = self._evaluate_answer(question, student_answer)
        points_earned = question.points if is_correct else 0

        # Record result
        result = AssessmentResult(
            question_id=question_id,
            student_answer=student_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            points_possible=question.points,
            time_taken=time_taken,
        )

        section_assessment.results.append(result)
        section_assessment.earned_points += points_earned

        self.logger.info(
            "Recorded answer for %s: %s (Correct: %s)",
            question_id,
            student_answer,
            is_correct,
        )

        return is_correct

    def _evaluate_answer(
        self, question: AssessmentQuestion, student_answer: Any
    ) -> bool:
        """Evaluate if a student's answer is correct."""
        if question.correct_answer is None:
            return True  # No wrong answer

        if question.question_type == "multiple_choice":
            return (
                str(student_answer).strip().lower()
                == str(question.correct_answer).strip().lower()
            )
        elif question.question_type == "numeric":
            try:
                # Allow for small floating point differences
                student_num = float(student_answer)
                correct_num = float(question.correct_answer)
                return abs(student_num - correct_num) < 1e-6
            except (ValueError, TypeError):
                return False
        elif question.question_type in ["text", "code"]:
            # Simple string comparison (could be enhanced with fuzzy matching)
            return (
                str(student_answer).strip().lower()
                == str(question.correct_answer).strip().lower()
            )
        else:
            return str(student_answer) == str(question.correct_answer)

    def complete_section(self) -> Dict[str, Any]:
        """
        Complete the current section and return results.

        Returns:
            Dictionary with section results
        """
        if not self.current_section:
            raise ValueError("No active section.")

        section_assessment = self.section_assessments[self.current_section]
        section_assessment.completion_time = time.time() - self.start_time

        # Update progress
        self.progress["sections_completed"].append(self.current_section)
        self.progress["current_section"] = None

        # Calculate results
        results = {
            "section_name": self.current_section,
            "total_questions": len(section_assessment.questions),
            "correct_answers": sum(
                1 for r in section_assessment.results if r.is_correct
            ),
            "total_points": section_assessment.total_points,
            "earned_points": section_assessment.earned_points,
            "score_percentage": section_assessment.calculate_score(),
            "completion_time": section_assessment.completion_time,
        }

        self.logger.info(
            "Completed section %s: %.1f%% (%d/%d points)",
            self.current_section,
            results["score_percentage"],
            results["earned_points"],
            results["total_points"],
        )

        self.current_section = None
        return results

    def get_overall_results(self) -> Dict[str, Any]:
        """Get overall assessment results."""
        total_points = sum(sa.total_points for sa in self.section_assessments.values())
        earned_points = sum(
            sa.earned_points for sa in self.section_assessments.values()
        )
        overall_score = (earned_points / total_points * 100) if total_points > 0 else 0

        total_time = time.time() - self.start_time

        results = {
            "script_name": self.script_name,
            "student_id": self.student_id,
            "total_sections": len(self.section_assessments),
            "completed_sections": len(self.progress["sections_completed"]),
            "total_questions": sum(
                len(sa.questions) for sa in self.section_assessments.values()
            ),
            "total_points": total_points,
            "earned_points": earned_points,
            "overall_score": overall_score,
            "total_time": total_time,
            "section_results": {
                name: {
                    "score": sa.calculate_score(),
                    "points": f"{sa.earned_points}/{sa.total_points}",
                    "questions": len(sa.questions),
                }
                for name, sa in self.section_assessments.items()
            },
        }

        self.progress["overall_score"] = overall_score

        return results

    def generate_report(self) -> None:
        """Generate and save assessment report."""
        results = self.get_overall_results()

        # Console report
        print("\n📊 Assessment Report")
        print("=" * 60)
        print(f"Script: {results['script_name']}")
        print(f"Student: {results['student_id']}")
        print(f"Overall Score: {results['overall_score']:.1f}%")
        print(f"Total Points: {results['earned_points']}/{results['total_points']}")
        print(f"Completion Time: {results['total_time']:.1f}s")

        print("\n📋 Section Breakdown:")
        for section_name, section_results in results["section_results"].items():
            print(
                f"  {section_name}: {section_results['score']:.1f}% "
                f"({section_results['points']} points, "
                f"{section_results['questions']} questions)"
            )

        print("=" * 60)

        # Save detailed JSON report
        report_file = self.output_dir / f"{self.script_name}_assessment_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info("Assessment report saved to: %s", report_file)

    def create_quick_assessment(
        self, section_name: str, questions_data: List[Dict[str, Any]]
    ) -> None:
        """
        Create a quick assessment for a section.

        Args:
            section_name: Name of the section
            questions_data: List of question dictionaries
        """
        self.start_section(section_name)

        for q_data in questions_data:
            self.add_question(
                question_id=q_data.get(
                    "id", f"q_{len(self.section_assessments[section_name].questions)}"
                ),
                question_text=q_data["text"],
                question_type=q_data.get("type", "multiple_choice"),
                options=q_data.get("options", []),
                correct_answer=q_data.get("answer"),
                points=q_data.get("points", 1),
                explanation=q_data.get("explanation", ""),
            )

    def auto_answer_demo_questions(self) -> None:
        """Automatically answer questions with correct answers for demo purposes."""
        if not self.current_section:
            return

        section_assessment = self.section_assessments[self.current_section]

        for question in section_assessment.questions:
            if question.correct_answer is not None:
                self.record_answer(question.question_id, question.correct_answer)
            else:
                # Provide a default answer based on question type
                if question.question_type == "multiple_choice" and question.options:
                    self.record_answer(question.question_id, question.options[0])
                elif question.question_type == "numeric":
                    self.record_answer(
                        question.question_id, 42
                    )  # Default numeric answer
                else:
                    self.record_answer(question.question_id, "Demo answer")

        self.logger.info(
            "Auto-answered all questions in section %s", self.current_section
        )
