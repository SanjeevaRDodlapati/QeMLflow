"""
ChemML Bootcamp Assessment Framework
Interactive assessment system for tracking learning progress
"""

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, clear_output, display
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


class BootcampAssessment:
    """
    Comprehensive assessment system for ChemML Bootcamp
    Tracks progress, provides feedback, and generates reports
    """

    def __init__(self, student_id: str, day: int, track: str = "standard"):
        self.student_id = student_id
        self.day = day
        self.track = track
        self.start_time = datetime.now()
        self.assessments = []
        self.session_data = {}

        # Create assessment directory
        self.assessment_dir = Path(f"assessments/{student_id}")
        self.assessment_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data if available
        self._load_existing_data()

        # Track configuration
        self.track_configs = {
            "quick": {"target_hours": 3, "min_completion": 0.7, "focus": "core"},
            "standard": {
                "target_hours": 4.5,
                "min_completion": 0.8,
                "focus": "comprehensive",
            },
            "intensive": {
                "target_hours": 6,
                "min_completion": 0.9,
                "focus": "advanced",
            },
            "extended": {"target_hours": 3, "min_completion": 0.8, "focus": "deep"},
        }

    def _load_existing_data(self):
        """Load existing assessment data"""
        data_file = self.assessment_dir / f"day_{self.day}_assessments.json"
        if data_file.exists():
            with open(data_file, "r") as f:
                existing_data = json.load(f)
                self.assessments = existing_data.get("assessments", [])
                self.session_data = existing_data.get("session_data", {})

    def save_progress(self):
        """Save current progress to file"""
        data = {
            "student_id": self.student_id,
            "day": self.day,
            "track": self.track,
            "start_time": self.start_time.isoformat(),
            "last_updated": datetime.now().isoformat(),
            "assessments": self.assessments,
            "session_data": self.session_data,
        }

        data_file = self.assessment_dir / f"day_{self.day}_assessments.json"
        with open(data_file, "w") as f:
            json.dump(data, f, indent=2)

    def add_checkpoint(
        self,
        section: str,
        task: str,
        status: str,
        code_quality: int,
        understanding: int,
        time_spent: float,
        notes: str = "",
        code_snippet: str = "",
    ):
        """Add assessment checkpoint"""
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "section": section,
            "task": task,
            "status": status,  # 'completed', 'partial', 'struggling', 'skipped'
            "code_quality": code_quality,  # 1-5 scale
            "understanding": understanding,  # 1-5 scale
            "time_spent_minutes": time_spent,
            "notes": notes,
            "code_snippet": code_snippet[:500] if code_snippet else "",  # Limit size
        }
        self.assessments.append(checkpoint)
        self.save_progress()
        return checkpoint

    def calculate_day_score(self) -> Dict[str, float]:
        """Calculate comprehensive day performance metrics"""
        if not self.assessments:
            return {
                "overall_score": 0.0,
                "completion_rate": 0.0,
                "code_quality_avg": 0.0,
                "understanding_avg": 0.0,
                "efficiency_score": 0.0,
                "recommendation": "Start completing assessments",
            }

        # Basic metrics
        total_assessments = len(self.assessments)
        completed = len([a for a in self.assessments if a["status"] == "completed"])
        partial = len([a for a in self.assessments if a["status"] == "partial"])
        struggling = len([a for a in self.assessments if a["status"] == "struggling"])

        # Weighted completion rate
        completion_rate = (
            completed + 0.5 * partial + 0.2 * struggling
        ) / total_assessments

        # Quality metrics
        code_quality_avg = np.mean([a["code_quality"] for a in self.assessments])
        understanding_avg = np.mean([a["understanding"] for a in self.assessments])

        # Time efficiency
        total_time = sum([a["time_spent_minutes"] for a in self.assessments])
        target_time = self.track_configs[self.track]["target_hours"] * 60
        efficiency_score = min(1.0, target_time / max(total_time, 1))

        # Overall score calculation
        overall_score = (
            completion_rate * 0.4
            + (code_quality_avg / 5) * 0.3
            + (understanding_avg / 5) * 0.2
            + efficiency_score * 0.1
        )

        # Generate recommendation
        recommendation = self._generate_recommendation(
            completion_rate, code_quality_avg, understanding_avg, efficiency_score
        )

        return {
            "overall_score": round(overall_score, 3),
            "completion_rate": round(completion_rate, 3),
            "code_quality_avg": round(code_quality_avg, 2),
            "understanding_avg": round(understanding_avg, 2),
            "efficiency_score": round(efficiency_score, 3),
            "total_time_hours": round(total_time / 60, 2),
            "recommendation": recommendation,
        }

    def _generate_recommendation(
        self, completion: float, quality: float, understanding: float, efficiency: float
    ) -> str:
        """Generate personalized recommendation"""
        if completion < 0.5:
            return "📚 Focus on completing more exercises. Consider switching to a slower track."
        elif understanding < 3.0:
            return "🤔 Review concepts before continuing. Consider additional study materials."
        elif quality < 3.0:
            return (
                "💻 Spend more time on code quality. Review best practices and examples."
            )
        elif efficiency < 0.3:
            return (
                "⏱️ Try to work more efficiently. Consider time management techniques."
            )
        elif completion > 0.9 and quality > 4.0 and understanding > 4.0:
            return "🚀 Excellent work! Consider advanced challenges or helping other students."
        else:
            return (
                "✅ Good progress! Keep up the current pace and focus on understanding."
            )

    def create_progress_visualization(self) -> go.Figure:
        """Create comprehensive progress visualization"""
        if not self.assessments:
            # Return empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No assessment data available yet.<br>Complete some checkpoints to see progress!",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(title="ChemML Bootcamp Progress Dashboard")
            return fig

        # Prepare data
        df = pd.DataFrame(self.assessments)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["cumulative_assessments"] = range(1, len(df) + 1)

        # Calculate rolling averages
        df["rolling_quality"] = (
            df["code_quality"].rolling(window=3, min_periods=1).mean()
        )
        df["rolling_understanding"] = (
            df["understanding"].rolling(window=3, min_periods=1).mean()
        )

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Progress Over Time",
                "Quality Metrics",
                "Status Distribution",
                "Time Efficiency",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"type": "pie"}, {"secondary_y": False}],
            ],
        )

        # Progress line chart
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["cumulative_assessments"],
                mode="lines+markers",
                name="Cumulative Progress",
                line=dict(color="#1f77b4", width=3),
            ),
            row=1,
            col=1,
        )

        # Quality metrics
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_quality"],
                mode="lines+markers",
                name="Code Quality",
                line=dict(color="#ff7f0e"),
            ),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["rolling_understanding"],
                mode="lines+markers",
                name="Understanding",
                line=dict(color="#2ca02c"),
            ),
            row=1,
            col=2,
        )

        # Status distribution pie chart
        status_counts = df["status"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=status_counts.index,
                values=status_counts.values,
                name="Status Distribution",
            ),
            row=2,
            col=1,
        )

        # Time efficiency bar chart
        section_times = df.groupby("section")["time_spent_minutes"].sum()
        fig.add_trace(
            go.Bar(
                x=section_times.index,
                y=section_times.values,
                name="Time per Section",
                marker_color="#d62728",
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"Day {self.day} Assessment Dashboard - {self.track.title()} Track",
            height=600,
            showlegend=True,
        )

        # Update y-axis labels
        fig.update_yaxes(title_text="Assessments Completed", row=1, col=1)
        fig.update_yaxes(title_text="Score (1-5)", row=1, col=2)
        fig.update_yaxes(title_text="Minutes", row=2, col=2)
        fig.update_xaxes(title_text="Assessment #", row=1, col=1)
        fig.update_xaxes(title_text="Assessment #", row=1, col=2)
        fig.update_xaxes(title_text="Section", row=2, col=2)

        return fig

    def record_activity(self, activity_name: str, activity_data: Dict[str, Any]):
        """Record a learning activity for progress tracking"""
        activity_record = {
            "timestamp": datetime.now().isoformat(),
            "activity_name": activity_name,
            "data": activity_data,
        }

        if "activities" not in self.session_data:
            self.session_data["activities"] = []

        self.session_data["activities"].append(activity_record)
        self.save_progress()

    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of current progress"""
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds() / 60  # minutes

        activities_completed = len(self.session_data.get("activities", []))
        concepts_completed = len(
            [a for a in self.assessments if a.get("status") == "completed"]
        )

        total_assessments = len(self.assessments)
        completion_rate = concepts_completed / max(total_assessments, 1)

        return {
            "elapsed_time": elapsed_time,
            "activities_completed": activities_completed,
            "concepts_completed": concepts_completed,
            "total_assessments": total_assessments,
            "completion_rate": completion_rate,
            "track": self.track,
            "day": self.day,
        }

    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive progress report"""
        progress = self.get_progress_summary()
        day_score = self.calculate_day_score()

        return {
            "total_time": progress["elapsed_time"],
            "total_concepts": progress["concepts_completed"],
            "total_activities": progress["activities_completed"],
            "overall_completion": progress["completion_rate"],
            "performance_score": day_score.get("overall_score", 0) * 20,  # Scale to 100
            "code_quality_avg": day_score.get("code_quality_avg", 0),
            "understanding_avg": day_score.get("understanding_avg", 0),
        }

    def save_final_report(self):
        """Save final comprehensive report"""
        report = self.get_comprehensive_report()
        report["completion_timestamp"] = datetime.now().isoformat()

        report_file = self.assessment_dir / f"day_{self.day}_final_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)


class AssessmentWidget:
    """
    Interactive assessment widget for Jupyter notebooks
    Provides user-friendly interface for self-assessment
    """

    def __init__(
        self,
        assessment_system: BootcampAssessment,
        section: str,
        task: str,
        expected_time: int = 30,
    ):
        self.assessment = assessment_system
        self.section = section
        self.task = task
        self.expected_time = expected_time
        self.start_time = datetime.now()
        self.widget_created = False

    def create_widget(self):
        """Create interactive assessment form"""
        if self.widget_created:
            return  # Prevent duplicate widgets

        # Header
        header = widgets.HTML(
            value=f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        color: white; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                <h3 style="margin: 0;">📊 Assessment Checkpoint</h3>
                <p style="margin: 5px 0 0 0;">
                    <strong>Section:</strong> {self.section} |
                    <strong>Task:</strong> {self.task} |
                    <strong>Expected Time:</strong> ~{self.expected_time} minutes
                </p>
            </div>
            """
        )

        # Status selection with descriptions
        self.status = widgets.RadioButtons(
            options=[
                (
                    "✅ Completed Successfully - I understood and implemented everything",
                    "completed",
                ),
                (
                    "🔄 Partially Completed - I got most of it but struggled with some parts",
                    "partial",
                ),
                ("😅 Struggling - I need help or more time to understand", "struggling"),
                ("⏭️ Skipped - I moved on without completing this", "skipped"),
            ],
            description="",
            style={"description_width": "initial"},
            layout=widgets.Layout(margin="10px 0"),
        )

        # Code quality with visual scale
        self.code_quality = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Code Quality:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )

        quality_labels = widgets.HTML(
            value="""
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: -5px;">
                <span>1: Broken</span>
                <span>2: Basic</span>
                <span>3: Good</span>
                <span>4: Excellent</span>
                <span>5: Production-ready</span>
            </div>
            """
        )

        # Understanding with visual scale
        self.understanding = widgets.IntSlider(
            value=3,
            min=1,
            max=5,
            step=1,
            description="Understanding:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="100%"),
        )

        understanding_labels = widgets.HTML(
            value="""
            <div style="display: flex; justify-content: space-between; font-size: 11px; margin-top: -5px;">
                <span>1: Confused</span>
                <span>2: Unclear</span>
                <span>3: Clear</span>
                <span>4: Deep</span>
                <span>5: Expert</span>
            </div>
            """
        )

        # Time tracking
        self.time_spent = widgets.IntText(
            value=self.expected_time,
            description="Time (mins):",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="200px"),
        )

        # Notes section
        self.notes = widgets.Textarea(
            value="",
            placeholder="Optional: Share your thoughts, questions, or challenges...",
            description="Notes:",
            layout=widgets.Layout(width="100%", height="80px"),
            style={"description_width": "initial"},
        )

        # Code snippet (optional)
        self.code_snippet = widgets.Textarea(
            value="",
            placeholder="Optional: Paste a code snippet you're proud of or need help with...",
            description="Code:",
            layout=widgets.Layout(width="100%", height="60px"),
            style={"description_width": "initial"},
        )

        # Submit button
        self.submit_btn = widgets.Button(
            description="Submit Assessment",
            button_style="primary",
            icon="check",
            layout=widgets.Layout(width="200px", height="40px"),
        )

        # Progress indicator
        self.progress_output = widgets.Output()

        # Event handlers
        self.submit_btn.on_click(self._submit_assessment)
        self.status.observe(self._update_time_suggestion, names="value")

        # Layout
        self.widget = widgets.VBox(
            [
                header,
                widgets.HTML("<h4>How did this section go?</h4>"),
                self.status,
                widgets.HTML("<h4>Rate your work:</h4>"),
                self.code_quality,
                quality_labels,
                self.understanding,
                understanding_labels,
                widgets.HTML("<h4>Additional details:</h4>"),
                widgets.HBox(
                    [
                        widgets.VBox(
                            [widgets.HTML("<b>Time spent:</b>"), self.time_spent]
                        ),
                        widgets.VBox(
                            [widgets.HTML("<b>&nbsp;</b>"), widgets.HTML("")]
                        ),  # Spacer
                    ]
                ),
                self.notes,
                self.code_snippet,
                widgets.HTML("<br>"),
                self.submit_btn,
                self.progress_output,
            ]
        )

        self.widget_created = True

    def _update_time_suggestion(self, change):
        """Update time suggestion based on status"""
        status_time_multipliers = {
            "completed": 1.0,
            "partial": 1.3,
            "struggling": 1.8,
            "skipped": 0.3,
        }

        if change["new"] in status_time_multipliers:
            suggested_time = int(
                self.expected_time * status_time_multipliers[change["new"]]
            )
            self.time_spent.value = suggested_time

    def _submit_assessment(self, button):
        """Handle assessment submission"""
        with self.progress_output:
            clear_output()

            # Calculate actual time spent
            actual_time = (datetime.now() - self.start_time).total_seconds() / 60

            # Use reported time if reasonable, otherwise use calculated
            reported_time = self.time_spent.value
            time_to_use = (
                reported_time if abs(reported_time - actual_time) < 30 else actual_time
            )

            try:
                # Submit assessment
                checkpoint = self.assessment.add_checkpoint(
                    section=self.section,
                    task=self.task,
                    status=self.status.value,
                    code_quality=self.code_quality.value,
                    understanding=self.understanding.value,
                    time_spent=time_to_use,
                    notes=self.notes.value,
                    code_snippet=self.code_snippet.value,
                )

                # Success feedback
                display(
                    HTML(
                        f"""
                <div style="background: #d4edda; border: 1px solid #c3e6cb;
                           color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>✅ Assessment Submitted Successfully!</strong><br>
                    <small>Timestamp: {datetime.now().strftime('%H:%M:%S')}</small>
                </div>
                """
                    )
                )

                # Show current progress
                scores = self.assessment.calculate_day_score()
                display(
                    HTML(
                        f"""
                <div style="background: #f8f9fa; border: 1px solid #dee2e6;
                           padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>📊 Current Progress:</strong><br>
                    Overall Score: {scores['overall_score']:.2f}/1.0 |
                    Completion: {scores['completion_rate']:.1%} |
                    Code Quality: {scores['code_quality_avg']:.1f}/5.0 |
                    Understanding: {scores['understanding_avg']:.1f}/5.0<br>
                    <em>{scores['recommendation']}</em>
                </div>
                """
                    )
                )

                # Disable widget after submission
                self.submit_btn.disabled = True
                self.submit_btn.description = "Submitted ✓"
                self.submit_btn.button_style = "success"

            except Exception as e:
                display(
                    HTML(
                        f"""
                <div style="background: #f8d7da; border: 1px solid #f5c6cb;
                           color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0;">
                    <strong>❌ Submission Error:</strong> {str(e)}
                </div>
                """
                    )
                )

    def display(self):
        """Display the assessment widget"""
        if not self.widget_created:
            self.create_widget()
        display(self.widget)


class ProgressDashboard:
    """
    Comprehensive progress tracking dashboard
    Provides analytics and insights across multiple days
    """

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.assessment_dir = Path(f"assessments/{student_id}")
        self.all_data = self._load_all_data()

    def _load_all_data(self) -> Dict[int, Dict]:
        """Load assessment data from all days"""
        all_data = {}

        if not self.assessment_dir.exists():
            return all_data

        for file_path in self.assessment_dir.glob("day_*_assessments.json"):
            try:
                day_num = int(file_path.stem.split("_")[1])
                with open(file_path, "r") as f:
                    all_data[day_num] = json.load(f)
            except (ValueError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load {file_path}: {e}")

        return all_data

    def create_comprehensive_dashboard(self) -> go.Figure:
        """Create comprehensive multi-day dashboard"""
        if not self.all_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No assessment data found.<br>Complete some daily assessments to see your progress!",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(
                title="ChemML Bootcamp - Comprehensive Progress Dashboard"
            )
            return fig

        # Calculate daily summaries
        daily_summaries = []
        for day, data in sorted(self.all_data.items()):
            assessments = data.get("assessments", [])
            if assessments:
                df = pd.DataFrame(assessments)

                completion_rate = len(df[df["status"] == "completed"]) / len(df)
                avg_quality = df["code_quality"].mean()
                avg_understanding = df["understanding"].mean()
                total_time = df["time_spent_minutes"].sum() / 60  # Convert to hours

                daily_summaries.append(
                    {
                        "day": day,
                        "completion_rate": completion_rate,
                        "code_quality": avg_quality,
                        "understanding": avg_understanding,
                        "total_time_hours": total_time,
                        "total_assessments": len(df),
                    }
                )

        if not daily_summaries:
            fig = go.Figure()
            fig.add_annotation(
                text="Assessment data exists but no valid entries found.",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font_size=16,
            )
            return fig

        summary_df = pd.DataFrame(daily_summaries)

        # Create comprehensive dashboard
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Daily Progress Overview",
                "Learning Curve",
                "Time Investment",
                "Completion Trends",
                "Quality & Understanding Evolution",
                "Weekly Summary",
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": True}, {"type": "table"}],
            ],
        )

        # 1. Daily Progress Overview (Radar Chart)
        if len(summary_df) > 0:
            categories = [
                "Completion Rate",
                "Code Quality",
                "Understanding",
                "Efficiency",
            ]

            # Normalize metrics for radar chart
            latest_day = summary_df.iloc[-1]
            efficiency = (
                min(1.0, 4.0 / latest_day["total_time_hours"])
                if latest_day["total_time_hours"] > 0
                else 0
            )

            values = [
                latest_day["completion_rate"],
                latest_day["code_quality"] / 5,
                latest_day["understanding"] / 5,
                efficiency,
            ]

            fig.add_trace(
                go.Scatterpolar(
                    r=values + [values[0]],  # Close the radar chart
                    theta=categories + [categories[0]],
                    fill="toself",
                    name=f'Day {latest_day["day"]} Performance',
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )

        # 2. Learning Curve
        fig.add_trace(
            go.Scatter(
                x=summary_df["day"],
                y=summary_df["code_quality"],
                mode="lines+markers",
                name="Code Quality",
                line=dict(color="#ff7f0e", width=3),
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=summary_df["day"],
                y=summary_df["understanding"],
                mode="lines+markers",
                name="Understanding",
                line=dict(color="#2ca02c", width=3),
            ),
            row=2,
            col=1,
        )

        # 3. Time Investment
        fig.add_trace(
            go.Bar(
                x=summary_df["day"],
                y=summary_df["total_time_hours"],
                name="Hours per Day",
                marker_color="#d62728",
            ),
            row=2,
            col=2,
        )

        # 4. Completion Trends
        fig.add_trace(
            go.Scatter(
                x=summary_df["day"],
                y=summary_df["completion_rate"] * 100,
                mode="lines+markers",
                name="Completion %",
                line=dict(color="#9467bd", width=3),
                yaxis="y",
            ),
            row=3,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=summary_df["day"],
                y=summary_df["total_assessments"],
                mode="lines+markers",
                name="Assessments Count",
                line=dict(color="#8c564b", width=2),
                yaxis="y2",
            ),
            row=3,
            col=1,
        )

        # 5. Summary Table
        table_data = []
        for _, row in summary_df.iterrows():
            table_data.append(
                [
                    f"Day {int(row['day'])}",
                    f"{row['completion_rate']:.1%}",
                    f"{row['code_quality']:.1f}/5",
                    f"{row['understanding']:.1f}/5",
                    f"{row['total_time_hours']:.1f}h",
                    f"{row['total_assessments']}",
                ]
            )

        fig.add_trace(
            go.Table(
                header=dict(
                    values=[
                        "Day",
                        "Completion",
                        "Code Quality",
                        "Understanding",
                        "Time",
                        "Assessments",
                    ],
                    fill_color="#f1f1f2",
                    align="center",
                    font=dict(size=12, color="black"),
                ),
                cells=dict(
                    values=list(zip(*table_data)) if table_data else [[]] * 6,
                    fill_color="white",
                    align="center",
                    font=dict(size=11, color="black"),
                ),
            ),
            row=3,
            col=2,
        )

        # Update layout
        fig.update_layout(
            title=f"ChemML Bootcamp - Comprehensive Progress Dashboard<br><sub>Student: {self.student_id}</sub>",
            height=900,
            showlegend=True,
        )

        # Update specific subplot layouts
        fig.update_yaxes(title_text="Score (1-5)", row=2, col=1)
        fig.update_yaxes(title_text="Hours", row=2, col=2)
        fig.update_yaxes(title_text="Completion %", row=3, col=1)
        fig.update_yaxes(title_text="Count", secondary_y=True, row=3, col=1)

        fig.update_xaxes(title_text="Day", row=2, col=1)
        fig.update_xaxes(title_text="Day", row=2, col=2)
        fig.update_xaxes(title_text="Day", row=3, col=1)

        return fig

    def generate_progress_report(self) -> str:
        """Generate comprehensive text progress report"""
        if not self.all_data:
            return "No assessment data available for report generation."

        # Calculate overall statistics
        all_assessments = []
        for day_data in self.all_data.values():
            all_assessments.extend(day_data.get("assessments", []))

        if not all_assessments:
            return "No individual assessments found for analysis."

        df = pd.DataFrame(all_assessments)

        # Overall metrics
        total_assessments = len(df)
        completion_rate = len(df[df["status"] == "completed"]) / total_assessments
        avg_quality = df["code_quality"].mean()
        avg_understanding = df["understanding"].mean()
        total_time_hours = df["time_spent_minutes"].sum() / 60

        # Daily breakdown
        days_completed = len(self.all_data)
        avg_assessments_per_day = (
            total_assessments / days_completed if days_completed > 0 else 0
        )

        # Trend analysis
        daily_completion = []
        for day in sorted(self.all_data.keys()):
            day_assessments = self.all_data[day].get("assessments", [])
            if day_assessments:
                day_df = pd.DataFrame(day_assessments)
                daily_completion.append(
                    len(day_df[day_df["status"] == "completed"]) / len(day_df)
                )

        trend = (
            "improving"
            if len(daily_completion) > 1 and daily_completion[-1] > daily_completion[0]
            else "stable"
        )

        # Generate report
        report = f"""
# ChemML Bootcamp Progress Report
**Student ID:** {self.student_id}
**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overall Performance Summary

### Key Metrics
- **Days Completed:** {days_completed}/7
- **Total Assessments:** {total_assessments}
- **Overall Completion Rate:** {completion_rate:.1%}
- **Average Code Quality:** {avg_quality:.2f}/5.0
- **Average Understanding:** {avg_understanding:.2f}/5.0
- **Total Time Invested:** {total_time_hours:.1f} hours
- **Average Time per Day:** {total_time_hours/days_completed:.1f} hours

### Performance Analysis
- **Learning Trend:** {trend.title()}
- **Efficiency:** {avg_assessments_per_day:.1f} assessments per day
- **Quality Score:** {"Excellent" if avg_quality >= 4 else "Good" if avg_quality >= 3 else "Needs Improvement"}
- **Understanding Level:** {"Deep" if avg_understanding >= 4 else "Solid" if avg_understanding >= 3 else "Developing"}

## 📈 Daily Breakdown
"""

        for day in sorted(self.all_data.keys()):
            day_data = self.all_data[day]
            assessments = day_data.get("assessments", [])

            if assessments:
                day_df = pd.DataFrame(assessments)
                day_completion = len(day_df[day_df["status"] == "completed"]) / len(
                    day_df
                )
                day_quality = day_df["code_quality"].mean()
                day_understanding = day_df["understanding"].mean()
                day_time = day_df["time_spent_minutes"].sum() / 60

                report += f"""
### Day {day}
- **Completion Rate:** {day_completion:.1%}
- **Code Quality:** {day_quality:.1f}/5.0
- **Understanding:** {day_understanding:.1f}/5.0
- **Time Spent:** {day_time:.1f} hours
- **Assessments:** {len(day_df)}
"""

        # Recommendations
        report += f"""
## 🎯 Recommendations

### Strengths
"""
        if completion_rate > 0.8:
            report += "- ✅ Excellent completion rate - you're staying on track!\n"
        if avg_quality > 4.0:
            report += "- ✅ High code quality - your implementations are excellent!\n"
        if avg_understanding > 4.0:
            report += "- ✅ Deep understanding - concepts are well-grasped!\n"

        report += """
### Areas for Improvement
"""
        if completion_rate < 0.7:
            report += "- 📚 Focus on completing more exercises before moving forward\n"
        if avg_quality < 3.0:
            report += "- 💻 Spend more time on code quality and best practices\n"
        if avg_understanding < 3.0:
            report += "- 🤔 Review concepts more thoroughly before proceeding\n"
        if total_time_hours / days_completed > 6:
            report += (
                "- ⏱️ Consider time management strategies to work more efficiently\n"
            )

        report += """
### Next Steps
1. **Continue** with current learning approach if performance is strong
2. **Review** previous material if understanding scores are low
3. **Practice** more coding exercises if code quality needs improvement
4. **Seek help** if completion rates are consistently low
5. **Challenge yourself** with advanced exercises if excelling

## 📞 Support Resources
- Review notebook markdown cells for concept explanations
- Check BOOTCAMP_QUICKSTART.md for troubleshooting tips
- Join discussion forums for peer support
- Attend virtual office hours if available

---
*This report is automatically generated based on your self-assessment data. Continue completing checkpoint assessments to track your progress!*
"""

        return report


# Factory function for easy widget creation
def create_assessment_widget(
    student_id: str,
    day: int,
    section: str,
    task: str,
    track: str = "standard",
    expected_time: int = 30,
) -> AssessmentWidget:
    """
    Factory function to create assessment widgets easily

    Usage in notebooks:
    ```python
    from assessment_framework import create_assessment_widget

    # Create assessment for current section
    assessment = create_assessment_widget(
        student_id="your_name",
        day=1,
        section="1.1",
        task="Molecular Representation Mastery",
        expected_time=45
    )
    assessment.display()
    ```
    """
    assessment_system = BootcampAssessment(student_id, day, track)
    widget = AssessmentWidget(assessment_system, section, task, expected_time)
    return widget


# Utility function for progress visualization
def show_progress_dashboard(student_id: str):
    """
    Show comprehensive progress dashboard

    Usage:
    ```python
    from assessment_framework import show_progress_dashboard
    show_progress_dashboard("your_name")
    ```
    """
    dashboard = ProgressDashboard(student_id)
    fig = dashboard.create_comprehensive_dashboard()
    fig.show()

    # Also display text report
    report = dashboard.generate_progress_report()
    display(HTML(f"<pre>{report}</pre>"))


# Factory Functions for Easy Integration
# =====================================


def create_assessment(
    student_id: str, day: int, track: str = "standard"
) -> BootcampAssessment:
    """
    Factory function to create a BootcampAssessment instance

    Args:
        student_id: Unique identifier for the student
        day: Day number (1-7)
        track: Learning track (quick/standard/intensive/extended)

    Returns:
        BootcampAssessment instance
    """
    return BootcampAssessment(student_id=student_id, day=day, track=track)


def create_widget(
    assessment: BootcampAssessment,
    section: str,
    concepts: List[str],
    activities: List[str],
    time_estimate: int = 60,
    checkpoint: bool = False,
    final_assessment: bool = False,
) -> "AssessmentWidget":
    """
    Factory function to create an AssessmentWidget instance

    Args:
        assessment: BootcampAssessment instance
        section: Name of the section being assessed
        concepts: List of concepts to evaluate
        activities: List of activities to check
        time_estimate: Estimated time in minutes
        checkpoint: Whether this is a checkpoint assessment
        final_assessment: Whether this is the final assessment

    Returns:
        AssessmentWidget instance
    """
    # Create a combined task description from concepts and activities
    task_description = f"Section: {section}"
    if concepts:
        task_description += f"\nConcepts: {', '.join(concepts)}"
    if activities:
        task_description += f"\nActivities: {', '.join(activities)}"

    widget = AssessmentWidget(
        assessment_system=assessment,
        section=section,
        task=task_description,
        expected_time=time_estimate,
    )

    # Add metadata for tracking
    widget.concepts = concepts
    widget.activities = activities
    widget.checkpoint = checkpoint
    widget.final_assessment = final_assessment

    return widget


def create_dashboard(assessment: BootcampAssessment) -> "ProgressDashboard":
    """
    Factory function to create a ProgressDashboard instance

    Args:
        assessment: BootcampAssessment instance

    Returns:
        ProgressDashboard instance
    """
    return ProgressDashboard(assessment.student_id)


# Convenience functions for common operations
def quick_assessment(
    student_id: str, day: int, section: str, concepts: List[str]
) -> "AssessmentWidget":
    """
    Quick assessment setup with minimal configuration
    """
    assessment = create_assessment(student_id, day, "quick")
    return create_widget(assessment, section, concepts, [])


def comprehensive_assessment(
    student_id: str, day: int, track: str = "standard"
) -> tuple:
    """
    Create a comprehensive assessment setup with widget and dashboard

    Returns:
        Tuple of (assessment, widget_factory, dashboard)
    """
    assessment = create_assessment(student_id, day, track)

    def widget_factory(
        section: str, concepts: List[str], activities: List[str], **kwargs
    ):
        return create_widget(assessment, section, concepts, activities, **kwargs)

    dashboard = create_dashboard(assessment)

    return assessment, widget_factory, dashboard


if __name__ == "__main__":
    # Example usage
    print("ChemML Bootcamp Assessment Framework")
    print("===================================")
    print("This module provides interactive assessment tools for the bootcamp.")
    print("Import the functions you need in your Jupyter notebooks.")
    print("\nExample usage:")
    print(
        "from assessment_framework import create_assessment, create_widget, create_dashboard"
    )
    print("\n# Create assessment for Day 1")
    print("assessment = create_assessment('student_123', day=1, track='standard')")
    print("\n# Create assessment widget")
    print(
        "widget = create_widget(assessment, 'Section 1', ['concept1'], ['activity1'])"
    )
    print("\n# Create progress dashboard")
    print("dashboard = create_dashboard(assessment)")
