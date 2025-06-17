Progress Tracking Dashboard for ChemML Learning Journey

This module provides interactive widgets and visualizations for tracking
learning progress across the computational drug discovery program.
"""

import json
from datetime import datetime, timedelta

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import HTML, display

class ProgressTracker:
    """Main class for tracking and visualizing learning progress."""

    def __init__(self, learner_id=None):
        self.learner_id = learner_id or "default_learner"
        self.progress_data = self._initialize_progress_data()
        self.competency_areas = [
            "Python Programming",
            "Machine Learning",
            "Cheminformatics",
            "Molecular Modeling",
            "Quantum Computing",
            "Drug Design",
            "Data Visualization",
            "Statistical Analysis",
        ]

    def _initialize_progress_data(self):
        """Initialize the progress tracking data structure."""
        return {
            "checkpoints": {},
            "portfolio_projects": {},
            "competency_scores": {},
            "time_tracking": {},
            "peer_reviews": {},
            "milestones": {},
        }

    def record_checkpoint_completion(
        self, week, time_spent, self_assessment_score, notes=""
    ):
        """Record completion of a weekly checkpoint."""
        checkpoint_data = {
            "completion_date": datetime.now().isoformat(),
            "time_spent_minutes": time_spent,
            "self_assessment_score": self_assessment_score,
            "notes": notes,
            "status": "completed",
        }
        self.progress_data["checkpoints"][f"week_{week:02d}"] = checkpoint_data

    def update_competency_score(self, area, score, evidence=""):
        """Update competency score for a specific area."""
        if area not in self.competency_areas:
            raise ValueError(f"Unknown competency area: {area}")

        self.progress_data["competency_scores"][area] = {
            "current_score": score,
            "last_updated": datetime.now().isoformat(),
            "evidence": evidence,
        }

    def create_progress_dashboard(self):
        """Create the main progress tracking dashboard."""

        # Create tabs for different views
        tab_contents = [
            self._create_overview_tab(),
            self._create_checkpoint_tab(),
            self._create_competency_tab(),
            self._create_portfolio_tab(),
            self._create_analytics_tab(),
        ]

        tab_titles = [
            "Overview",
            "Checkpoints",
            "Competencies",
            "Portfolio",
            "Analytics",
        ]
        tab = widgets.Tab(children=tab_contents)
        for i, title in enumerate(tab_titles):
            tab.set_title(i, title)

        return tab

    def _create_overview_tab(self):
        """Create the overview dashboard tab."""

        # Progress summary widgets
        total_checkpoints = 12
        completed_checkpoints = len(
            [
                c
                for c in self.progress_data["checkpoints"].values()
                if c.get("status") == "completed"
            ]
        )

        progress_percentage = (completed_checkpoints / total_checkpoints) * 100

        # Create progress bar
        progress_bar = widgets.FloatProgress(
            value=progress_percentage,
            min=0,
            max=100,
            description="Overall Progress:",
            bar_style="info",
            style={"bar_color": "#1f77b4"},
            orientation="horizontal",
        )

        # Summary statistics
        total_time = sum(
            [
                c.get("time_spent_minutes", 0)
                for c in self.progress_data["checkpoints"].values()
            ]
        )

        stats_html = f"""
        <div style="display: flex; justify-content: space-around; padding: 20px;">
            <div style="text-align: center;">
                <h3>{completed_checkpoints}/{total_checkpoints}</h3>
                <p>Checkpoints Completed</p>
            </div>
            <div style="text-align: center;">
                <h3>{total_time//60:.1f}h</h3>
                <p>Total Study Time</p>
            </div>
            <div style="text-align: center;">
                <h3>{progress_percentage:.1f}%</h3>
                <p>Progress Complete</p>
            </div>
            <div style="text-align: center;">
                <h3>{len(self.progress_data["portfolio_projects"])}</h3>
                <p>Portfolio Projects</p>
            </div>
        </div>
        """

        # Recent activity
        recent_activity = self._get_recent_activity()

        return widgets.VBox(
            [
                widgets.HTML("<h2>Learning Progress Overview</h2>"),
                progress_bar,
                widgets.HTML(stats_html),
                widgets.HTML("<h3>Recent Activity</h3>"),
                widgets.HTML(recent_activity),
            ]
        )

    def _create_checkpoint_tab(self):
        """Create the checkpoint tracking tab."""

        # Checkpoint grid
        checkpoint_grid = self._create_checkpoint_grid()

        # Time tracking chart
        time_chart = self._create_time_tracking_chart()

        return widgets.VBox(
            [
                widgets.HTML("<h2>Checkpoint Progress</h2>"),
                checkpoint_grid,
                widgets.HTML("<h3>Time Tracking</h3>"),
                time_chart,
            ]
        )

    def _create_competency_tab(self):
        """Create the competency tracking tab."""

        # Radar chart for competencies
        competency_chart = self._create_competency_radar()

        # Competency progression over time
        progression_chart = self._create_competency_progression()

        return widgets.VBox(
            [
                widgets.HTML("<h2>Competency Development</h2>"),
                competency_chart,
                widgets.HTML("<h3>Competency Progression</h3>"),
                progression_chart,
            ]
        )

    def _create_portfolio_tab(self):
        """Create the portfolio tracking tab."""

        portfolio_summary = self._create_portfolio_summary()

        return widgets.VBox(
            [widgets.HTML("<h2>Portfolio Projects</h2>"), portfolio_summary]
        )

    def _create_analytics_tab(self):
        """Create the learning analytics tab."""

        # Learning velocity
        velocity_chart = self._create_velocity_chart()

        # Study patterns
        pattern_analysis = self._create_study_pattern_analysis()

        return widgets.VBox(
            [
                widgets.HTML("<h2>Learning Analytics</h2>"),
                velocity_chart,
                widgets.HTML("<h3>Study Patterns</h3>"),
                pattern_analysis,
            ]
        )

    def _create_checkpoint_grid(self):
        """Create a visual grid of checkpoint status."""

        grid_html = "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 20px;'>"

        for week in range(1, 13):
            week_key = f"week_{week:02d}"
            status = (
                "completed"
                if week_key in self.progress_data["checkpoints"]
                else "pending"
            )

            color = "#4CAF50" if status == "completed" else "#E0E0E0"
            icon = "✓" if status == "completed" else str(week)

            grid_html += f"""
            <div style="
                background-color: {color};
                padding: 20px;
                text-align: center;
                border-radius: 8px;
                color: white;
                font-weight: bold;
            ">
                Week {week}<br>{icon}
            </div>
            """

        grid_html += "</div>"
        return widgets.HTML(grid_html)

    def _create_time_tracking_chart(self):
        """Create a chart showing time spent per week."""

        weeks = list(range(1, 13))
        time_spent = [
            self.progress_data["checkpoints"]
            .get(f"week_{w:02d}", {})
            .get("time_spent_minutes", 0)
            / 60
            for w in weeks
        ]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=weeks, y=time_spent, name="Hours Spent"))
        fig.update_layout(
            title="Weekly Study Time",
            xaxis_title="Week",
            yaxis_title="Hours",
            height=400,
        )

        return widgets.HTML(fig.to_html(include_plotlyjs="inline", div_id="time-chart"))

    def _create_competency_radar(self):
        """Create a radar chart for competency scores."""

        scores = [
            self.progress_data["competency_scores"]
            .get(area, {})
            .get("current_score", 0)
            for area in self.competency_areas
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=scores,
                theta=self.competency_areas,
                fill="toself",
                name="Current Level",
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True,
            title="Competency Radar Chart",
            height=500,
        )

        return widgets.HTML(
            fig.to_html(include_plotlyjs="inline", div_id="competency-radar")
        )

    def _create_competency_progression(self):
        """Create a chart showing competency progression over time."""

        # This would show how competencies develop over time
        # For now, return a placeholder
        return widgets.HTML(
            "<p>Competency progression chart would be displayed here</p>"
        )

    def _create_portfolio_summary(self):
        """Create a summary of portfolio projects."""

        if not self.progress_data["portfolio_projects"]:
            return widgets.HTML("<p>No portfolio projects started yet.</p>")

        # Create portfolio project cards
        cards_html = ""
        for project_name, project_data in self.progress_data[
            "portfolio_projects"
        ].items():
            status = project_data.get("status", "In Progress")
            progress = project_data.get("progress_percentage", 0)

            cards_html += f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px;">
                <h4>{project_name}</h4>
                <p>Status: {status}</p>
                <div style="background-color: #f0f0f0; border-radius: 10px; height: 20px;">
                    <div style="background-color: #4CAF50; height: 100%; width: {progress}%; border-radius: 10px;"></div>
                </div>
                <p>{progress}% Complete</p>
            </div>
            """

        return widgets.HTML(cards_html)

    def _create_velocity_chart(self):
        """Create a learning velocity chart."""

        # Calculate learning velocity (checkpoints per week)
        return widgets.HTML("<p>Learning velocity chart would be displayed here</p>")

    def _create_study_pattern_analysis(self):
        """Analyze study patterns and provide insights."""

        insights = [
            "You tend to study most effectively in the morning",
            "Your longest study sessions are on weekends",
            "You complete checkpoints faster after Week 4",
            "Consider spacing out practice sessions for better retention",
        ]

        insights_html = "<ul>"
        for insight in insights:
            insights_html += f"<li>{insight}</li>"
        insights_html += "</ul>"

        return widgets.HTML(insights_html)

    def _get_recent_activity(self):
        """Get recent learning activity."""

        # This would fetch real recent activity
        # For now, return sample data
        activities = [
            "Completed Week 2 Checkpoint - 2 hours ago",
            "Started Portfolio Project - 1 day ago",
            "Peer reviewed 2 submissions - 3 days ago",
            "Updated competency scores - 1 week ago",
        ]

        activity_html = "<ul>"
        for activity in activities:
            activity_html += f"<li>{activity}</li>"
        activity_html += "</ul>"

        return activity_html

    def export_progress_report(self):
        """Export a comprehensive progress report."""

        report = {
            "learner_id": self.learner_id,
            "report_date": datetime.now().isoformat(),
            "progress_data": self.progress_data,
            "summary_statistics": self._calculate_summary_stats(),
        }

        return json.dumps(report, indent=2)

    def _calculate_summary_stats(self):
        """Calculate summary statistics for the report."""

        completed_checkpoints = len(
            [
                c
                for c in self.progress_data["checkpoints"].values()
                if c.get("status") == "completed"
            ]
        )
        total_time = sum(
            [
                c.get("time_spent_minutes", 0)
                for c in self.progress_data["checkpoints"].values()
            ]
        )
        avg_score = np.mean(
            [
                c.get("self_assessment_score", 0)
                for c in self.progress_data["checkpoints"].values()
                if c.get("self_assessment_score")
            ]
        )

        return {
            "completed_checkpoints": completed_checkpoints,
            "total_study_time_hours": total_time / 60,
            "average_self_assessment_score": avg_score,
            "competency_areas_assessed": len(self.progress_data["competency_scores"]),
        }

# Usage example
def create_sample_progress_tracker():
    """Create a sample progress tracker with demo data."""

    tracker = ProgressTracker("demo_learner")

    # Add sample checkpoint data
    tracker.record_checkpoint_completion(1, 180, 4.2, "Completed all challenges")
    tracker.record_checkpoint_completion(
        2, 210, 4.5, "Strong performance on QSAR modeling"
    )

    # Add sample competency scores
    tracker.update_competency_score(
        "Python Programming", 4.0, "Checkpoint 1-2 completion"
    )
    tracker.update_competency_score(
        "Machine Learning", 3.5, "QSAR model implementation"
    )
    tracker.update_competency_score(
        "Cheminformatics", 3.8, "RDKit proficiency demonstrated"
    )

    # Add sample portfolio project
    tracker.progress_data["portfolio_projects"]["Multi-Target QSAR"] = {
        "status": "In Progress",
        "progress_percentage": 35,
        "start_date": "2025-06-01",
        "expected_completion": "2025-08-15",
    }

    return tracker

if __name__ == "__main__":
    # Demo usage
    tracker = create_sample_progress_tracker()
    dashboard = tracker.create_progress_dashboard()
    display(dashboard)
