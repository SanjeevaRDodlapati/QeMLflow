"""
Simple Progress Tracker for ChemML 7-Day QuickStart Bootcamp

Simplified assessment framework focusing on daily completion tracking
and basic progress visualization. Removes enterprise over-engineering
while maintaining essential learning progress indicators.
"""

import json
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


class QuickStartProgressTracker:
    """Simplified progress tracker for 7-day bootcamp."""

    def __init__(self, learner_name: str = "Bootcamp Learner"):
        self.learner_name = learner_name
        self.start_date = datetime.now().date()
        self.daily_progress = {}
        self.bootcamp_days = [
            "Day 1: Environment Setup & Basic ML",
            "Day 2: Molecular Data Processing",
            "Day 3: Feature Engineering & QSAR",
            "Day 4: Advanced ML Models",
            "Day 5: Quantum ML Foundations (Module 1)",
            "Day 6: Quantum ML Advanced (Module 2)",
            "Day 7: Production Integration (Module 3)",
        ]

    def complete_day(
        self, day: int, time_spent_minutes: int, self_rating: int, notes: str = ""
    ):
        """Record completion of a bootcamp day.

        Args:
            day: Day number (1-7)
            time_spent_minutes: Time spent in minutes
            self_rating: Self-assessment rating (1-5 scale)
            notes: Optional completion notes
        """
        if day not in range(1, 8):
            raise ValueError("Day must be between 1 and 7")
        if self_rating not in range(1, 6):
            raise ValueError("Self rating must be between 1 and 5")

        self.daily_progress[day] = {
            "completed": True,
            "completion_date": datetime.now().isoformat(),
            "time_spent_minutes": time_spent_minutes,
            "self_rating": self_rating,
            "notes": notes,
            "day_title": self.bootcamp_days[day - 1],
        }

    def get_completion_percentage(self) -> float:
        """Get overall completion percentage."""
        completed_days = len(
            [d for d in self.daily_progress.values() if d.get("completed", False)]
        )
        return (completed_days / 7) * 100

    def get_total_time_spent(self) -> float:
        """Get total time spent in hours."""
        total_minutes = sum(
            [d.get("time_spent_minutes", 0) for d in self.daily_progress.values()]
        )
        return total_minutes / 60

    def get_average_rating(self) -> float:
        """Get average self-assessment rating."""
        ratings = [
            d.get("self_rating", 0)
            for d in self.daily_progress.values()
            if d.get("completed", False)
        ]
        return np.mean(ratings) if ratings else 0

    def display_progress_summary(self):
        """Display a simple progress summary."""
        completion_pct = self.get_completion_percentage()
        total_hours = self.get_total_time_spent()
        avg_rating = self.get_average_rating()

        print(f"🚀 ChemML 7-Day QuickStart Progress for {self.learner_name}")
        print("=" * 50)
        print(
            f"📊 Overall Progress: {completion_pct:.1f}% ({len(self.daily_progress)}/7 days)"
        )
        print(f"⏱️  Total Time Spent: {total_hours:.1f} hours")
        print(f"⭐ Average Self-Rating: {avg_rating:.1f}/5.0")
        print()

        # Day-by-day breakdown
        print("📅 Daily Progress:")
        for day in range(1, 8):
            if day in self.daily_progress:
                data = self.daily_progress[day]
                status = "✅ COMPLETED"
                time_str = f"({data['time_spent_minutes']}min)"
                rating_str = f"Rating: {data['self_rating']}/5"
            else:
                status = "⏳ PENDING"
                time_str = ""
                rating_str = ""

            day_title = self.bootcamp_days[day - 1]
            print(f"  Day {day}: {status} {day_title} {time_str} {rating_str}")

    def create_simple_visualization(self):
        """Create a simple progress visualization."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Progress bar chart
        days = list(range(1, 8))
        completed = [1 if day in self.daily_progress else 0 for day in days]
        colors = ["green" if c else "lightgray" for c in completed]

        ax1.bar(days, [1] * 7, color=colors)
        ax1.set_xlabel("Bootcamp Day")
        ax1.set_ylabel("Completion Status")
        ax1.set_title("Daily Completion Progress")
        ax1.set_ylim(0, 1.2)
        ax1.set_xticks(days)

        # Time spent per day
        time_spent = [
            self.daily_progress.get(day, {}).get("time_spent_minutes", 0) / 60
            for day in days
        ]
        ax2.bar(days, time_spent, color="steelblue")
        ax2.set_xlabel("Bootcamp Day")
        ax2.set_ylabel("Hours Spent")
        ax2.set_title("Time Spent Per Day")
        ax2.set_xticks(days)

        plt.tight_layout()
        return fig

    def generate_completion_badges(self) -> List[str]:
        """Generate completion badges for achievements."""
        badges = []

        # Daily completion badges
        for day, data in self.daily_progress.items():
            if data.get("completed", False):
                badges.append(f"🏆 Day {day} Completed")

                # Special achievement badges
                if data.get("time_spent_minutes", 0) >= 240:  # 4+ hours
                    badges.append(f"⚡ Deep Dive Day {day}")
                if data.get("self_rating", 0) >= 5:
                    badges.append(f"🌟 Mastery Day {day}")

        # Milestone badges
        completion_pct = self.get_completion_percentage()
        if completion_pct >= 50:
            badges.append("🎯 Halfway Hero")
        if completion_pct >= 100:
            badges.append("🎓 Bootcamp Graduate")

        total_hours = self.get_total_time_spent()
        if total_hours >= 20:
            badges.append("💪 Dedicated Learner")
        if total_hours >= 35:
            badges.append("🔥 Power Learner")

        return badges

    def export_simple_report(self) -> str:
        """Export a simple completion report."""
        report = {
            "learner_name": self.learner_name,
            "bootcamp_start_date": self.start_date.isoformat(),
            "report_generated": datetime.now().isoformat(),
            "completion_percentage": self.get_completion_percentage(),
            "total_hours_spent": self.get_total_time_spent(),
            "average_self_rating": self.get_average_rating(),
            "daily_progress": self.daily_progress,
            "achievements": self.generate_completion_badges(),
        }

        return json.dumps(report, indent=2)

    def save_progress(self, filename: str = None):
        """Save progress to a JSON file."""
        if filename is None:
            filename = f"chemml_bootcamp_progress_{self.learner_name.replace(' ', '_').lower()}.json"

        with open(filename, "w") as f:
            f.write(self.export_simple_report())

        print(f"Progress saved to {filename}")


# Usage Examples and Demo
def demo_progress_tracker():
    """Demo the simplified progress tracker."""

    # Create tracker
    tracker = QuickStartProgressTracker("Demo Learner")

    # Simulate some completed days
    tracker.complete_day(1, 180, 4, "Great introduction to ML concepts")
    tracker.complete_day(2, 210, 5, "Loved working with molecular data")
    tracker.complete_day(3, 195, 4, "QSAR modeling was challenging but rewarding")
    tracker.complete_day(5, 240, 5, "Quantum ML Module 1 - mind-blowing!")

    # Display progress
    tracker.display_progress_summary()
    print()

    # Show badges
    badges = tracker.generate_completion_badges()
    print("🏆 Achievements Unlocked:")
    for badge in badges:
        print(f"  {badge}")

    return tracker


if __name__ == "__main__":
    demo_tracker = demo_progress_tracker()

    # Create visualization
    fig = demo_tracker.create_simple_visualization()
    plt.show()

    # Save progress
    demo_tracker.save_progress()
