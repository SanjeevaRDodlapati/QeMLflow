"""
Visual Completion Badges for ChemML 7-Day QuickStart Bootcamp

Simple, motivating visual indicators for progress tracking.
Provides immediate feedback and achievement recognition.
"""

from typing import Dict, List

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle


class CompletionBadges:
    """Generate visual badges for bootcamp completion."""

    def __init__(self):
        self.badge_colors = {
            "completed": "#4CAF50",  # Green
            "pending": "#E0E0E0",  # Light Gray
            "current": "#FF9800",  # Orange
            "excellence": "#2196F3",  # Blue
            "mastery": "#9C27B0",  # Purple
        }

        self.day_titles = [
            "Setup & Basic ML",
            "Data Processing",
            "QSAR Modeling",
            "Advanced ML",
            "Quantum ML Foundations",
            "Quantum ML Advanced",
            "Production Integration",
        ]

    def create_progress_badges(
        self, completed_days: List[int], current_day: int = None
    ) -> plt.Figure:
        """Create visual progress badges for completed days.

        Args:
            completed_days: List of completed day numbers (1-7)
            current_day: Current day being worked on

        Returns:
            matplotlib Figure with badge visualization
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))

        # Create grid layout for badges
        cols = 4
        rows = 2

        for day in range(1, 8):
            row = (day - 1) // cols
            col = (day - 1) % cols

            # Calculate position
            x = col * 3.5
            y = (1 - row) * 3.5

            # Determine badge status
            if day in completed_days:
                color = self.badge_colors["completed"]
                status_text = "✅"
                border_color = self.badge_colors["completed"]
            elif day == current_day:
                color = self.badge_colors["current"]
                status_text = "🔄"
                border_color = self.badge_colors["current"]
            else:
                color = self.badge_colors["pending"]
                status_text = "⏳"
                border_color = self.badge_colors["pending"]

            # Draw badge circle
            circle = Circle(
                (x + 1.25, y + 1.25),
                1.0,
                facecolor=color,
                edgecolor=border_color,
                linewidth=3,
            )
            ax.add_patch(circle)

            # Add day number
            ax.text(
                x + 1.25,
                y + 1.5,
                f"Day {day}",
                ha="center",
                va="center",
                fontweight="bold",
                fontsize=12,
            )

            # Add status icon
            ax.text(
                x + 1.25, y + 1.0, status_text, ha="center", va="center", fontsize=20
            )

            # Add title below badge
            title = self.day_titles[day - 1]
            ax.text(
                x + 1.25,
                y + 0.3,
                title,
                ha="center",
                va="center",
                fontsize=10,
                wrap=True,
                style="italic",
            )

        # Set plot properties
        ax.set_xlim(-0.5, 14)
        ax.set_ylim(-0.5, 7.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add title
        fig.suptitle(
            "🚀 ChemML 7-Day QuickStart Progress Badges",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )

        # Add legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.badge_colors["completed"],
                markersize=10,
                label="Completed",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.badge_colors["current"],
                markersize=10,
                label="In Progress",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=self.badge_colors["pending"],
                markersize=10,
                label="Pending",
            ),
        ]
        ax.legend(
            handles=legend_elements, loc="upper right", bbox_to_anchor=(1.0, 0.15)
        )

        plt.tight_layout()
        return fig

    def create_achievement_badges(self, achievements: List[str]) -> plt.Figure:
        """Create achievement badges for special accomplishments.

        Args:
            achievements: List of achievement names

        Returns:
            matplotlib Figure with achievement badges
        """
        if not achievements:
            achievements = ["Start your journey to unlock achievements!"]

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Achievement badge designs
        achievement_icons = {
            "Day 1 Completed": "🏆",
            "Day 2 Completed": "🧪",
            "Day 3 Completed": "📊",
            "Day 4 Completed": "🤖",
            "Day 5 Completed": "⚛️",
            "Day 6 Completed": "🔬",
            "Day 7 Completed": "🚀",
            "Halfway Hero": "🎯",
            "Bootcamp Graduate": "🎓",
            "Dedicated Learner": "💪",
            "Power Learner": "🔥",
            "Deep Dive": "⚡",
            "Mastery": "🌟",
        }

        # Create achievement badges in grid
        cols = 4
        for i, achievement in enumerate(achievements[:12]):  # Limit to 12 badges
            row = i // cols
            col = i % cols

            x = col * 2.8
            y = (2 - row) * 2.0

            # Get icon for achievement
            icon = "🏅"  # Default icon
            for key, val in achievement_icons.items():
                if key in achievement:
                    icon = val
                    break

            # Draw achievement badge
            rect = Rectangle(
                (x, y),
                2.5,
                1.5,
                facecolor=self.badge_colors["excellence"],
                edgecolor="gold",
                linewidth=2,
                alpha=0.8,
            )
            ax.add_patch(rect)

            # Add icon
            ax.text(x + 1.25, y + 1.0, icon, ha="center", va="center", fontsize=24)

            # Add achievement text
            achievement_text = achievement.replace("Day ", "D").replace(
                "Completed", "✓"
            )
            ax.text(
                x + 1.25,
                y + 0.3,
                achievement_text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )

        # Set plot properties
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 5.0)
        ax.set_aspect("equal")
        ax.axis("off")

        # Add title
        fig.suptitle(
            "🏆 Achievement Badges Unlocked", fontsize=16, fontweight="bold", y=0.95
        )

        plt.tight_layout()
        return fig

    def create_completion_certificate(
        self,
        learner_name: str,
        completion_percentage: float,
        total_hours: float,
        avg_rating: float,
    ) -> plt.Figure:
        """Create a completion certificate for bootcamp graduates.

        Args:
            learner_name: Name of the learner
            completion_percentage: Percentage of bootcamp completed
            total_hours: Total hours spent
            avg_rating: Average self-assessment rating

        Returns:
            matplotlib Figure with completion certificate
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Certificate border
        border = Rectangle(
            (0.5, 0.5), 11, 7, facecolor="white", edgecolor="gold", linewidth=5
        )
        ax.add_patch(border)

        # Inner decorative border
        inner_border = Rectangle(
            (1, 1),
            10,
            6,
            facecolor="none",
            edgecolor=self.badge_colors["excellence"],
            linewidth=2,
            linestyle="--",
        )
        ax.add_patch(inner_border)

        # Certificate content
        if completion_percentage >= 100:
            certificate_title = "🎓 CERTIFICATE OF COMPLETION"
            subtitle = "ChemML 7-Day QuickStart Bootcamp"
        else:
            certificate_title = "📜 CERTIFICATE OF PARTICIPATION"
            subtitle = f"ChemML 7-Day QuickStart Bootcamp ({completion_percentage:.0f}% Complete)"

        # Title
        ax.text(
            6,
            6.5,
            certificate_title,
            ha="center",
            va="center",
            fontsize=18,
            fontweight="bold",
            color=self.badge_colors["mastery"],
        )

        # Subtitle
        ax.text(6, 5.8, subtitle, ha="center", va="center", fontsize=14, style="italic")

        # Learner name
        ax.text(6, 4.8, f"Presented to", ha="center", va="center", fontsize=12)
        ax.text(
            6,
            4.3,
            learner_name,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=self.badge_colors["excellence"],
        )

        # Achievement details
        details = f"""
        📊 Completion: {completion_percentage:.1f}%
        ⏱️ Time Invested: {total_hours:.1f} hours
        ⭐ Average Rating: {avg_rating:.1f}/5.0
        📅 Completed: {plt.datetime.datetime.now().strftime('%B %d, %Y')}
        """

        ax.text(
            6,
            3.2,
            details,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=self.badge_colors["pending"]),
        )

        # Signature line
        ax.text(
            6,
            1.8,
            "ChemML Development Team",
            ha="center",
            va="center",
            fontsize=12,
            style="italic",
        )
        ax.plot([4, 8], [1.5, 1.5], "k-", linewidth=1)

        # Decorative elements
        ax.text(1.5, 6.5, "🧪", fontsize=30, alpha=0.3)
        ax.text(10.5, 6.5, "⚛️", fontsize=30, alpha=0.3)
        ax.text(1.5, 1.5, "🤖", fontsize=30, alpha=0.3)
        ax.text(10.5, 1.5, "🚀", fontsize=30, alpha=0.3)

        # Set plot properties
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_aspect("equal")
        ax.axis("off")

        plt.tight_layout()
        return fig

    def create_progress_bar_widget(self, completion_percentage: float) -> plt.Figure:
        """Create a simple progress bar visualization.

        Args:
            completion_percentage: Completion percentage (0-100)

        Returns:
            matplotlib Figure with progress bar
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 2))

        # Background bar
        bg_bar = Rectangle(
            (1, 0.3),
            8,
            0.4,
            facecolor=self.badge_colors["pending"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(bg_bar)

        # Progress bar
        progress_width = (completion_percentage / 100) * 8
        progress_bar = Rectangle(
            (1, 0.3),
            progress_width,
            0.4,
            facecolor=self.badge_colors["completed"],
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(progress_bar)

        # Progress text
        ax.text(
            5,
            0.5,
            f"{completion_percentage:.1f}%",
            ha="center",
            va="center",
            fontweight="bold",
            fontsize=14,
        )

        # Day markers
        for day in range(1, 8):
            x_pos = 1 + (day - 1) * (8 / 7)
            ax.plot([x_pos, x_pos], [0.2, 0.8], "k-", linewidth=1, alpha=0.5)
            ax.text(x_pos, 1.0, f"D{day}", ha="center", va="center", fontsize=10)

        # Set plot properties
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 1.3)
        ax.axis("off")

        # Title
        ax.text(
            5,
            1.2,
            "🚀 Bootcamp Progress",
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        return fig


# Demo and usage examples
def demo_badges():
    """Demonstrate the badge system."""
    badges = CompletionBadges()

    # Demo progress badges
    completed_days = [1, 2, 3, 5]
    current_day = 6

    # Create visualizations
    progress_fig = badges.create_progress_badges(completed_days, current_day)
    plt.show()

    # Demo achievement badges
    achievements = [
        "Day 1 Completed",
        "Day 2 Completed",
        "Day 3 Completed",
        "Day 5 Completed",
        "Halfway Hero",
        "Deep Dive Day 5",
        "Mastery Day 2",
        "Dedicated Learner",
    ]

    achievement_fig = badges.create_achievement_badges(achievements)
    plt.show()

    # Demo completion certificate
    cert_fig = badges.create_completion_certificate("Demo Learner", 85.7, 22.5, 4.2)
    plt.show()

    # Demo progress bar
    progress_bar_fig = badges.create_progress_bar_widget(85.7)
    plt.show()


if __name__ == "__main__":
    demo_badges()
