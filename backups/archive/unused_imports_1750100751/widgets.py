"""
Interactive Widgets for ChemML Tutorials
=======================================

Interactive widgets and educational tools to enhance learning experiences in computational chemistry and machine learning tutorials.

Key Features:
- Interactive assessments and quizzes
- Progress tracking dashboards
- Molecular visualization widgets
- Parameter tuning interfaces
- Real-time feedback systems
"""

import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import HTML, clear_output, display

    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    warnings.warn("ipywidgets not available. Interactive features will be limited.")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class InteractiveAssessment:
    """
    Interactive assessment widget for tutorial sections.

    This class creates interactive quizzes, concept checks, and knowledge
    assessments that can be embedded in notebooks.
    """

    def __init__(
        self, section: str, concepts: List[str], activities: List[Dict[str, Any]]
    ):
        """
        Initialize interactive assessment.

        Args:
            section (str): Tutorial section name
            concepts (List[str]): List of concepts being assessed
            activities (List[Dict]): Assessment activities configuration
        """
        self.section = section
        self.concepts = concepts
        self.activities = activities
        self.responses = {}
        self.start_time = datetime.now()
        self.widgets = {}

    def display(self):
        """Display the interactive assessment widget."""
        if not WIDGETS_AVAILABLE:
            self._display_fallback()
            return

        # Create assessment container
        assessment_container = widgets.VBox()

        # Header
        header = widgets.HTML(
            f"""
        <h3>🎯 Interactive Assessment: {self.section}</h3>
        <p><strong>Concepts:</strong> {', '.join(self.concepts)}</p>
        <hr>
        """
        )

        assessment_items = [header]

        # Create widgets for each activity
        for i, activity in enumerate(self.activities):
            activity_widget = self._create_activity_widget(i, activity)
            assessment_items.append(activity_widget)

        # Submit button
        submit_btn = widgets.Button(
            description="Submit Assessment", button_style="success", icon="check"
        )
        submit_btn.on_click(self._on_submit)

        # Progress indicator
        self.progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=len(self.activities),
            description="Progress:",
            bar_style="info",
        )

        assessment_items.extend([self.progress_bar, submit_btn])
        assessment_container.children = assessment_items

        display(assessment_container)

    def collect_feedback(self) -> Dict[str, Any]:
        """
        Collect user feedback and assessment results.

        Returns:
            Dict[str, Any]: Assessment results and feedback
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        results = {
            "section": self.section,
            "concepts": self.concepts,
            "responses": self.responses,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "completion_rate": (
                len(self.responses) / len(self.activities) if self.activities else 0
            ),
        }

        # Calculate score if applicable
        if self.responses:
            correct_count = sum(
                1 for resp in self.responses.values() if resp.get("correct", False)
            )
            results["score"] = correct_count / len(self.activities)

        return results

    def _create_activity_widget(
        self, index: int, activity: Dict[str, Any]
    ) -> widgets.Widget:
        """Create widget for a specific assessment activity."""
        activity_type = activity.get("type", "multiple_choice")

        if activity_type == "multiple_choice":
            return self._create_multiple_choice(index, activity)
        elif activity_type == "true_false":
            return self._create_true_false(index, activity)
        elif activity_type == "text_input":
            return self._create_text_input(index, activity)
        elif activity_type == "slider":
            return self._create_slider(index, activity)
        else:
            return widgets.HTML(f"<p>Unknown activity type: {activity_type}</p>")

    def _create_multiple_choice(
        self, index: int, activity: Dict[str, Any]
    ) -> widgets.Widget:
        """Create multiple choice question widget."""
        question = activity.get("question", f"Question {index + 1}")
        options = activity.get("options", [])

        question_html = widgets.HTML(f"<h4>Q{index + 1}: {question}</h4>")

        radio_buttons = widgets.RadioButtons(options=options, disabled=False)

        # Store widget reference
        self.widgets[f"activity_{index}"] = radio_buttons

        # Add change handler
        def on_change(change):
            self.responses[index] = {
                "type": "multiple_choice",
                "answer": change["new"],
                "correct": change["new"] == activity.get("correct_answer"),
            }
            self._update_progress()

        radio_buttons.observe(on_change, names="value")

        return widgets.VBox([question_html, radio_buttons])

    def _create_true_false(
        self, index: int, activity: Dict[str, Any]
    ) -> widgets.Widget:
        """Create true/false question widget."""
        question = activity.get("question", f"Question {index + 1}")

        question_html = widgets.HTML(f"<h4>Q{index + 1}: {question}</h4>")

        toggle = widgets.ToggleButtons(
            options=[("True", True), ("False", False)],
            description="Answer:",
            disabled=False,
            button_style="",
            tooltips=["True", "False"],
        )

        self.widgets[f"activity_{index}"] = toggle

        def on_change(change):
            self.responses[index] = {
                "type": "true_false",
                "answer": change["new"],
                "correct": change["new"] == activity.get("correct_answer"),
            }
            self._update_progress()

        toggle.observe(on_change, names="value")

        return widgets.VBox([question_html, toggle])

    def _create_text_input(
        self, index: int, activity: Dict[str, Any]
    ) -> widgets.Widget:
        """Create text input question widget."""
        question = activity.get("question", f"Question {index + 1}")
        placeholder = activity.get("placeholder", "Enter your answer...")

        question_html = widgets.HTML(f"<h4>Q{index + 1}: {question}</h4>")

        text_input = widgets.Text(
            value="", placeholder=placeholder, description="Answer:", disabled=False
        )

        self.widgets[f"activity_{index}"] = text_input

        def on_change(change):
            # For text inputs, we might have multiple correct answers
            correct_answers = activity.get("correct_answers", [])
            is_correct = change["new"].lower().strip() in [
                ans.lower() for ans in correct_answers
            ]

            self.responses[index] = {
                "type": "text_input",
                "answer": change["new"],
                "correct": is_correct,
            }
            self._update_progress()

        text_input.observe(on_change, names="value")

        return widgets.VBox([question_html, text_input])

    def _create_slider(self, index: int, activity: Dict[str, Any]) -> widgets.Widget:
        """Create slider question widget."""
        question = activity.get("question", f"Question {index + 1}")
        min_val = activity.get("min", 0)
        max_val = activity.get("max", 100)
        step = activity.get("step", 1)

        question_html = widgets.HTML(f"<h4>Q{index + 1}: {question}</h4>")

        slider = widgets.FloatSlider(
            value=min_val,
            min=min_val,
            max=max_val,
            step=step,
            description="Value:",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
        )

        self.widgets[f"activity_{index}"] = slider

        def on_change(change):
            # For sliders, check if answer is within acceptable range
            correct_range = activity.get("correct_range", [min_val, max_val])
            is_correct = correct_range[0] <= change["new"] <= correct_range[1]

            self.responses[index] = {
                "type": "slider",
                "answer": change["new"],
                "correct": is_correct,
            }
            self._update_progress()

        slider.observe(on_change, names="value")

        return widgets.VBox([question_html, slider])

    def _update_progress(self):
        """Update progress bar based on completed responses."""
        if hasattr(self, "progress_bar"):
            self.progress_bar.value = len(self.responses)

    def _on_submit(self, button):
        """Handle assessment submission."""
        results = self.collect_feedback()

        # Display results
        score = results.get("score", 0) * 100
        duration = results.get("duration_seconds", 0)

        results_html = f"""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0;">
        <h4>📊 Assessment Results</h4>
        <p><strong>Score:</strong> {score:.1f}%</p>
        <p><strong>Time taken:</strong> {duration:.1f} seconds</p>
        <p><strong>Completion rate:</strong> {results['completion_rate']:.1%}</p>
        </div>
        """

        display(HTML(results_html))

    def _display_fallback(self):
        """Display fallback assessment when widgets are not available."""
        print(f"\n🎯 Assessment: {self.section}")
        print(f"Concepts: {', '.join(self.concepts)}")
        print("-" * 50)

        for i, activity in enumerate(self.activities):
            print(f"\nQ{i + 1}: {activity.get('question', 'Question')}")

            if activity.get("type") == "multiple_choice":
                for j, option in enumerate(activity.get("options", [])):
                    print(f"  {chr(65 + j)}) {option}")
            elif activity.get("type") == "true_false":
                print("  Answer: True or False")

        print("\n💡 Note: Interactive features require ipywidgets installation")


class ProgressDashboard:
    """
    Progress tracking dashboard for learning analytics.

    This class creates interactive dashboards to visualize student progress,
    time tracking, and concept mastery across tutorial sessions.
    """

    def __init__(self, student_id: str = "demo"):
        """
        Initialize progress dashboard.

        Args:
            student_id (str): Unique identifier for the student
        """
        self.student_id = student_id
        self.session_data = []
        self.concept_mastery = {}

    def add_session_data(self, session_data: Dict[str, Any]):
        """Add data from a completed tutorial session."""
        session_data["timestamp"] = datetime.now().isoformat()
        self.session_data.append(session_data)

        # Update concept mastery
        if "concepts" in session_data:
            for concept in session_data["concepts"]:
                if concept not in self.concept_mastery:
                    self.concept_mastery[concept] = []

                score = session_data.get("score", 0)
                self.concept_mastery[concept].append(score)

    def create_time_tracking_plot(self) -> Any:
        """Create time tracking visualization."""
        if not self.session_data:
            print("No session data available for visualization")
            return None

        if PLOTLY_AVAILABLE:
            return self._create_plotly_time_plot()
        else:
            return self._create_matplotlib_time_plot()

    def create_concept_mastery_radar(self) -> Any:
        """Create concept mastery radar chart."""
        if not self.concept_mastery:
            print("No concept mastery data available")
            return None

        if PLOTLY_AVAILABLE:
            return self._create_plotly_radar()
        else:
            return self._create_matplotlib_radar()

    def create_daily_comparison(self) -> Any:
        """Create daily progress comparison chart."""
        if len(self.session_data) < 2:
            print("Need at least 2 sessions for comparison")
            return None

        if PLOTLY_AVAILABLE:
            return self._create_plotly_comparison()
        else:
            return self._create_matplotlib_comparison()

    def display_summary_stats(self):
        """Display summary statistics."""
        if not self.session_data:
            print("No data available")
            return

        total_time = sum(
            session.get("duration_seconds", 0) for session in self.session_data
        )
        avg_score = np.mean([session.get("score", 0) for session in self.session_data])
        total_concepts = len(self.concept_mastery)

        if WIDGETS_AVAILABLE:
            stats_html = f"""
            <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px;">
            <h3>📈 Learning Progress Summary</h3>
            <div style="display: flex; justify-content: space-around;">
                <div><strong>Total Study Time:</strong><br>{total_time/60:.1f} minutes</div>
                <div><strong>Average Score:</strong><br>{avg_score:.1%}</div>
                <div><strong>Concepts Covered:</strong><br>{total_concepts}</div>
                <div><strong>Sessions Completed:</strong><br>{len(self.session_data)}</div>
            </div>
            </div>
            """
            display(HTML(stats_html))
        else:
            print("\n📈 Learning Progress Summary")
            print(f"Total Study Time: {total_time/60:.1f} minutes")
            print(f"Average Score: {avg_score:.1%}")
            print(f"Concepts Covered: {total_concepts}")
            print(f"Sessions Completed: {len(self.session_data)}")

    def _create_plotly_time_plot(self):
        """Create time tracking plot using Plotly."""
        dates = [session.get("timestamp", "") for session in self.session_data]
        durations = [
            session.get("duration_seconds", 0) / 60 for session in self.session_data
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=durations,
                mode="lines+markers",
                name="Study Time",
                line=dict(color="blue", width=2),
                marker=dict(size=8),
            )
        )

        fig.update_layout(
            title="Study Time Tracking",
            xaxis_title="Date",
            yaxis_title="Time (minutes)",
            hovermode="x",
        )

        return fig

    def _create_matplotlib_time_plot(self):
        """Create time tracking plot using Matplotlib."""
        durations = [
            session.get("duration_seconds", 0) / 60 for session in self.session_data
        ]
        sessions = list(range(1, len(self.session_data) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(sessions, durations, "bo-", linewidth=2, markersize=8)
        plt.title("Study Time Tracking")
        plt.xlabel("Session Number")
        plt.ylabel("Time (minutes)")
        plt.grid(True, alpha=0.3)
        plt.show()

        return plt.gcf()

    def _create_plotly_radar(self):
        """Create concept mastery radar chart using Plotly."""
        concepts = list(self.concept_mastery.keys())
        avg_scores = [np.mean(scores) for scores in self.concept_mastery.values()]

        fig = go.Figure()

        fig.add_trace(
            go.Scatterpolar(
                r=avg_scores, theta=concepts, fill="toself", name="Concept Mastery"
            )
        )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Concept Mastery Overview",
        )

        return fig

    def _create_matplotlib_radar(self):
        """Create concept mastery radar chart using Matplotlib."""
        concepts = list(self.concept_mastery.keys())
        scores = [np.mean(scores) for scores in self.concept_mastery.values()]

        # Angles for radar chart
        angles = np.linspace(0, 2 * np.pi, len(concepts), endpoint=False)
        scores_plot = scores + [scores[0]]  # Complete the circle
        angles_plot = np.concatenate((angles, [angles[0]]))

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
        ax.plot(angles_plot, scores_plot, "o-", linewidth=2)
        ax.fill(angles_plot, scores_plot, alpha=0.25)
        ax.set_xticks(angles)
        ax.set_xticklabels(concepts)
        ax.set_ylim(0, 1)
        ax.set_title("Concept Mastery Overview", y=1.08)
        plt.show()

        return fig

    def _create_plotly_comparison(self):
        """Create daily comparison chart using Plotly."""
        # Group sessions by day
        daily_scores = {}
        for session in self.session_data:
            date = session.get("timestamp", "")[:10]  # Get date part
            score = session.get("score", 0)
            if date not in daily_scores:
                daily_scores[date] = []
            daily_scores[date].append(score)

        dates = list(daily_scores.keys())
        avg_scores = [np.mean(scores) for scores in daily_scores.values()]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=dates, y=avg_scores, name="Daily Average Score"))

        fig.update_layout(
            title="Daily Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Average Score",
            yaxis=dict(range=[0, 1]),
        )

        return fig

    def _create_matplotlib_comparison(self):
        """Create daily comparison chart using Matplotlib."""
        sessions = list(range(1, len(self.session_data) + 1))
        scores = [session.get("score", 0) for session in self.session_data]

        plt.figure(figsize=(10, 6))
        plt.bar(sessions, scores, alpha=0.7, color="green")
        plt.title("Session Performance Comparison")
        plt.xlabel("Session Number")
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.show()

        return plt.gcf()


class MolecularVisualizationWidget:
    """
    Interactive molecular visualization widget.

    This class provides interactive molecular visualization capabilities
    for educational purposes in computational chemistry tutorials.
    """

    def __init__(self):
        """Initialize molecular visualization widget."""
        self.molecules = {}
        self.current_molecule = None

    def add_molecule(self, name: str, smiles: str):
        """Add a molecule to the visualization widget."""
        if RDKIT_AVAILABLE:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                self.molecules[name] = {"smiles": smiles, "mol": mol}
            else:
                warnings.warn(f"Invalid SMILES: {smiles}")
        else:
            self.molecules[name] = {"smiles": smiles, "mol": None}

    def create_molecule_selector(self) -> Any:
        """Create interactive molecule selector widget."""
        if not WIDGETS_AVAILABLE:
            self._display_molecule_list()
            return None

        if not self.molecules:
            return widgets.HTML("<p>No molecules loaded</p>")

        # Molecule selector
        selector = widgets.Dropdown(
            options=list(self.molecules.keys()), description="Molecule:", disabled=False
        )

        # Output area for visualization
        output = widgets.Output()

        def on_molecule_change(change):
            with output:
                clear_output(wait=True)
                self.visualize_molecule(change["new"])

        selector.observe(on_molecule_change, names="value")

        # Initialize with first molecule
        if self.molecules:
            first_mol = list(self.molecules.keys())[0]
            with output:
                self.visualize_molecule(first_mol)

        return widgets.VBox([selector, output])

    def visualize_molecule(self, name: str):
        """Visualize a specific molecule."""
        if name not in self.molecules:
            print(f"Molecule {name} not found")
            return

        molecule_data = self.molecules[name]
        self.current_molecule = name

        if RDKIT_AVAILABLE and molecule_data["mol"] is not None:
            # Use RDKit for visualization
            img = Draw.MolToImage(molecule_data["mol"], size=(400, 400))
            display(img)

            # Display molecular properties
            mol = molecule_data["mol"]
            properties_html = f"""
            <div style="margin-top: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;">
            <h4>{name}</h4>
            <p><strong>SMILES:</strong> {molecule_data['smiles']}</p>
            <p><strong>Molecular Formula:</strong> {Chem.rdMolDescriptors.CalcMolFormula(mol)}</p>
            <p><strong>Molecular Weight:</strong> {Chem.rdMolDescriptors.CalcExactMolWt(mol):.2f}</p>
            <p><strong>Number of Atoms:</strong> {mol.GetNumAtoms()}</p>
            <p><strong>Number of Bonds:</strong> {mol.GetNumBonds()}</p>
            </div>
            """

            if WIDGETS_AVAILABLE:
                display(HTML(properties_html))
            else:
                print(f"Molecule: {name}")
                print(f"SMILES: {molecule_data['smiles']}")
        else:
            # Fallback display
            print(f"Molecule: {name}")
            print(f"SMILES: {molecule_data['smiles']}")
            print("Note: Install RDKit for molecular visualization")

    def create_property_comparison(self, properties: List[str]) -> Any:
        """Create interactive property comparison widget."""
        if not self.molecules or not WIDGETS_AVAILABLE:
            return None

        # Calculate properties for all molecules
        prop_data = {}
        for name, mol_data in self.molecules.items():
            if RDKIT_AVAILABLE and mol_data["mol"] is not None:
                mol = mol_data["mol"]
                props = {}
                for prop in properties:
                    if prop == "molecular_weight":
                        props[prop] = Chem.rdMolDescriptors.CalcExactMolWt(mol)
                    elif prop == "num_atoms":
                        props[prop] = mol.GetNumAtoms()
                    elif prop == "num_bonds":
                        props[prop] = mol.GetNumBonds()

                prop_data[name] = props

        if not prop_data:
            return widgets.HTML("<p>No property data available</p>")

        # Create comparison visualization
        output = widgets.Output()

        with output:
            if PLOTLY_AVAILABLE:
                self._create_plotly_property_comparison(prop_data, properties)
            else:
                self._create_matplotlib_property_comparison(prop_data, properties)

        return output

    def _display_molecule_list(self):
        """Display molecule list as fallback."""
        print("\n🧪 Available Molecules:")
        print("-" * 30)
        for name, data in self.molecules.items():
            print(f"{name}: {data['smiles']}")

    def _create_plotly_property_comparison(
        self, prop_data: Dict, properties: List[str]
    ):
        """Create property comparison using Plotly."""
        molecules = list(prop_data.keys())

        fig = make_subplots(rows=1, cols=len(properties), subplot_titles=properties)

        for i, prop in enumerate(properties):
            values = [prop_data[mol].get(prop, 0) for mol in molecules]

            fig.add_trace(go.Bar(x=molecules, y=values, name=prop), row=1, col=i + 1)

        fig.update_layout(title="Molecular Property Comparison", showlegend=False)

        fig.show()

    def _create_matplotlib_property_comparison(
        self, prop_data: Dict, properties: List[str]
    ):
        """Create property comparison using Matplotlib."""
        molecules = list(prop_data.keys())

        fig, axes = plt.subplots(1, len(properties), figsize=(15, 5))

        if len(properties) == 1:
            axes = [axes]

        for i, prop in enumerate(properties):
            values = [prop_data[mol].get(prop, 0) for mol in molecules]

            axes[i].bar(molecules, values)
            axes[i].set_title(prop)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()


# Convenience functions for easy widget creation


def create_assessment(
    section: str, concepts: List[str], questions: List[Dict[str, Any]]
) -> InteractiveAssessment:
    """
    Create an interactive assessment widget.

    Args:
        section (str): Tutorial section name
        concepts (List[str]): List of concepts being assessed
        questions (List[Dict]): List of question configurations

    Returns:
        InteractiveAssessment: Interactive assessment widget
    """
    return InteractiveAssessment(section, concepts, questions)


def create_progress_dashboard(student_id: str = "demo") -> ProgressDashboard:
    """
    Create a progress tracking dashboard.

    Args:
        student_id (str): Student identifier

    Returns:
        ProgressDashboard: Progress dashboard widget
    """
    return ProgressDashboard(student_id)


def create_molecule_viewer(molecules: Dict[str, str]) -> MolecularVisualizationWidget:
    """
    Create a molecular visualization widget.

    Args:
        molecules (Dict[str, str]): Dictionary of molecule names to SMILES

    Returns:
        MolecularVisualizationWidget: Molecular visualization widget
    """
    viewer = MolecularVisualizationWidget()
    for name, smiles in molecules.items():
        viewer.add_molecule(name, smiles)

    return viewer


def check_widget_requirements() -> Dict[str, bool]:
    """
    Check if widget requirements are met.

    Returns:
        Dict[str, bool]: Status of widget dependencies
    """
    return {
        "ipywidgets": WIDGETS_AVAILABLE,
        "plotly": PLOTLY_AVAILABLE,
        "rdkit": RDKIT_AVAILABLE,
    }
