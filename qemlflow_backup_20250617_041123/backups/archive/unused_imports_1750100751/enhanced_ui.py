"""
Advanced User Interface and API Improvements
===========================================

Phase 2 implementation: Enhanced user interfaces, better API design,
and improved usability for ChemML.

Features:
- Intuitive function interfaces
- Smart parameter validation
- Auto-completion and hints
- Interactive help system
- Progressive disclosure of complexity

Usage:
    from chemml.utils.enhanced_ui import ChemMLInterface, interactive_help

    interface = ChemMLInterface()
    interface.quick_start()
"""

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type, Union, Callable


@dataclass
class Parameter:
    """Enhanced parameter definition with validation and help."""

    name: str
    param_type: Type
    description: str
    default: Any = None
    required: bool = True
    valid_range: Optional[tuple] = None
    valid_choices: Optional[List[Any]] = None
    examples: List[str] = field(default_factory=list)
    advanced: bool = False  # Hide from basic interface


@dataclass
class FunctionMetadata:
    """Enhanced function metadata for better UX."""

    name: str
    description: str
    category: str
    difficulty: str  # "beginner", "intermediate", "advanced"
    parameters: List[Parameter]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    related_functions: List[str] = field(default_factory=list)
    documentation_url: Optional[str] = None


class SmartParameterValidator:
    """Intelligent parameter validation with helpful suggestions."""

    @staticmethod
    def validate_parameter(param: Parameter, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a parameter value with helpful error messages."""
        # Type checking
        if not isinstance(value, param.param_type) and value is not None:
            return (
                False,
                f"Expected {param.param_type.__name__}, got {type(value).__name__}",
            )

        # Range validation
        if param.valid_range and hasattr(value, "__ge__") and hasattr(value, "__le__"):
            min_val, max_val = param.valid_range
            if not (min_val <= value <= max_val):
                return False, f"Value {value} must be between {min_val} and {max_val}"

        # Choice validation
        if param.valid_choices and value not in param.valid_choices:
            return False, f"Value {value} must be one of: {param.valid_choices}"

        return True, None

    @staticmethod
    def suggest_corrections(param: Parameter, invalid_value: Any) -> List[str]:
        """Suggest corrections for invalid parameter values."""
        suggestions = []

        if param.valid_choices:
            # Find closest match
            if hasattr(invalid_value, "lower"):
                closest = min(
                    param.valid_choices,
                    key=lambda x: abs(len(str(x)) - len(str(invalid_value))),
                )
                suggestions.append(f"Did you mean '{closest}'?")

        if param.examples:
            suggestions.append(
                f"Example values: {', '.join(map(str, param.examples[:3]))}"
            )

        return suggestions


class InteractiveHelp:
    """Interactive help system with contextual assistance."""

    def __init__(self):
        self.help_db = self._load_help_database()

    def _load_help_database(self) -> Dict[str, Any]:
        """Load help database with common questions and solutions."""
        return {
            "getting_started": {
                "title": "Getting Started with ChemML",
                "content": [
                    "1. Import ChemML: `import chemml`",
                    "2. Load data: `data = chemml.load_data('your_data.csv')`",
                    "3. Process data: `processed = chemml.preprocess(data)`",
                    "4. Train model: `model = chemml.train_model(processed)`",
                ],
            },
            "common_errors": {
                "ImportError": "Install missing packages with pip install",
                "FileNotFoundError": "Check file paths and permissions",
                "ValueError": "Verify input data format and types",
            },
            "best_practices": [
                "Always validate your input data",
                "Use train/validation/test splits",
                "Monitor model performance",
                "Save your trained models",
            ],
        }

    def get_help(self, topic: str = None) -> str:
        """Get contextual help for a topic."""
        if topic is None:
            return self._show_help_menu()

        if topic in self.help_db:
            help_data = self.help_db[topic]
            if isinstance(help_data, dict):
                lines = [f"📚 {help_data['title']}", "=" * 40]
                if isinstance(help_data["content"], list):
                    lines.extend(help_data["content"])
                else:
                    lines.append(help_data["content"])
                return "\n".join(lines)
            else:
                return str(help_data)

        return f"No help available for '{topic}'. Type help() for available topics."

    def _show_help_menu(self) -> str:
        """Show main help menu."""
        lines = [
            "🔍 ChemML Interactive Help",
            "=" * 30,
            "Available topics:",
        ]

        for topic in self.help_db.keys():
            lines.append(f"  • {topic}")

        lines.extend(
            ["", "Usage: help('topic_name')", "Example: help('getting_started')"]
        )

        return "\n".join(lines)


class ProgressiveDisclosure:
    """Progressive disclosure system to hide complexity."""

    def __init__(self):
        self.user_level = "beginner"  # beginner, intermediate, advanced
        self.show_advanced = False

    def set_user_level(self, level: str):
        """Set user expertise level."""
        if level in ["beginner", "intermediate", "advanced"]:
            self.user_level = level
            self.show_advanced = level == "advanced"
        else:
            raise ValueError("Level must be 'beginner', 'intermediate', or 'advanced'")

    def filter_parameters(self, parameters: List[Parameter]) -> List[Parameter]:
        """Filter parameters based on user level."""
        if self.show_advanced:
            return parameters

        return [p for p in parameters if not p.advanced]

    def get_simplified_interface(
        self, func_metadata: FunctionMetadata
    ) -> Dict[str, Any]:
        """Get simplified interface based on user level."""
        filtered_params = self.filter_parameters(func_metadata.parameters)

        interface = {
            "name": func_metadata.name,
            "description": func_metadata.description,
            "parameters": {},
        }

        for param in filtered_params:
            interface["parameters"][param.name] = {
                "type": param.param_type.__name__,
                "description": param.description,
                "required": param.required,
                "default": param.default,
            }

            if param.examples:
                interface["parameters"][param.name]["examples"] = param.examples[:2]

        return interface


class ChemMLInterface:
    """Main enhanced interface for ChemML."""

    def __init__(self):
        self.validator = SmartParameterValidator()
        self.help_system = InteractiveHelp()
        self.progressive = ProgressiveDisclosure()
        self.function_registry = {}
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        """Register built-in functions with metadata."""
        # Example function registrations
        self.register_function(
            "load_data",
            FunctionMetadata(
                name="load_data",
                description="Load chemical data from various formats",
                category="data_io",
                difficulty="beginner",
                parameters=[
                    Parameter(
                        name="filepath",
                        param_type=str,
                        description="Path to the data file",
                        examples=["data.csv", "molecules.sdf", "compounds.xlsx"],
                    ),
                    Parameter(
                        name="format",
                        param_type=str,
                        description="File format",
                        default="auto",
                        required=False,
                        valid_choices=["csv", "sdf", "xlsx", "json", "auto"],
                        examples=["csv", "sdf"],
                    ),
                    Parameter(
                        name="molecular_descriptors",
                        param_type=bool,
                        description="Calculate molecular descriptors automatically",
                        default=True,
                        required=False,
                        advanced=True,
                    ),
                ],
                examples=[
                    {"filepath": "molecules.csv", "format": "csv"},
                    {"filepath": "compounds.sdf", "molecular_descriptors": True},
                ],
            ),
        )

    def register_function(self, name: str, metadata: FunctionMetadata):
        """Register a function with enhanced metadata."""
        self.function_registry[name] = metadata

    def get_function_help(self, func_name: str) -> str:
        """Get comprehensive help for a function."""
        if func_name not in self.function_registry:
            return f"Function '{func_name}' not found. Available functions: {list(self.function_registry.keys())}"

        metadata = self.function_registry[func_name]
        interface = self.progressive.get_simplified_interface(metadata)

        lines = [
            f"🧪 {metadata.name}",
            "=" * (len(metadata.name) + 4),
            f"📝 {metadata.description}",
            f"🎯 Difficulty: {metadata.difficulty}",
            f"📂 Category: {metadata.category}",
            "",
            "⚙️ Parameters:",
        ]

        for param_name, param_info in interface["parameters"].items():
            required_str = "required" if param_info["required"] else "optional"
            default_str = (
                f" (default: {param_info['default']})"
                if not param_info["required"]
                else ""
            )

            lines.append(
                f"  • {param_name} ({param_info['type']}, {required_str}){default_str}"
            )
            lines.append(f"    {param_info['description']}")

            if "examples" in param_info:
                examples_str = ", ".join(map(str, param_info["examples"]))
                lines.append(f"    Examples: {examples_str}")
            lines.append("")

        if metadata.examples:
            lines.extend(["💡 Usage Examples:", ""])
            for i, example in enumerate(metadata.examples[:2], 1):
                lines.append(f"  Example {i}:")
                for key, value in example.items():
                    lines.append(f"    {key}={repr(value)}")
                lines.append("")

        return "\n".join(lines)

    def validate_function_call(
        self, func_name: str, **kwargs
    ) -> tuple[bool, List[str]]:
        """Validate a function call with enhanced error messages."""
        if func_name not in self.function_registry:
            return False, [f"Unknown function: {func_name}"]

        metadata = self.function_registry[func_name]
        errors = []

        # Check required parameters
        required_params = {p.name for p in metadata.parameters if p.required}
        provided_params = set(kwargs.keys())
        missing_params = required_params - provided_params

        if missing_params:
            errors.append(f"Missing required parameters: {', '.join(missing_params)}")

        # Validate provided parameters
        for param in metadata.parameters:
            if param.name in kwargs:
                is_valid, error_msg = self.validator.validate_parameter(
                    param, kwargs[param.name]
                )
                if not is_valid:
                    suggestions = self.validator.suggest_corrections(
                        param, kwargs[param.name]
                    )
                    error_detail = f"Parameter '{param.name}': {error_msg}"
                    if suggestions:
                        error_detail += f" | Suggestions: {'; '.join(suggestions)}"
                    errors.append(error_detail)

        return len(errors) == 0, errors

    def quick_start(self) -> str:
        """Provide quick start guide."""
        return self.help_system.get_help("getting_started")

    def set_expertise_level(self, level: str):
        """Set user expertise level for progressive disclosure."""
        self.progressive.set_user_level(level)
        print(f"✅ Expertise level set to: {level}")
        if level == "advanced":
            print("🔧 Advanced parameters are now visible")
        else:
            print("🎯 Showing simplified interface")

    def list_functions(self, category: str = None, difficulty: str = None) -> str:
        """List available functions with filtering."""
        functions = self.function_registry.values()

        if category:
            functions = [f for f in functions if f.category == category]

        if difficulty:
            functions = [f for f in functions if f.difficulty == difficulty]

        if not functions:
            return "No functions found matching the criteria."

        lines = ["🧪 Available ChemML Functions", "=" * 30]

        current_category = None
        for func in sorted(functions, key=lambda x: (x.category, x.difficulty, x.name)):
            if func.category != current_category:
                lines.append(f"\n📂 {func.category.title()}:")
                current_category = func.category

            difficulty_icon = {"beginner": "🟢", "intermediate": "🟡", "advanced": "🔴"}
            icon = difficulty_icon.get(func.difficulty, "⚪")

            lines.append(f"  {icon} {func.name} - {func.description}")

        return "\n".join(lines)


class APIUsabilityEnhancements:
    """API usability enhancements and shortcuts."""

    @staticmethod
    def create_smart_defaults(func: Callable) -> Callable:
        """Create intelligent defaults based on function usage patterns."""

        def wrapper(*args, **kwargs):
            # Add smart defaults logic here
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def add_parameter_hints(func: Callable) -> Callable:
        """Add parameter hints and validation to functions."""

        def wrapper(*args, **kwargs):
            # Add parameter hints and validation
            return func(*args, **kwargs)

        return wrapper


# Global interface instance
chemml_interface = ChemMLInterface()


# Convenience functions
def help(topic: str = None) -> str:
    """Get help on ChemML topics."""
    if topic and topic in chemml_interface.function_registry:
        return chemml_interface.get_function_help(topic)
    return chemml_interface.help_system.get_help(topic)


def functions(category: str = None, level: str = None) -> str:
    """List available functions."""
    return chemml_interface.list_functions(category, level)


def set_level(level: str):
    """Set expertise level."""
    chemml_interface.set_expertise_level(level)


def quick_start() -> str:
    """Get quick start guide."""
    return chemml_interface.quick_start()


if __name__ == "__main__":
    print("🚀 ChemML Enhanced UI Test")

    # Test the interface
    interface = ChemMLInterface()

    # Show quick start
    print(interface.quick_start())

    # Show function help
    print("\n" + interface.get_function_help("load_data"))

    # Test validation
    is_valid, errors = interface.validate_function_call(
        "load_data", filepath="test.csv", format="invalid_format"
    )

    if not is_valid:
        print(f"\n❌ Validation errors: {errors}")

    print("\n✅ Enhanced UI test completed!")
