"""
QeMLflow Auto-Documentation Generator
=================================

Automatically generates comprehensive API documentation with examples.
Scans the codebase and creates interactive, searchable documentation.
"""

import ast
import importlib.util
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class APIDocGenerator:
    """Automatically generate comprehensive API documentation."""

    def __init__(
        self, source_dir: str = "src/qemlflow", output_dir: str = "docs/api_auto"
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.modules_info = {}
        self.functions_info = {}
        self.classes_info = {}

    def scan_and_document(self) -> Dict[str, Any]:
        """Scan codebase and generate comprehensive documentation."""
        print("🔍 Scanning QeMLflow codebase for API documentation...")

        # Discover all Python modules
        modules = self._discover_modules()
        print(f"📦 Found {len(modules)} modules to document")

        # Extract information from each module
        for module_path in modules:
            try:
                self._analyze_module(module_path)
            except Exception as e:
                print(f"⚠️  Warning: Could not analyze {module_path}: {e}")

        # Generate documentation
        self._generate_module_docs()
        self._generate_function_docs()
        self._generate_class_docs()
        self._generate_search_index()
        self._generate_main_index()

        print(f"✅ API documentation generated in {self.output_dir}")
        return {
            "modules": len(self.modules_info),
            "functions": len(self.functions_info),
            "classes": len(self.classes_info),
        }

    def _discover_modules(self) -> List[Path]:
        """Discover all Python modules in the source directory."""
        modules = []
        for py_file in self.source_dir.rglob("*.py"):
            if not py_file.name.startswith("_") or py_file.name == "__init__.py":
                modules.append(py_file)
        return sorted(modules)

    def _analyze_module(self, module_path: Path):
        """Analyze a Python module and extract API information."""
        # Read the source code
        try:
            source_code = module_path.read_text(encoding="utf-8")
        except Exception:
            return

        # Parse AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return

        # Extract module information
        module_name = self._get_module_name(module_path)
        module_info = {
            "name": module_name,
            "path": str(module_path),
            "docstring": ast.get_docstring(tree),
            "functions": [],
            "classes": [],
            "imports": [],
        }

        # Walk the AST to extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = self._extract_function_info(node, module_name)
                module_info["functions"].append(func_info)
                self.functions_info[f"{module_name}.{func_info['name']}"] = func_info

            elif isinstance(node, ast.ClassDef):
                class_info = self._extract_class_info(node, module_name)
                module_info["classes"].append(class_info)
                self.classes_info[f"{module_name}.{class_info['name']}"] = class_info

        self.modules_info[module_name] = module_info

    def _get_module_name(self, module_path: Path) -> str:
        """Convert file path to module name."""
        relative_path = module_path.relative_to(self.source_dir)
        if relative_path.name == "__init__.py":
            parts = relative_path.parent.parts
        else:
            parts = relative_path.with_suffix("").parts
        return ".".join(parts)

    def _extract_function_info(
        self, node: ast.FunctionDef, module_name: str
    ) -> Dict[str, Any]:
        """Extract information about a function."""
        # Get function signature
        args = []
        for arg in node.args.args:
            arg_info = {"name": arg.arg}
            if arg.annotation:
                arg_info["type"] = (
                    ast.unparse(arg.annotation) if hasattr(ast, "unparse") else "Any"
                )
            args.append(arg_info)

        # Get return type
        return_type = "Any"
        if node.returns:
            return_type = (
                ast.unparse(node.returns) if hasattr(ast, "unparse") else "Any"
            )

        return {
            "name": node.name,
            "module": module_name,
            "docstring": ast.get_docstring(node),
            "args": args,
            "return_type": return_type,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "decorators": [
                ast.unparse(dec) if hasattr(ast, "unparse") else "decorator"
                for dec in node.decorator_list
            ],
            "lineno": node.lineno,
        }

    def _extract_class_info(
        self, node: ast.ClassDef, module_name: str
    ) -> Dict[str, Any]:
        """Extract information about a class."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self._extract_function_info(item, module_name)
                method_info["is_method"] = True
                methods.append(method_info)

        return {
            "name": node.name,
            "module": module_name,
            "docstring": ast.get_docstring(node),
            "bases": [
                ast.unparse(base) if hasattr(ast, "unparse") else "base"
                for base in node.bases
            ],
            "methods": methods,
            "decorators": [
                ast.unparse(dec) if hasattr(ast, "unparse") else "decorator"
                for dec in node.decorator_list
            ],
            "lineno": node.lineno,
        }

    def _generate_examples(self, item_info: Dict[str, Any]) -> List[str]:
        """Generate code examples for functions and classes."""
        examples = []

        # Basic usage example
        if item_info.get("name"):
            if "args" in item_info:  # Function
                args_example = ", ".join(
                    [f"{arg['name']}=..." for arg in item_info["args"][:3]]
                )
                examples.append(f"result = {item_info['name']}({args_example})")
            else:  # Class
                examples.append(f"instance = {item_info['name']}()")

        # Generate domain-specific examples based on function name
        name = item_info.get("name", "").lower()
        if any(keyword in name for keyword in ["molecular", "smiles", "mol"]):
            examples.append(
                '# Example with molecular data\nsmiles = "CCO"\nresult = {}(smiles)'.format(
                    item_info["name"]
                )
            )

        if any(keyword in name for keyword in ["train", "fit", "model"]):
            examples.append(
                "# Example with training data\nX_train, y_train = load_data()\nmodel.{}(X_train, y_train)".format(
                    item_info["name"]
                )
            )

        if any(keyword in name for keyword in ["predict", "score", "evaluate"]):
            examples.append(
                "# Example with prediction\nX_test = load_test_data()\npredictions = model.{}(X_test)".format(
                    item_info["name"]
                )
            )

        return examples

    def _generate_function_docs(self):
        """Generate documentation for all functions."""
        functions_html = self._generate_html_template("Functions", "🔧")

        content = ""
        for func_name, func_info in sorted(self.functions_info.items()):
            examples = self._generate_examples(func_info)

            content += f"""
            <div class="api-item">
                <h3 class="api-name">🔧 {func_info['name']}</h3>
                <div class="api-meta">
                    <span class="module-name">{func_info['module']}</span>
                    {"<span class='async-badge'>async</span>" if func_info.get('is_async') else ""}
                </div>
                <div class="api-signature">
                    <code>{func_info['name']}({', '.join([arg['name'] + (f": {arg.get('type', 'Any')}" if 'type' in arg else '') for arg in func_info.get('args', [])])})</code>
                    <span class="return-type">→ {func_info.get('return_type', 'Any')}</span>
                </div>
                <div class="api-description">
                    {func_info.get('docstring', 'No description available.') or 'No description available.'}
                </div>
                {"<div class='api-examples'><h4>Examples:</h4>" + ''.join([f"<pre><code>{ex}</code></pre>" for ex in examples]) + "</div>" if examples else ""}
            </div>
            """

        functions_html = functions_html.replace("{{CONTENT}}", content)

        with open(self.output_dir / "functions.html", "w", encoding="utf-8") as f:
            f.write(functions_html)

    def _generate_class_docs(self):
        """Generate documentation for all classes."""
        classes_html = self._generate_html_template("Classes", "🏗️")

        content = ""
        for class_name, class_info in sorted(self.classes_info.items()):
            examples = self._generate_examples(class_info)

            methods_html = ""
            for method in class_info.get("methods", []):
                methods_html += f"""
                <div class="method-item">
                    <h5>{method['name']}</h5>
                    <code>{method['name']}({', '.join([arg['name'] for arg in method.get('args', [])])})</code>
                    <p>{method.get('docstring', 'No description available.') or 'No description available.'}</p>
                </div>
                """

            content += f"""
            <div class="api-item">
                <h3 class="api-name">🏗️ {class_info['name']}</h3>
                <div class="api-meta">
                    <span class="module-name">{class_info['module']}</span>
                    {f"<span class='bases'>Inherits: {', '.join(class_info.get('bases', []))}</span>" if class_info.get('bases') else ""}
                </div>
                <div class="api-description">
                    {class_info.get('docstring', 'No description available.') or 'No description available.'}
                </div>
                {"<div class='api-examples'><h4>Examples:</h4>" + ''.join([f"<pre><code>{ex}</code></pre>" for ex in examples]) + "</div>" if examples else ""}
                {"<div class='methods-section'><h4>Methods:</h4>" + methods_html + "</div>" if methods_html else ""}
            </div>
            """

        classes_html = classes_html.replace("{{CONTENT}}", content)

        with open(self.output_dir / "classes.html", "w", encoding="utf-8") as f:
            f.write(classes_html)

    def _generate_module_docs(self):
        """Generate documentation for all modules."""
        modules_html = self._generate_html_template("Modules", "📦")

        content = ""
        for module_name, module_info in sorted(self.modules_info.items()):
            content += f"""
            <div class="api-item">
                <h3 class="api-name">📦 {module_name}</h3>
                <div class="api-meta">
                    <span class="path">{module_info['path']}</span>
                </div>
                <div class="api-description">
                    {module_info.get('docstring', 'No description available.') or 'No description available.'}
                </div>
                <div class="module-contents">
                    <div class="content-section">
                        <h4>Functions ({len(module_info.get('functions', []))})</h4>
                        <ul>
                            {''.join([f"<li><a href='functions.html#{func['name']}'>{func['name']}</a></li>" for func in module_info.get('functions', [])])}
                        </ul>
                    </div>
                    <div class="content-section">
                        <h4>Classes ({len(module_info.get('classes', []))})</h4>
                        <ul>
                            {''.join([f"<li><a href='classes.html#{cls['name']}'>{cls['name']}</a></li>" for cls in module_info.get('classes', [])])}
                        </ul>
                    </div>
                </div>
            </div>
            """

        modules_html = modules_html.replace("{{CONTENT}}", content)

        with open(self.output_dir / "modules.html", "w", encoding="utf-8") as f:
            f.write(modules_html)

    def _generate_search_index(self):
        """Generate search index for the documentation."""
        search_data = []

        # Add functions to search index
        for func_name, func_info in self.functions_info.items():
            search_data.append(
                {
                    "type": "function",
                    "name": func_info["name"],
                    "full_name": func_name,
                    "module": func_info["module"],
                    "description": (
                        func_info.get("docstring", "")[:200]
                        if func_info.get("docstring")
                        else ""
                    ),
                    "url": f"functions.html#{func_info['name']}",
                }
            )

        # Add classes to search index
        for class_name, class_info in self.classes_info.items():
            search_data.append(
                {
                    "type": "class",
                    "name": class_info["name"],
                    "full_name": class_name,
                    "module": class_info["module"],
                    "description": (
                        class_info.get("docstring", "")[:200]
                        if class_info.get("docstring")
                        else ""
                    ),
                    "url": f"classes.html#{class_info['name']}",
                }
            )

        # Save search index
        with open(self.output_dir / "search_index.json", "w", encoding="utf-8") as f:
            json.dump(search_data, f, indent=2)

    def _generate_main_index(self):
        """Generate main index page."""
        stats = {
            "modules": len(self.modules_info),
            "functions": len(self.functions_info),
            "classes": len(self.classes_info),
        }

        index_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QeMLflow API Documentation</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>🧬 QeMLflow API Documentation</h1>
            <p>Comprehensive, auto-generated API reference</p>
        </header>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{stats['modules']}</div>
                <div class="stat-label">Modules</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['functions']}</div>
                <div class="stat-label">Functions</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{stats['classes']}</div>
                <div class="stat-label">Classes</div>
            </div>
        </div>

        <div class="navigation-grid">
            <a href="modules.html" class="nav-card">
                <h3>📦 Modules</h3>
                <p>Browse all QeMLflow modules and their organization</p>
            </a>
            <a href="functions.html" class="nav-card">
                <h3>🔧 Functions</h3>
                <p>Explore all available functions with examples</p>
            </a>
            <a href="classes.html" class="nav-card">
                <h3>🏗️ Classes</h3>
                <p>Discover classes and their methods</p>
            </a>
        </div>

        <div class="search-section">
            <h2>🔍 Search API</h2>
            <input type="text" id="search-input" placeholder="Search functions, classes, or modules...">
            <div id="search-results"></div>
        </div>
    </div>

    <script src="search.js"></script>
</body>
</html>
        """

        with open(self.output_dir / "index.html", "w", encoding="utf-8") as f:
            f.write(index_html)

        # Generate CSS and JavaScript
        self._generate_styles()
        self._generate_search_script()

    def _generate_html_template(self, title: str, icon: str) -> str:
        """Generate HTML template for documentation pages."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - QeMLflow API</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <div class="container">
        <header class="header">
            <nav class="breadcrumb">
                <a href="index.html">Home</a> > <span>{title}</span>
            </nav>
            <h1>{icon} {title}</h1>
        </header>

        <div class="content">
            {{{{CONTENT}}}}
        </div>
    </div>
</body>
</html>
        """

    def _generate_styles(self):
        """Generate CSS styles for documentation."""
        css_content = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f8f9fa;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 40px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 12px;
        }

        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 700;
        }

        .breadcrumb {
            margin-bottom: 20px;
            font-size: 14px;
        }

        .breadcrumb a {
            color: rgba(255, 255, 255, 0.8);
            text-decoration: none;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .stat-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .stat-number {
            font-size: 3em;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 10px;
        }

        .stat-label {
            font-size: 1.1em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .navigation-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .nav-card {
            background: white;
            padding: 30px;
            border-radius: 12px;
            text-decoration: none;
            color: inherit;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .nav-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.15);
        }

        .nav-card h3 {
            margin: 0 0 15px 0;
            color: #667eea;
        }

        .api-item {
            background: white;
            margin-bottom: 30px;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .api-name {
            margin: 0 0 15px 0;
            color: #333;
            font-size: 1.5em;
        }

        .api-meta {
            margin-bottom: 15px;
            font-size: 14px;
            color: #666;
        }

        .module-name {
            background: #e3f2fd;
            padding: 4px 8px;
            border-radius: 4px;
            font-family: monospace;
        }

        .async-badge {
            background: #fff3e0;
            color: #f57c00;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 10px;
        }

        .api-signature {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            border-left: 4px solid #667eea;
        }

        .return-type {
            color: #666;
            margin-left: 10px;
        }

        .api-examples {
            margin-top: 20px;
        }

        .api-examples h4 {
            color: #667eea;
            margin-bottom: 10px;
        }

        .api-examples pre {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 10px;
        }

        .search-section {
            background: white;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        #search-input {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            margin-bottom: 20px;
        }

        #search-input:focus {
            outline: none;
            border-color: #667eea;
        }

        #search-results {
            max-height: 400px;
            overflow-y: auto;
        }

        .search-result {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            cursor: pointer;
        }

        .search-result:hover {
            background: #f8f9fa;
        }

        .methods-section {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .method-item {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #e0e0e0;
        }

        .method-item:last-child {
            border-bottom: none;
        }
        """

        with open(self.output_dir / "style.css", "w", encoding="utf-8") as f:
            f.write(css_content)

    def _generate_search_script(self):
        """Generate JavaScript for search functionality."""
        js_content = """
        let searchIndex = [];

        // Load search index
        fetch('search_index.json')
            .then(response => response.json())
            .then(data => {
                searchIndex = data;
            });

        // Search functionality
        document.getElementById('search-input').addEventListener('input', function(e) {
            const query = e.target.value.toLowerCase();
            const results = document.getElementById('search-results');

            if (query.length < 2) {
                results.innerHTML = '';
                return;
            }

            const matches = searchIndex.filter(item =>
                item.name.toLowerCase().includes(query) ||
                item.module.toLowerCase().includes(query) ||
                item.description.toLowerCase().includes(query)
            ).slice(0, 10);

            results.innerHTML = matches.map(item => `
                <div class="search-result" onclick="window.location.href='${item.url}'">
                    <h4>${item.type === 'function' ? '🔧' : '🏗️'} ${item.name}</h4>
                    <p><strong>${item.module}</strong></p>
                    <p>${item.description}</p>
                </div>
            `).join('');
        });
        """

        with open(self.output_dir / "search.js", "w", encoding="utf-8") as f:
            f.write(js_content)


# Command-line interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QeMLflow API documentation")
    parser.add_argument(
        "--source", default="src/qemlflow", help="Source directory to scan"
    )
    parser.add_argument(
        "--output", default="docs/api_auto", help="Output directory for documentation"
    )

    args = parser.parse_args()

    generator = APIDocGenerator(args.source, args.output)
    stats = generator.scan_and_document()

    print("\n📊 Documentation Statistics:")
    print(f"   📦 Modules: {stats['modules']}")
    print(f"   🔧 Functions: {stats['functions']}")
    print(f"   🏗️ Classes: {stats['classes']}")
    print(
        f"\n🌐 Open {args.output}/index.html in your browser to view the documentation!"
    )
