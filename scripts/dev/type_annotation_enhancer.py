#!/usr/bin/env python3
"""
QeMLflow Type Annotation Enhancement Tool
========================================

Systematically adds type annotations to Python files to improve code quality,
IDE support, and documentation.

Usage:
    python tools/type_annotation_enhancer.py [--file PATH] [--directory PATH] [--dry-run]
"""

import ast
import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any


class TypeAnnotationEnhancer:
    """Systematically enhance Python files with type annotations."""
    
    def __init__(self):
        self.common_imports = {
            'typing': {'Dict', 'List', 'Set', 'Tuple', 'Optional', 'Any', 'Union', 'Callable'},
            'numpy': {'ndarray'},
            'pandas': {'DataFrame', 'Series'},
        }
        
        self.common_patterns = {
            # Function return patterns
            r'def.*load.*\(': 'Dict[str, Any]',
            r'def.*create.*\(': 'Any',
            r'def.*get.*\(': 'Any',
            r'def.*process.*\(': 'Any',
            r'def.*evaluate.*\(': 'Dict[str, float]',
            r'def.*predict.*\(': 'np.ndarray',
            r'def.*fit.*\(': 'None',
            r'def.*transform.*\(': 'np.ndarray',
            r'def.*split.*\(': 'Tuple[Any, ...]',
            
            # Parameter patterns
            'data': 'pd.DataFrame',
            'X': 'np.ndarray',
            'y': 'np.ndarray', 
            'model': 'Any',
            'config': 'Dict[str, Any]',
            'params': 'Dict[str, Any]',
            'features': 'List[str]',
            'molecules': 'List[str]',
            'smiles': 'List[str]',
            'results': 'Dict[str, Any]',
            'metrics': 'List[str]',
            'path': 'Union[str, Path]',
            'filepath': 'Union[str, Path]',
            'filename': 'str',
            'name': 'str',
            'verbose': 'bool',
            'seed': 'int',
            'n_samples': 'int',
            'test_size': 'float',
        }

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file for missing type annotations."""
        if not file_path.exists():
            return {'error': f'File not found: {file_path}'}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            analyzer = FunctionAnalyzer(str(file_path))
            analyzer.visit(tree)
            
            return {
                'file': str(file_path),
                'functions_without_hints': analyzer.functions_without_hints,
                'total_functions': analyzer.total_functions,
                'needs_typing_import': analyzer.needs_typing_import,
                'current_imports': analyzer.current_imports,
            }
            
        except Exception as e:
            return {'error': f'Error analyzing {file_path}: {e}'}

    def enhance_file(self, file_path: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Add type annotations to a Python file."""
        analysis = self.analyze_file(file_path)
        
        if 'error' in analysis:
            return analysis
            
        if not analysis['functions_without_hints']:
            return {'status': 'No enhancements needed', 'file': str(file_path)}
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            enhanced_lines = self._add_type_hints(lines, analysis)
            enhanced_content = ''.join(enhanced_lines)
            
            if dry_run:
                return {
                    'status': 'Dry run - changes not applied',
                    'file': str(file_path),
                    'functions_enhanced': len(analysis['functions_without_hints']),
                    'preview': enhanced_content[:500] + '...' if len(enhanced_content) > 500 else enhanced_content
                }
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(enhanced_content)
                    
                return {
                    'status': 'Enhanced successfully',
                    'file': str(file_path),
                    'functions_enhanced': len(analysis['functions_without_hints'])
                }
                
        except Exception as e:
            return {'error': f'Error enhancing {file_path}: {e}'}

    def _add_type_hints(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Add type hints to function definitions."""
        enhanced_lines = lines[:]
        
        # Add typing imports if needed
        if analysis['needs_typing_import']:
            enhanced_lines = self._add_typing_imports(enhanced_lines, analysis)
            
        # Process each function that needs type hints
        for func_info in analysis['functions_without_hints']:
            enhanced_lines = self._enhance_function(enhanced_lines, func_info)
            
        return enhanced_lines

    def _add_typing_imports(self, lines: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Add necessary typing imports."""
        typing_imports = {'Dict', 'List', 'Optional', 'Any', 'Union', 'Tuple'}
        current_typing = analysis['current_imports'].get('typing', set())
        needed_imports = typing_imports - current_typing
        
        if not needed_imports:
            return lines
            
        # Find where to insert the import
        insert_line = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_line = i + 1
            elif line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                break
                
        import_statement = f"from typing import {', '.join(sorted(needed_imports))}\n"
        lines.insert(insert_line, import_statement)
        
        return lines

    def _enhance_function(self, lines: List[str], func_info: Dict[str, Any]) -> List[str]:
        """Add type hints to a specific function."""
        func_line = func_info['line'] - 1  # Convert to 0-based indexing
        
        if func_line >= len(lines):
            return lines
            
        original_line = lines[func_line]
        
        # Skip if already has type hints or is too complex
        if '->' in original_line or 'def __' in original_line:
            return lines
            
        # Simple pattern-based type hint addition
        enhanced_line = self._add_simple_type_hints(original_line, func_info)
        lines[func_line] = enhanced_line
        
        return lines

    def _add_simple_type_hints(self, line: str, func_info: Dict[str, Any]) -> str:
        """Add simple type hints based on common patterns."""
        # Extract function signature
        match = re.match(r'(\s*def\s+\w+\s*\([^)]*\)\s*)(.*)', line)
        if not match:
            return line
            
        signature, rest = match.groups()
        
        # Add return type hint if none exists
        if '->' not in rest and ':' in rest:
            func_name = func_info.get('name', '')
            
            # Determine return type based on function name patterns
            return_type = self._infer_return_type(func_name, line)
            
            if return_type:
                # Insert return type hint before the colon
                if rest.strip().endswith(':'):
                    enhanced_rest = rest.replace(':', f' -> {return_type}:')
                else:
                    enhanced_rest = rest
                return signature + enhanced_rest
                
        return line

    def _infer_return_type(self, func_name: str, line: str) -> Optional[str]:
        """Infer return type based on function name and context."""
        # Common return type patterns
        if any(pattern in func_name.lower() for pattern in ['load', 'get', 'create']):
            return 'Any'
        elif any(pattern in func_name.lower() for pattern in ['predict', 'transform']):
            return 'np.ndarray'
        elif any(pattern in func_name.lower() for pattern in ['fit', 'train', 'update']):
            return 'None'
        elif any(pattern in func_name.lower() for pattern in ['evaluate', 'score']):
            return 'Dict[str, float]'
        elif any(pattern in func_name.lower() for pattern in ['split']):
            return 'Tuple[Any, ...]'
        elif func_name.startswith('is_') or func_name.startswith('has_'):
            return 'bool'
        elif func_name.startswith('__') and func_name.endswith('__'):
            # Skip magic methods for now
            return None
        else:
            return 'Any'

    def enhance_directory(self, directory: Path, dry_run: bool = False) -> Dict[str, Any]:
        """Enhance all Python files in a directory."""
        if not directory.exists():
            return {'error': f'Directory not found: {directory}'}
            
        python_files = list(directory.rglob('*.py'))
        results = []
        
        for file_path in python_files:
            # Skip __pycache__ and test files for now
            if '__pycache__' in str(file_path) or 'test_' in file_path.name:
                continue
                
            result = self.enhance_file(file_path, dry_run)
            results.append(result)
            
        enhanced_count = sum(1 for r in results if r.get('status', '').startswith('Enhanced'))
        total_files = len(results)
        
        return {
            'status': f'Processed {total_files} files, enhanced {enhanced_count}',
            'results': results,
            'enhanced_count': enhanced_count,
            'total_files': total_files
        }


class FunctionAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze functions for missing type annotations."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.functions_without_hints: List[Dict[str, Any]] = []
        self.total_functions = 0
        self.current_imports: Dict[str, Any] = {'typing': set()}
        self.needs_typing_import = False
        
    def visit_Import(self, node):
        """Track import statements."""
        for alias in node.names:
            if alias.name == 'typing':
                self.current_imports['typing'].add('typing')
                
    def visit_ImportFrom(self, node):
        """Track from imports."""
        if node.module == 'typing':
            for alias in node.names:
                self.current_imports['typing'].add(alias.name)
                
    def visit_FunctionDef(self, node):
        """Analyze function definitions."""
        self.total_functions += 1
        
        # Skip private methods and magic methods for now
        if node.name.startswith('_'):
            return
            
        missing_hints = []
        
        # Check return annotation
        if node.returns is None and node.name != '__init__':
            missing_hints.append('return')
            
        # Check parameter annotations
        for arg in node.args.args:
            if arg.annotation is None and arg.arg != 'self':
                missing_hints.append(f'parameter:{arg.arg}')
                
        if missing_hints:
            self.functions_without_hints.append({
                'name': node.name,
                'line': node.lineno,
                'missing': missing_hints,
                'is_public': not node.name.startswith('_')
            })
            self.needs_typing_import = True


def main():
    """Main entry point for the type annotation enhancer."""
    parser = argparse.ArgumentParser(description='Enhance Python files with type annotations')
    parser.add_argument('--file', type=str, help='Path to a specific file to enhance')
    parser.add_argument('--directory', type=str, help='Path to directory to enhance')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    enhancer = TypeAnnotationEnhancer()
    
    if args.file:
        result = enhancer.enhance_file(Path(args.file), args.dry_run)
        print(f"Result: {result}")
    elif args.directory:
        result = enhancer.enhance_directory(Path(args.directory), args.dry_run)
        print(f"Directory enhancement: {result['status']}")
        if args.verbose:
            for r in result['results']:
                if r.get('status', '').startswith('Enhanced'):
                    print(f"  ✅ {r['file']}: {r['functions_enhanced']} functions enhanced")
    else:
        print("Please specify either --file or --directory")


if __name__ == '__main__':
    main()
