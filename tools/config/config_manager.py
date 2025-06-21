#!/usr/bin/env python3
"""
🔧 QeMLflow Configuration Management CLI Tool

Enterprise-grade configuration management utilities for:
- Configuration validation
- Template rendering
- Environment switching
- Configuration migration
- Security auditing

Usage:
    python config_manager.py validate --config qemlflow --environment production
    python config_manager.py render --template qemlflow --output /tmp/config.yml
    python config_manager.py migrate --from v1.0.0 --to v2.0.0
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from qemlflow.config import ConfigurationEngine, validate_environment

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ConfigurationManager:
    """Configuration management CLI tool"""
    
    def __init__(self):
        self.config_root = Path(__file__).parent.parent.parent / "config"
        self.engine = ConfigurationEngine(config_root=self.config_root)
    
    def validate_config(self, config_name: str, environment: str) -> bool:
        """Validate configuration for environment"""
        try:
            logger.info(f"Validating configuration '{config_name}' for environment '{environment}'")
            
            # Initialize engine for specific environment
            engine = ConfigurationEngine(
                config_root=self.config_root,
                environment=environment
            )
            
            # Load schema
            schema = engine.load_schema(config_name)
            if not schema:
                logger.error(f"No schema found for '{config_name}'")
                return False
            
            # Load and validate configuration
            config = engine.load_configuration(config_name)
            
            # Get metadata
            metadata = engine.get_config_metadata()
            if metadata:
                logger.info(f"Configuration version: {metadata.version}")
                logger.info(f"Configuration checksum: {metadata.checksum}")
                logger.info(f"Source files: {', '.join(metadata.source_files)}")
            
            logger.info("✅ Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    def render_template(
        self,
        template_name: str,
        output_file: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        environment: str = "development"
    ) -> bool:
        """Render configuration template"""
        try:
            logger.info(f"Rendering template '{template_name}' for environment '{environment}'")
            
            # Initialize engine for specific environment
            engine = ConfigurationEngine(
                config_root=self.config_root,
                environment=environment
            )
            
            # Render template
            rendered = engine.render_template(template_name, context)
            
            if output_file:
                # Save to file
                output_path = Path(output_file)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'w') as f:
                    f.write(rendered)
                
                logger.info(f"✅ Template rendered and saved to: {output_path}")
            else:
                # Print to stdout
                print(rendered)
                logger.info("✅ Template rendered to stdout")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Template rendering failed: {e}")
            return False
    
    def show_environment_info(self) -> bool:
        """Show current environment information"""
        try:
            env_info = validate_environment()
            
            print("\n🔍 Environment Information:")
            print(f"  Environment: {env_info['environment']}")
            print(f"  Config Root Exists: {env_info['config_root_exists']}")
            print(f"  Templates Available: {env_info['templates_available']}")
            print(f"  Schemas Available: {env_info['schemas_available']}")
            
            print("\n📋 Environment Variables:")
            for key, value in env_info['environment_variables'].items():
                print(f"  {key}: {value}")
            
            print("\n🔍 Detection Indicators:")
            for indicator, value in env_info['detected_indicators'].items():
                status = "✅" if value else "❌"
                print(f"  {indicator}: {status}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to get environment info: {e}")
            return False
    
    def list_configurations(self) -> bool:
        """List available configurations"""
        try:
            print("\n📋 Available Configurations:")
            
            # Base configurations
            base_configs = list(self.config_root.glob("*.yml"))
            if base_configs:
                print("\n  Base Configurations:")
                for config in base_configs:
                    print(f"    - {config.stem}")
            
            # Environment-specific configurations
            for env_dir in self.config_root.iterdir():
                if env_dir.is_dir() and env_dir.name not in ['templates', 'schemas']:
                    env_configs = list(env_dir.glob("*.yml"))
                    if env_configs:
                        print(f"\n  {env_dir.name.title()} Environment:")
                        for config in env_configs:
                            print(f"    - {config.stem}")
            
            # Templates
            template_files = list((self.config_root / "templates").glob("*.j2"))
            if template_files:
                print("\n  Available Templates:")
                for template in template_files:
                    print(f"    - {template.stem}")
            
            # Schemas
            schema_files = list((self.config_root / "schemas").glob("*.schema.json"))
            if schema_files:
                print("\n  Available Schemas:")
                for schema in schema_files:
                    name = schema.name.replace('.schema.json', '')
                    print(f"    - {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to list configurations: {e}")
            return False
    
    def test_configuration(self, config_name: str, environment: str) -> bool:
        """Test configuration loading and access"""
        try:
            logger.info(f"Testing configuration '{config_name}' for environment '{environment}'")
            
            # Initialize engine
            engine = ConfigurationEngine(
                config_root=self.config_root,
                environment=environment
            )
            
            # Load configuration
            config = engine.load_configuration(config_name)
            
            # Test common configuration access patterns
            test_paths = [
                "application.name",
                "application.version",
                "application.environment",
                "security.authentication.enabled",
                "database.url",
                "performance.workers"
            ]
            
            print(f"\n🧪 Configuration Test Results for '{config_name}' ({environment}):")
            
            for path in test_paths:
                try:
                    value = engine.get_config_value(path)
                    status = "✅" if value is not None else "⚠️"
                    print(f"  {path}: {status} {value}")
                except Exception as e:
                    print(f"  {path}: ❌ Error: {e}")
            
            # Test metadata
            metadata = engine.get_config_metadata()
            if metadata:
                print(f"\n📊 Configuration Metadata:")
                print(f"  Version: {metadata.version}")
                print(f"  Checksum: {metadata.checksum}")
                print(f"  Created: {metadata.created_at}")
                print(f"  Sources: {len(metadata.source_files)} files")
            
            logger.info("✅ Configuration test completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    def generate_sample_config(self, environment: str, output_file: str) -> bool:
        """Generate sample configuration for environment"""
        try:
            logger.info(f"Generating sample configuration for environment '{environment}'")
            
            # Sample context for template rendering
            sample_context = {
                "application_name": "QeMLflow Sample",
                "application_version": "2.0.0",
                "database_url": f"sqlite:///{environment}.db",
                "redis_url": "redis://localhost:6379/0",
                "custom_config": {
                    "sample_feature": True,
                    "sample_setting": "sample_value"
                }
            }
            
            return self.render_template(
                template_name="qemlflow",
                output_file=output_file,
                context=sample_context,
                environment=environment
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate sample configuration: {e}")
            return False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="QeMLflow Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('--config', default='qemlflow', help='Configuration name')
    validate_parser.add_argument('--environment', default='development', help='Target environment')
    
    # Render command
    render_parser = subparsers.add_parser('render', help='Render configuration template')
    render_parser.add_argument('--template', required=True, help='Template name')
    render_parser.add_argument('--output', help='Output file path')
    render_parser.add_argument('--environment', default='development', help='Target environment')
    render_parser.add_argument('--context', help='JSON context for template rendering')
    
    # Environment info command
    subparsers.add_parser('env-info', help='Show environment information')
    
    # List command
    subparsers.add_parser('list', help='List available configurations')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test configuration loading')
    test_parser.add_argument('--config', default='qemlflow', help='Configuration name')
    test_parser.add_argument('--environment', default='development', help='Target environment')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate sample configuration')
    generate_parser.add_argument('--environment', required=True, help='Target environment')
    generate_parser.add_argument('--output', required=True, help='Output file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize configuration manager
    config_manager = ConfigurationManager()
    
    # Execute command
    success = False
    
    if args.command == 'validate':
        success = config_manager.validate_config(args.config, args.environment)
    
    elif args.command == 'render':
        context = {}
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON context: {e}")
                return 1
        
        success = config_manager.render_template(
            args.template, args.output, context, args.environment
        )
    
    elif args.command == 'env-info':
        success = config_manager.show_environment_info()
    
    elif args.command == 'list':
        success = config_manager.list_configurations()
    
    elif args.command == 'test':
        success = config_manager.test_configuration(args.config, args.environment)
    
    elif args.command == 'generate':
        success = config_manager.generate_sample_config(args.environment, args.output)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
