#!/usr/bin/env python3
"""
QeMLflow Resource Management CLI Tool

A comprehensive command-line interface for managing QeMLflow's resource management system.
Provides tools for monitoring, optimization, scaling, and alerting.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Import QeMLflow resource management modules
try:
    from qemlflow.resources import (
        MemoryManager,
        ComputeManager, 
        AutoScaler,
        ResourceMonitor,
        ResourceDashboard,
        create_auto_scaler,
        create_resource_monitor,
        setup_basic_alerting,
        get_memory_usage,
        get_system_resources
    )
except ImportError as e:
    print(f"Error importing QeMLflow resources: {e}")
    print("Please ensure QeMLflow is properly installed.")
    sys.exit(1)


class ResourceManagerCLI:
    """Main CLI class for resource management operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self, config_path: Optional[str] = None) -> None:
        """Load resource management configuration."""
        if config_path is None:
            # Look for config in common locations
            possible_paths = [
                "config/resources.yml",
                "qemlflow/config/resources.yml", 
                "/etc/qemlflow/resources.yml",
                Path.home() / ".qemlflow" / "resources.yml"
            ]
            
            for path in possible_paths:
                if Path(str(path)).exists():
                    config_path = str(path)
                    break
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        else:
            self.logger.warning("No configuration file found, using defaults")
    
    def setup_logging(self, log_level: str = "INFO") -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def cmd_status(self, args) -> None:
        """Show current system resource status."""
        print("QeMLflow Resource Management Status")
        print("=" * 40)
        
        # Memory status
        try:
            memory_usage = get_memory_usage()
            print("\n📊 Memory Usage:")
            print(f"  System Memory: {memory_usage['system_memory_gb']:.1f} GB total")
            print(f"  Available: {memory_usage['available_memory_gb']:.1f} GB")
            print(f"  Usage: {memory_usage['memory_usage_percent']:.1f}%")
            print(f"  Process Memory: {memory_usage['process_memory_mb']:.1f} MB")
            print(f"  Process Usage: {memory_usage['process_memory_percent']:.1f}%")
        except Exception as e:
            print(f"  ❌ Memory status unavailable: {e}")
        
        # System resources
        try:
            system_resources = get_system_resources()
            cpu_status = system_resources.get('cpu_status', {})
            print("\n💻 CPU Status:")
            print(f"  Usage: {cpu_status.get('usage_percent', 0):.1f}%")
            print(f"  Load Average: {cpu_status.get('load_average', [])}")
            
            if 'gpu_status' in system_resources:
                gpu_status = system_resources['gpu_status']
                if gpu_status.get('available', False):
                    print("\n🎮 GPU Status:")
                    print(f"  GPU Count: {gpu_status.get('gpu_count', 0)}")
                    for gpu in gpu_status.get('gpus', []):
                        print(f"  GPU {gpu['id']}: {gpu['gpu_utilization_percent']:.1f}% util, "
                              f"{gpu['memory_usage']['utilization_percent']:.1f}% memory")
                else:
                    print("\n🎮 GPU Status: Not available")
        except Exception as e:
            print(f"  ❌ System status unavailable: {e}")
        
        # Configuration status
        print("\n⚙️  Configuration:")
        print(f"  Config loaded: {bool(self.config)}")
        if self.config:
            print(f"  Memory management: {self.config.get('memory', {}).get('profiling_enabled', False)}")
            print(f"  Auto-scaling: {self.config.get('scaling', {}).get('enabled', False)}")
            print(f"  Monitoring: {self.config.get('monitoring', {}).get('enabled', False)}")
    
    def cmd_monitor(self, args) -> None:
        """Start resource monitoring."""
        print("Starting resource monitoring...")
        
        # Create and configure monitor
        interval = args.interval or self.config.get('monitoring', {}).get('monitoring_interval_seconds', 10.0)
        monitor = create_resource_monitor(monitoring_interval=interval)
        
        # Set up alerting if enabled
        if args.alerts or self.config.get('monitoring', {}).get('alerting', {}).get('enabled', True):
            setup_basic_alerting(monitor)
            print("✓ Alerting enabled")
        
        # Start monitoring
        monitor.start_monitoring()
        print(f"✓ Monitoring started (interval: {interval}s)")
        
        try:
            if args.duration:
                print(f"Monitoring for {args.duration} seconds...")
                time.sleep(args.duration)
            else:
                print("Monitoring indefinitely. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
                    
                    # Show periodic updates
                    if hasattr(args, 'verbose') and args.verbose:
                        metrics = monitor.get_current_metrics()
                        if metrics:
                            print(f"CPU: {metrics.cpu_percent:.1f}%, Memory: {metrics.memory_percent:.1f}%")
        
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
        
        finally:
            monitor.stop_monitoring()
            
            # Show final summary
            summary = monitor.get_performance_summary()
            print("\n📊 Monitoring Summary:")
            print(f"  Data points collected: {summary.get('data_points_collected', 0)}")
            
            if 'alerts' in summary:
                alerts = summary['alerts']
                print(f"  Active alerts: {alerts.get('total_active_alerts', 0)}")
    
    def cmd_optimize(self, args) -> None:
        """Run resource optimization."""
        print("Running resource optimization...")
        
        # Memory optimization
        if args.memory or args.all:
            print("\n🧠 Memory Optimization:")
            try:
                memory_manager = MemoryManager()
                result = memory_manager.cleanup_and_optimize()
                
                if 'cleanup' in result:
                    cleanup = result['cleanup']
                    print(f"  Memory freed: {cleanup.get('memory_freed_mb', 0):.1f} MB")
                    print(f"  GC collected: {cleanup.get('gc_collected', [])}")
                
                print("  ✓ Memory optimization completed")
            except Exception as e:
                print(f"  ❌ Memory optimization failed: {e}")
        
        # Compute optimization
        if args.compute or args.all:
            print("\n💻 Compute Optimization:")
            try:
                compute_manager = ComputeManager()
                
                # Get optimization recommendations
                report = compute_manager.get_performance_report()
                recommendations = report.get('cpu_recommendations', [])
                
                print("  Recommendations:")
                for rec in recommendations:
                    print(f"    • {rec}")
                
                # Optimize for specified workload type
                workload_type = args.workload_type or "mixed"
                optimization = compute_manager.optimize_for_workload(workload_type)
                
                print(f"  Optimized for: {workload_type}")
                print(f"  Optimal workers: {optimization.get('optimal_worker_count', 'N/A')}")
                
                print("  ✓ Compute optimization completed")
            except Exception as e:
                print(f"  ❌ Compute optimization failed: {e}")
        
        # Auto-scaling optimization
        if args.scaling or args.all:
            print("\n📈 Auto-scaling Setup:")
            try:
                scaler = create_auto_scaler()
                status = scaler.get_status_report()
                
                print(f"  Current instances: {status.get('current_instances', 1)}")
                print(f"  CPU usage: {status.get('current_metrics', {}).get('cpu_percent', 0):.1f}%")
                print(f"  Memory usage: {status.get('current_metrics', {}).get('memory_percent', 0):.1f}%")
                
                print("  ✓ Auto-scaling status checked")
            except Exception as e:
                print(f"  ❌ Auto-scaling setup failed: {e}")
    
    def cmd_dashboard(self, args) -> None:
        """Generate resource dashboard."""
        print("Generating resource dashboard...")
        
        try:
            # Create monitor and collect some data
            monitor = create_resource_monitor()
            monitor.start_monitoring()
            
            print("Collecting metrics...")
            time.sleep(args.duration or 5)
            
            # Generate dashboard
            dashboard = ResourceDashboard(monitor)
            
            output_path = args.output or "resource_dashboard.html"
            
            if dashboard.generate_html_report(output_path):
                print(f"✓ Dashboard generated: {output_path}")
                
                # Also generate JSON data if requested
                if args.json:
                    json_path = output_path.replace('.html', '.json')
                    data = dashboard.get_dashboard_data()
                    with open(json_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"✓ JSON data saved: {json_path}")
            else:
                print("❌ Dashboard generation failed")
                
            monitor.stop_monitoring()
            
        except Exception as e:
            print(f"❌ Dashboard generation error: {e}")
    
    def cmd_config(self, args) -> None:
        """Configuration management."""
        if args.show:
            print("Current Configuration:")
            print("=" * 30)
            if self.config:
                print(yaml.dump(self.config, default_flow_style=False))
            else:
                print("No configuration loaded")
        
        elif args.validate:
            print("Validating configuration...")
            
            required_sections = ['memory', 'compute', 'scaling', 'monitoring']
            missing_sections = []
            
            for section in required_sections:
                if section not in self.config:
                    missing_sections.append(section)
            
            if missing_sections:
                print(f"❌ Missing sections: {missing_sections}")
            else:
                print("✓ Configuration is valid")
        
        elif args.template:
            template_path = args.template
            print(f"Generating configuration template: {template_path}")
            
            template_config = {
                'memory': {
                    'profiling_enabled': True,
                    'optimization_enabled': True,
                    'auto_cleanup': True,
                    'auto_gc_threshold': 80.0
                },
                'compute': {
                    'cpu_monitoring_enabled': True,
                    'gpu_monitoring_enabled': True,
                    'cpu_warning_threshold': 80.0
                },
                'scaling': {
                    'enabled': True,
                    'default_policy': {
                        'min_instances': 1,
                        'max_instances': 5,
                        'target_cpu_percent': 70.0
                    }
                },
                'monitoring': {
                    'enabled': True,
                    'monitoring_interval_seconds': 10.0,
                    'alerting': {
                        'enabled': True
                    }
                }
            }
            
            with open(template_path, 'w') as f:
                yaml.dump(template_config, f, default_flow_style=False)
            
            print(f"✓ Template generated: {template_path}")
    
    def cmd_alerts(self, args) -> None:
        """Alert management."""
        print("Alert Management")
        print("=" * 20)
        
        # Create monitor with alerting
        monitor = create_resource_monitor()
        setup_basic_alerting(monitor)
        
        if args.test:
            print("Testing alert system...")
            
            # Trigger test alerts
            if monitor.alert_manager:
                from qemlflow.resources.monitoring import AlertSeverity
                
                test_alert = monitor.alert_manager.create_alert(
                    metric_name="test_metric",
                    current_value=95.0,
                    threshold_value=80.0,
                    severity=AlertSeverity.WARNING,
                    message="Test alert from CLI tool"
                )
                
                if test_alert:
                    print(f"✓ Test alert created: {test_alert.id}")
                else:
                    print("❌ Failed to create test alert")
        
        elif args.list:
            monitor.start_monitoring()
            time.sleep(5)  # Collect some data
            
            summary = monitor.get_performance_summary()
            if 'alerts' in summary:
                alerts = summary['alerts']
                print(f"Active alerts: {alerts.get('total_active_alerts', 0)}")
                
                for alert in alerts.get('recent_alerts', []):
                    timestamp = datetime.fromtimestamp(alert['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"  [{alert['severity'].upper()}] {alert['message']} ({timestamp})")
            else:
                print("No alerts available")
            
            monitor.stop_monitoring()
    
    def cmd_export(self, args) -> None:
        """Export resource data."""
        print(f"Exporting resource data to {args.output}...")
        
        try:
            monitor = create_resource_monitor()
            monitor.start_monitoring()
            
            # Collect data for specified duration
            duration = args.duration or 60
            print(f"Collecting data for {duration} seconds...")
            time.sleep(duration)
            
            # Export data
            if monitor.export_metrics(args.output, hours=args.hours or 1):
                print(f"✓ Data exported to {args.output}")
            else:
                print("❌ Export failed")
            
            monitor.stop_monitoring()
            
        except Exception as e:
            print(f"❌ Export error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QeMLflow Resource Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show resource status')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start resource monitoring')
    monitor_parser.add_argument('--interval', type=float, help='Monitoring interval in seconds')
    monitor_parser.add_argument('--duration', type=int, help='Monitoring duration in seconds')
    monitor_parser.add_argument('--alerts', action='store_true', help='Enable alerts')
    monitor_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run resource optimization')
    optimize_parser.add_argument('--memory', action='store_true', help='Optimize memory')
    optimize_parser.add_argument('--compute', action='store_true', help='Optimize compute')
    optimize_parser.add_argument('--scaling', action='store_true', help='Optimize scaling')
    optimize_parser.add_argument('--all', action='store_true', help='Optimize all resources')
    optimize_parser.add_argument('--workload-type', choices=['cpu_bound', 'io_bound', 'mixed'], 
                                default='mixed', help='Workload type for optimization')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Generate resource dashboard')
    dashboard_parser.add_argument('--output', default='resource_dashboard.html', 
                                 help='Output file path')
    dashboard_parser.add_argument('--duration', type=int, default=5, 
                                 help='Data collection duration')
    dashboard_parser.add_argument('--json', action='store_true', 
                                 help='Also generate JSON data')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--validate', action='store_true', help='Validate configuration')
    config_parser.add_argument('--template', help='Generate configuration template')
    
    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Alert management')
    alerts_parser.add_argument('--test', action='store_true', help='Test alert system')
    alerts_parser.add_argument('--list', action='store_true', help='List current alerts')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export resource data')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('--duration', type=int, default=60, 
                              help='Data collection duration')
    export_parser.add_argument('--hours', type=int, default=1, 
                              help='Hours of historical data to export')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize CLI
    cli = ResourceManagerCLI()
    cli.setup_logging(args.log_level)
    
    if args.config:
        cli.load_config(args.config)
    
    # Execute command
    try:
        command_method = getattr(cli, f'cmd_{args.command}')
        command_method(args)
    except AttributeError:
        print(f"Unknown command: {args.command}")
        parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error executing command: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
