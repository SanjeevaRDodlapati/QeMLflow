"""
QeMLflow Core Health Monitor
===========================

Real-time monitoring of core framework health and performance.
Implements philosophy: "Performance monitoring and analytics"
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import psutil


class CoreHealthMonitor:
    """Monitor core framework performance and health metrics."""

    def __init__(self):
        self.start_time = time.time()
        self.import_times = {}
        self.memory_baseline = psutil.Process().memory_info().rss / 1024 / 1024  # MB

    def measure_import_time(self, module_name: str) -> float:
        """Measure and record module import time."""
        start = time.time()
        try:
            __import__(module_name)
            import_time = time.time() - start
            self.import_times[module_name] = import_time
            return import_time
        except ImportError as e:
            self.import_times[module_name] = f"FAILED: {e}"
            return -1

    def get_core_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive core health metrics."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Test core imports
        core_modules = [
            "qemlflow.core.models",
            "qemlflow.core.featurizers",
            "qemlflow.core.evaluation",
            "qemlflow.core.data",
        ]

        import_health = {}
        total_import_time = 0

        for module in core_modules:
            import_time = self.measure_import_time(module)
            import_health[module] = {
                "time": import_time,
                "status": "✅ OK"
                if import_time > 0 and import_time < 1.0
                else "⚠️ SLOW"
                if import_time > 0
                else "❌ FAILED",
            }
            if import_time > 0:
                total_import_time += import_time

        return {
            "core_import_time": f"{total_import_time:.3f}s",
            "memory_usage": f"{current_memory:.1f} MB",
            "memory_growth": f"{current_memory - self.memory_baseline:.1f} MB",
            "framework_uptime": f"{time.time() - self.start_time:.1f}s",
            "import_health": import_health,
            "philosophy_compliance": {
                "sub_5s_imports": "✅ PASS" if total_import_time < 5.0 else "❌ FAIL",
                "lean_core": "✅ PASS" if current_memory < 100 else "⚠️ REVIEW",
                "robust_loading": "✅ PASS"
                if all(h["status"].startswith("✅") for h in import_health.values())
                else "❌ FAIL",
            },
        }

    def generate_health_report(self) -> str:
        """Generate human-readable health report."""
        health = self.get_core_health_summary()

        report = f"""
🏥 QeMLflow Core Health Report
=====================================

📊 Performance Metrics:
  • Total Core Import Time: {health['core_import_time']}
  • Memory Usage: {health['memory_usage']}
  • Memory Growth: {health['memory_growth']}
  • Framework Uptime: {health['framework_uptime']}

🧩 Module Health:"""

        for module, status in health["import_health"].items():
            module_short = module.split(".")[-1]
            report += f"\n  • {module_short}: {status['time']:.3f}s {status['status']}"

        report += f"""

🎯 Philosophy Compliance:
  • Sub-5s Imports: {health['philosophy_compliance']['sub_5s_imports']}
  • Lean Core: {health['philosophy_compliance']['lean_core']}
  • Robust Loading: {health['philosophy_compliance']['robust_loading']}

Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report


def quick_health_check() -> None:
    """Quick health check for core framework."""
    monitor = CoreHealthMonitor()
    print(monitor.generate_health_report())


if __name__ == "__main__":
    quick_health_check()
