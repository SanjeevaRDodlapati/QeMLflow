from typing import Any, Dict

# !/usr/bin/env python3
"""
Configuration Caching Activator for QeMLflow
Activates advanced caching features for improved performance
"""

import json
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qemlflow.config.unified_config import UnifiedConfig
from qemlflow.utils.config_cache import ConfigurationCache


class CachingActivator:
    """Activates and configures advanced caching features"""

    def __init__(self):
        self.cache_dir = Path.home() / ".qemlflow" / "cache"
        self.config_cache = ConfigurationCache()
        self.performance_metrics = {}

    def setup_cache_infrastructure(self) -> bool:
        """Set up caching infrastructure"""
        print("🔧 Setting up caching infrastructure...")

        try:
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # Create cache subdirectories
            subdirs = ["config", "models", "data", "performance"]
            for subdir in subdirs:
                (self.cache_dir / subdir).mkdir(exist_ok=True)

            # Initialize cache metadata
            metadata_file = self.cache_dir / "metadata.json"
            if not metadata_file.exists():
                metadata = {
                    "version": "1.0",
                    "created": time.time(),
                    "last_cleanup": time.time(),
                    "cache_stats": {"hits": 0, "misses": 0, "size_mb": 0},
                }

                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

            print("✅ Cache infrastructure ready")
            return True

        except Exception as e:
            print(f"❌ Failed to setup cache infrastructure: {e}")
            return False

    def activate_config_caching(self) -> bool:
        """Activate configuration caching"""
        print("⚡ Activating configuration caching...")

        try:
            # Test configuration loading with caching
            config_path = (
                Path(__file__).parent.parent / "config" / "advanced_config.yaml"
            )

            if config_path.exists():
                # Test cached loading
                start_time = time.time()
                _cached_config = self.config_cache.get_config(str(config_path))
                cache_time = time.time() - start_time

                # Test direct loading
                start_time = time.time()
                _direct_config = UnifiedConfig.load_config(str(config_path))
                direct_time = time.time() - start_time

                self.performance_metrics["config_cache_speedup"] = (
                    direct_time / cache_time if cache_time > 0 else 1.0
                )

                print(
                    f"📊 Config loading: Direct {direct_time:.3f}s, Cached {cache_time:.3f}s"
                )
                print(
                    f"🚀 Cache speedup: {self.performance_metrics['config_cache_speedup']:.1f}x"
                )

            print("✅ Configuration caching activated")
            return True

        except Exception as e:
            print(f"❌ Failed to activate config caching: {e}")
            return False

    def setup_performance_monitoring(self) -> bool:
        """Set up performance monitoring"""
        print("📈 Setting up performance monitoring...")

        try:
            # Create performance tracking system
            perf_dir = self.cache_dir / "performance"

            # Initialize performance database
            perf_db = perf_dir / "metrics.json"
            if not perf_db.exists():
                initial_metrics = {
                    "import_times": [],
                    "function_calls": {},
                    "memory_usage": [],
                    "cache_performance": {"hits": 0, "misses": 0, "hit_rate": 0.0},
                }

                with open(perf_db, "w") as f:
                    json.dump(initial_metrics, f, indent=2)

            # Set environment variable to enable monitoring
            os.environ["QEMLFLOW_PERFORMANCE_MONITORING"] = "true"
            os.environ["QEMLFLOW_CACHE_DIR"] = str(self.cache_dir)

            print("✅ Performance monitoring ready")
            return True

        except Exception as e:
            print(f"❌ Failed to setup performance monitoring: {e}")
            return False

    def create_optimization_profiles(self) -> bool:
        """Create optimization profiles for different use cases"""
        print("🎯 Creating optimization profiles...")

        try:
            profiles_dir = self.cache_dir / "profiles"
            profiles_dir.mkdir(exist_ok=True)

            # Development profile (fast imports, moderate caching)
            dev_profile = {
                "name": "development",
                "import_strategy": "lazy_aggressive",
                "cache_ttl": 3600,  # 1 hour
                "memory_limit_mb": 1024,
                "enable_monitoring": True,
                "features": {
                    "auto_cleanup": True,
                    "performance_tracking": True,
                    "smart_preloading": False,
                },
            }

            # Production profile (optimized for performance)
            prod_profile = {
                "name": "production",
                "import_strategy": "optimized",
                "cache_ttl": 86400,  # 24 hours
                "memory_limit_mb": 2048,
                "enable_monitoring": False,
                "features": {
                    "auto_cleanup": True,
                    "performance_tracking": False,
                    "smart_preloading": True,
                },
            }

            # Research profile (full features, extended caching)
            research_profile = {
                "name": "research",
                "import_strategy": "full",
                "cache_ttl": 604800,  # 1 week
                "memory_limit_mb": 4096,
                "enable_monitoring": True,
                "features": {
                    "auto_cleanup": False,
                    "performance_tracking": True,
                    "smart_preloading": True,
                },
            }

            # Save profiles
            for profile in [dev_profile, prod_profile, research_profile]:
                profile_file = profiles_dir / f"{profile['name']}.json"
                with open(profile_file, "w") as f:
                    json.dump(profile, f, indent=2)

            # Set default profile
            default_profile_file = self.cache_dir / "active_profile.json"
            with open(default_profile_file, "w") as f:
                json.dump({"active_profile": "development"}, f)

            print("✅ Optimization profiles created")
            return True

        except Exception as e:
            print(f"❌ Failed to create optimization profiles: {e}")
            return False

    def test_caching_performance(self) -> Dict[str, float]:
        """Test caching performance improvements"""
        print("🧪 Testing caching performance...")

        results = {}

        try:
            # Test configuration caching
            config_path = (
                Path(__file__).parent.parent / "config" / "advanced_config.yaml"
            )

            if config_path.exists():
                # Cold load (no cache)
                self.config_cache.clear_cache()
                start_time = time.time()
                _config1 = self.config_cache.get_config(str(config_path))
                cold_time = time.time() - start_time

                # Warm load (from cache)
                start_time = time.time()
                _config2 = self.config_cache.get_config(str(config_path))
                warm_time = time.time() - start_time

                results["config_cold_load"] = cold_time
                results["config_warm_load"] = warm_time
                results["config_speedup"] = (
                    cold_time / warm_time if warm_time > 0 else 1.0
                )

            # Test memory cache performance
            start_time = time.time()
            for i in range(100):
                self.config_cache._memory_cache[f"test_{i}"] = (
                    f"value_{i}",
                    time.time(),
                )
            memory_write_time = time.time() - start_time

            start_time = time.time()
            for i in range(100):
                _ = self.config_cache._memory_cache.get(f"test_{i}")
            memory_read_time = time.time() - start_time

            results["memory_write_time"] = memory_write_time
            results["memory_read_time"] = memory_read_time

            print("📊 Performance Results:")
            for key, value in results.items():
                print(f"   {key}: {value:.4f}s")

            return results

        except Exception as e:
            print(f"❌ Caching performance test failed: {e}")
            return {}

    def run_activation(self) -> Dict[str, Any]:
        """Run complete caching activation suite"""
        print("🚀 QeMLflow Advanced Caching Activation")
        print("=" * 50)

        results = {
            "infrastructure": False,
            "config_caching": False,
            "performance_monitoring": False,
            "optimization_profiles": False,
            "performance_tests": {},
        }

        # 1. Setup infrastructure
        results["infrastructure"] = self.setup_cache_infrastructure()

        # 2. Activate config caching
        if results["infrastructure"]:
            results["config_caching"] = self.activate_config_caching()

        # 3. Setup performance monitoring
        if results["config_caching"]:
            results["performance_monitoring"] = self.setup_performance_monitoring()

        # 4. Create optimization profiles
        if results["performance_monitoring"]:
            results["optimization_profiles"] = self.create_optimization_profiles()

        # 5. Test performance
        if results["optimization_profiles"]:
            results["performance_tests"] = self.test_caching_performance()

        return results


def main():
    """Run caching activation"""
    activator = CachingActivator()
    results = activator.run_activation()

    print("\n" + "=" * 50)
    print("📊 CACHING ACTIVATION RESULTS")
    print("=" * 50)

    success_count = sum(1 for k, v in results.items() if k != "performance_tests" and v)
    total_count = len(results) - 1  # Exclude performance_tests from count

    print(f"✅ Components activated: {success_count}/{total_count}")

    if results.get("performance_tests"):
        perf = results["performance_tests"]
        if "config_speedup" in perf:
            print(f"⚡ Config cache speedup: {perf['config_speedup']:.1f}x")

    if success_count == total_count:
        print("🏆 SUCCESS: Advanced caching fully activated!")
        print("\n💡 Benefits enabled:")
        print("   • Smart configuration caching")
        print("   • Performance monitoring")
        print("   • Optimization profiles")
        print("   • Memory-efficient operations")
    else:
        print("🔄 PARTIAL: Some features activated, continuing optimization...")


if __name__ == "__main__":
    main()
