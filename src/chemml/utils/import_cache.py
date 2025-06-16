"""
Import result caching for ChemML
Caches expensive import operations
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Dict, Optional


class ImportCache:
    """Cache import results to speed up subsequent loads"""

    def __init__(self, cache_dir: Optional[str] = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".chemml" / "cache" / "imports"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}

    def _get_cache_key(self, module_name: str) -> str:
        """Generate cache key for module"""
        return hashlib.md5(module_name.encode()).hexdigest()

    def _get_cache_file(self, module_name: str) -> Path:
        """Get cache file path for module"""
        key = self._get_cache_key(module_name)
        return self.cache_dir / f"{key}.pkl"

    def is_cached(self, module_name: str) -> bool:
        """Check if module result is cached"""
        if module_name in self._memory_cache:
            return True

        cache_file = self._get_cache_file(module_name)
        return cache_file.exists()

    def get_cached(self, module_name: str) -> Optional[Any]:
        """Get cached module result"""
        # Try memory cache first
        if module_name in self._memory_cache:
            return self._memory_cache[module_name]

        # Try disk cache
        cache_file = self._get_cache_file(module_name)
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                # Check if cache is still valid (1 hour TTL)
                if time.time() - cached_data["timestamp"] < 3600:
                    result = cached_data["result"]
                    self._memory_cache[module_name] = result
                    return result

            except Exception:
                # Invalid cache, remove it
                cache_file.unlink(missing_ok=True)

        return None

    def cache_result(self, module_name: str, result: Any) -> None:
        """Cache module import result"""
        # Store in memory
        self._memory_cache[module_name] = result

        # Store on disk for persistence
        cache_data = {"timestamp": time.time(), "result": result}

        cache_file = self._get_cache_file(module_name)
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except Exception:
            pass  # Fail silently if can't cache to disk


# Global cache instance
_import_cache = ImportCache()


def cached_import(module_name: str) -> Any:
    """Import with caching support"""
    cached_result = _import_cache.get_cached(module_name)
    if cached_result is not None:
        return cached_result

    # Not cached, do actual import
    import importlib

    result = importlib.import_module(module_name)

    # Cache the result
    _import_cache.cache_result(module_name, result)

    return result
