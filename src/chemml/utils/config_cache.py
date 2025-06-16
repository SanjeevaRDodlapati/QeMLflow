"""
ChemML Configuration Caching System
Provides smart caching for configuration files and settings to improve startup performance.
"""

import hashlib
import os
import pickle
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class CacheMetadata:
    """Metadata for cached configuration."""

    file_path: str
    file_hash: str
    cached_time: float
    cache_version: str = "1.0"


class ConfigurationCache:
    """Smart caching system for ChemML configurations."""

    def __init__(self, cache_dir: Optional[str] = None, ttl: int = 3600) -> None:
        """
        Initialize configuration cache.

        Args:
            cache_dir: Directory to store cache files (default: ~/.chemml/cache)
            ttl: Time-to-live for cache entries in seconds (default: 1 hour)
        """
        self.ttl = ttl
        self._lock = threading.Lock()

        if cache_dir is None:
            cache_dir = Path.home() / ".chemml" / "cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for fastest access
        self._memory_cache: Dict[str, tuple] = {}

    def _get_file_hash(self, filepath: str) -> str:
        """Get SHA256 hash of file contents."""
        try:
            with open(filepath, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (FileNotFoundError, PermissionError):
            return ""

    def _get_cache_path(self, key: str) -> Path:
        """Get path for cache file."""
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{safe_key}.cache"

    def _is_cache_valid(self, metadata: CacheMetadata, current_hash: str) -> bool:
        """Check if cached data is still valid."""
        # Check file hash
        if metadata.file_hash != current_hash:
            return False

        # Check TTL
        if time.time() - metadata.cached_time > self.ttl:
            return False

        return True

    def get_yaml_config(
        self, filepath: str, use_cache: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Load YAML configuration with caching.

        Args:
            filepath: Path to YAML configuration file
            use_cache: Whether to use cached version if available

        Returns:
            Parsed configuration dictionary or None if file not found
        """
        if not use_cache:
            return self._load_yaml_direct(filepath)

        cache_key = f"yaml:{filepath}"

        # Check memory cache first
        if cache_key in self._memory_cache:
            cached_data, metadata = self._memory_cache[cache_key]
            current_hash = self._get_file_hash(filepath)
            if self._is_cache_valid(metadata, current_hash):
                return cached_data
            else:
                del self._memory_cache[cache_key]

        # Check disk cache
        with self._lock:
            cache_path = self._get_cache_path(cache_key)
            current_hash = self._get_file_hash(filepath)

            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_data, metadata = pickle.load(f)

                    if self._is_cache_valid(metadata, current_hash):
                        # Store in memory cache for next time
                        self._memory_cache[cache_key] = (cached_data, metadata)
                        return cached_data

                except (pickle.PickleError, EOFError):
                    # Cache corrupted, remove it
                    cache_path.unlink(missing_ok=True)

            # Load fresh data and cache it
            config_data = self._load_yaml_direct(filepath)
            if config_data is not None:
                metadata = CacheMetadata(
                    file_path=filepath, file_hash=current_hash, cached_time=time.time()
                )

                # Cache to disk
                try:
                    with open(cache_path, "wb") as f:
                        pickle.dump((config_data, metadata), f)
                except Exception:
                    pass  # Caching failed, but continue with loaded data

                # Cache to memory
                self._memory_cache[cache_key] = (config_data, metadata)

            return config_data

    def _load_yaml_direct(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load YAML file directly without caching."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except (FileNotFoundError, yaml.YAMLError, PermissionError):
            return None

    def cache_computed_config(
        self, key: str, data: Dict[str, Any], metadata: Optional[Dict] = None
    ) -> None:
        """
        Cache computed configuration data.

        Args:
            key: Unique key for the cached data
            data: Configuration data to cache
            metadata: Optional metadata about the computation
        """
        cache_key = f"computed:{key}"
        cache_metadata = CacheMetadata(
            file_path=key,
            file_hash=hashlib.sha256(str(data).encode()).hexdigest(),
            cached_time=time.time(),
        )

        with self._lock:
            # Store in memory
            self._memory_cache[cache_key] = (data, cache_metadata)

            # Store to disk
            cache_path = self._get_cache_path(cache_key)
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump((data, cache_metadata), f)
            except Exception:
                pass  # Caching failed, but data is still in memory

    def get_computed_config(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached computed configuration.

        Args:
            key: Unique key for the cached data

        Returns:
            Cached configuration data or None if not found/expired
        """
        cache_key = f"computed:{key}"

        # Check memory cache
        if cache_key in self._memory_cache:
            cached_data, metadata = self._memory_cache[cache_key]
            if time.time() - metadata.cached_time <= self.ttl:
                return cached_data
            else:
                del self._memory_cache[cache_key]

        # Check disk cache
        with self._lock:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as f:
                        cached_data, metadata = pickle.load(f)

                    if time.time() - metadata.cached_time <= self.ttl:
                        self._memory_cache[cache_key] = (cached_data, metadata)
                        return cached_data
                    else:
                        cache_path.unlink(missing_ok=True)

                except (pickle.PickleError, EOFError):
                    cache_path.unlink(missing_ok=True)

        return None

    def clear_cache(self, pattern: Optional[str] = None) -> int:
        """
        Clear cached configurations.

        Args:
            pattern: Optional pattern to match cache keys (None clears all)

        Returns:
            Number of cache entries cleared
        """
        cleared_count = 0

        with self._lock:
            # Clear memory cache
            if pattern is None:
                cleared_count += len(self._memory_cache)
                self._memory_cache.clear()
            else:
                keys_to_remove = [
                    key for key in self._memory_cache.keys() if pattern in key
                ]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                cleared_count += len(keys_to_remove)

            # Clear disk cache
            for cache_file in self.cache_dir.glob("*.cache"):
                if pattern is None:
                    cache_file.unlink()
                    cleared_count += 1
                else:
                    # For disk cache, we'd need to check the content which is expensive
                    # So for now, clear all when pattern is specified
                    cache_file.unlink()
                    cleared_count += 1

        return cleared_count

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            memory_entries = len(self._memory_cache)
            disk_entries = len(list(self.cache_dir.glob("*.cache")))

            cache_size = 0
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_size += cache_file.stat().st_size
                except OSError:
                    pass

            return {
                "memory_entries": memory_entries,
                "disk_entries": disk_entries,
                "cache_dir": str(self.cache_dir),
                "cache_size_bytes": cache_size,
                "cache_size_mb": cache_size / (1024 * 1024),
                "ttl_seconds": self.ttl,
            }


# Global cache instance
_global_cache = None
_cache_lock = threading.Lock()


def get_config_cache(
    cache_dir: Optional[str] = None, ttl: int = 3600
) -> ConfigurationCache:
    """Get the global configuration cache instance."""
    global _global_cache

    if _global_cache is None:
        with _cache_lock:
            if _global_cache is None:
                _global_cache = ConfigurationCache(cache_dir, ttl)

    return _global_cache


def cached_yaml_load(filepath: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
    """Convenience function to load YAML with caching."""
    cache = get_config_cache()
    return cache.get_yaml_config(filepath, use_cache)


def cache_config(key: str, data: Dict[str, Any]) -> None:
    """Convenience function to cache computed configuration."""
    cache = get_config_cache()
    cache.cache_computed_config(key, data)


def get_cached_config(key: str) -> Optional[Dict[str, Any]]:
    """Convenience function to retrieve cached configuration."""
    cache = get_config_cache()
    return cache.get_computed_config(key)
