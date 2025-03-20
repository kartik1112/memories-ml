import os
from diskcache import Cache
import hashlib
from datetime import timedelta

class CacheManager:
    def __init__(self, cache_dir='cache'):
        os.makedirs(cache_dir, exist_ok=True)
        self.cache = Cache(cache_dir)
        # Set default expiration to 24 hours
        self.default_expiry = timedelta(hours=24)

    def get_key(self, url: str) -> str:
        """Generate a cache key from URL"""
        return hashlib.md5(url.encode()).hexdigest()

    def get_features(self, url: str):
        """Get cached features for a URL"""
        key = self.get_key(url)
        return self.cache.get(key)

    def set_features(self, url: str, features):
        """Cache features for a URL"""
        key = self.get_key(url)
        self.cache.set(key, features, expire=self.default_expiry)

    def clear(self):
        """Clear all cached items"""
        self.cache.clear()

    def __del__(self):
        """Cleanup on deletion"""
        self.cache.close()
