import time
import threading

class CacheManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_cache_size=100):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(CacheManager, cls).__new__(cls)
                    cls._instance.cache = {}
                    cls._instance.max_cache_size = max_cache_size
        return cls._instance

    def normalize_key(self, key):
        return key.strip().lower()

    def add_to_cache(self, key, value, embedding=None):
        normalized_key = self.normalize_key(key)
        if len(self.cache) >= self.max_cache_size:
            self.evict_cache()
        self.cache[normalized_key] = {
            "response": value,
            "timestamp": time.time(),
            "embedding": embedding
        }

    def get_from_cache(self, key):
        normalized_key = self.normalize_key(key)
        return self.cache.get(normalized_key, {}).get("response", None)

    def get_embedding(self, key):
        normalized_key = self.normalize_key(key)
        return self.cache.get(normalized_key, {}).get("embedding", None)

    def evict_cache(self):
        if self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["timestamp"])
            del self.cache[oldest_key]

    def clear_cache(self):
        self.cache.clear()
