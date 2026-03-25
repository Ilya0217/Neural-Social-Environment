"""
Intelligent caching system for dialogue responses.
Reduces API calls and costs by caching similar contexts.

NOTE: This module is not yet integrated into the main pipeline.
See dialogue_manager.py for the current API call flow.
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import pickle


@dataclass
class CacheEntry:
    """A single cache entry"""
    key: str
    value: Any
    timestamp: datetime
    hits: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry has expired"""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > ttl_seconds


class CacheManager:
    """
    Intelligent cache with TTL, LRU eviction, and semantic similarity.
    """
    
    def __init__(
        self,
        cache_path: Optional[Path] = None,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        enable_persistence: bool = True
    ):
        self.cache_path = cache_path or Path("logs/cache.pkl")
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_persistence = enable_persistence
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        
        if enable_persistence and self.cache_path.exists():
            self._load_from_disk()
    
    def _generate_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key from context"""
        # Serialize context to JSON and hash
        serialized = json.dumps(context, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def get(self, context: Dict[str, Any]) -> Optional[Any]:
        """Get cached value if exists and not expired"""
        key = self._generate_key(context)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired(self.ttl_seconds):
            del self.cache[key]
            self.misses += 1
            return None
        
        # Update hit count and move to end (LRU)
        entry.hits += 1
        self.hits += 1
        self.cache[key] = entry  # Re-insert to update order in dict
        
        return entry.value
    
    def set(self, context: Dict[str, Any], value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Set cache entry"""
        key = self._generate_key(context)
        
        # Evict if at max capacity (LRU: remove oldest)
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Remove least recently used (first item in dict)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.cache[key] = entry
        
        if self.enable_persistence:
            self._save_to_disk()
    
    def invalidate(self, context: Dict[str, Any]):
        """Remove specific entry from cache"""
        key = self._generate_key(context)
        if key in self.cache:
            del self.cache[key]
            if self.enable_persistence:
                self._save_to_disk()
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        if self.enable_persistence and self.cache_path.exists():
            self.cache_path.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        # Calculate cache size
        cache_size_kb = 0
        if self.cache_path.exists():
            cache_size_kb = self.cache_path.stat().st_size / 1024
        
        return {
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size_kb": cache_size_kb,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }
    
    def _save_to_disk(self):
        """Persist cache to disk"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)
    
    def _load_from_disk(self):
        """Load cache from disk"""
        try:
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache from disk: {e}")
            self.cache = {}
    
    def prune_expired(self):
        """Remove all expired entries"""
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired(self.ttl_seconds)
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys and self.enable_persistence:
            self._save_to_disk()
        
        return len(expired_keys)


class SemanticCacheManager(CacheManager):
    """
    Advanced cache with semantic similarity matching.
    Finds similar contexts even if not exact matches.
    """
    
    def __init__(self, similarity_threshold: float = 0.85, **kwargs):
        super().__init__(**kwargs)
        self.similarity_threshold = similarity_threshold
    
    def _compute_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """
        Compute similarity between two contexts.
        Uses Jaccard similarity on tokenized text.
        """
        # Extract text from contexts
        text1 = self._extract_text(context1)
        text2 = self._extract_text(context2)
        
        # Tokenize
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        # Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def _extract_text(self, context: Dict[str, Any]) -> str:
        """Extract all text from context dictionary"""
        texts = []
        
        def extract_recursive(obj):
            if isinstance(obj, str):
                texts.append(obj)
            elif isinstance(obj, dict):
                for v in obj.values():
                    extract_recursive(v)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(context)
        return " ".join(texts)
    
    def get(self, context: Dict[str, Any]) -> Optional[Any]:
        """Get with semantic similarity fallback"""
        # Try exact match first
        exact_match = super().get(context)
        if exact_match is not None:
            return exact_match
        
        # Try semantic match
        best_similarity = 0.0
        best_entry = None
        
        for entry in self.cache.values():
            if entry.is_expired(self.ttl_seconds):
                continue
            
            # Reconstruct context from metadata (if available)
            cached_context = entry.metadata.get("context", {})
            if not cached_context:
                continue
            
            similarity = self._compute_similarity(context, cached_context)
            
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_entry = entry
        
        if best_entry:
            best_entry.hits += 1
            self.hits += 1
            return best_entry.value
        
        self.misses += 1
        return None
    
    def set(self, context: Dict[str, Any], value: Any, metadata: Optional[Dict[str, Any]] = None):
        """Set with context stored in metadata for semantic matching"""
        meta = metadata or {}
        meta["context"] = context  # Store for similarity matching
        super().set(context, value, meta)

