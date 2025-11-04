# Phase 27 Optimization Implementation Guide

**Phase 27.5-27.7**: Latency Optimization (p95 < 50ms target)

---

## Task 5.2: NumPy Vectorization Optimization

**Target**: 20-30% latency reduction

### Current Implementation (Loop-based)

```python
# ml/drum_pattern_recommender.py (example)
def _calculate_pattern_distances(query: DrumQuery, candidates: List[DrumPattern]):
    distances = []
    for pattern in candidates:
        dist = calculate_distance(query, pattern)
        distances.append(dist)
    return distances

def calculate_distance(query: DrumQuery, pattern: DrumPattern) -> float:
    tempo_diff = abs(query.tempo_bpm - pattern.tempo_bpm)
    energy_diff = abs(query.target_energy - pattern.energy)
    section_match = 1.0 if query.section == pattern.section else 0.0
    
    return tempo_diff * 0.4 + energy_diff * 0.4 + (1 - section_match) * 0.2
```

### Optimized Implementation (Vectorized)

```python
import numpy as np

def _calculate_pattern_distances_vectorized(
    query: DrumQuery, 
    candidates: List[DrumPattern]
) -> np.ndarray:
    """
    Vectorized distance calculation using NumPy broadcasting.
    
    Performance: ~25-30% faster than loop-based approach.
    """
    # Extract features into NumPy arrays (1-time operation)
    n = len(candidates)
    
    tempo_vec = np.array([p.tempo_bpm for p in candidates])
    energy_vec = np.array([p.energy for p in candidates])
    section_vec = np.array([p.section for p in candidates])
    
    # Query features (broadcast-ready)
    query_tempo = query.tempo_bpm
    query_energy = query.target_energy
    query_section = query.section
    
    # Vectorized distance calculation
    tempo_diff = np.abs(tempo_vec - query_tempo)
    energy_diff = np.abs(energy_vec - query_energy)
    section_match = (section_vec == query_section).astype(float)
    
    # Weighted sum (vectorized)
    distances = tempo_diff * 0.4 + energy_diff * 0.4 + (1 - section_match) * 0.2
    
    return distances
```

### Feature Extraction Optimization

```python
# Before (loop-based)
def extract_features(patterns: List[DrumPattern]) -> List[List[float]]:
    features = []
    for p in patterns:
        feat = [
            p.tempo_bpm,
            p.energy,
            p.density,
            p.kick_downbeat_rate,
            p.snare_backbeat_acc,
        ]
        features.append(feat)
    return features

# After (vectorized)
def extract_features_vectorized(patterns: List[DrumPattern]) -> np.ndarray:
    """
    Vectorized feature extraction.
    
    Performance: ~20-25% faster than loop-based approach.
    """
    n = len(patterns)
    X = np.empty((n, 5), dtype=np.float32)
    
    # Column-wise assignment (cache-friendly)
    X[:, 0] = [p.tempo_bpm for p in patterns]
    X[:, 1] = [p.energy for p in patterns]
    X[:, 2] = [p.density for p in patterns]
    X[:, 3] = [p.kick_downbeat_rate for p in patterns]
    X[:, 4] = [p.snare_backbeat_acc for p in patterns]
    
    return X
```

### Implementation Steps

1. **Replace loop-based distance calculation**:
   - `ml/drum_pattern_recommender.py`: `_calculate_pattern_distances()`
   - `ml/guitar_pattern_recommender.py`: `_calculate_pattern_distances()` (if exists)
   - `ml/bass_pattern_recommender.py`: `_calculate_pattern_distances()` (if exists)
   - `ml/piano_pattern_recommender.py`: `_calculate_pattern_distances()` (if exists)

2. **Vectorize feature extraction**:
   - All recommender classes: `extract_features()`

3. **Benchmark**:
   ```bash
   python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
   # Expected: p95 ~70-80ms (before: ~100ms)
   ```

---

## Task 5.3: ML Model Cache Implementation

**Target**: 10-15% latency reduction

### Strategy

1. **Query Hashing**: Convert `DrumQuery` to hashable key
2. **LRU Cache**: `functools.lru_cache(maxsize=1024)`
3. **Cache Invalidation**: Automatic (LRU eviction)

### Implementation

```python
from functools import lru_cache
from typing import Tuple

class DrumPatternRecommender:
    def __init__(self, model, pattern_dict, config):
        self.model = model
        self.pattern_dict = pattern_dict
        self.config = config
    
    def recommend(
        self, 
        query: DrumQuery, 
        min_proba: float = 0.15, 
        min_margin: float = 0.10
    ) -> DrumPattern:
        """
        Recommend drum pattern with ML inference caching.
        """
        # 1. Query hashing
        query_key = self._hash_query(query)
        
        # 2. Cached ML inference
        probas = self._predict_cached(query_key)
        
        # 3. Pattern selection (not cached, query-specific)
        pattern = self._select_pattern(probas, query, min_proba, min_margin)
        
        return pattern
    
    def _hash_query(self, query: DrumQuery) -> Tuple:
        """
        Convert query to hashable key.
        
        Important: Only include features used in ML inference.
        Exclude query-specific fields (e.g., song_id).
        """
        return (
            query.tempo_bpm,
            query.time_sig_slots,
            query.section,
            round(query.target_energy, 2),  # Round to avoid cache misses
        )
    
    @lru_cache(maxsize=1024)
    def _predict_cached(self, query_key: Tuple) -> np.ndarray:
        """
        Cached ML inference.
        
        Cache size: 1024 (recent queries)
        Performance: ~10-15% latency reduction (cache hit rate ~60-70%)
        """
        # Reconstruct query from key
        tempo_bpm, time_sig_slots, section, target_energy = query_key
        
        # Extract features
        features = np.array([
            tempo_bpm,
            time_sig_slots,
            self._encode_section(section),
            target_energy,
        ]).reshape(1, -1)
        
        # ML inference
        probas = self.model.predict_proba(features)[0]
        
        return probas
    
    def _encode_section(self, section: str) -> float:
        """Section encoding (example)"""
        mapping = {"Intro": 0.0, "Verse": 0.25, "Chorus": 0.5, "Bridge": 0.75, "Outro": 1.0}
        return mapping.get(section, 0.5)
```

### Cache Monitoring

```python
def get_cache_stats(self) -> dict:
    """
    Get cache statistics for monitoring.
    
    Returns:
        {
            "cache_size": 1024,
            "cache_hits": 650,
            "cache_misses": 350,
            "hit_rate": 0.65
        }
    """
    info = self._predict_cached.cache_info()
    
    return {
        "cache_size": info.maxsize,
        "cache_hits": info.hits,
        "cache_misses": info.misses,
        "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0,
    }
```

### Implementation Steps

1. **Add query hashing**:
   - All recommender classes: `_hash_query()`

2. **Implement cached inference**:
   - All recommender classes: `_predict_cached()` with `@lru_cache`

3. **Update recommend method**:
   - All recommender classes: `recommend()` to use `_predict_cached()`

4. **Add cache monitoring**:
   - All recommender classes: `get_cache_stats()`

5. **Benchmark**:
   ```bash
   python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
   # Expected: p95 ~60-70ms (before: ~70-80ms)
   ```

---

## Task 5.4: Batch Processing Optimization

**Target**: 30-40% latency reduction (for multiple queries)

### Strategy

1. **Batch API**: `recommend_batch(queries: List[Query])`
2. **Batch ML Inference**: `model.predict_proba(X_batch)`
3. **Parallel Pattern Selection**: Vectorized filtering

### Implementation

```python
class DrumPatternRecommender:
    def recommend_batch(
        self,
        queries: List[DrumQuery],
        min_proba: float = 0.15,
        min_margin: float = 0.10
    ) -> List[DrumPattern]:
        """
        Batch recommendation for multiple queries.
        
        Performance: ~30-40% faster than individual recommend() calls.
        
        Use case: Real-time generation of multiple sections (e.g., Verse + Chorus)
        """
        if not queries:
            return []
        
        # 1. Batch feature extraction
        X_batch = self._extract_features_batch(queries)
        
        # 2. Batch ML inference
        probas_batch = self.model.predict_proba(X_batch)
        
        # 3. Batch pattern selection
        results = []
        for probas, query in zip(probas_batch, queries):
            pattern = self._select_pattern(probas, query, min_proba, min_margin)
            results.append(pattern)
        
        return results
    
    def _extract_features_batch(self, queries: List[DrumQuery]) -> np.ndarray:
        """
        Vectorized feature extraction for batch queries.
        """
        n = len(queries)
        X = np.empty((n, 4), dtype=np.float32)
        
        # Vectorized extraction
        X[:, 0] = [q.tempo_bpm for q in queries]
        X[:, 1] = [q.time_sig_slots for q in queries]
        X[:, 2] = [self._encode_section(q.section) for q in queries]
        X[:, 3] = [q.target_energy for q in queries]
        
        return X
```

### Benchmark Script Update

```python
# scripts/benchmark_ml_latency.py

def benchmark_drums_batch(pickle_path: Path, iterations: int = 1000) -> Dict[str, float]:
    """Benchmark batch processing"""
    print(f"[INFO] Benchmarking Drums (batch mode, iterations: {iterations})")
    
    rec = DrumPatternRecommender.from_pickle(pickle_path)
    
    # Batch queries (e.g., 10 queries per batch)
    batch_size = 10
    test_batches = []
    
    for i in range(0, iterations, batch_size):
        batch = [
            DrumQuery(
                tempo_bpm=120 + j % 60,
                time_sig_slots=16,
                section="Chorus",
                target_energy=0.5 + (j % 10) * 0.05
            )
            for j in range(i, min(i + batch_size, iterations))
        ]
        test_batches.append(batch)
    
    # Benchmark
    latencies = []
    
    for batch in test_batches:
        t0 = time.perf_counter()
        results = rec.recommend_batch(batch, min_proba=0.15, min_margin=0.10)
        t_elapsed = time.perf_counter() - t0
        
        # Per-query latency
        latency_per_query = t_elapsed / len(batch)
        latencies.append(latency_per_query)
    
    # Stats
    stats = {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "mean": float(np.mean(latencies)),
        "batch_size": batch_size,
    }
    
    print(f"\n[RESULT] Drums Batch Latency Stats (per-query):")
    print(f"  - p50: {format_ms(stats['p50'])}")
    print(f"  - p95: {format_ms(stats['p95'])} {'✅ PASS' if stats['p95'] < 0.050 else '❌ FAIL'}")
    print(f"  - batch_size: {batch_size}")
    
    return stats
```

### Implementation Steps

1. **Add batch API**:
   - All recommender classes: `recommend_batch(queries)`

2. **Implement batch feature extraction**:
   - All recommender classes: `_extract_features_batch(queries)`

3. **Update benchmark script**:
   - `scripts/benchmark_ml_latency.py`: Add `benchmark_drums_batch()`

4. **Benchmark**:
   ```bash
   # Single-query mode (baseline)
   python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
   
   # Batch mode (optimized)
   python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000 --batch-mode
   # Expected: p95 ~40-50ms (before: ~60-70ms)
   ```

---

## Combined Performance Target

**Optimization Pipeline**:

1. **Baseline**: p95 ~100ms
2. **After NumPy Vectorization** (Task 5.2): p95 ~70-80ms (-20-30%)
3. **After ML Cache** (Task 5.3): p95 ~60-70ms (-10-15%)
4. **After Batch Processing** (Task 5.4): p95 ~40-50ms (-30-40% from step 3)

**Final Target**: p95 < 50ms ✅

---

## Implementation Priority

1. **Task 5.2** (NumPy Vectorization): Implement first (foundation)
2. **Task 5.3** (ML Cache): Implement second (independent)
3. **Task 5.4** (Batch Processing): Implement last (requires 5.2)

**Estimated Total Time**: ~4-6 hours (all 3 tasks)

---

## Testing & Validation

### Benchmark Command Sequence

```bash
# 1. Baseline (before optimization)
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
# Record: p95_baseline

# 2. After Task 5.2 (NumPy Vectorization)
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
# Expected: p95 ~70-80ms

# 3. After Task 5.3 (ML Cache)
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000
# Expected: p95 ~60-70ms

# 4. After Task 5.4 (Batch Processing)
python scripts/benchmark_ml_latency.py --instrument drums --iterations 1000 --batch-mode
# Expected: p95 ~40-50ms ✅
```

### Validation Criteria

- ✅ p95 < 50ms (primary target)
- ✅ p99 < 100ms (secondary target)
- ✅ Cache hit rate > 60% (Task 5.3)
- ✅ No KPI degradation (accent_score, chord_fit, etc.)

---

**Next**: Implement Task 5.2 (NumPy Vectorization)
