# LAMDa Unified Database Build - Execution Checklist
# ===================================================

## ✅ Pre-Execution Checklist

### 1. GCS Data Validation
- [ ] Verify GCS bucket access: `gsutil ls gs://otobon/lamda/`
- [ ] Confirm data uploaded:
  - [ ] CHORDS_DATA.tar.gz (575MB)
  - [ ] KILO_CHORDS_DATA/ (602MB)
  - [ ] SIGNATURES_DATA/ (290MB)
  - [ ] TOTALS_MATRIX/ (33MB)
  - [ ] CODE/ (15MB)

### 2. Vertex AI Environment
- [ ] Vertex AI Colab Enterprise instance created
  - Instance: `shimogami88-Default`
  - Machine type: `e2-standard-4` (4 vCPU, 16GB RAM)
  - Region: `us-central1`
- [ ] Credit balance confirmed: ¥155,305 available

### 3. Repository Preparation
- [ ] `lamda_unified_analyzer.py` created ✅
- [ ] `scripts/build_lamda_unified_db.py` created ✅
- [ ] `docs/vertex_ai_lamda_unified_guide.py` created ✅
- [ ] `docs/LAMDA_UNIFIED_ARCHITECTURE.md` created ✅
- [ ] README.md updated with LAMDa section ✅

### 4. Code Review
- [ ] Review `lamda_unified_analyzer.py`:
  - [ ] LAMDaEvent class (data structure parsing)
  - [ ] LAMDaUnifiedAnalyzer class (integration logic)
  - [ ] build_unified_database() method (main processing)
- [ ] Review database schema:
  - [ ] progressions table (CHORDS_DATA)
  - [ ] kilo_sequences table (KILO_CHORDS_DATA)
  - [ ] signatures table (SIGNATURES_DATA)
  - [ ] hash_id indexing

## 🚀 Execution Steps

### Option A: Python Script Execution
```bash
# 1. Push code to GitHub
git add lamda_unified_analyzer.py scripts/build_lamda_unified_db.py
git add docs/vertex_ai_lamda_unified_guide.py docs/LAMDA_UNIFIED_ARCHITECTURE.md
git add README.md
git commit -m "Add LAMDa unified architecture integration"
git push origin main

# 2. Open Vertex AI Colab Enterprise
# Navigate to: https://console.cloud.google.com/vertex-ai/colab

# 3. Create new notebook or terminal session

# 4. Run the script
python scripts/build_lamda_unified_db.py
```

**Estimated time**: 90-120 minutes  
**Estimated cost**: ¥30-50

### Option B: Notebook Guided Execution (RECOMMENDED)
```python
# 1. Open new Colab Notebook in Vertex AI

# 2. Copy content from docs/vertex_ai_lamda_unified_guide.py

# 3. Execute cells in order:
#    Cell 1: Environment check + GCS authentication
#    Cell 2: Repository clone + dependencies
#    Cell 3: Download CHORDS_DATA (575MB → 15GB)
#    Cell 4: Download KILO, SIGNATURES, TOTALS
#    Cell 5: Build unified database (60-90 min)
#    Cell 6: Upload to GCS
#    Cell 7: Summary stats

# 4. Monitor progress with tqdm bars

# 5. Verify output: gs://otobon/lamda/lamda_unified.db
```

## 📊 Expected Output

### Database Structure
```
lamda_unified.db (estimated 500MB - 2GB)
├── progressions (CHORDS_DATA derived)
│   ├── hash_id
│   ├── progression (JSON)
│   ├── total_events
│   ├── chord_events
│   └── source_file
├── kilo_sequences (KILO_CHORDS_DATA)
│   ├── hash_id (PRIMARY KEY)
│   ├── sequence
│   └── sequence_length
└── signatures (SIGNATURES_DATA)
    ├── hash_id (PRIMARY KEY)
    ├── pitch_distribution
    └── top_pitches
```

### Statistics (Example)
```
Total progressions: ~150,000-200,000
Total kilo sequences: ~180,000
Total signatures: ~180,000
```

## ✅ Post-Execution Checklist

### 1. Verification
- [ ] Database file created: `lamda_unified.db`
- [ ] Database size reasonable (500MB - 2GB)
- [ ] GCS upload successful: `gs://otobon/lamda/lamda_unified.db`

### 2. Quality Checks
```python
import sqlite3

conn = sqlite3.connect('lamda_unified.db')
cursor = conn.cursor()

# Check progressions
cursor.execute("SELECT COUNT(*) FROM progressions")
prog_count = cursor.fetchone()[0]
print(f"Progressions: {prog_count:,}")

# Check kilo_sequences
cursor.execute("SELECT COUNT(*) FROM kilo_sequences")
kilo_count = cursor.fetchone()[0]
print(f"Kilo sequences: {kilo_count:,}")

# Check signatures
cursor.execute("SELECT COUNT(*) FROM signatures")
sig_count = cursor.fetchone()[0]
print(f"Signatures: {sig_count:,}")

# Verify hash_id linkage
cursor.execute("""
    SELECT COUNT(*) FROM progressions p
    INNER JOIN kilo_sequences k ON p.hash_id = k.hash_id
""")
linked_count = cursor.fetchone()[0]
print(f"Linked records: {linked_count:,}")

conn.close()
```

- [ ] All counts > 0
- [ ] Linked records exist (hash_id join works)

### 3. Download for Local Use
```bash
# Download from GCS
gsutil cp gs://otobon/lamda/lamda_unified.db ./

# Test query
sqlite3 lamda_unified.db "SELECT COUNT(*) FROM progressions;"
```

### 4. Integration Testing
```python
from lamda_unified_analyzer import LAMDaUnifiedAnalyzer
from pathlib import Path

# Test analyzer
analyzer = LAMDaUnifiedAnalyzer(Path('data/Los-Angeles-MIDI'))

# Load components
kilo_data = analyzer.load_kilo_chords()
print(f"Loaded {len(kilo_data)} kilo sequences")

signatures = analyzer.load_signatures()
print(f"Loaded {len(signatures)} signatures")

# Test database query
import sqlite3
conn = sqlite3.connect('lamda_unified.db')
cursor = conn.cursor()

# Sample progression
cursor.execute("SELECT * FROM progressions LIMIT 1")
sample = cursor.fetchone()
print(f"Sample progression: {sample}")

conn.close()
```

## 🐛 Troubleshooting

### Issue: GCS Permission Denied
```bash
# Solution: Authenticate
gcloud auth application-default login
```

### Issue: Memory Error during CHORDS_DATA processing
```python
# Solution: Process in smaller batches
# Modify build_unified_database() to commit every 10 files
```

### Issue: Database locked
```bash
# Solution: Close all connections
rm lamda_unified.db  # Delete and rebuild
```

### Issue: Extraction taking too long
```bash
# Normal: CHORDS_DATA extraction is 15GB
# Expected: 5-10 minutes for extraction
# If > 20 minutes, check disk I/O
```

## 📈 Performance Optimization

### For future runs:
1. **Cache extracted CHORDS_DATA**: Skip extraction if already exists
2. **Parallel processing**: Use multiprocessing for pickle analysis
3. **Batch commits**: Commit every N files instead of per-file
4. **Index creation**: Create indexes AFTER bulk insert

## 🎯 Next Steps After Completion

1. **Implement recommendation system**:
   - Use kilo_sequences for fast pattern matching
   - Use signatures for similarity scoring

2. **Build search interface**:
   - Query by chord progression
   - Filter by key (from signatures)
   - Sort by similarity

3. **Integrate with composer**:
   - Use LAMDa progressions as templates
   - Apply style transfer from signatures
   - Generate variations using patterns

## 📚 Documentation References

- Architecture: `docs/LAMDA_UNIFIED_ARCHITECTURE.md`
- Analyzer API: `lamda_unified_analyzer.py` docstrings
- Execution guide: `docs/vertex_ai_lamda_unified_guide.py`
- README section: Search for "LAMDa Integration"

## ⏰ Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Clone repo + install deps | 2-3 min | 3 min |
| Download CHORDS_DATA | 3-5 min | 8 min |
| Extract CHORDS_DATA | 5-10 min | 18 min |
| Download KILO/SIG/TOTALS | 2-3 min | 21 min |
| Build database | 60-90 min | 111 min |
| Upload to GCS | 1-2 min | 113 min |
| **Total** | **~2 hours** | |

## 💰 Cost Estimate

| Resource | Rate | Usage | Cost |
|----------|------|-------|------|
| e2-standard-4 | ¥23/hour | 2 hours | ¥46 |
| GCS egress | ¥0.15/GB | 2GB | ¥0.30 |
| GCS storage | ¥0.023/GB/month | 1.5GB | ¥0.03 |
| **Total** | | | **~¥50** |

## ✅ Final Approval

Before execution, confirm:
- [ ] All code reviewed and tested locally (if possible)
- [ ] GCS data validated
- [ ] Vertex AI instance ready
- [ ] Budget approved (¥50)
- [ ] Time allocated (2 hours)

**Ready to execute? Let's build the unified database! 🚀**
