#!/usr/bin/env python3
"""
LAMDa Test Sample Creator
小規模テストデータを作成してローカルで高速反復テストを実行

Usage:
    python scripts/create_test_sample.py

Output:
    data/Los-Angeles-MIDI/TEST_SAMPLE/
    ├── CHORDS_DATA/
    │   └── sample_100.pickle (100サンプル)
    ├── KILO_CHORDS_DATA/
    │   └── sample_100.pickle
    ├── SIGNATURES_DATA/
    │   └── sample_100.pickle
    └── TOTALS_MATRIX/
        └── sample.pickle
"""

import pickle
from pathlib import Path
import shutil

# tqdm is optional
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


def create_test_sample(source_dir: Path, output_dir: Path, num_samples: int = 100):
    """テストサンプルを作成"""

    print(f"📦 Creating test sample with {num_samples} entries...")
    print(f"   Source: {source_dir}")
    print(f"   Output: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # CHORDS_DATA から最初のnum_samplesを抽出
    print("\n1️⃣ Processing CHORDS_DATA...")
    chords_source = source_dir / "CHORDS_DATA"
    chords_output = output_dir / "CHORDS_DATA"
    chords_output.mkdir(exist_ok=True)

    # 最初のpickleファイルから抽出
    first_pickle = list(chords_source.glob("*.pickle"))[0]
    print(f"   Reading: {first_pickle.name}")

    with open(first_pickle, "rb") as f:
        full_data = pickle.load(f)

    sample_data = full_data[:num_samples]

    output_pickle = chords_output / f"sample_{num_samples}.pickle"
    with open(output_pickle, "wb") as f:
        pickle.dump(sample_data, f)

    print(f"   ✅ Created: {output_pickle}")
    print(f"      Samples: {len(sample_data)}")

    # hash_idリストを取得
    sample_hash_ids = set([item[0] for item in sample_data])
    print(f"      Hash IDs: {len(sample_hash_ids)}")

    # KILO_CHORDS_DATA から対応するサンプルを抽出
    print("\n2️⃣ Processing KILO_CHORDS_DATA...")
    kilo_source = source_dir / "KILO_CHORDS_DATA" / "LAMDa_KILO_CHORDS_DATA.pickle"
    kilo_output = output_dir / "KILO_CHORDS_DATA"
    kilo_output.mkdir(exist_ok=True)

    if kilo_source.exists():
        with open(kilo_source, "rb") as f:
            kilo_data = pickle.load(f)

        # hash_idが一致するものだけ抽出
        kilo_sample = [item for item in kilo_data if item[0] in sample_hash_ids]

        output_kilo = kilo_output / f"sample_{num_samples}.pickle"
        with open(output_kilo, "wb") as f:
            pickle.dump(kilo_sample, f)

        print(f"   ✅ Created: {output_kilo}")
        print(f"      Samples: {len(kilo_sample)}")
    else:
        print(f"   ⚠️  Source not found: {kilo_source}")

    # SIGNATURES_DATA から対応するサンプルを抽出
    print("\n3️⃣ Processing SIGNATURES_DATA...")
    sig_source = source_dir / "SIGNATURES_DATA" / "LAMDa_SIGNATURES_DATA.pickle"
    sig_output = output_dir / "SIGNATURES_DATA"
    sig_output.mkdir(exist_ok=True)

    if sig_source.exists():
        with open(sig_source, "rb") as f:
            sig_data = pickle.load(f)

        # hash_idが一致するものだけ抽出
        sig_sample = [item for item in sig_data if item[0] in sample_hash_ids]

        output_sig = sig_output / f"sample_{num_samples}.pickle"
        with open(output_sig, "wb") as f:
            pickle.dump(sig_sample, f)

        print(f"   ✅ Created: {output_sig}")
        print(f"      Samples: {len(sig_sample)}")
    else:
        print(f"   ⚠️  Source not found: {sig_source}")

    # TOTALS_MATRIX はそのままコピー
    print("\n4️⃣ Processing TOTALS_MATRIX...")
    totals_source = source_dir / "TOTALS_MATRIX" / "LAMDa_TOTALS.pickle"
    totals_output = output_dir / "TOTALS_MATRIX"
    totals_output.mkdir(exist_ok=True)

    if totals_source.exists():
        shutil.copy2(totals_source, totals_output / "LAMDa_TOTALS.pickle")
        print(f"   ✅ Copied: TOTALS_MATRIX")
    else:
        print(f"   ⚠️  Source not found: {totals_source}")

    # サマリー
    print("\n" + "=" * 70)
    print("✅ Test sample created successfully!")
    print("=" * 70)
    print(f"📁 Location: {output_dir}")
    print(f"📊 Samples: {num_samples}")
    print(f"🔗 Linked hash_ids: {len(sample_hash_ids)}")

    return output_dir


if __name__ == "__main__":
    # 設定
    SOURCE_DIR = Path("data/Los-Angeles-MIDI")
    OUTPUT_DIR = Path("data/Los-Angeles-MIDI/TEST_SAMPLE")
    NUM_SAMPLES = 100  # 100サンプルでテスト

    # 実行
    create_test_sample(SOURCE_DIR, OUTPUT_DIR, NUM_SAMPLES)

    print("\n🚀 Ready for local testing!")
    print("   Run: python scripts/test_local_build.py")
