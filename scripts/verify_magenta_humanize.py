#!/usr/bin/env python3
"""
Phase A検証: Magenta humanize効果自動チェック
Velocity std / IOI std が grooved > seed になっていることを確認
"""
import sys
from pathlib import Path
import numpy as np

try:
    import pretty_midi
except ImportError:
    print("❌ pretty_midi not installed")
    sys.exit(1)


def stats(mid_path: Path):
    """Calculate velocity std and IOI std from drum MIDI."""
    if not mid_path.exists():
        return None, None
    
    pm = pretty_midi.PrettyMIDI(str(mid_path))
    inst = [i for i in pm.instruments if i.is_drum]
    notes = [n for i in inst for n in i.notes]
    
    if not notes:
        return 0.0, 0.0
    
    vels = np.array([n.velocity for n in notes])
    onsets = np.array([n.start for n in notes])
    
    vel_std = float(vels.std())
    
    if len(onsets) > 1:
        ioi = np.diff(np.sort(onsets))
        ioi_std = float(ioi.std())
    else:
        ioi_std = 0.0
    
    return vel_std, ioi_std


def main():
    if len(sys.argv) < 2:
        print("Usage: verify_magenta_humanize.py <song_dir>")
        print("Example: verify_magenta_humanize.py song_packages/suno_project/song_001")
        sys.exit(1)
    
    song_dir = Path(sys.argv[1])
    seed = song_dir / "drums_seed.mid"
    grooved = song_dir / "drums_grooved.mid"
    
    if not (seed.exists() and grooved.exists()):
        print(f"⚠️  Missing seed/grooved MIDI in {song_dir}")
        print(f"   seed: {seed.exists()}, grooved: {grooved.exists()}")
        sys.exit(0)
    
    sv, si = stats(seed)
    gv, gi = stats(grooved)
    
    if sv is None or gv is None:
        print("❌ Failed to load MIDI files")
        sys.exit(1)
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Magenta Humanize Effect Check")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Velocity std: seed={sv:.2f} → grooved={gv:.2f}")
    print(f"IOI std     : seed={si:.4f}s → grooved={gi:.4f}s")
    
    # Check improvement
    vel_improved = gv > sv
    ioi_improved = gi > si
    
    if vel_improved or ioi_improved:
        print("✅ Humanization detected:")
        if vel_improved:
            vel_increase = ((gv - sv) / sv * 100) if sv > 0 else 0
            print(f"   - Velocity variance increased: {sv:.2f} → {gv:.2f} (+{vel_increase:.1f}%)")
        if ioi_improved:
            ioi_increase = ((gi - si) / si * 100) if si > 0 else 0
            print(f"   - Timing variance increased: {si:.4f}s → {gi:.4f}s (+{ioi_increase:.1f}%)")
    else:
        print("⚠️  No humanization improvement detected")
        print("   (Expected: grooved std > seed std)")
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
