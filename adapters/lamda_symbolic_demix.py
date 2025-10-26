"""
Symbolic MIDI Demix v1.1 (heuristic, NO-OP safe)
- Adds per-instrument feature extraction & role scores:
  * avg_pitch, p10_pitch, dur_mean_ql, short_ratio, monophony(=1-poly), stepwise_ratio
- Melody: high pitch, long sustain, monophonic, stepwise phrase
- Bass: low p10, monophonic
- Ornaments: short notes dominant
- Harmony: residual
CLI:
  python -m adapters.lamda_symbolic_demix --midi input.mid --out-dir out --write-midi 1 --json-out out/roles.json
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional

def _sec_to_ql(sec: float, bpm: float) -> float:
    return float(sec) * float(bpm) / 60.0 * 4.0

def _pct10(vals: List[float]) -> float:
    if not vals: return 0.0
    s = sorted(vals)
    k = max(0, int(len(s)*0.1)-1)
    return float(s[k])

def _feat_for_instrument(ins, bpm: float) -> Dict[str, float]:
    # compute features; handle empty safely
    notes = getattr(ins, "notes", []) or []
    if not notes:
        return {"avg_pitch":0.0,"p10_pitch":0.0,"dur_mean_ql":0.0,"short_ratio":0.0,"poly_ratio":0.0,"stepwise_ratio":0.0}
    pitches = [n.pitch for n in notes]
    durs_ql = [_sec_to_ql(n.end - n.start, bpm) for n in notes]
    avg_pitch = sum(pitches)/len(pitches)
    p10_pitch = _pct10(pitches)
    dur_mean = sum(durs_ql)/len(durs_ql)
    short_ratio = sum(1 for d in durs_ql if d <= 1.0) / float(len(durs_ql))  # <= quarter note
    # poly approx: chords if many same-time onsets
    onsets = sorted([n.start for n in notes])
    overlaps = 0
    thr = 0.02
    for i in range(1, len(onsets)):
        if abs(onsets[i] - onsets[i-1]) < thr: overlaps += 1
    poly_ratio = overlaps / max(1, len(onsets)-1)
    # stepwise ratio: |Δpitch| <= 2 semitones between consecutive onset-sorted notes (monophonic bias)
    step = 0; comp = 0
    prev_p = None; prev_t = None
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if prev_t is not None and (n.start - prev_t) >= 0.01:  # new event
            if prev_p is not None:
                comp += 1
                if abs(n.pitch - prev_p) <= 2: step += 1
        prev_p, prev_t = n.pitch, n.start
    stepwise_ratio = (step/comp) if comp else 0.0
    return {
        "avg_pitch": float(avg_pitch),
        "p10_pitch": float(p10_pitch),
        "dur_mean_ql": float(dur_mean),
        "short_ratio": float(short_ratio),
        "poly_ratio": float(poly_ratio),
        "stepwise_ratio": float(stepwise_ratio),
    }

def demix_roles(midi_path: str, write_midi: bool=False, out_dir: Optional[str]=None,
                melody_weight: Tuple[float,float,float,float] = (0.35, 0.25, 0.25, 0.15)) -> Dict[str, Any]:
    """
    melody_weight = (pitch_w, sustain_w, mono_w, step_w)  # weights sum to 1.0
    """
    out: Dict[str, Any] = {"file": midi_path, "roles": {}, "notes": 0, "features": {}, "scores": {}}
    try:
        import pretty_midi as pm
        m = pm.PrettyMIDI(midi_path)
        tempo_changes, tempi = m.get_tempo_changes()
        bpm = float(tempi[0]) if len(tempi) else 120.0

        # features per instrument
        feats = {}
        for idx, ins in enumerate(m.instruments):
            feats[str(idx)] = _feat_for_instrument(ins, bpm)
        out["features"] = feats

        role_map = {}
        # 1) drums
        for i, ins in enumerate(m.instruments):
            if ins.is_drum:
                role_map[i] = "drums"

        # non-drums list
        nd = [(i, m.instruments[i]) for i in range(len(m.instruments)) if not m.instruments[i].is_drum]
        # 2) bass by low register + monophony
        if nd:
            nd_stats = []
            for i, ins in nd:
                f = feats[str(i)]
                bass_score = (100 - f["p10_pitch"]) + (1.0 - f["poly_ratio"])*20.0
                nd_stats.append((bass_score, i))
            nd_stats.sort(reverse=True)  # highest score
            bass_i = nd_stats[0][1] if nd_stats else None
            if bass_i is not None and feats[str(bass_i)]["p10_pitch"] < 55:  # ~G3
                role_map[bass_i] = "bass"
                out["scores"][str(bass_i)] = {"bass_score": float(nd_stats[0][0])}

        # 3) melody scoring
        mel_scores = []
        for i, ins in nd:
            if i in role_map: continue
            f = feats[str(i)]
            # normalize terms
            pitch_term = max(0.0, (f["avg_pitch"] - 60.0) / 24.0)  # C4..C6 -> 0..1
            sustain_term = min(1.0, f["dur_mean_ql"] / 2.0)       # >= half note ~ good
            mono_term = 1.0 - min(1.0, f["poly_ratio"])
            step_term = f["stepwise_ratio"]
            w = melody_weight
            score = (w[0]*pitch_term + w[1]*sustain_term + w[2]*mono_term + w[3]*step_term)
            mel_scores.append((score, i))
        mel_scores.sort(reverse=True)
        if mel_scores:
            best_score, mel_i = mel_scores[0]
            if best_score >= 0.35:  # minimal bar
                role_map[mel_i] = "melody"
                out["scores"][str(mel_i)] = {"melody_score": float(best_score)}

        # 4) ornaments: dominant short notes
        for i, ins in nd:
            if i in role_map: continue
            f = feats[str(i)]
            if f["short_ratio"] >= 0.7:
                role_map[i] = "ornaments"

        # 5) remaining -> harmony
        for i, ins in nd:
            if i not in role_map:
                role_map[i] = "harmony"

        out["roles"] = {str(i): r for i, r in role_map.items()}
        out["notes"] = sum(len(ins.notes) for ins in m.instruments)

        # optional split MIDIs
        if write_midi and out_dir:
            import os
            os.makedirs(out_dir, exist_ok=True)
            by_role: Dict[str, pm.PrettyMIDI] = {}
            for i, ins in enumerate(m.instruments):
                r = role_map.get(i, "harmony")
                if r not in by_role:
                    by_role[r] = pm.PrettyMIDI()
                clone = pm.Instrument(program=ins.program, is_drum=ins.is_drum, name=ins.name or f"track{i}")
                clone.notes = [pm.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end) for n in ins.notes]
                by_role[r].instruments.append(clone)
            for r, mm in by_role.items():
                mm.write(os.path.join(out_dir, f"{r}.mid"))
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

if __name__ == "__main__":
    import argparse, json, os
    ap = argparse.ArgumentParser(description="Symbolic MIDI Demix v1.1")
    ap.add_argument("--midi", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--write-midi", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()
    res = demix_roles(args.midi, write_midi=bool(args.write_midi), out_dir=args.out_dir)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(res, ensure_ascii=False, indent=2))
