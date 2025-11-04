#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, sys, os, hashlib, pickle, json
from pathlib import Path

def md5_bytes(b: bytes) -> str:
    m = hashlib.md5(); m.update(b); return m.hexdigest()

def safe_imports():
    mods = {}
    try:
        import numpy as np
        mods["np"] = np
    except Exception:
        mods["np"] = None
    try:
        import soundfile as sf
        mods["sf"] = sf
    except Exception:
        mods["sf"] = None
    try:
        from scipy.signal import resample_poly
        mods["resample_poly"] = resample_poly
    except Exception:
        mods["resample_poly"] = None
    return mods

def read_audio(path, mods):
    sf = mods["sf"]; np = mods["np"]
    if sf is None or np is None:
        return None, None
    try:
        x, sr = sf.read(str(path), always_2d=True)
        if x.size == 0: return None, None
        x = x.astype("float32").mean(axis=1)
        return sr, x
    except Exception:
        return None, None

def resample_audio(sr, x, target_sr, mods):
    if sr == target_sr:
        return x
    rp = mods["resample_poly"]; np = mods["np"]
    if rp is not None:
        import math
        g = math.gcd(sr, target_sr)
        up = target_sr // g; down = sr // g
        return rp(x, up, down).astype("float32")
    if np is None:
        return x
    t_old = np.linspace(0, 1, num=len(x), endpoint=False, dtype="float32")
    t_new = np.linspace(0, 1, num=int(len(x) * (target_sr / sr)), endpoint=False, dtype="float32")
    return np.interp(t_new, t_old, x).astype("float32")

def peak_normalize(x, target_peak=0.98, eps=1e-9, np=None):
    if np is None: return x
    peak = float(np.max(np.abs(x)) + eps)
    return (x / peak * target_peak).astype("float32")

def trim_silence(x, sr, thr_db=-50.0, win_ms=20.0, np=None):
    if np is None: return x, (0.0, len(x)/max(1,sr))
    win = max(1, int(sr * win_ms * 0.001))
    if len(x) < win:
        return x, (0.0, len(x)/max(1,sr))
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(x, win).astype("float32")
    env = ( (w*w).mean(axis=1) + 1e-9 ) ** 0.5
    pad = win - 1
    import numpy as np
    env = np.pad(env, (pad//2, pad - pad//2), mode="edge")
    thr = 10.0 ** (thr_db / 20.0)
    mask = env > thr
    if not mask.any():
        return x, (0.0, len(x)/max(1,sr))
    i0 = int(np.argmax(mask))
    i1 = int(len(mask) - np.argmax(mask[::-1]) - 1)
    return x[i0:i1+1], (i0/sr, (i1+1)/sr)

def audio_stats(x, sr, np=None):
    if np is None or x is None or sr is None or len(x)==0:
        return {"dur_s": None, "rms": None, "peak": None, "onset_rate_hz": None, "clip_ratio": None}
    import numpy as np
    dur = len(x)/sr
    rms = float(np.sqrt(np.mean(x*x)))
    peak = float(np.max(np.abs(x)))
    frame = max(1, int(sr*0.02)); H = frame//2
    onset_rate = 0.0
    if len(x) >= frame*4:
        from numpy.fft import rfft
        E = []
        for i in range(0, len(x)-frame, H):
            spec = np.abs(rfft(x[i:i+frame]))
            E.append(spec.sum())
        diff = np.diff(np.array(E))
        onset_rate = float((diff > (diff.mean()+2*diff.std())).sum() / max(1, dur))
    clips = float((np.abs(x) > 0.999).sum())/max(1,len(x))
    return {"dur_s": dur, "rms": rms, "peak": peak, "onset_rate_hz": onset_rate, "clip_ratio": clips}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--peak", type=float, default=0.98)
    ap.add_argument("--trim-db", type=float, default=-50.0)
    ap.add_argument("--min-dur", type=float, default=1.5)
    ap.add_argument("--max-dur", type=float, default=120.0)
    ap.add_argument("--write-audio", action="store_true")
    ap.add_argument("--index-name", default="wav_index.pkl")
    ap.add_argument("--csv-name", default="wav_index.csv")
    ap.add_argument("--log-json", default="wav_cleaning_summary.json")
    args = ap.parse_args()

    mods = safe_imports(); np = mods["np"]

    in_root = Path(args.input)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted([p for p in in_root.rglob("*") if p.suffix.lower() in (".wav",".wave")])
    seen_md5 = set()
    idx = []; dup = 0; ok = 0; err = 0

    for p in wavs:
        try:
            raw = p.read_bytes()
            md5 = md5_bytes(raw)
            if md5 in seen_md5:
                idx.append({"original_path": str(p), "md5_raw": md5, "status":"duplicate"})
                dup += 1; continue
            seen_md5.add(md5)

            sr, x = read_audio(p, mods)
            s_in = audio_stats(x, sr, np=np) if x is not None else {}
            flags = []
            if x is None:
                flags.append("decode_failed")
            else:
                x = resample_audio(sr, x, args.sr, mods) if sr else None
                x = peak_normalize(x, args.peak, np=np) if x is not None else None
                x, (t0,t1) = trim_silence(x, args.sr, args.trim_db, np=np) if x is not None else (None,(0.0,0.0))
                s_out = audio_stats(x, args.sr, np=np) if x is not None else {}
                if s_out.get("dur_s") and s_out["dur_s"] < args.min_dur: flags.append("too_short")
                if s_out.get("dur_s") and s_out["dur_s"] > args.max_dur: flags.append("too_long")
                if s_out.get("clip_ratio") and s_out["clip_ratio"] > 0.01: flags.append("clipping")
                if s_out.get("rms") and s_out["rms"] < 0.01: flags.append("too_quiet")
                if args.write_audio and x is not None and mods["sf"] is not None:
                    out_rel = p.relative_to(in_root)
                    out_path = out_root / "cleaned" / out_rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    mods["sf"].write(str(out_path), x, args.sr, subtype="PCM_16")
            rec = {
                "original_path": str(p),
                "md5_raw": md5,
                "sr_in": sr,
                "duration_in_s": s_in.get("dur_s"),
                "sr_out": args.sr if x is not None else None,
                "duration_out_s": (s_out.get("dur_s") if 's_out' in locals() else None),
                "flags": flags,
                "status": "ok" if x is not None else "decode_failed",
            }
            idx.append(rec); ok += 1 if x is not None else 0
        except Exception as e:
            idx.append({"original_path": str(p), "md5_raw": None, "status":"error", "error": str(e)})
            err += 1

    import pandas as pd
    (out_root / "index").mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(idx)
    df.to_csv(out_root/"index"/args.csv_name, index=False)
    with open(out_root/"index"/args.index_name, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)
    summary = {
        "scanned": len(wavs),
        "unique": len(seen_md5),
        "deduped": dup,
        "ok": ok,
        "error": err,
        "out_index": str(out_root/"index"/args.index_name),
        "out_csv": str(out_root/"index"/args.csv_name),
        "wrote_audio": bool(args.write_audio),
    }
    with open(out_root/args.log_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
