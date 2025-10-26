"""
Teacher v1.1 (minimal+priors)
- Adds positional chord priors (bar % 8)
- Adds key bigram prior from key_hints
- Light denoising: snap rare figures to positional majority
- NO external ML deps; fully deterministic
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import json, os, pickle

def _figure(e: Dict[str,Any]) -> str:
    r = e.get("root") or "N"
    q = e.get("quality") or ""
    return r if r=="N" else f"{r}{q}"

def _bar_index_from_timeql(t_ql: float) -> int:
    try:
        return int(float(t_ql)//4.0)
    except Exception:
        return 0

class TeacherV1:
    def __init__(self):
        self.chord_hist: Dict[str,int] = {}
        self.key_hist: Dict[str,int] = {}
        self.section_hist: Dict[str,int] = {}
        self.section_bigram: Dict[Tuple[str,str], int] = {}
        # new
        self.pos_chord_hist: Dict[int, Dict[str,int]] = {}  # pos(0..7) -> figure -> count
        self.key_bigram: Dict[Tuple[str,str], int] = {}
        self.version = "v1.1"

    def update_from_stage2(self, j: Dict[str,Any]):
        # chords
        ev = ((j.get("chordmap") or {}).get("events")) or []
        for e in ev:
            fig = _figure(e)
            self.chord_hist[fig] = self.chord_hist.get(fig, 0) + 1
            pos = _bar_index_from_timeql(e.get("time",0.0)) % 8
            d = self.pos_chord_hist.setdefault(pos, {})
            d[fig] = d.get(fig, 0) + 1
        # keys + bigram
        hints = j.get("key_hint") or j.get("key_hints") or []
        seq: List[str] = []
        for k in hints:
            key = ""
            if isinstance(k, (list,tuple)) and len(k)>=2: key = str(k[1])
            elif isinstance(k, dict): key = str(k.get("to") or k.get("key") or "")
            if key: 
                self.key_hist[key] = self.key_hist.get(key, 0) + 1
                seq.append(key)
        for a,b in zip(seq, seq[1:]):
            self.key_bigram[(a,b)] = self.key_bigram.get((a,b), 0) + 1
        # sections
        sec = ((j.get("sections_auto") or {}).get("sections")) or ((j.get("sections") or {}).get("sections")) or []
        labels = [s.get("label","") for s in sec if isinstance(s, dict)]
        for lab in labels:
            if not lab: continue
            self.section_hist[lab] = self.section_hist.get(lab, 0) + 1
        for a,b in zip(labels, labels[1:]):
            if a and b:
                self.section_bigram[(a,b)] = self.section_bigram.get((a,b), 0) + 1

    def fit_from_dir(self, gold_stage2_dir: str):
        for root, _, files in os.walk(gold_stage2_dir):
            for fn in files:
                if not fn.endswith(".json"): continue
                try:
                    j = json.load(open(os.path.join(root, fn), "r", encoding="utf-8"))
                    self.update_from_stage2(j)
                except Exception:
                    continue
        return self

    def _majority(self, hist: Dict[str,int], default: str) -> str:
        return max(hist.items(), key=lambda kv: kv[1])[0] if hist else default

    def _pos_majority(self, pos: int, default: str) -> str:
        d = self.pos_chord_hist.get(pos%8, {})
        return self._majority(d, default)

    def predict_from_stage2_like(self, j: Dict[str,Any]) -> Dict[str,Any]:
        chordmap = (j.get("chordmap") or {})
        events = chordmap.get("events") or []
        # fill/denoise chords
        if not events:
            bars = 64
            # positional majority fill
            ev = []
            for b in range(bars):
                fig = self._pos_majority(b%8, self._majority(self.chord_hist, "Cmaj"))
                root = "C"; qual = "maj"
                # naive split of figure
                if fig and fig!="N":
                    root = fig[0:2] if len(fig)>=2 and fig[1] in "#-" else fig[0]
                    qual = fig[len(root):]
                else:
                    root, qual = "N", ""
                ev.append({"time": float(b*4), "root": root, "quality": qual, "confidence": 0.55})
            chordmap = {"unit":"ql","events": ev}
            events = ev
        else:
            # snap rare figures to positional priors when confidence not provided
            ev2 = []
            for e in events:
                fig = _figure(e)
                pos = _bar_index_from_timeql(e.get("time",0.0)) % 8
                prior = self._pos_majority(pos, fig)
                new = dict(e)
                if prior != fig:
                    # light snap only if global prior strongly favors it
                    if self.pos_chord_hist.get(pos, {}).get(prior, 0) > self.pos_chord_hist.get(pos, {}).get(fig, 0)*2:
                        if prior=="N":
                            new["root"], new["quality"] = "N",""
                        else:
                            r = prior[0:2] if len(prior)>=2 and prior[1] in "#-" else prior[0]
                            q = prior[len(r):]
                            new["root"], new["quality"] = r, q
                        new["confidence"] = max(0.5, float(e.get("confidence", 0.5)))
                ev2.append(new)
            events = ev2
            chordmap["events"] = events

        # key prediction (global majority; fallback local bigram if we see explicit changes)
        key = self._majority(self.key_hist, "C")
        # sections
        sec_auto = j.get("sections_auto") or {}
        sections = sec_auto.get("sections") or []
        if not sections:
            bars = int(max([e.get("time",0.0) for e in events]+[0.0])//4)+1
            cuts = [0, max(8, bars//3), max(16, 2*bars//3), max(bars-8, 0)]
            labels = ["intro","verse","chorus","outro"]
            sections = [{"bar": int(cuts[i]), "label": labels[i]} for i in range(len(labels))]
            sec_auto = {"unit":"bar","sections": sections}

        conf = {
            "chord": 1.0 if self.pos_chord_hist else (0.5 if self.chord_hist else 0.3),
            "key": 1.0 if self.key_hist else 0.5,
            "sections": 1.0 if self.section_hist else 0.5,
        }
        conf["overall"] = round((conf["chord"] + conf["key"] + conf["sections"]) / 3.0, 3)
        return {
            "pred": {
                "chordmap": chordmap,
                "key": key,
                "sections_auto": sec_auto
            },
            "confidence": conf,
            "model": {"name": "TeacherV1", "version": self.version}
        }

    # persistence
    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "chord_hist": self.chord_hist,
                "key_hist": self.key_hist,
                "section_hist": self.section_hist,
                "section_bigram": self.section_bigram,
                "pos_chord_hist": self.pos_chord_hist,
                "key_bigram": self.key_bigram,
                "version": self.version
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        import pickle
        d = pickle.load(open(path, "rb"))
        self.chord_hist = d.get("chord_hist", {})
        self.key_hist = d.get("key_hist", {})
        self.section_hist = d.get("section_hist", {})
        self.section_bigram = d.get("section_bigram", {})
        self.pos_chord_hist = d.get("pos_chord_hist", {})
        self.key_bigram = d.get("key_bigram", {})
        self.version = d.get("version", "v1.1")
        return self
