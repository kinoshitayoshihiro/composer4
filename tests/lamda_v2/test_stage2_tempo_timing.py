from __future__ import annotations
import math
import pretty_midi as pm

from scripts.lamda_v2.tempo_timing import (
    build_beat_grid,
    sec_to_ql,
    ql_to_sec,
    merge_min_dwell,
    snap_times_to_grid,
)


def _mk_pm_constant(bpm: float = 120.0) -> pm.PrettyMIDI:
    m = pm.PrettyMIDI(initial_tempo=bpm)
    # default 4/4 timesig
    ts = pm.containers.TimeSignature(4, 4, 0.0)
    m.time_signature_changes.append(ts)
    # Add a dummy note to ensure downbeats are generated
    inst = pm.Instrument(program=0)
    note = pm.Note(velocity=64, pitch=60, start=0.0, end=1.0)
    inst.notes.append(note)
    m.instruments.append(inst)
    return m


def _mk_pm_two_tempi(t0_bpm=120.0, t1_time=4.0, t1_bpm=60.0) -> pm.PrettyMIDI:
    m = pm.PrettyMIDI(initial_tempo=t0_bpm)
    # Add tempo change at t1_time
    # Note: PrettyMIDI stores tempo changes internally, we'll create via attributes
    # For testing, we'll use a workaround: create notes and rely on get_tempo_changes
    # Actually, let's use the fact that PM constructor accepts initial_tempo
    # but we need to inject a second tempo change. Let's try a different approach:
    # Create a minimal MIDI with tempo changes
    import mido

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    # Set initial tempo
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t0_bpm), time=0))
    # Add second tempo change at t1_time (convert to ticks)
    ticks_per_beat = mid.ticks_per_beat
    t1_ticks = int(t1_time * ticks_per_beat * 2)  # rough estimate
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(t1_bpm), time=t1_ticks))
    # End of track
    track.append(mido.MetaMessage("end_of_track", time=0))

    # Save to temporary file and load with pretty_midi
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        mid.save(f.name)
        m = pm.PrettyMIDI(f.name)

    return m


def test_sec_ql_roundtrip_constant_tempo():
    midi = _mk_pm_constant(120.0)
    grid = build_beat_grid(midi)
    tmap = grid["tempo_map"]
    # 120 BPM => 1 sec = 8 QL
    for t in [0.0, 0.25, 0.5, 1.0, 2.0, 3.5]:
        ql = sec_to_ql(t, tmap)
        assert math.isclose(ql, t * 8.0, rel_tol=1e-9, abs_tol=1e-9)
        rt = ql_to_sec(ql, tmap)
        assert math.isclose(rt, t, rel_tol=1e-9, abs_tol=1e-9)


def test_build_beat_grid_downbeats_and_ql():
    midi = _mk_pm_constant(90.0)
    grid = build_beat_grid(midi)
    assert grid["tempo_map"][0][1] == 90.0
    # downbeats should start at 0.0 and be strictly increasing
    dbs = grid["downbeats_sec"]
    assert len(dbs) >= 1 and dbs[0] == 0.0
    assert all(b2 > b1 for b1, b2 in zip(dbs, dbs[1:]))
    # 90 BPM => 1 sec = 6 QL
    ql0 = grid["downbeats_ql"][0]
    assert math.isclose(ql0, 0.0, abs_tol=1e-9)


def test_merge_min_dwell_and_dedup():
    events = [
        {"time": 0.0, "root": "C", "quality": "maj"},
        {"time": 2.0, "root": "C", "quality": "maj"},
        {"time": 4.0, "root": "A", "quality": "min"},
        {"time": 6.0, "root": "A", "quality": "min"},
    ]
    merged = merge_min_dwell(events, min_ql=2.0)
    # consecutive duplicates are removed: expect C at 0, A at 4
    assert [round(e["time"]) for e in merged] == [0, 4]
    assert (merged[0]["root"], merged[0]["quality"]) == ("C", "maj")
    assert (merged[1]["root"], merged[1]["quality"]) == ("A", "min")


def test_snap_times_to_grid():
    grid_ql = [0.0, 4.0, 8.0, 12.0]
    src = [0.1, 3.9, 7.7, 8.4]
    out = snap_times_to_grid(src, grid_ql)
    assert out == [0.0, 4.0, 8.0, 8.0]


def test_tempo_change_map_and_roundtrip():
    midi = _mk_pm_two_tempi(120.0, 4.0, 60.0)
    grid = build_beat_grid(midi)
    tmap = grid["tempo_map"]
    # two segments
    assert len(tmap) == 2
    # roundtrip a time before and after the change
    for t in [2.0, 6.0]:
        ql = sec_to_ql(t, tmap)
        rt = ql_to_sec(ql, tmap)
        assert math.isclose(rt, t, rel_tol=1e-9, abs_tol=1e-9)
