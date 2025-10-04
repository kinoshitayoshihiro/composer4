from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, cast

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None  # type: ignore[assignment]
try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    # ------------------------------------------------------------------
    # Fallback stub so the module can still be imported when streamlit
    # is not available (e.g. CI environments, headless servers).
    # Only the APIs actually referenced in this file are stubbed.
    # ------------------------------------------------------------------
    from functools import lru_cache

    class _StreamlitStub:  # pylint: disable=too-few-public-methods
        """Minimal stub that exposes dummy decorators / functions."""

        @staticmethod
        def cache_data(func=None, **_kwargs):  # type: ignore[no-self-use]
            """Fallback for @st.cache_data.

            - 使い方①: @st.cache_data        ← func が渡ってくる
            - 使い方②: @st.cache_data(ttl=1) ← func は None、kwargs あり
            """

            def _wrap(f):
                return lru_cache(maxsize=None)(f)  # 超簡易メモ化

            # デコレータに引数がある場合は二段階構造に
            return _wrap(func) if callable(func) else _wrap

        # 任意に呼ばれる可能性のある関数を no-op で生やしておく
        def __getattr__(self, _name):  # noqa: D401
            def _dummy(*_a, **_kw):  # pylint: disable=unused-argument
                raise RuntimeError(
                    "streamlit is not installed – "
                    "run `pip install streamlit` to enable GUI mode."
                )

            return _dummy

    st = _StreamlitStub()  # type: ignore[assignment]

from utilities import groove_sampler_ngram, groove_sampler_rnn
from utilities.groove_sampler_ngram import Event
from utilities.streaming_sampler import sd


@st.cache_data  # type: ignore[misc]
def _to_json(events: list[Event]) -> str:
    return json.dumps(events)


def _play_events(events: list[Event]) -> None:
    data = _to_json(events)
    st.components.v1.html(
        f"""
        <script src="https://cdn.jsdelivr.net/npm/tone@14"></script>
        <button onclick="play()">Play</button>
        <script>
        const events = {data};
        async function play() {{
            await Tone.start();
            const synth = new Tone.MembraneSynth().toDestination();
            const start = Tone.now();
            for (const ev of events) {{
                const dur = ev.duration || 0.25;
                synth.triggerAttackRelease('C2', dur, start + ev.offset, ev.velocity/127);
            }}
        }}
        </script>
        """,
        height=80,
    )


def main() -> None:
    st.set_page_config(layout="wide")
    sidebar = st.sidebar
    file = sidebar.file_uploader("Model", type=["pkl", "pt"])
    backend = sidebar.radio("Backend", ["ngram", "rnn"], index=0)
    bars = sidebar.slider("Bars", 1, 32, 4)
    temp = sidebar.slider("Temperature", 0.0, 1.5, 1.0)
    sections = ["intro", "verse", "chorus", "bridge"]
    intensities = ["low", "mid", "high"]
    if file is not None:
        path = Path(file.name)
        path.write_bytes(file.getbuffer())
        meta: dict[str, Any]
        if backend == "rnn" and path.suffix == ".pt":
            _, meta = groove_sampler_rnn.load(path)
        else:
            meta = cast(dict[str, Any], groove_sampler_ngram.load(path))
        combos = cast(dict[Any, tuple[str, str, str]], meta.get("aux_cache", {}))
        if combos:
            sections = sorted({tup[0] for tup in combos.values()})
            intensities = sorted({tup[2] for tup in combos.values()})
    section = sidebar.selectbox("Section", sections)
    intensity = sidebar.selectbox("Intensity", intensities)
    preset_name = sidebar.text_input("Preset name")
    if sidebar.button("Save Preset"):
        st.session_state.setdefault("presets", {})[preset_name] = {
            "backend": backend,
            "bars": bars,
            "temp": temp,
            "section": section,
            "intensity": intensity,
        }
    load = sidebar.selectbox("Load Preset", [""] + list(st.session_state.get("presets", {}).keys()))
    if load:
        pre = st.session_state["presets"][load]
        backend = pre["backend"]
        bars = pre["bars"]
        temp = pre["temp"]
        section = pre["section"]
        intensity = pre["intensity"]
    generate = sidebar.button("Generate")

    if not generate or file is None:
        return

    path = Path(file.name)
    path.write_bytes(file.getbuffer())
    events: list[Event]
    if backend == "rnn" and path.suffix == ".pt":
        model_rnn, meta = groove_sampler_rnn.load(path)
        events = groove_sampler_rnn.sample(
            model_rnn, meta, bars=bars, temperature=temp
        )
    else:
        model_ng = groove_sampler_ngram.load(path)
        events = groove_sampler_ngram.sample(model_ng, bars=bars, temperature=temp,
            cond={"section": section, "intensity": intensity})
    events = list(events)
    xs = [ev["offset"] for ev in events]
    ys = [ev.get("pitch", 36) if isinstance(ev, dict) else 36 for ev in events]
    vs = [ev.get("velocity", 100) for ev in events]
    fig = go.Figure(
        go.Heatmap(x=xs, y=ys, z=vs, colorscale="Viridis")
    )
    st.plotly_chart(fig, use_container_width=True)
    if sd is None:
        _play_events(events)
    else:
        if st.button("Play"):
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "tmp.mid"
                pm = groove_sampler_ngram.events_to_midi(events)
                pm.write(tmp)
                st.audio(tmp.read_bytes(), format="audio/midi")


if __name__ == "__main__":  # pragma: no cover - UI entry
    main()
