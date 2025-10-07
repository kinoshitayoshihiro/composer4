#!/usr/bin/env python3
"""
Suno Project Importer
=====================

Sunoエクスポートフォルダから楽曲データをインポート

Expected folder structure:
--------------------------
data/suno_samples/song_001/
├── complete.wav              # 完成曲（参考用）
├── vocals.wav                # Vocalステム
├── vocals.mid                # Vocal MIDI
├── guitar.wav                # Guitarステム
├── guitar.mid                # Guitar MIDI
├── bass.wav                  # Bassステム
├── bass.mid                  # Bass MIDI
├── drums.wav                 # Drumsステム
├── drums.mid                 # Drums MIDI
├── ... (他の楽器)
└── metadata.json             # メタデータ

metadata.json format:
--------------------
{
    "title": "遠い日の記憶",
    "lyrics": "遠い日の記憶が\\n心に降る雨のように...",
    "emotion": {
        "valence": -0.4,
        "arousal": 0.3,
        "intensity": 0.6,
        "mood": "melancholic"
    },
    "tempo": 80,
    "key": "Am",
    "time_signature": "4/4"
}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import wave


@dataclass
class StemData:
    """楽器ステムデータ"""

    name: str
    wav_path: Optional[Path] = None
    midi_path: Optional[Path] = None

    def exists(self) -> bool:
        """ファイルが存在するか"""
        return (self.wav_path and self.wav_path.exists()) or (
            self.midi_path and self.midi_path.exists()
        )

    def has_wav(self) -> bool:
        return self.wav_path and self.wav_path.exists()

    def has_midi(self) -> bool:
        return self.midi_path and self.midi_path.exists()


@dataclass
class SunoProject:
    """Sunoプロジェクト"""

    project_dir: Path
    title: str
    lyrics: Optional[str] = None
    emotion: Optional[Dict] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    time_signature: str = "4/4"

    # ステム
    vocals: Optional[StemData] = None
    instruments: Dict[str, StemData] = field(default_factory=dict)
    complete_wav: Optional[Path] = None

    def get_all_stems(self) -> List[StemData]:
        """全ステムリスト取得"""
        stems = []
        if self.vocals and self.vocals.exists():
            stems.append(self.vocals)
        stems.extend([s for s in self.instruments.values() if s.exists()])
        return stems

    def get_instrument_names(self) -> List[str]:
        """楽器名リスト"""
        return list(self.instruments.keys())


class SunoProjectImporter:
    """Sunoプロジェクトインポーター"""

    # 標準的な楽器名
    STANDARD_INSTRUMENTS = [
        "vocals",  # メインボーカル
        "backing_vocals",  # バッキングボーカル
        "guitar",
        "bass",
        "drums",
        "piano",
        "keyboard",
        "synth",
        "strings",
        "brass",
        "percussion",
        "fx",
        "ambient",
    ]

    def __init__(self):
        self.project = None

    def import_project(self, project_dir: Path) -> SunoProject:
        """
        プロジェクトフォルダをインポート

        Args:
            project_dir: Sunoエクスポートフォルダ

        Returns:
            SunoProject
        """
        if not project_dir.exists():
            raise FileNotFoundError(f"Project directory not found: {project_dir}")

        print(f"📂 Importing Suno project: {project_dir.name}")

        # メタデータ読み込み
        metadata = self._load_metadata(project_dir)

        # プロジェクト初期化
        project = SunoProject(
            project_dir=project_dir,
            title=metadata.get("title", project_dir.name),
            lyrics=metadata.get("lyrics"),
            emotion=metadata.get("emotion"),
            tempo=metadata.get("tempo"),
            key=metadata.get("key"),
            time_signature=metadata.get("time_signature", "4/4"),
        )

        # 完成曲WAV (complete.wav または full.wav)
        complete_path = project_dir / "complete.wav"
        if not complete_path.exists():
            complete_path = project_dir / "full.wav"

        if complete_path.exists():
            project.complete_wav = complete_path
            print(f"  ✅ Complete WAV: {complete_path.name}")

        # Vocalステム
        project.vocals = self._load_stem(project_dir, "vocals")
        if project.vocals and project.vocals.exists():
            print(f"  🎤 Vocals: WAV={project.vocals.has_wav()} MIDI={project.vocals.has_midi()}")

        # 楽器ステム
        for instrument in self.STANDARD_INSTRUMENTS:
            stem = self._load_stem(project_dir, instrument)
            if stem and stem.exists():
                project.instruments[instrument] = stem
                print(
                    f"  🎸 {instrument.capitalize()}: WAV={stem.has_wav()} MIDI={stem.has_midi()}"
                )

        # その他のファイル検出
        self._detect_additional_files(project_dir, project)

        self.project = project

        print(f"\n✅ Import complete!")
        print(f"   Title: {project.title}")
        print(f"   Tempo: {project.tempo} BPM")
        print(f"   Key: {project.key}")
        print(f"   Vocals: {'Yes' if project.vocals else 'No'}")
        print(f"   Instruments: {len(project.instruments)}")

        return project

    def _load_metadata(self, project_dir: Path) -> Dict:
        """メタデータ読み込み"""
        metadata_path = project_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        else:
            print("  ⚠️  metadata.json not found, using defaults")
            return {}

    def _load_stem(self, project_dir: Path, stem_name: str) -> Optional[StemData]:
        """ステムデータ読み込み（フラット構造 or ディレクトリ構造対応）"""
        # パターン1: フラット構造 (vocals.wav, guitar.wav)
        wav_path = project_dir / f"{stem_name}.wav"
        midi_path = project_dir / f"{stem_name}.mid"

        # パターン2: ディレクトリ構造 (stemswav_001/Vocals.wav)
        # または (stemswav_001/★★曲名★★ (Vocals).wav)
        if not wav_path.exists():
            for stems_dir in project_dir.glob("stems*wav*"):
                # 単純な名前でチェック (Vocals.wav)
                candidate = stems_dir / f"{stem_name.capitalize()}.wav"
                if candidate.exists():
                    wav_path = candidate
                    break

                # スペースを含む名前 (Backing Vocals.wav)
                friendly = stem_name.replace("_", " ").title()
                candidate = stems_dir / f"{friendly}.wav"
                if candidate.exists():
                    wav_path = candidate
                    break

                # 曲名プレフィックス付き (★★曲名★★ (Vocals).wav)
                for wav_file in stems_dir.glob("*wav"):
                    filename = wav_file.name
                    # (Vocals).wav または (Backing Vocals).wav
                    if f"({stem_name.capitalize()})" in filename:
                        wav_path = wav_file
                        break
                    if f"({friendly})" in filename:
                        wav_path = wav_file
                        break
                if wav_path.exists():
                    break

        if not midi_path.exists():
            for stems_dir in project_dir.glob("stem*midi*"):
                candidate = stems_dir / f"{stem_name.capitalize()}.mid"
                if candidate.exists():
                    midi_path = candidate
                    break

                friendly = stem_name.replace("_", " ").title()
                candidate = stems_dir / f"{friendly}.mid"
                if candidate.exists():
                    midi_path = candidate
                    break

                # 曲名プレフィックス付き
                for mid_file in stems_dir.glob("*.mid"):
                    filename = mid_file.name
                    if f"({stem_name.capitalize()})" in filename:
                        midi_path = mid_file
                        break
                    if f"({friendly})" in filename:
                        midi_path = mid_file
                        break
                if midi_path.exists():
                    break

        if wav_path.exists() or midi_path.exists():
            return StemData(
                name=stem_name,
                wav_path=wav_path if wav_path.exists() else None,
                midi_path=midi_path if midi_path.exists() else None,
            )

        return None

    def _detect_additional_files(self, project_dir: Path, project: SunoProject):
        """追加ファイル検出（標準以外の楽器）"""
        for wav_file in project_dir.glob("*.wav"):
            stem_name = wav_file.stem

            # すでに処理済みならスキップ
            if stem_name in ["complete", "vocals"] or stem_name in project.instruments:
                continue

            # 追加楽器として登録
            stem = self._load_stem(project_dir, stem_name)
            if stem and stem.exists():
                project.instruments[stem_name] = stem
                print(
                    f"  🎹 {stem_name.capitalize()} (additional): WAV={stem.has_wav()} MIDI={stem.has_midi()}"
                )

    def export_summary(self, output_path: Path):
        """プロジェクトサマリーをJSON出力"""
        if not self.project:
            raise ValueError("No project loaded. Run import_project() first.")

        summary = {
            "title": self.project.title,
            "lyrics": self.project.lyrics,
            "emotion": self.project.emotion,
            "tempo": self.project.tempo,
            "key": self.project.key,
            "time_signature": self.project.time_signature,
            "vocals": {
                "wav": (
                    str(self.project.vocals.wav_path)
                    if self.project.vocals and self.project.vocals.has_wav()
                    else None
                ),
                "midi": (
                    str(self.project.vocals.midi_path)
                    if self.project.vocals and self.project.vocals.has_midi()
                    else None
                ),
            },
            "instruments": {
                name: {
                    "wav": str(stem.wav_path) if stem.has_wav() else None,
                    "midi": str(stem.midi_path) if stem.has_midi() else None,
                }
                for name, stem in self.project.instruments.items()
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Summary exported: {output_path}")


# ============================================================================
# コマンドラインインターフェース
# ============================================================================


def main():
    """デモ実行"""
    import argparse

    parser = argparse.ArgumentParser(description="Import Suno project from export folder")
    parser.add_argument("project_dir", type=Path, help="Path to Suno export folder")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/suno_project_summary.json"),
        help="Output summary JSON path",
    )

    args = parser.parse_args()

    # インポート実行
    importer = SunoProjectImporter()
    project = importer.import_project(args.project_dir)

    # サマリー出力
    args.output.parent.mkdir(parents=True, exist_ok=True)
    importer.export_summary(args.output)

    print("\n" + "=" * 70)
    print("🎉 Suno project imported successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
