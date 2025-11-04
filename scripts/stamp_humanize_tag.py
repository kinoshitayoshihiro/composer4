#!/usr/bin/env python3
"""
[レビュー提案5] Humanize再現性タグ生成

YAML humanizeセクションのハッシュを計算し、MIDIメタデータに焼き込む
→ 音源差分の追跡が容易になる
"""

import hashlib
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List


def extract_humanize_sections(config: Dict[str, Any], sections: List[str]) -> Dict[str, Any]:
    """指定されたセクションのみを抽出"""
    extracted = {}
    for section in sections:
        if section in config:
            extracted[section] = config[section]
    return extracted


def compute_humanize_hash(config_path: Path, sections: List[str] = None) -> str:
    """humanizeセクションのハッシュを計算

    Args:
        config_path: plan_humanize.yaml のパス
        sections: 対象セクション（デフォルト: bar_features, performance, roles）

    Returns:
        8文字の短縮ハッシュ（例: "abc12345"）
    """
    if sections is None:
        sections = ["bar_features", "performance", "roles"]

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 対象セクションのみ抽出
    humanize_config = extract_humanize_sections(config, sections)

    # JSON化（キー順序を固定して再現性確保）
    json_str = json.dumps(humanize_config, sort_keys=True, ensure_ascii=False)

    # SHA256ハッシュ → 8文字短縮
    hash_full = hashlib.sha256(json_str.encode("utf-8")).hexdigest()
    hash_short = hash_full[:8]

    return hash_short


def generate_humanize_tag(config_path: Path, version: str = "v2") -> str:
    """人間味タグを生成

    Returns:
        例: "humanize_v2_abc12345"
    """
    hash_short = compute_humanize_hash(config_path)
    return f"humanize_{version}_{hash_short}"


def embed_tag_in_midi_meta(midi_path: Path, tag: str, track_name_suffix: bool = True):
    """MIDIメタデータにタグを埋め込む

    Args:
        midi_path: 出力MIDIファイル
        tag: タグ文字列（例: "humanize_v2_abc12345"）
        track_name_suffix: Trueならトラック名の末尾に追記
    """
    import mido

    mid = mido.MidiFile(midi_path)

    # メタイベントとして追加
    meta_msg = mido.MetaMessage("text", text=tag, time=0)
    if mid.tracks:
        mid.tracks[0].insert(0, meta_msg)

    # トラック名にも追記（オプション）
    if track_name_suffix:
        for track in mid.tracks:
            for msg in track:
                if msg.type == "track_name":
                    msg.name = f"{msg.name} [{tag}]"
                    break

    mid.save(midi_path)
    print(f"✅ タグ埋め込み完了: {tag} → {midi_path}")


def main():
    """使用例"""
    import argparse

    parser = argparse.ArgumentParser(description="Humanize再現性タグ生成")
    parser.add_argument("--config", required=True, help="plan_humanize.yaml のパス")
    parser.add_argument("--midi", help="タグ埋め込み先MIDI（オプション）")
    parser.add_argument("--version", default="v2", help="バージョン文字列（デフォルト: v2）")

    args = parser.parse_args()

    config_path = Path(args.config)
    tag = generate_humanize_tag(config_path, args.version)

    print(f"生成タグ: {tag}")
    print(f"ハッシュ元: bar_features, performance, roles")

    if args.midi:
        embed_tag_in_midi_meta(Path(args.midi), tag)


if __name__ == "__main__":
    main()
