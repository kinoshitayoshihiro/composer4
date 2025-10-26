"""
cache_util.py — 短命キャッシュ（TTL・キー生成・フォーマット安全）

「正本＝JSON/YAML/Parquet、DB＝索引、pickleは使わない。キャッシュは任意・短命・再計算可能」
方式に準拠したキャッシュユーティリティ。

使い方:
    from cache_util import CacheStore, CacheConfig
    
    cache = CacheStore(CacheConfig(
        enable=True,
        dir="data/.cache/local_lamda",
        ttl_hours=168
    ))
    
    # キー生成（manifest + code_version + params）
    key = cache.make_key(
        manifest=manifest_dict,
        code_version="script@abc123",
        params={"sr": 48000, "hop_ms": 10}
    )
    
    # 読み込み
    cached, meta = cache.load("chroma", key)
    if cached is not None:
        return cached
    
    # 計算 → 保存
    result = compute_heavy_features(...)
    cache.save("chroma", key, result, ext="npz", meta={"song_id": "xxx"})
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheConfig:
    """キャッシュ設定"""
    enable: bool = False
    dir: str = "data/.cache/local_lamda"
    ttl_hours: int = 168                 # 7 days
    formats: Tuple[str, ...] = ("npz", "parquet")  # 原則 pickle 不使用


class CacheStore:
    """短命キャッシュストア（TTL管理・フォーマット安全）"""
    
    def __init__(self, cfg: CacheConfig):
        self.cfg = cfg
        self.base = Path(cfg.dir)

    # ---- key generation -----------------------------------------------------
    def make_key(self, *, manifest: Dict, code_version: str, params: Dict) -> str:
        """
        再現性のある"中身ハッシュ"。manifest(SoT) + code_version + params を結合。
        
        Args:
            manifest: canonical manifest (roles/segments) の辞書
            code_version: "script@git-hash" 形式
            params: {"sr": 48000, "hop_ms": 10, ...} 等の処理パラメータ
        
        Returns:
            24桁のhexハッシュ（衝突確率は極めて低い）
        """
        payload = {
            "manifest": manifest,               # canonical manifest (roles/segments)
            "code_version": code_version,       # script@hash
            "params": params                    # 閾値やSRなど
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:24]  # 24桁で十分

    # ---- path helpers -------------------------------------------------------
    def _kind_dir(self, kind: str) -> Path:
        """種別ごとのディレクトリ"""
        return self.base / kind

    def path_for(self, kind: str, key: str, ext: str) -> Path:
        """データファイルのパス"""
        assert ext in self.cfg.formats, f"unsupported format: {ext}"
        return self._kind_dir(kind) / f"{key}.{ext}"

    def meta_path_for(self, kind: str, key: str) -> Path:
        """メタファイル（JSON）のパス"""
        return self._kind_dir(kind) / f"{key}.meta.json"

    # ---- freshness & purge --------------------------------------------------
    def is_fresh(self, path: Path) -> bool:
        """TTL内か判定"""
        if not path.exists():
            return False
        age_h = (time.time() - path.stat().st_mtime) / 3600.0
        return age_h <= self.cfg.ttl_hours

    def purge_expired(self, kind: Optional[str] = None) -> int:
        """
        TTL切れのファイル(.npz/.parquet/.meta.json)を削除。削除数を返す。
        
        Args:
            kind: 指定すると特定種別のみ削除、Noneなら全種別
        
        Returns:
            削除したファイル数
        """
        if not self.base.exists():
            return 0
        
        exts = set(self.cfg.formats) | {"meta.json"}
        n = 0
        dirs = [self._kind_dir(kind)] if kind else [p for p in self.base.iterdir() if p.is_dir()]
        
        for d in dirs:
            if not d.exists():
                continue
            for p in d.iterdir():
                if p.suffix.lstrip(".") in exts or p.name.endswith(".meta.json"):
                    if not self.is_fresh(p):
                        p.unlink(missing_ok=True)
                        n += 1
        return n

    # ---- IO (npz/parquet/json meta) ----------------------------------------
    def save(self, kind: str, key: str, obj: Any, *, ext: str, meta: Optional[Dict] = None) -> Path:
        """
        objの型に応じて書く。DataFrame→parquet, dict/arrays→npz。
        
        Args:
            kind: 種別（"chroma", "onset", "cqt" 等）
            key: キャッシュキー（make_key で生成）
            obj: 保存するオブジェクト（DataFrame or ndarray or dict[str, ndarray]）
            ext: "parquet" or "npz"
            meta: サイドカーに書くメタ情報（song_id, sr, etc）
        
        Returns:
            保存したファイルのパス
        """
        path = self.path_for(kind, key, ext)
        path.parent.mkdir(parents=True, exist_ok=True)

        if ext == "parquet":
            import pandas as pd
            assert hasattr(obj, "to_parquet"), "parquet save requires pandas DataFrame-like"
            obj.to_parquet(path, index=False)
        elif ext == "npz":
            import numpy as np
            # obj: dict[str, np.ndarray] or np.ndarray
            if isinstance(obj, dict):
                np.savez_compressed(path, **obj)
            else:
                np.savez_compressed(path, arr=obj)
        else:
            raise ValueError(f"unsupported ext {ext}")

        # side meta
        mp = self.meta_path_for(kind, key)
        meta = meta or {}
        mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def load(self, kind: str, key: str) -> Tuple[Optional[Any], Optional[Dict]]:
        """
        extの優先順で探す→TTLチェック→ロード。obj, meta を返す。ない/失効なら(None, None)。
        
        Args:
            kind: 種別
            key: キャッシュキー
        
        Returns:
            (obj, meta): オブジェクトとメタ情報のタプル。キャッシュミス時は (None, None)
        """
        for ext in self.cfg.formats:
            p = self.path_for(kind, key, ext)
            if p.exists() and self.is_fresh(p):
                mp = self.meta_path_for(kind, key)
                meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else None
                
                if ext == "parquet":
                    import pandas as pd
                    return pd.read_parquet(p), meta
                elif ext == "npz":
                    import numpy as np
                    data = np.load(p, allow_pickle=False)
                    # 単一配列は 'arr'、辞書保存は keys()
                    if "arr" in data.files and len(data.files) == 1:
                        return data["arr"], meta
                    return {k: data[k] for k in data.files}, meta
        
        return None, None


# ========== 使用例 ==========
if __name__ == "__main__":
    # 初期化
    cache = CacheStore(CacheConfig(
        enable=True,
        dir="data/.cache/local_lamda",
        ttl_hours=168
    ))
    
    # サンプルmanifest（セグメント情報）
    manifest = {
        "version": "ok-audio-1.0",
        "song_id": "test-song-001",
        "role": "guitar",
        "segments": [
            {"relpath": "guitar/seg_0.wav", "size": 12345, "sha256": "abc...", "start_sec": 0.0}
        ]
    }
    
    # キー生成
    key = cache.make_key(
        manifest=manifest,
        code_version="chroma_extractor@v1.2.3",
        params={"sr": 48000, "hop_ms": 10}
    )
    print(f"Cache key: {key}")
    
    # 読み込み試行
    cached, meta = cache.load("chroma", key)
    if cached is not None:
        print("Cache hit!")
        print(f"Meta: {meta}")
    else:
        print("Cache miss - computing...")
        import numpy as np
        # ダミー計算
        chroma_dict = {
            "C": np.random.rand(12, 100),
            "G": np.random.rand(12, 100)
        }
        # 保存
        cache.save("chroma", key, chroma_dict, ext="npz", meta={
            "song_id": "test-song-001",
            "role": "guitar",
            "sr": 48000
        })
        print("Cached!")
    
    # 期限切れ削除（テスト）
    deleted = cache.purge_expired()
    print(f"Deleted {deleted} expired files")
