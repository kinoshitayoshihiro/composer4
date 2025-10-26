"""
db_registry.py — DB索引スキーマ（artifacts テーブル）

「正本＝JSON/YAML/Parquet、DB＝索引、pickleは使わない」
方式に準拠したレジストリ管理。

成果物（beat_grid.json, bars.parquet, etc）のパスとIDのみをDBに登録。
実体はファイルシステム上のJSON/YAML/Parquetに保存。

使い方:
    from db_registry import init_registry, register_artifact, query_artifacts
    
    # DB初期化
    init_registry("data/local_lamda_registry.db")
    
    # 成果物登録
    register_artifact(
        db_path="data/local_lamda_registry.db",
        song_id="9653a690-c28c-4e8f-962e-ff7ed18b8ee9",
        run_id="local-2025-10-25T12:34:56",
        kind="beat_grid",
        path="data/local_lamda_wav_features/moisesdb/9653a690.../beat_grid.json",
        file_id="2ead80e890c4"
    )
    
    # 検索
    results = query_artifacts(
        db_path="data/local_lamda_registry.db",
        song_id="9653a690-c28c-4e8f-962e-ff7ed18b8ee9",
        kind="beat_grid"
    )
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any


# ========== スキーマ定義 ==========
ARTIFACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
  song_id TEXT NOT NULL,
  run_id  TEXT NOT NULL,
  kind    TEXT NOT NULL,         -- 'beat_grid'|'accent_grid'|'audio_chordmap'|'bars_parquet'|'vocal_features'|'mix_diagnostics'|'manifest'|'stage1_clean_mid'|'stage1_clean_json'|...
  path    TEXT NOT NULL,         -- ファイルへの相対/絶対パス
  file_id TEXT,                  -- sha256(canonical_manifest)[:12] (WAV) or content_id (MIDI)
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (song_id, run_id, kind)
);

CREATE INDEX IF NOT EXISTS idx_artifacts_song_id ON artifacts(song_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(kind);
CREATE INDEX IF NOT EXISTS idx_artifacts_file_id ON artifacts(file_id);
"""


def init_registry(db_path: str, wal_mode: bool = True) -> None:
    """
    レジストリDBを初期化。テーブル・インデックス作成。
    
    Args:
        db_path: SQLiteファイルのパス
        wal_mode: WALモード有効化（推奨）
    """
    db_path_obj = Path(db_path)
    db_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    con = sqlite3.connect(db_path)
    try:
        if wal_mode:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
        
        # テーブル作成
        con.executescript(ARTIFACTS_SCHEMA)
        con.commit()
        print(f"✓ Registry initialized: {db_path}")
    finally:
        con.close()


def register_artifact(
    db_path: str,
    song_id: str,
    run_id: str,
    kind: str,
    path: str,
    file_id: Optional[str] = None
) -> None:
    """
    成果物をレジストリに登録（INSERT OR REPLACE）。
    
    Args:
        db_path: SQLiteファイルのパス
        song_id: 曲ID
        run_id: 実行ID（local-2025-10-25T12:34:56 等）
        kind: 成果物種別（beat_grid, accent_grid, bars_parquet, etc）
        path: ファイルパス（相対または絶対）
        file_id: file_id（WAV系）または content_id（MIDI系）
    """
    con = sqlite3.connect(db_path)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute(
            """INSERT OR REPLACE INTO artifacts(song_id, run_id, kind, path, file_id)
               VALUES(?, ?, ?, ?, ?)""",
            (song_id, run_id, kind, path, file_id)
        )
        con.commit()
    finally:
        con.close()


def query_artifacts(
    db_path: str,
    song_id: Optional[str] = None,
    run_id: Optional[str] = None,
    kind: Optional[str] = None,
    file_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    成果物を検索。
    
    Args:
        db_path: SQLiteファイルのパス
        song_id: 曲ID（部分一致）
        run_id: 実行ID（部分一致）
        kind: 成果物種別（完全一致）
        file_id: file_id/content_id（完全一致）
    
    Returns:
        マッチした行のリスト（辞書形式）
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    
    try:
        query = "SELECT * FROM artifacts WHERE 1=1"
        params = []
        
        if song_id:
            query += " AND song_id LIKE ?"
            params.append(f"%{song_id}%")
        if run_id:
            query += " AND run_id LIKE ?"
            params.append(f"%{run_id}%")
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        if file_id:
            query += " AND file_id = ?"
            params.append(file_id)
        
        query += " ORDER BY created_at DESC"
        
        cursor = con.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        con.close()


def delete_artifacts(
    db_path: str,
    song_id: Optional[str] = None,
    run_id: Optional[str] = None
) -> int:
    """
    成果物を削除（物理ファイルは削除しない）。
    
    Args:
        db_path: SQLiteファイルのパス
        song_id: 削除対象の曲ID
        run_id: 削除対象の実行ID
    
    Returns:
        削除した行数
    """
    con = sqlite3.connect(db_path)
    try:
        query = "DELETE FROM artifacts WHERE 1=1"
        params = []
        
        if song_id:
            query += " AND song_id = ?"
            params.append(song_id)
        if run_id:
            query += " AND run_id = ?"
            params.append(run_id)
        
        cursor = con.execute(query, params)
        con.commit()
        return cursor.rowcount
    finally:
        con.close()


# ========== CLI サンプル ==========
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python db_registry.py <command> [args...]")
        print("Commands:")
        print("  init <db_path>")
        print("  register <db_path> <song_id> <run_id> <kind> <path> [file_id]")
        print("  query <db_path> [--song-id=X] [--kind=X]")
        print("  delete <db_path> [--song-id=X] [--run-id=X]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "init":
        db_path = sys.argv[2]
        init_registry(db_path)
    
    elif cmd == "register":
        db_path = sys.argv[2]
        song_id = sys.argv[3]
        run_id = sys.argv[4]
        kind = sys.argv[5]
        path = sys.argv[6]
        file_id = sys.argv[7] if len(sys.argv) > 7 else None
        register_artifact(db_path, song_id, run_id, kind, path, file_id)
        print(f"✓ Registered: {kind} for {song_id}")
    
    elif cmd == "query":
        db_path = sys.argv[2]
        kwargs = {}
        for arg in sys.argv[3:]:
            if arg.startswith("--song-id="):
                kwargs["song_id"] = arg.split("=", 1)[1]
            elif arg.startswith("--kind="):
                kwargs["kind"] = arg.split("=", 1)[1]
            elif arg.startswith("--run-id="):
                kwargs["run_id"] = arg.split("=", 1)[1]
        
        results = query_artifacts(db_path, **kwargs)
        print(f"Found {len(results)} artifacts:")
        for r in results:
            print(f"  {r['song_id']} | {r['kind']} | {r['path']}")
    
    elif cmd == "delete":
        db_path = sys.argv[2]
        kwargs = {}
        for arg in sys.argv[3:]:
            if arg.startswith("--song-id="):
                kwargs["song_id"] = arg.split("=", 1)[1]
            elif arg.startswith("--run-id="):
                kwargs["run_id"] = arg.split("=", 1)[1]
        
        n = delete_artifacts(db_path, **kwargs)
        print(f"✓ Deleted {n} artifacts")
    
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
