"""pickle互換ユーティリティ

古いpickleが異なるモジュール名でクラスを保存している場合に備え、
モジュール名をリマップして安全にアンピクルする小さなラッパを提供します。

使い方:
    from utilities.pickle_compat import load, loads
    with open(path, 'rb') as f:
        obj = load(f)

また、簡易のresave_pickle関数を提供し、既存のpickleをcanonicalなモジュール名で再保存するために使えます。
"""
from __future__ import annotations

import pickle
from io import BytesIO
from typing import BinaryIO


class RenamingUnpickler(pickle.Unpickler):
    """モジュール名を別名にリマップしてアンピクルするUnpickler。

    例: '__main__' に保存されたクラスを 'scripts.extract_stage2_patterns' にマップする。
    """

    def __init__(self, file, rename_map: dict[str, str] | None = None):
        super().__init__(file)
        # legacy module name -> current module name
        self.rename_map = rename_map or {"__main__": "extract_stage2_patterns"}

    def find_class(self, module, name):
        # モジュール名がrename_mapにある場合は置換する
        if module in self.rename_map:
            module = self.rename_map[module]
        return super().find_class(module, name)


def load(file: BinaryIO, rename_map: dict[str, str] | None = None):
    """ファイルオブジェクトから安全に読み込む（rename_mapを適用）。"""
    unpickler = RenamingUnpickler(file, rename_map=rename_map)
    return unpickler.load()


def loads(data: bytes, rename_map: dict[str, str] | None = None):
    """bytesから安全に読み込む（rename_mapを適用）。"""
    bio = BytesIO(data)
    return load(bio, rename_map=rename_map)


def resave_pickle(path_in: str | bytes | BinaryIO, path_out: str | None = None, *, rename_map: dict[str, str] | None = None):
    """既存pickleを読み込み、canonicalなモジュール名で再保存するユーティリティ。

    path_in: 入力ファイルパスまたはbytes/ファイルオブジェクト
    path_out: 出力ファイルパス（Noneの場合は上書き）
    rename_map: 読み込み時のモジュールリマップ。読み込み後は通常のpickle.dumpで再保存する。
    """
    close_after = False
    if hasattr(path_in, 'read'):
        f_in = path_in
    else:
        f_in = open(path_in, 'rb')
        close_after = True

    try:
        obj = load(f_in, rename_map=rename_map)
    finally:
        if close_after:
            f_in.close()

    out_path = path_out or path_in
    with open(out_path, 'wb') as f_out:
        pickle.dump(obj, f_out, protocol=pickle.HIGHEST_PROTOCOL)


__all__ = ["RenamingUnpickler", "load", "loads", "resave_pickle"]
