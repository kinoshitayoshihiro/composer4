"""
LOCAL LAMDA生成システム
======================

公式LAMDAを触らず、ステムMIDIから**LAMDA互換データ**を自作生成。
既存Stage2/Stage3に非破壊で合流させる。

生成ファイル:
- LOCAL_KILO_CHORDS_DATA.pickle (bar単位コード進行)
- LOCAL_META_DATA_*.pickle (notes/CC/PB/tempo/patches/統計)
- LOCAL_SIGNATURES_DATA.pickle (拍子情報)
- LOCAL_TOTALS.pickle (pitch/dur/velヒストグラム、外れ値検出用)

ID形式: LOCAL:<sha1(path+size+bars+bpm0)[:20]>
- 公式file_idと衝突回避
- 4資源間でjoin可能
"""
