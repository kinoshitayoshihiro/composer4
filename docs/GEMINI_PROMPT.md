# Gemini用プロンプト: Google Colabでの機械学習トレーニングセットアップ

## 背景
私はPythonで音楽生成AIの開発をしており、5つの楽器（ギター、ベース、ピアノ、ストリングス、ドラム）のDUV（Duration, Velocity）モデルをトレーニングする必要があります。

MacBook AirのMPSでは処理が重すぎるため、Google Colabでトレーニングを実行したいと思います。

## 現在の状況
- プロジェクトフォルダ: `composer2-3`
- Google DriveのOthercomputers経由でアクセス可能
- パス: `/content/drive/Othercomputers/マイ MacBook Air/composer2-3`
- トレーニングデータ: `data/phrase_csv/` に5つの楽器の `*_train_raw.csv` と `*_val_raw.csv`
- メインスクリプト: `scripts/train_phrase.py`

## 必要な作業
1. Google Driveをマウントしてプロジェクトディレクトリにアクセス
2. 必要な依存関係のインストール (`requirements.txt`)
3. GPU（CUDA）の確認
4. トレーニングスクリプトの実行（5つの楽器を順番に15エポックずつ）

## トレーニング設定
- アーキテクチャ: Transformer (d_model=512, nhead=8, layers=4)
- バッチサイズ: 128 (ColabのGPU用に最適化)
- モード: regression only (`duv-mode=reg`)
- デバイス: `cuda`
- エポック数: 15
- 学習率: 1e-4

## 期待する出力
- 各楽器ごとに `checkpoints/*_duv_raw.best.ckpt` (約99MB)
- 正常なトレーニングログ: `train_batches/epoch > 10,000` かつ `vel_mae` と `dur_mae` が減少

## 質問
Google Colabでこのプロジェクトを実行するための完全なコードを段階的に教えてください。特に以下の点について：

1. Google Driveマウントとディレクトリナビゲーション
2. パスにスペースと日本語が含まれる問題の対処法
3. 依存関係のインストール
4. 5つの楽器のトレーニングを効率的に実行する方法
5. トレーニング進捗の監視方法
6. 生成されたチェックポイントファイルのダウンロード方法

可能であれば、コピー&ペーストですぐに使えるコードセルを提供してください。