# Copilot Chat カスタム指示設定

## 設定方法

VS Code設定 → `GitHub Copilot Chat: Custom Instructions` に以下を貼り付け。

## カスタム指示内容

```
常に以下を守る:

1. 変更提案フォーマット
   - まず "Plan / Risks / Options" を出し、即コード提出しない
   - Plan: 何を変えるか、なぜ必要か
   - Risks: 既存機能への影響、後方互換性
   - Options: A/B案を比較、推奨案を明示
   - Diff: 実際の差分コード（最小）
   - Tests: CI/手動検証の具体的コマンド

2. E2E Playbook遵守
   - docs/E2E_PLAYBOOK.md の禁止事項を厳守
   - 直MIDI書き出し不可（midi_writer.py必須）
   - json2midi.py は使用禁止（旧スクリプト）
   - provenance必須（meta.provenance に AI適用情報を刻む）

3. 最小差分原則
   - 変更は最小差分
   - E2E本線に合流させる
   - NO-OP時は理由を明記

4. タスク実行
   - コマンド実行は .vscode/tasks.json のタスク名を使う
   - Run: E2E (strict)
   - Verify: CI (strict only)
   - Export: MIDI from Plan (補助動線)

5. Provenance必須フィールド
   - meta.provenance.writer: "midi_writer.py"
   - meta.provenance.bass_f0: {enabled, file, bars}
   - meta.provenance.oaf_piano: {enabled, file, notes}
   - meta.provenance.emotion_ai: {enabled, profile}
   - meta.provenance.harmony_ai: {enabled, usage_db}
   - meta.provenance.magenta: {enabled, engine}

6. CREPE/OaF検証
   - CREPE frames数確認: bass_f0.meta.json の frames（47,997等）
   - OaF適用確認: piano_plan.json の context_sources.oaf_piano = true
   - 解析のみで終わらせない

7. CI検証必須
   - すべての変更後に ci_verify_music_package.py --strict 実行
   - 11/11 PASS が必須
```

## Continue拡張機能との連携

`.continue/config.json` で以下のスラッシュコマンドが利用可能：

- `/deep` - 深掘りレビュー（Plan→Risks→Options→Diff→Tests）
- `/strict-e2e` - E2E本線での安全実行（疑似思考モード）

## 効果

- ✅ 安易な解決に飛びつかない
- ✅ 必ず代替案を比較
- ✅ 最小差分で後方互換性を保つ
- ✅ CI検証で品質を担保

---

**Phase 121**: 2025年11月8日
