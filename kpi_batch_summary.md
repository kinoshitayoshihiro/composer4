# KPI Gate集計レポート

**生成日時**: 2025-10-30 20:42:02

**集計対象**: 1 曲

---

## 📊 全体統計

| 指標 | 値 |
|-----|-----|
| **Total bars** | 150 |
| **Pass bars** | 149 (99.3%) |
| **Fail bars** | 1 (0.7%) |
| **Warning bars** | 0 (0.0%) |
| **Pass Rate（加重平均）** | 99.3% |
| **Fail Rate（加重平均）** | 0.7% |
| **Warning Rate（加重平均）** | 0.0% |
| **曲単位Pass平均（参考）** | 99.3% |
| **曲単位Warning平均（参考）** | 0.0% |
| **section_override適用** | 0 件 |
| **Safe-Kit推奨** | 1 (0.7%) |

---

## ✅ SLO（Service Level Objective）判定

| SLO指標 | 目標値 | 実測値 | 判定 |
|--------|--------|--------|------|
| **Post-gen Pass率** | ≥ 90% | 99.3% | ✅ PASS |
| **Warning率** | 0-5% | 0.0% | ✅ PASS |
| **Safe-Kit率** | ≤ 15% | 0.7% | ✅ PASS |

### 🎉 **All SLO PASS!**

---

## 📋 曲別詳細

| 曲名 | Total | Pass | Fail | Warning | Pass率 | section_override | Safe-Kit |
|-----|-------|------|------|---------|--------|------------------|----------|
| suno_project/song_001 | 150 | 149 | 1 | 0 | 99.3% | 0 | 1 |

---

## 🔍 Fail理由Top10

| 順位 | Fail理由 | 件数 |
|-----|---------|------|
| 1 | backbeat_strength too high: 1.00 > 0.9 | 1 |
| 2 | notes_per_bar too low: 7.00 < 8.0 | 1 |

---

## 🎯 推奨アクション

- ✅ **全てのSLO達成**: Phase 14（VioPTT本番実装）へ移行可能

