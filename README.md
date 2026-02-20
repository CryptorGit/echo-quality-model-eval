# 心エコー画像の品質評価モデル（Phase1）と埋め込み評価（Phase2）

## まず最初に（このリポでできること / できないこと）
このリポジトリは「心エコー画像の品質を数値化するモデル（Phase1）」と、「そのモデルの embedding がある程度まともかを検証する枠組み（Phase2）」をまとめたものです。

できること:
- 入力: 心エコー静止画像（ビュー: A4C / PL / PSAV / PSMV / SC）
- 出力: (Phase1) 品質スコア $\hat{q}$（概ね 0〜9） / (Phase2) embedding の検証指標（静的構造・方向安定性・改善可能性）
- 用途: データ選別（品質ゲート）・品質の定量化・将来の誘導研究の前段の評価

できないこと:
- データセット本体（CACTUS 等）の同梱はしません（各データの規約に従って取得してください）
- Phase3/Phase4（プローブ誘導・シミュレーター等）は含みません
- 臨床利用を想定した保証やバリデーションは提供しません

全体の流れ（最小）:
```
CACTUS画像 + qualityラベル
  └─ (Phase1) 品質回帰モデル（multihead） → 品質スコア q̂
       └─ (Phase2) embedding 抽出 → 構造/安定性/改善可能性を検証
```

## 要旨（Abstract）
本リポジトリは、**プローブ操作の術者依存性が大きい心エコーに対して、画像品質（断面の良さ）を定量化するモデル**を構築し（Phase1）、そのモデルが **一貫した品質表現（embedding 空間）を持つか**を評価した（Phase2）作業を、コードと再現手順としてまとめたものです。

公開範囲は **Phase1 と Phase2 のみ**です。

本リポジトリには、推論・評価のための学習済み重み（checkpoint）を含みます（**Git LFS** 管理）。利用条件は [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) を参照してください（**非商用のみ**）。

注意:
- モデル（重み）の利用可否・利用範囲は、元データセットの利用規約に強く依存します。
- もし **元データの規約上「派生物（重み等）の再配布」が許容されない**場合、ここに同梱している重みは利用できません（必要に応じてローカルから削除し、再学習/再現手順のみを使ってください）。

---

## 0. Quickstart（まず動かす）
前提: **Python 3.10.x（例: 3.10.13）** / Git / Git LFS

### 0.1 Clone（重みを含む）
```powershell
git clone https://github.com/CryptorGit/echo-quality-model-eval.git
cd echo-quality-model-eval
git lfs install
git lfs pull
```

### 0.2 Install
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

（bash の場合）
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

### 0.3 データ配置（このパス前提です）
CACTUS を展開して、少なくとも次が存在するように配置してください:
- `datasets/CACTUS/extracted/Cactus Dataset/Grades/`（`*_grades.csv` が入っている）
- `datasets/CACTUS/extracted/Cactus Dataset/` 配下に画像ファイル（png/jpg 等）

想定ディレクトリ例（最小）:
```
echo-quality-model-eval/
  datasets/
    CACTUS/
      extracted/
        Cactus Dataset/
          Grades/
            A4C_grades.csv
            PL_grades.csv
            ...
          (images...)
          (subfolders...)
```

補足:
- `make_cactus_manifest.py` は `Grades/*_grades.csv` を読み、画像ファイルを探索して `datasets/CACTUS/manifests/*.csv` を生成します。
- 画像パスの書き方がCSVと一致しない場合でも、basename（ファイル名）で探索して解決を試みます（ただし衝突が多い場合は解決に失敗します）。

### 0.4 前処理（manifest / split 生成。初回のみ）
```powershell
python phases/01_quality/scripts/make_cactus_manifest.py
python phases/01_quality/scripts/make_cactus_groupsplit.py
python phases/01_quality/scripts/prepare_cactus_consensus_dataset.py --resize_mode pad
python phases/01_quality/scripts/make_cactus_groupsplit_stratified.py --exclude_random
python phases/01_quality/scripts/make_phase1_kfold_v1.py --base_split_dir datasets/CACTUS/manifests_group_consensus_stratified --exclude_random
```

### 0.5 Phase2 を回す（採用モデル w14, fold0-4）
```powershell
python phases/02_embedding/scripts/run_phase2_pipeline.py --run_tag loss_reweight_view_band_w14 --folds 0,1,2,3,4
```

Quickstart が通らない場合は、まず `python phases/02_embedding/scripts/run_phase2_pipeline.py --help` が動くか確認してください。

## 1. 背景
心エコーは、同一被写体でもプローブの当て方・角度・押し当て・位置で画像品質が大きく変動します。将来のプローブ誘導（「どちらへ動かすべきか」）を成立させるには、まず **各ビューにおける“品質”を安定に推定できる基盤モデル**が必要です。

---

## 2. 目的
1) **Phase1**: 心エコー画像品質を $\hat{q}$（概ね 0〜9 の連続値）として推定するモデルを構築し、k-fold で安定に評価する。

2) **Phase2**: Phase1 の採用モデルが、embedding 空間上で **妥当な順位構造・方向安定性・改善可能性**を満たすかを定量評価する。

---

## 3. リポジトリ構成（公開用に最小限）
- `phases/01_quality/scripts/`: Phase1（品質評価モデルの学習・評価・前処理）
- `phases/02_embedding/scripts/`: Phase2（embedding 抽出と検証）
- `shared/runs/phase1/`: 学習済み重み（`.pt`、Git LFS 管理）

注:
- `datasets/` や `splits/` は **ローカルで生成/配置する前提**です（データ由来ファイルは同梱しません）。
- 本リポジトリは意図的に **大規模出力（csv/npy）・ログ・画像**を同梱しません（手元で生成してください）。

---

## 4. データと前処理（Phase1共通）
### 4.1 データ（想定）
- データセット: CACTUS
- 配布ページ（例）: https://www.frdr-dfdr.ca/repo/dataset/86beb91e-c0ba-496b-ad38-732b67e3d5f6
- 対象ビュー: `A4C, PL, PSAV, PSMV, SC`
- `Random` は Phase1 では除外して運用
- ラベル: `quality`（float、概ね 0〜9）

注意:
- 本リポジトリは **データセット本体を同梱しません**。上記ページから取得してください。
- **利用条件（研究用途可否、再配布可否、商用可否、派生物の公開可否）は配布元の規約が優先**です。

### 4.2 幾何の整合（pad2）
テンプレート生成・テンプレート適用・モデル入力を、すべて **アスペクト比維持 + letterbox pad**（KeepRatioPad）で統一します。縦横別リサイズ（warp）が混入すると、sector（扇形）領域の幾何が崩れ、学習・評価の整合性が壊れるためです。

### 4.3 GroupSplit（stratified v3: リーク抑制 + 分布整合）
画像の近縁サンプルが train/test に跨らないよう、ファイル名から `group` を抽出して group-aware に分割します。加えて、split 間で view×quality 分布が極端にズレないよう、group 単位の (view, grade) カウントを用いた greedy 割当で stratify します。

pad2 split stats（実績; exclude_random=true）:
- rows: train 22139 / val 3902 / test 3804
- groups: train 89 / val 11 / test 12
- overlap: train-val/train-test/val-test は全て 0

---

## 5. Phase1: 品質評価モデル
### 5.1 タスク定義
入力: 画像 1枚（RGB）

出力: 品質スコア $\hat{q}$（連続値）

### 5.2 モデル概要（実装ベース）
- encoder: ResNet 系
- multihead: encoder 共有 + view 別 head
- ordinal head: 9 logits に対して $\hat{q}=\sum \sigma(o_k)$（実装上の近似）

### 5.3 評価指標
- 予測性能: QWK（Quadratic Weighted Kappa）, MAE, RMSE（All / per-view）
- 方向安定性: `global_top20_cos`（top20%良品方向の cosine）

### 5.4 実験（k-fold 本番 11実験のうち主要比較）
代表的な比較軸:
- baseline（pad2 + group/view/grade sampler）
- weak-view 強化 head
- view×band の loss reweight
- SupCon（小規模グリッド）

### 5.5 結果
#### 5.5.1 本番モデル（上位）

| Model | All QWK | All MAE | All RMSE | global_top20_cos | weak-view avg QWK |
|---|---:|---:|---:|---:|---:|
| loss_reweight_view_band_w14 | 0.8094 ± 0.0387 | 0.7713 ± 0.0659 | 0.9406 ± 0.0929 | 0.7478 ± 0.0569 | 0.4725 |
| loss_reweight_view_band | 0.8063 ± 0.0345 | 0.7814 ± 0.0505 | 0.9527 ± 0.0610 | 0.7454 ± 0.0337 | 0.4727 |
| baseline_runB_pad2_p1_p2_group_view_grade | 0.7972 ± 0.0272 | 0.7822 ± 0.0693 | 0.9603 ± 0.0775 | 0.7681 ± 0.0568 | 0.4470 |

#### 5.5.2 w12 と w14 の最終比較

| Metric | w12 | w14 | Δ (w14 - w12) |
|---|---:|---:|---:|
| QWK | 0.7878 ± 0.0334 | 0.8094 ± 0.0387 | +0.0216 |
| MAE | 0.7954 ± 0.0567 | 0.7713 ± 0.0659 | -0.0241 |
| RMSE | 0.9738 ± 0.0781 | 0.9406 ± 0.0929 | -0.0332 |
| global_top20_cos | 0.7159 ± 0.1006 | 0.7478 ± 0.0569 | +0.0319 |

弱ビュー（A4C/SC/PSAV）平均QWK:
- w12: 0.4431
- w14: 0.4725（+0.0294）

結論: **採用モデルは `loss_reweight_view_band_w14`**（All / weak-view / 方向安定性の優位を優先）。

---

## 6. Phase2: embedding によるモデル評価
### 6.1 方針
Phase1 の採用モデル（fold別 best）で embedding を抽出し、view ごとの代表集合 `G_view` を構築して、
- 距離指標 `d_view(z)`
- ガイダンス指標 `g(z)`
を付与した上で、静的構造・方向安定性・改善可能性を検証します。

補足: 現行 Phase1 multihead は view 入力前提のため、Phase2 では `view_pred` は `view_true` を格納します（暫定）。

### 6.2 合否基準と結果（最終）
- 静的構造（Spearman<0 が 4/5 view 以上）: **PASS**
- 方向安定性（global top20 mean cos $\ge 0.7$）: **PASS**（0.7210）
- 改善可能性（global $>0.6$ かつ weak $>0.55$）: **PASS**（global 0.6673, weak 0.7094）

---

## 7. 再現手順（データ配置はローカル前提）
### 7.1 依存パッケージ
Python は **3.10.x** を推奨します。

依存は [requirements.txt](requirements.txt) を参照してください（PyTorch は環境（CUDA/CPU）に合わせて導入してください）。

重みを含む clone には Git LFS が必要です:
```powershell
git lfs install
git lfs pull
```

### 7.2 Phase1（例）
データ準備（例）:
```powershell
python phases/01_quality/scripts/make_cactus_manifest.py
python phases/01_quality/scripts/make_cactus_groupsplit.py
python phases/01_quality/scripts/make_cactus_groupsplit_stratified.py
python phases/01_quality/scripts/prepare_cactus_consensus_dataset.py --resize_mode pad
```

学習/評価（例）:
```powershell
python phases/01_quality/scripts/train_phase1.py --help
python phases/01_quality/scripts/eval_phase1.py --help
python phases/01_quality/scripts/direction_stability_test.py --help
```

### 7.3 Phase2（例）
```powershell
python phases/02_embedding/scripts/run_phase2_pipeline.py --run_tag loss_reweight_view_band_w14 --folds 0,1,2,3,4
```

---

## 8. 制約と今後
このリポジトリは **Phase1/Phase2 までのコードと検証枠組み**に限定して公開します。

また、原則として **本リポジトリに関する今後の更新は行いません**。
ただし、(i) **オープンなプローブ位置（位置・姿勢）に対応づく心断面のデータセット**、または (ii) **心エコー画像を出力できるシミュレーター**を作成・取得できた場合に限り、追加の更新を行い、次段階の作業へ進める予定です。

## 用語ミニ辞書（最小）
- **QWK**: Quadratic Weighted Kappa。ラベルのズレを重く罰する一致度（大きいほど良い）。
- **MAE / RMSE**: 回帰誤差指標（小さいほど良い）。
- **pad2 / KeepRatioPad**: アスペクト比を維持し、letterbox で正方形入力へ整形する前処理。
- **GroupSplit**: 近縁サンプル（同一患者/系列など）が train/test を跨がないように `group` 単位で分割。
- **global_top20_cos**: 「高品質（上位20%）」方向の embedding の整列度合いを cosine で測る指標（大きいほど“方向が安定”）。

---

## 参考文献
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition (CVPR).
