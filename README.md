# ベイズ統計学研究プロジェクト

歪んだサイコロを題材に、**ブラックボックス観測だけで確率を推定する**ベイズ学習プロジェクトです。

## 学べること
1. Dirichlet-Categorical モデルによるベイズ更新
2. 観測回数が増えると推定がどう安定するか
3. 予測確率だけでなく、不確実性も読む方法

## クイックスタート
### 1. 基本デモ
プロジェクト直下で次を実行します。

python run_demo.py

### 2. 収束の確認
観測が増えると予測がどう変わるかを確認します。

python analyze_convergence.py

### 3. テスト
実装の正しさは次で確認できます。

python -m unittest discover -s tests -v

## プロジェクト構成
- [docs/bayesian_dice_prediction_design.md](docs/bayesian_dice_prediction_design.md): 数式ベースの設計書
- [docs/tutorial.md](docs/tutorial.md): 初学者向けチュートリアル
- [run_demo.py](run_demo.py): 最小の実行例
- [analyze_convergence.py](analyze_convergence.py): 学習過程の確認
- [src/bayesian_dice](src/bayesian_dice): 実装本体

## 学習の進め方
1. [docs/tutorial.md](docs/tutorial.md) を読む
2. [run_demo.py](run_demo.py) を実行する
3. [analyze_convergence.py](analyze_convergence.py) で収束を確認する
4. [docs/bayesian_dice_prediction_design.md](docs/bayesian_dice_prediction_design.md) の数式と照らし合わせる

## モデル概要
- サイコロの内部確率は見えない
- 観測結果だけを使って確率を更新する
- 初期版は対称 Dirichlet 事前分布を使う
- 出力は各面の予測確率と不確実性の要約

## 参考文献
- Gelman, A., et al. Bayesian Data Analysis. 3rd ed., 2014.
- McElreath, R. Statistical Rethinking. 2nd ed., 2020.
- Murphy, K. P. Machine Learning: A Probabilistic Perspective. 2012.
