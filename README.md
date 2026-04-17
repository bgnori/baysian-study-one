# ベイズ統計学研究プロジェクト

歪んだサイコロを題材に、**ブラックボックス観測だけで確率を推定する**ベイズ学習プロジェクトです。

## 学べること
1. Dirichlet-Categorical モデルによるベイズ更新
2. 観測回数が増えると推定がどう安定するか
3. 予測確率だけでなく、不確実性も読む方法

## 理解度チェック
### 問題
1. このプロジェクトでいう「ブラックボックス観測」とは、何が見えていて何が見えていない状態ですか。
2. 観測回数が少ないとき、予測確率や判断はなぜ揺れやすくなりますか。
3. 最も出やすい面だけでなく、不確実性も同時に読むことがなぜ大切ですか。

### 答え
1. 見えているのは各試行で実際に出た目だけで、サイコロ内部の真の確率は直接は見えません。そのため、予測器は観測結果だけを使って推定を更新します。
2. データが少ないと偶然の偏りが大きく見えやすく、まだ十分な根拠を持てないためです。観測が増えるほど、推定は安定しやすくなります。
3. 同じ予測でも、強い確信を持っている場合と、まだ迷いが大きい場合があります。不確実性を見ることで、その予測をどの程度信頼してよいか判断できます。

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
- [docs/advanced_entropy_estimation.md](docs/advanced_entropy_estimation.md): エントロピー推定の発展課題
- [docs/advanced_bayesian_topics.md](docs/advanced_bayesian_topics.md): 感度分析・公平性比較・時変歪み検知の発展課題
- [run_demo.py](run_demo.py): 最小の実行例
- [analyze_convergence.py](analyze_convergence.py): 学習過程の確認
- [entropy_posterior_demo.py](entropy_posterior_demo.py): 乱数源エントロピー推定の発展デモ
- [prior_sensitivity_study.py](prior_sensitivity_study.py): 事前分布感度分析の発展デモ
- [posterior_predictive_study.py](posterior_predictive_study.py): 事後予測シミュレーションの発展デモ
- [fairness_comparison_demo.py](fairness_comparison_demo.py): 公平なサイコロとの比較デモ
- [time_varying_bias_demo.py](time_varying_bias_demo.py): 時変歪み検知の発展デモ
- [src/bayesian_dice](src/bayesian_dice): 実装本体

## 学習の進め方
1. [docs/tutorial.md](docs/tutorial.md) を読む
2. [run_demo.py](run_demo.py) を実行する
3. [analyze_convergence.py](analyze_convergence.py) で収束を確認する
4. [docs/bayesian_dice_prediction_design.md](docs/bayesian_dice_prediction_design.md) の数式と照らし合わせる
5. 発展課題として [docs/advanced_entropy_estimation.md](docs/advanced_entropy_estimation.md) と [entropy_posterior_demo.py](entropy_posterior_demo.py) に進む
6. さらに [docs/advanced_bayesian_topics.md](docs/advanced_bayesian_topics.md) を読み、感度分析・公平性比較・時変歪み検知を試す

## モデル概要
- サイコロの内部確率は見えない
- 観測結果だけを使って確率を更新する
- 初期版は対称 Dirichlet 事前分布を使う
- 出力は各面の予測確率と不確実性の要約

## 参考文献
- Gelman, A., et al. Bayesian Data Analysis. 3rd ed., 2014.
- McElreath, R. Statistical Rethinking. 2nd ed., 2020.
- Murphy, K. P. Machine Learning: A Probabilistic Perspective. 2012.
