# ベイズ的サイコロ予測プロジェクト設計文書

## 予測サイドアーキテクチャ
本ドキュメントは、サイコロシミュレーターをブラックボックスとして扱い、ベイズ的アプローチに基づく予測サイドのアーキテクチャを記述します。

### ジェネレーティブモデル
サイコロの出目 $X$ は、パラメータ $\theta$ に基づき、確率的に生成されます。ここで、$\theta$ はサイコロのバイアスを表します。
\[ X | \theta \sim \text{Categorical}(\theta) \]

### プライヤー
$\theta$ に対する事前分布は、非情報的な一様分布として設定します。
\[ \theta \sim \text{Uniform}(0, 1) \]

### ライクリフッド
観測データ $D$ を条件にした場合のライクリフッドは次のように表されます。
\[ P(D | \theta) = \prod_{i=1}^{n} P(x_i | \theta) \]

### ポスタリアー更新
ベイズの定理を使用して、ポスタリアー分布は次のように更新されます。
\[ P(\theta | D) = \frac{P(D | \theta) P(\theta)}{P(D)} \]

### 予測分布
新しい出目 $X_{new}$ の予測分布は以下のように表現されます。
\[ P(X_{new} | D) = \int P(X_{new} | \theta) P(\theta | D) d\theta \]

### 評価指標
1. 精度（Accuracy）
2. 再現率（Recall）
3. F1スコア（F1 Score）

### 参考文献
- Gelman, A., & Hill, J. (2006). Data Analysis Using Regression and Multilevel/Hierarchical Models. Cambridge University Press.
- Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.  
- Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. MIT Press.
