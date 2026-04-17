# ベイズ的サイコロ予測プロジェクト設計文書

## 1. 目的
本プロジェクトでは、歪んだ6面サイコロをブラックボックスとして扱い、観測された出目列だけから各面の出現確率を推定する。予測側はシミュレーター内部の確率設定を参照せず、観測データのみに基づいて更新を行う。

## 2. ブラックボックス制約
予測器が利用できる情報は、各試行で得られた出目
$$
X_t \in \{1,2,3,4,5,6\}
$$
のみとする。内部パラメータ
$$
\theta = (\theta_1,\theta_2,\theta_3,\theta_4,\theta_5,\theta_6)
$$
は未知であり、直接参照してはならない。

## 3. 生成モデル
各試行の出目はカテゴリ分布に従うと仮定する。
$$
X_t \mid \theta \sim \mathrm{Categorical}(\theta_1,\dots,\theta_6)
$$
ここで各パラメータは
$$
\theta_k \ge 0, \qquad \sum_{k=1}^{6} \theta_k = 1
$$
を満たす。

## 4. 事前分布
6個の確率を同時に扱うため、事前分布には Dirichlet 分布を用いる。
$$
\theta \sim \mathrm{Dirichlet}(\alpha_1,\alpha_2,\alpha_3,\alpha_4,\alpha_5,\alpha_6)
$$
初期実装では対称事前分布
$$
\alpha_1 = \cdots = \alpha_6 = 1
$$
を採用する。これは全ての面を等しく事前に扱う設定である。

## 5. 尤度
観測データを
$$
D = (x_1,\dots,x_n)
$$
とし、各面の出現回数を
$$
n_k = \sum_{t=1}^{n} \mathbf{1}(x_t = k)
$$
と置く。このとき尤度は多項分布の形で
$$
P(D \mid \theta) \propto \prod_{k=1}^{6} \theta_k^{n_k}
$$
と書ける。

## 6. 事後分布
Dirichlet 分布はカテゴリ分布に対する共役事前分布なので、事後分布は閉形式で更新できる。
$$
\theta \mid D \sim \mathrm{Dirichlet}(\alpha_1 + n_1, \dots, \alpha_6 + n_6)
$$
これにより、実装ではサンプル列を受け取るたびに各面のカウントを増やすだけでよい。

## 7. 事後平均と予測分布
各面の事後平均は
$$
\mathbb{E}[\theta_k \mid D] = \frac{\alpha_k + n_k}{\sum_{j=1}^{6}(\alpha_j + n_j)}
$$
となる。次の1回の出目の事後予測分布も同じ形で
$$
P(X_{n+1}=k \mid D) = \frac{\alpha_k + n_k}{\sum_{j=1}^{6}(\alpha_j + n_j)}
$$
となるため、初期版ではこの値をそのまま予測確率として採用する。

## 8. 実装方針
予測側は以下の責務を持つ。

1. 観測結果を1件ずつ受け取る
2. 各面の出現回数を更新する
3. 事後平均を計算する
4. 次回出目の予測分布を返す
5. 最も確からしい面を返す

一方、モック側は歪んだサイコロのロール結果のみを公開する。これにより、予測器はブラックボックス制約を保ったまま検証できる。

## 9. 評価指標
初期実装では以下を評価対象とする。

- 事後予測確率の収束
- 真の分布とのずれ（シミュレーター側でのみ比較可能）
- 予測対数損失
- 最頻面の予測精度

Recall や F1 のような分類指標は主目的ではないため、補助的な位置づけに留める。

## 10. 将来拡張
- 事前分布の感度分析
- 観測回数に対する収束速度の比較
- 非対称な事前知識の導入
- MCMC によるより一般的な推論への拡張

## 11. 参考文献
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. Bayesian Data Analysis. 3rd ed., 2014.
- McElreath, R. Statistical Rethinking. 2nd ed., 2020.
- Murphy, K. P. Machine Learning: A Probabilistic Perspective. 2012.
- Minka, T. Estimating a Dirichlet Distribution. 2003.
- Vehtari, A., Gelman, A., & Gabry, J. Practical Bayesian Model Evaluation Using Leave-One-Out Cross-Validation and WAIC. 2017.
