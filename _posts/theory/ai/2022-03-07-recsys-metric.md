---
title: "추천 시스템 metric"
categories: [Theory, AI]
tags: [RecSys, Metric]
img_path: /assets/img/posts/theory/ai/recsys_metric/
author: gshan
math: true
---

성능을 평가하는 관점으로는 2가지가 있다.
- 비즈니스/서비스 관점: 매출, PV(page view), CTR 증가.
- 품질 관점: 연관성(relevance), 다양성(diversity), 새로움(novelty), 참신함(serendipity)

추천 시스템의 목적은 위의 2가지 관점을 만족시키며 특정 user/item에게 적합한 item/user를 추천하는 것이다. 이러한 추천 시스템의 성능을 정량적으로 평가하기 위해 offline/online test를 진행한다.

## Ⅰ. Offline Test

새로운 추천 시스템을 검증하기 위해 가장 우선적으로 offline test가 수행된다.Offline test를 위해 user-item 상호 작용을 나타낼 score 값이 필요하다. Score 값으로 크게 2가지 유형이 있다.
- Ranking: user에게 적합한 item Top K개를 추천
  - Top K개를 선정하기 위한 기준 혹은 스코어가 필요하지만, 정확한 선호도를 구할 필요는 없다.
  - 평가지표로는 Precision@K, Recall@K, MAP@K, nDCG@K, Hit Rate@K 등이 있다.
- Prediction: user가 item 선호하는 정도를 정확하게 예측
  - Explicit Feedback: 평점 값을 예측
  - Implicit Feedback: 물건을 조회하거나 구매할 확률 값을 예측
  - User-item 행렬을 채우는 문제로 평가 지표로 MAE, RMSE, AUC 등이 있다.

### ⅰ. 추천된 item ranking 고려 x

#### Precision/Recall@K

Precision@K는 추천한 K개 item 가운데 실제 user가 관심있는 item의 비율이고, Recall@K는 user가 관심있는 전체 item 가운데 추천된 item의 비율이다. Precision과 Recall은 추천된 순서를 신경쓰지 않는다는 단점이 있다.

![](1.png)
_Precision@K, Recall@K 예시 [출처][1]{:target="_blank"}_

- 추천된 item 개추(K) = 5, user가 관심있는 item 개수 = 3, user가 관심있는 전체 item 개수 = 6
- Precision@5 = $\frac{3}{5}$, Recall@5 = $\frac{3}{6}$

#### Hit Rate@K

Hit rate는 전체 user 수 대비 적중한 user 수를 의미한다. 여기서 적중한 user는 추천 받은 items 중 user가 선호하는 item이 하나라도 있는 경우를 의미한다.

$$
\textrm{Hit Rate@K} = \frac{\# \textrm{ of Hit user}}{\# \textrm{ of user}}
$$

![](4.jpg)
_Hit Rate 예시 [출처][1]{:target="_blank"}_

위에서 살펴본 Hit Rate는 순서를 고려하지 않는다. 
순서를 고려하는 Hit Rate로 ARHR(Average Reciprocal Hit Rate), cHR(Cumulative Hit Rate), rHR(Rating Hit Rate) 등이 있다.

#### MAE, RMSE, AUC

MAE, RMSE, AUC 등 특정 값을 예측하는 것에 대한 평가는 실제 추천 시스템이 이뤄지는 환경에 적합하지 않아 많이 사용되지 않는다. 
추후에 따로 해당 metric을 다루겠다.

> AUC는 CTR 예측에서 사용된다.


### ⅱ. 추천된 item ranking 고려 o

#### MAP(Mean Average Precision)@K

Ranking 순서를 고려하지 않는 precision과 recall의 단점을 보완한 metric이다. MAP는 average precision의 mean을 의미한다.

$$
\begin{align*}
\textrm{AP@K} &= \frac{1}{\textrm{min}(K, T)}\sum_{i=1}^K\textrm{precision}@i \times \textrm{rel}(i) \\  
\textrm{rel}(i) &= 
  \begin{cases}
    1 & \textrm{if } i^{\textrm{th}} \text{ is True Positive} \\
    0 & \textrm{otherwise} 
  \end{cases}
\end{align*}
$$

- $T$: 전체 item set에서 user가 관심있는 전체 item 수.
- $\text{rel}(i)$: relevence를 나타내는 것으로 $i$번째 item을 사용자가 선호하는지 여부를 나타냄.
  - Relevence를 곱해주는 이유는 해당 순위에만 영향력을 주기 위함이다.
- $\textrm{K}$: 추천 시스템이 추천하는 item의 수.

AP(Average Precision)@K는 한 user에 대해 Precision@1부터 Precision@K까지의 평균값이다. Precision@K와 달리 관련 item을 더 높은 순위에 추천할수록 점수가 상승한다. 즉, <span class="text-color-yellow">추천되는 순서를 반영</span>한다.

$$
\textrm{MAP@K} = \frac{1}{|U|}\sum_{u=1}^{|U|}(\textrm{AP@K})_u
$$

- $\|U\|$는 user의 수를 나타낸다.

MAP@K는 모든 user에 대한 AP@K의 평균이다.

![](2.jpg)
_AP@K, MAP@K 예시 [출처][1]{:target="_blank"}_

User A가 선호하는 전체 item은 A, F이고 A, B, C를 추천 받았다. B가 선호하는 전체 item은 B, F이고 D, E, F를 추천 받았다 따라서 user A와 B의 AP@3, MAP@3는 위의 예시와 같다.

#### NDCG(Normalized Discounted Cumulative Gain)@K

NDCG는 원래 검색(information retrieval)에서 등장한 지표로 추천 시스템에서 가장 많이 사용되는 지표 중 하나이다. Top K개의 item을 추천받고 MAP@K와 마찬가지로 추천의 순서에 가중치를 두어 성능을 평가한다. MAP와 달리, 연성으로 이진(binary) 값이 아닌 수치도 사용할 수 있기 때문에 user에게 얼마나 더 관련 있는 item을 상위로 노출시키는지 알 수 있다.

- **CG(Cumulative Gain)**

  $$
  \textrm{CG}_K = \sum_{i=1}^K \textrm{rel}(i)
  $$
  
  상위 K개 item에 대하여 관련도(relevence)를 합한 것으로 순서에 따라 discount하지 않고 동일하게 더한 값이다. 즉, 순서를 고려하지 않았다. rel($i$)는 binary value(implicit feedback: 클릭 유무 등)이거나 문제에 따라 특정한 값(explicit feedback: rating 등)을 가질 수 있다.

- **DCG(Discounted Cumulative Gain)**

  $$
  \textrm{DCG}_K = \sum_{i=1}^K \frac{\textrm{rel}(i)}{\textrm{log}_2(i+1)}
  $$

  추천된 순서에 따라 CG를 discount한다.

- **IDCG(Ideal DCG)**

  $$
  \textrm{IDCG}_K = \sum_{i=1}^K \frac{\textrm{rel}(i)^{\textrm{opt}}}{\textrm{log}_2(i+1)}
  $$

  이상적인 추천이 일어났을 때의 DCG값으로, 가능한 DCG 값 중 제일 크다.

- **NDCG(Normalized DCG)**

  $$
  \textrm{NDCG}_K = \frac{\textrm{DCG}_K}{\textrm{IDCG}_K}
  $$

  추천 결과에 따라 구해진 DCG를 IDCG로 나눈 값이다.

![](3.jpg)
_NDCG@K 예시 [출처][1]{:target="_blank"}_


#### MRR(Mean Reciprocal Rank)

Reciprocal (역수) rank는 말 그대로 target item의 rank의 역수이다.

$$
\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
$$

추천시스템에선 순위가 가장 높은 item에만 관심있을 때 사용하는 metric이다.
예를 들어, 구글 검색에선 뒤쪽에 나타나는 페이지에는 관심이 없을 것이다.

![](6.jpg)


## Ⅱ. Online Test

앞에서 다양한 평가 방법을 살펴 보았지만, 상황/문화/사업 목표 등에 따라 적합한 방법론이 달리진다. 따라서 최종적으로 추천 시스템을 평가할 땐 실제 users가 추천 목록에 어떻게 반응하는지 측정해야 한다. 이 중 online A/B test가 있다. 이외에 사용자에게 직접적으로 추천 목록을 평가해달라고 요청하는 perceived quality 등이 있다.

![](5.jpg)
_Online A/B test_

Online A/B test 란, offline test에서 검증된 가설이나 모델을 이용해 실제 추천 결과를 serving하는 단계이다.
- 시간을 기준으로 추천 시스템 변경 전후의 성능을 비교하는 것이 아니라, 동시에 대조군(A)과 실험군(B)의 성능을 평가한다. (대조군과 실험군의 환경은 **최대한 동일**해야 함.)
- 실제 서비스를 통해 얻어지는 결과를 통해 최종 의사결정이 이루어진다.

대부분 현업에서 의사결정에 사용하는 최종 지표는 모델 성능이 아닌 매출, CTR 등의 비즈니스/서비스 지표를 사용한다.

## References

1. [[추천시스템] 성능 평가 방법 - Precision, Recall, NDCG, Hit Rate, MAE, RMSE][1]{:target="_blank"}
2. [How mean Average Precision at k (mAP@k) can be more useful than other evaluation metrics][2]{:target="_blank"}
3. [Wiki - Mean reciprocal rank][3]{:target="_blank"}
4. [YouTube - Amazon personalized 소개][4]{:target="_blank"}

[1]: https://sungkee-book.tistory.com/11
[2]: https://medium.com/@misty.mok/how-mean-average-precision-at-k-map-k-can-be-more-useful-than-other-evaluation-metrics-6881e0ee21a9
[3]: https://en.wikipedia.org/wiki/Mean_reciprocal_rank
[4]: https://www.youtube.com/watch?v=Och2ml4mB0s