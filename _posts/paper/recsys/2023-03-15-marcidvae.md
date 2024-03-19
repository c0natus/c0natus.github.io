---
title: Learning Disentangled Representations for Recommendation, (NeurIPS'19)
categories: [Paper, RecSys]
tags: [Collaborative Filtering, Disentangle, Variational Autoencoder]
img_path: /assets/img/posts/paper/recsys/marcidvae/
author: gshan
math: true
---

|[Paper Link][1]{:target="_blank"}|[Official Github][2]{:target="_blank"}|

## Abstract

Challenges
: 1. User의 선호도(low-level)과 의사 결정을 한 의도(high-level) 간의 hierarchical 관계를 유지해야 함.  
2. Data가 discrete하고 sparse하다. 그래서 고차원인 representation space에서 대부분의 vector가 training data와 관련 없다. 이는 interpretability에 대한 조사를 할 수 없음을 의미함.

Address
: 1. Item과 interaction이 생긴다면, implicit하게 학습한 item의 concept과 관련된 user preference representation $z_u^{(k)}$만 update한다.
2. Disentangled representation의 모든 entries가 사람이 이해 가능한 것이 아니지만, 특정 entries는 사람이 이해 가능한 부분(color, size, etc)으로 학습 가능하다.

## 1. Introduction

많은 연구들을 user의 행동에 기반해 user의 선호도를 나타낸다.
이러한 기존의 연구들은 user의 의사 결정 뒤에 숨겨진, 복잡하게 얽혀있는 <span class="text-color-yellow">latent factor 간의 interaction을 반영하지 못한다.</span>
- Latent factors는 한 session 동안 user의 주된 의도를 담는 macro(high-level: 옷, 휴대폰, 등)부터 user의 세분화된 선호도인 micro(low-level: 사이즈, 색, 등)까지 다양하다.

기존의 연구들은 latent factors를 disentangle하는데 실패했고, 학습된 representation은 교란 변수(confounding of the factors)를 보존하게 된다.
이로 인해 non-robustness와 low interpretability를 가진다.

[Disentangled representation learning][3]{:target="_blank"}의 목표는 관찰된 data에 숨어있는 latent explanatory factors를 발견하는 factorized representations을 학습하는 것이다.
Disentangle된 representation은 더 robust하고 강화된 interpretability, controllability를 가진다.

Robust하다는 것은 bias가 존재할 수밖에 없는 training data으로 도출된 misleading correlations에 덜 민감하다는 뜻이다.
그리고 강화된 interpretability를 통해 transparent advertising(광고가 나타난 이유를 알림), 설명가능한 추천 등이 가능해진다.
또한, controllability는 user에게 더 interactive한 경험(소비할 가능성이 높은 item)을 제공한다.

> 예를 들어, user의 수입에 대한 latent factor를 나타낼 수 있다면 수입이 증가했을 때 더 소비할 확률이 높은 비싸지만 품질이 좋은 item을 추천해줄 수 있다. 즉, 추천되는 item에 대한 control이 가능하다. 해당 예시는 [Causal Representation Learning for Out-of-Distribution Recommendation][4]{:target="_blank"} 논문에서 발췌했다.

Computer vision에서는 disentangled representation learning 연구가 진행되고 있다.
하지만 discrete relational data 중 하나인 user behavior data (user-item interactions)는 image data와 근본적으로 다르다.
Discrete relational data를 disentangle하는 데 2가지 challenges가 존재한다.

1. User의 선호도(low-level)과 item 소비 의도(high-level) 사이의 hierarchical 관계를 유지하면서 disentanglement를 수행해야 한다.
2. User behavior data는 discrete하고 sparse하지만, 학습된 representation은 continuous하다.
즉, 높은 차원의 representation space에서 대부분의 vector point가 어떠한 behavior와도 연관되어 있지 않다.
이는 representation vector에서 다른 element의 값을 고정시킨 채, 특정 element를 변경해가며 interpretability를 조사하려고 할 때 문제가 된다.

> 2 번째는 manifold와 관련된 내용인 것 같다.  
> 예를 들어, item representation vector에서 특정 element를 변경했을 때, 그게 item에 어떤 영향을 주는 지 알 수 없다.
> 예를 들어, 잘 disentangle되어 있다면 특정 element를 바꾸면 형태는 비슷하지만 색만 달리진 item representation이 될 수 있다.
> Section 2.4를 참고하자.
> 
> 하지만, image data도 마찬가지 아닌가...? VAE로 찾은 manifold에서 어느 정도 해석이 가능하다...  
> 추천 data의 특성 때문에 manifold 찾는 것 자체가 어렵다는 것인가...?

본 논문에서는 MACRo-mIcro Disentangled Variational Auto-Encoder (MacridVAE)를 제안한다.
MacridVAE는 macro, micro factors를 explicit하게 disentangle한다.
- User의 의도와 관련된 high-level concepts를 식별하고, 각 concept별 user의 선호도를 학습해 macro disentanglement를 수행한다.
- Micro disentanglement를 위해 VAE의 regularizer를 강화한다. Information-theoretic 관점으로 VAEs를 해석하면, 이를 통해 vector의 각 element가 independent한 micro factor를 반영하도록 만든다.

> Regularization을 약화하는 Mult-VAE$^\text{PR}$과 다르다.  
> 하지만 구현된 code를 참고하면, Mult-VAE$^\text{PR}$과 같은데...?

Sparse한 discrete data와 dense한 continuous representation 간의 trade-off를 다루기 위해 beam-search를 사용해 smooth한 trajectory를 찾는다.
이를 통해 각 element의 interpretability를 조사할 수 있게 된다.

> Image였다면 특정 element를 변경한 뒤 decoder에 feed하여 generation을 진행할 수 있다.  
> 하지만 추천은 실제 있는 item을 표현해야 한다.
> 그래서 실제 있는 item 중 변경된 vector와 가장 유사한 item을 찾아야 하는데, 이때 효율을 위해 beam search를 사용한다.

## 2. Method

![](1.png)
_Figure 1: Our framework. Macro disentanglement is achieved by learning a set of prototypes, based on which the user intention related with each item is inferred, and then capturing the preference of a user about the different intentions separately. Micro disentanglement is achieved by magnifying the KL divergence, from which a term that penalizes total correlation can be separated, with a factor of $\beta$._

### 2.1. Notations and Problem Formulation

User는 총 N명이 있고, item은 총 M개가 있다.
User $u$와 item $i$ 간 interaction이 있으면 $x_{u,i} = 1$이고 없으면 $x_{u,i} = 0$이다.
User $u$가 소비한 item 목록은 $\textbf{x}_u \in \mathbb{R}^M$으로 나타낸다.
본 논문에서는 macro, micro disentanglement를 반영한 user representations $\textbf{z}_u$를 학습해야 한다.
$\theta$는 MarcidVAE에 모든 trainable parameters를 의미한다.

<span class="text-coloar-bold">**Macro disentanglement.**</span>
User는 다양한 곳에 흥미를 가질 것이고, user가 interaction한 items은 많은 high-level concepts(product categories 등)에 속할 것이다.

$K$개의 high-level concepts이 있을 때, 저자들은 user $u$의 factorized representation $\textbf{z}_u = \[\textbf{z}_u^{(1)};\cdots;\textbf{z}_u^{(K)}\] \in \mathbb{R}^{d^\prime}$을 학습시키며 macro disentanglement를 수행한다. $d^\prime = Kd$이고, $\textbf{z}_u^{(k)} \in \mathbb{R}^d$는 $k$번째 concept과 관련된 user의 선호도를 나타낸다.

추가로 item $i$가 concept $k$에 속하는지 안 하는지 추론한다.
만약 속한다면 $c_{i,k} = 1$이고 $k^\prime \neq k$인 $k^\prime$에 대해 $c_{i,k^\prime} = 0$이다.
이를 각 item마다 one-hot vector $\textbf{c}_i$로 나타내고, 이들을 모아 $\textbf{C} \in \mathbb{R}^{M \times K}$로 표기한다.

저자들은 모든 $\textbf{z}_u$와 $\textbf{c}_i$를 unsupervised로 추론한다.

> 각 concept이 뭘 뜻하는지, 각 item이 어떤 concept에 속하는지 알지 못하기 때문에 unsupervised로 볼 수 있다.

<span class="text-coloar-bold">**Micro disentanglement.**</span>
저자들은 한 item에 대한 user의 선호도를 더 세분화되는 것을 원한다.
예를 들어, concept $k$가 옷이라고 했을 때 $\textbf{z}_u^{(k)}$ vector의 각 element가 개별적으로 user가 선호하는 옷 size나 color 등을 capture하기를 원한다.

### 2.2. Model

Macro disentanglement를 위해 generative model을 제안한다.
User $u$의 관찰된 data는 아래의 generative model에 의해 생성된다고 가정한다.

$$
\begin{align}
  &p_\theta(\textbf{x}_u) = \mathbb{E}_{p_\theta(\textbf{C})}\bigg[ \int p_\theta(\textbf{x}_u|\textbf{z}_u, \textbf{C})p_\theta(\textbf{z}_u)d\textbf{z}_u \bigg], \ \ p_\theta(\textbf{x}_u|\textbf{z}_u, \textbf{C}) = \frac{p_\theta(\textbf{x}_u,\textbf{z}_u|\textbf{C})}{p_\theta(\textbf{z}_u|\textbf{C})}\\

  &p_\theta(\textbf{x}_u|\textbf{z}_u, \textbf{C}) = \prod_{x_{u,i} \in \textbf{x}_u}p_\theta(x_{u,i}|\textbf{z}_u, \textbf{C})
\end{align}
$$

$p_\theta(z_u) = p_\theta(z_u\|C)$라고 가정한다. 즉, $z_u$와 $C$는 서로 독립된 distirubiton에서 생성된다.
$p_\theta(x_{u,i}\|z_u,C)$는 $M$개의 item에 대한 user $u$의 선호도를 나타내는 categorical distribution으로 아래와 같이 정의된다.

$$
\begin{split}
  &p_\theta(\textbf{x}_{u,i}|\textbf{z}_u,\textbf{C}) = Z^{-1}_u \cdot \sum_{k=1}^K c_{i,k} \cdot g_\theta^{(i)}(\textbf{z}_u^{(k)})\\
  &Z_u = \sum_{i=1}^M\sum_{k=1}^K c_{i,k} \cdot g_\theta^{(i)}(\textbf{z}_u^{(k)})\\
  &g_\theta^{(i)}: \mathbb{R}^d \rightarrow \mathbb{R}_+
\end{split}
$$

$g_\theta^{(i)}$는 얉은 neural network로 user의 item $i$에 대한 선호도를 추정한다.
만약 $M$이 매우 크다면 몇 개의 sample된 items으로 $Z_u$를 추정하기 위해 [sampled softmax][5]{:target="_blank"}를 사용한다.

> $Z_u$를 구할 때 $\sum_i^M$ 대신 $\sum_i^{M^\prime}$을 한다는 뜻이다. 
> 추천에서 모든 negative item에 대한 역전파하는 것의 cost가 상당하기 때문에 negative sampling하는 이유와 유사하다.
> 
> $g_\theta^{(i)}$에서 $\theta$는 item i의 representation vector인 $h_i$를 뜻한다.
> 뒤에서 더 자세히 살펴볼 건데, $\textbf{z}_u^{(k)}$가 주어졌을 때 item embedding과의 Cosine similarity로 구한다.

<span class="text-coloar-bold">**Macro disentanglement.**</span>
User representation $\textbf{z}_u$는 user와 interaction할 items를 예측하기에 sufficient하다고 가정한다.
그리고 interaction을 예측할 item이 $k$ concept이라면 $\textbf{z}_u^{(k)}$만으로도 sufficient하다고 가정한다.
이는 추론된 concept assignment matrix $\textbf{C}$가 의미 있다면, $\textbf{z}_u^{(k)}$가 $k$ 번째 concept에 관한 선호도를 explicit하게 capture하도록 만든다.

> $\textbf{C}$가 의미 있다는 뜻은 items가 어떤 concept에 속하는지 잘 학습했다는 뜻인 것 같다.

이후에 $p_\theta(C), p_\theta(z_u), g_\theta^{(i)}(z_u^{(k)})$ 구현에 대해 살펴볼 것이다.
특히, mode collapse를 방지하기 위해 $p_\theta(C)$를 잘 design해야 한다.

> Model collapse란 대부분의 item이 하나의 concept에 할당되는 문제이다.

<span class="text-coloar-bold">**Variational inference.**</span>
VAE paradigm에 따라 $\sum_u \text{ln }p_\theta(\textbf{x}_u)$의 lower bound를 최대화하는 $\theta$를 찾아야 한다.

각 user에 대한 $\text{ln }p_\theta(\textbf{x}_u)$의 lower bound는 아래와 같다.

$$
\begin{equation}
\text{ln }p_\theta(\textbf{x}_u) 
  \geq \mathbb{E}_{p_\theta(\textbf{C})}
    [
      \mathbb{E}_{q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})}
        [
          \text{ln }p_\theta(\textbf{x}_u|\textbf{z}_u, \textbf{C})
        ]
      - D_\text{KL}(q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})||p_\theta(\textbf{z}_u))
    ]
\end{equation}
$$

- Lower bound를 구하는 과정은 해당 논문의 supplementary를 살펴보자.

$p_\theta(C)$와 $q_\theta(z_u\|x_u, C)$를 end-to-end로 학습시키기 위해 각각에 대해 Gumbel-Softmax trick과 Gaussian re-parameterization trick을 사용한다.
Training이 끝났을 때 $p_\theta(C)$의 mode를 $C$로 사용하고 $q_\theta(z_u\|x_u, C)$의 mode를 $z_u$로 사용한다.

> 코드 구현상으로 train에서는 soft gumbel softmax로 sampling을 하고, test에서는 mode를 사용하기위해 단순 softmax를 활용한다.

<span class="text-coloar-bold">**Micro disentanglement.**</span>
Micro disentanglement를 수행하는 자연스러운 방법은 latent vector의 각 element들이 statistical independence $q_\theta(z_u^{(k)}) \approx \prod_{j=1}^d q_\theta(z_{u,j}^{(k)}\|C)$ 하도록 만드는 것이다.
운이 좋게도, Eq. 3의 regularization term이 independent하게 만드는 역할을 한다.

$$
\begin{align}
\mathbb{E}_{p_\text{data}(\textbf{x}_u)} [D_\text{KL}(q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})||p_\theta(\textbf{z}_u))]
  &= \mathbb{I}_q(\textbf{x}_u;\textbf{z}_u) + D_\text{KL}(q_\theta(\textbf{z}_u|\textbf{C})||p_\theta(\textbf{z}_u))\\
\nonumber &\text{where } q_\theta(\textbf{z}_u|\textbf{C}) = \int q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})p_\text{data}(\textbf{x}_u)d\textbf{x}_u
\end{align}
$$

- 자세한 증명은 해당 논문의 supplementary를 살펴보자.

만약 prior가 $p_\theta(z_u) \approx \prod_{j=1}^{d^\prime} p_\theta(z_{u,j}\|C)$를 만족하면, KL term인 Eq. 4를 최소화하는 것은 elements들이 independent하도록 만든다.
그리고 mutual information term $\mathbb{I}_q(\textbf{x}_u;\textbf{z}_u)$을 최소화하는 것은 user representation $\textbf{z}_u$에 noise를 없애고 minimal sufficient 정보만 가지도록 만든다.

> Eq. 3의 reconstruction error로 $\textbf{z}_u$의 성능은 보장한다. 동시에 KL term에서 $\textbf{x}_u$과 $\textbf{z}_u$이 서로 독립하도록 만든다. 이는 Information bottleneck principle과 같다.

그래서 저자들은 $\beta$-VAE에서 $\beta \gg 1$로 설정해 regularization term을 강화한다.

> Mult-VAE$^\text{PR}$과 다르게 regularization을 강화하는 이유는 independent를 보장하기 prior와 유사해야 하기 때문이다.

$$
\begin{equation}
\mathbb{E}_{p_\theta(\textbf{C})}
    [
      \mathbb{E}_{q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})}
        [
          \text{ln }p_\theta(\textbf{x}_u|\textbf{z}_u, \textbf{C})
        ]
      - \beta \cdot D_\text{KL}(q_\theta(\textbf{z}_u|\textbf{x}_u, \textbf{C})||p_\theta(\textbf{z}_u))
    ]
\end{equation}
$$

### 2.3. Implementation

![](2.png)

Concept assignment $p_\theta(C)$, decoder $p_\theta(x_{u,i}\|z_u, C)$, prior $p_\theta(z_u)$, encoder $q_\theta(z_u\|x_u, C)$, 그리고 model collapse 방지 전략을 설명한다.

Adam을 사용해 training objective인 Eq. 5를 최대화 하도록 $\theta$를 학습한다.
최적화 해야 할 $\theta$는 다음과 같다. 
- K concept prototypes {$m_k$}$^K_{k=1} \in \mathbb{R}^{K \times d}$
- Decoder에 쓰이는 $M$ items representations {$h_i$}$^M_{i=1} \in \mathbb{R}^{M \times d}$
- Encoder에 쓰이는 $M$ context representations {$t_i$}$^M_{i=1} \in \mathbb{R}^{M \times d}$
- Neural network $f_{nn}:\mathbb{R}^d \rightarrow \mathbb{R}^{2d}$의 parameter

<span class="text-coloar-bold">**Prototype-based concept assignment.**</span>
가장 직관적인 방법은 $p_\theta(C) = \prod_{i=1}^Mp(c_i)$라고 가정한 뒤, 각 categorical distribution $p(c_i)$의 $K-1$개의 parameter를 최적화하는 것이다. 그러나 이 방법을 사용하면 parameter의 개수가 많아지고 sampling의 효율도 떨어진다.

> 확률은 1이므로 $K-1$개의 parameter만 있어도 각 $K$개에 대한 확률을 알 수 있다.

그래서 저자들은 prototype-based implementation을 제안한다.
$K$ concept prototypes {$m_l$}$^K_{k=1}$를 도입하고 decoder에 쓰이는 {$h_i$}$^M_{i=1}$를 재사용한다.
이들을 사용해 one-hot vector $c_i$는 categorical distribution인 $p_\theta(c_i)$로부터 sampling된다고 가정한다.

$$
\begin{equation}
\textbf{c}_i \sim \text{Cat}(\text{Softmax}([s_{i,1};s_{i,2};\dots;s_{i,K}])), \ \ s_{i,k} = \text{Cosine}(\textbf{h}_i, \textbf{m}_k) / \tau
\end{equation}
$$

- $\text{Cosine}(\cdot)$는 cosine similarity를 의미한다.
- $\tau$는 range $[-1/\tau, 1/\tau]$를 조절하는 hyper-parameter이다. 저자들은 $\tau=0.1$로 설정해 더 skew된 distirubiton을 얻는다.

> 비슷한 user에게 소비되는 items는 같은 cluster에 포함이 될 것이다.

<span class="text-coloar-bold">**Preventing mode collapse.**</span>
기존 work에서 similarity 측정으로 많이 사용되는 inner product대신 cosine similarity를 선택했다.
이는 model collapse를 예방하는 데 중요하다.

Inner product를 사용하면, Euclidean space에서 items이 올바르게 cluster 되어 있더라도, 대부분의 items들은 매우 큰 norm 값을 가지는 하나의 concept $m_{k^\prime}, \|\|m_{k^\prime}\|\|_2 \rightarrow \infty$에 할당된다. (Fig. 2e 참고)
Cosine similarity는 normalization 덕분에 이러한 현상을 피할 수 있다.

> 경험적인 관점에서 popular한 item의 norm 값은 그렇지 않은 것보다 상대적으로 더 크다.
> Norm 값이 크다면, inner product는 angle에 덜 영향을 받는다.
> 단순히 inner product를 사용한다면, 대부분의 item이 popular한 item이 속해 있는 concept $m_{k^\prime}$ 을 가진다고 생각될 수 있다.
>
> 따라서 cosine similarity를 사용해 normalization을 진행해 model collapse를 방지한다.

게다가 cosine similarity는 unit hypershphrer에서의 Euclidean distance와 관련이 있다.
그리고 Euclidean distance는 inner product와 비교해서 cluster structure를 추론하는 데 더 적합하다.

<span class="text-coloar-bold">**Decoder.**</span>
Decoder는 user representation $z_u = [z_u^{(1)};\dots;z_u^{(K)}]$과 one-hot concept assignments {$c_i$}$_{i=1}^M$이 주어졌을 때, $M$개의 items 중 user가 가장 클릭할 가능성이 높은 item을 예측한다.

저자들은 user $u$의 $M$개의 item에 대한 선호도를 나타내는 categorical distribution을 $p_\theta(x_{u,i}\|z_u,C) \propto \sum_{k=1}^K c_{i,k} \cdot g_\theta^{(i)}(z_u^{(k)})$라고 가정한다.
그리고 $g_\theta^{(i)}(z_u^{(k)}) = \text{exp}(\text{Cosine}(z_u^{(k)}, h_i)/\tau)$라 정의한다.

> Decoder에는 MLP가 없고, parameter로는 item embedding $h_i$만 필요하다.  
> 그리고 솔직히 $\text{exp}(\cdot)$가 왜 있는지 모르겠다.  
> 이걸 하기 때문에, 제일 마지막에 있는 Algorithm 1 마지막에서 log를 두번이나 해주게 된다. (마지막 log는 Eq. 5에서 나온 것이다.)  
> 이걸 하지 않고, Eq. 5에 따라 log를 한번만 씌어도 성능이 비슷하게 나온다.

이는 item $i$에 대해 $c_{i,k} = 1$일 때, $z_u^{(k)}$이 micro-disentangle 가능하다면, $h_i$도 micro-disentangle이 될 것이라는 것을 의미한다.

> $k$ 번째 concept에 속하는 item $i$의 representation $h_i$이 $z_u^{(k)}$와 inner product로 곱해져 user의 item에 대한 선호도가 구해진다.
> 따라서 $k$ 번째 concept에 대한 user representation $z_u^{(k)}$이 micro-disentangle이 가능하다고 가정하면, $h_i$도 micro-disentangle이 가능하다.
>
> 예를 들어, $z_u^{(k)}$의 0, 1번째 entries가 user가 color 선호도에 대한 정보를 담고 있다면, innder product로 인해 $h_i$의 0, 1 번째 entries는 item color에 대한 정보를 가지게 될 것이다.

<span class="text-coloar-bold">**Prior & Encoder.**</span>
Prior는 micro disentanglement를 위해 factorize ($p_\theta(z_u) \approx \prod_{j=1}^{d^\prime} p_\theta(z_{u,j}\|C)$)가 가능해야 한다.
그래서 저자들은 $p_\theta(z_u)$를 $\mathcal{N}(0, \sigma^2_0 I)$로 설정한다.

> Covariance matrix가 diagonal이므로 $\textbf{z}_u$의 entries이 서로 독립적이다.
> 즉, $\textbf{z}_u$의 distribution은 $d^\prime$개의 univariate Gaussian distirubtion의 곱이 된다. [참고][6]{:target="_blank"}  
> Mult-VAE$^\text{PR}$에서도 prior는 covariance가 diagonal matrix인 Gaussian distribution이다.

Encoder $q_\theta(z_u\|x_u, C)$는 user의 behavior data $x_u$가 주어졌을 때 user의 representation $z_u$를 계산한다.
Encoder는 Mult-VAE$^\text{PR}$처럼 decoder에서 사용되는 item representation {$h_i$}$^M_{i=1}$을 재사용하는 대신 context representations {$t_i$}$^M_{i=1}$를 추가적으로 도입한다.

> Mult-VAE$^\text{PR}$에서는 encoder에서 $\textbf{z}_u$를 구할 때 $\phi$ parameter를 사용하고, decoder에서 $\textbf{x}_u$를 구할 때 $\theta$ parameter를 구한다.
> 
> 이 관점에서 MarcidVAE 안에서는 encoder에서 parameter $t_i$와 MLP($f_{nn}$)를 바탕으로 $\textbf{z}_u$를 구하고, decoder에서 parameter $h_i$를 바탕으로 $\textbf{x}_u$를 구한다.
> 그렇다고 encoder에서 $h_i$가 사용되지 않는 것도 아니다. $\textbf{C}$를 구할 때 필요하다.

저자들은 $q_\theta(z_u\|x_u, C) = \prod_{k=1}^K q_\theta(z_u^{(k)} \| x_u, C)$라고 가정하고, 각 $q_\theta(z_u^{(k)}\|x_u, C)$는 diagonal covariance matrix $\mathcal{N}(\mu_u^{(k)}, \[\text{diag}(\sigma_u^{(k)}\]^2)$를 가지는 multivariate Gaussian distribution이다.
Mean과 standard devitaion은 아래와 같이 neural network $f_{nn}:\mathbb{R}^d \rightarrow \mathbb{R}^{2d}$로 구한다.

$$
\begin{equation}
(\textbf{a}_u^{(k)}, \textbf{b}_u^{(k)}) = f_{nn}\bigg(\frac{\sum_{i:x_{u,i}=+1} c_{i,k}\cdot\textbf{t}_i}{\sqrt{\sum_{i:x_{u,i}=+1} c_{i,k}^2}}\bigg), 
\boldsymbol{\mu}_u^{(k)} = \frac{\textbf{a}_u^{(k)}}{||\textbf{a}_u^{(k)}||_2},
\boldsymbol{\sigma}_u^{(k)} \leftarrow \sigma_0 \cdot \text{exp}\bigg(-\frac{1}{2}\textbf{b}_u^{(k)}\bigg)
\end{equation}
$$

> $\sum_{i:x_{u,i}=+1}$는 $x_{u,i}=1$이면 sum한다는 뜻이다.

지금까지 사용한 cosine similarity와 달리 $K$ concept에서 공유되는 neural network $f_{nn}$는 non-linearity를 capture한다.

Representations을 unit hypersphere에 project하는 cosine similarity의 사용과 일치하도록 mean을 정규화한다. 학습된 representations이 정규화되었으므로 $\sigma_0$은 작은 값(예: 약 0.1)으로 설정되어야 한다.

## 3. Empirical Results

논문을 참고하자.

[1]: https://arxiv.org/pdf/1910.14238.pdf
[2]: https://jianxinma.github.io/disentangle-recsys.html
[3]: https://arxiv.org/pdf/1206.5538.pdf
[4]: https://dl.acm.org/doi/pdf/10.1145/3485447.3512251
[5]: https://arxiv.org/pdf/1412.2007.pdf
[6]: https://stats.stackexchange.com/questions/419671/what-is-a-factorized-gaussian-distribution