---
title: "[CS234] 5. Value Function Approximation"
categories: [Course, CS234]
tags: [Reinforcement Learning, VFA]
img_path: /assets/img/posts/course/cs234/value_function_approx/
author: gshan
math: true
---

임시노트 나중에 정리 예정.

이전 강의에선 sequential decision-making의 경우에서 좋은 policy를 학습하였다.
- Optimization: 최적의 policy 찾기 (MC, SARSA, Q-learning)
- Delayed consequences: state value/ action-state value function 사용
- Exploration: $\epsilon$-greedy 사용

이때, finite state/action으로 value를 추적하기 위한 computation과 memory가 feasible하다.
하지만 real world에서는 state와 action이 매우 많거나, discrete하지 않고 continuous할 수 있기 때문에 tabular로 representation하기 불충분하다.
이런 경우를 다루기 위해 generalization이 필요하다.
- 이전에 경험을 본 적 없는 state action pair에 대한 value도 generalization한다.


# Value Function Approximation (VFA)

State/state-action value function을 parameterized function으로 표현.

![](1.png)
_Generalization_

Input이 들어오면 이전 강의처럼 table에서 찾는 게 아니라 parameter $w$를 가진 function을 거치게 된다.
Function은 polynomial, DNN 등이 될 수 있다.

VFA의 motivation은 모든 dynamics, reward model, state/state-action value, policy를 explicit하게 저장하지 않는다는 것이다.
즉, 일반화할 수 있는 compact한 representation을 원한다.

당연히 true $V^\pi$를 알 수 있는 oracle이 존재한다면, VFA는 필요가 없다.
따라서 model-free value function approximation에 대해 살펴 볼 것이다.

> State/State-Action value 값은 policy를 update할 때 사용된다.

SARSA나 Q-learning과 다른 점은 VFA는 본 적 없는 state/state-action value도 추정이 가능하다는 점이다.

![](2.png)

먼저 feature vectors가 필요하다.

예를 들어, 로봇의 센서가 $180^\circ$ 범위로 물체와의 거리를 측정한다고 하자.
그러면 센서의 1 단위의 degree가 $x_i(s)$를 의미하게 될 것이다.
즉, 총 180개의 features를 가지는 vector가 된다.
이들을 모두 합해 현재 앞에 있는 물체와의 거리를 알 수 있다.
이 예시는 markov property를 가지지 않는다.
만약 로봇이 건물의 복도를 지나간다고 하면, 양 옆은 막혀있지만 전방은 막혀있지 않은 상태가 지속될 것이다.
이를 partial aliasing이라고 한다.
Partial aliasing이 있으면 markov가 아니다.
하지만 state representation이 history를 모두 포함하게 된다면 markov가 된다.
즉, 각 observation이 alias되더라도 전체 state representation은 그렇지 않게 된다.

> Aliasing is an effect that causes different signals to become indistinguishable from each other during sampling.

어쨌든 features를 적절히 고르는 것은 중요하다.
즉, feature engineering이 필요하다.
이는 시간이 많이 소요되므로 DNN을 사용한다.

# Linear VFA with An Oracle

![](3.png)

$\mathbb{E}_\pi$는 policy $\pi$에 대한 states $s_1, \cdots, s_n$ state value의 square loss 평균값을 의미한다.
Oracel $V^\pi$를 알 수 없으므로 MC나 TD를 이용해 $G_t$ 값으로 parameter를 update한다.

> 그렇다면 $w$는 policy $\pi$마다 달라야 할 것 같다.  
> 일단 policy가 바뀌지 않는다고 가정하고 즉, control이 없다고 가정하고 VFA를 한다.  

# Monte Carlo VFA

Oracle은 알 수 없다.
Return $G_t$는 timestep $t$에서 episode 끝까지의 discount sum이다.
$G_t$는 $V^\pi(s_t)$의 unbiased noisy sample이다.
즉, $V^\pi(s_t) = \mathbb{E}[G_t\|s_t]$이다.

그래서 MC에서 $G_t$를 $V^\pi(s_t)$의 estimator로 사용해 supervised learning을 한다.

![](4.png)
_Algorithm 1: First Visit Monte Carlo Linear Value Function Approximation for Policy Evaluation_

## Convergence Guarantees for MC VFA

특정 policy $\pi$를 가지는 MDP로 정의된 Markov Chain은 state $s$에 대한 stationary distribution인 $d(s)$로 수렴한다.
$d(s)$는 state $s$에 머물게 될 time 비율로, $\sum_s d(s)=1$이다.

$$
d(s\prime) = \sum_s\sum_a\pi(a|s)p(s\prime|s,a)d(s)
$$

Value function의 수렴은 $d(s)$ 관한 VFA의 mean squared error와 관련 있다.

$$
\text{MSVE}(w) = \sum_{s\in\mathcal{S}}d(s)(V^\pi(s) - \hat{V^\pi}(s;w))^2
$$

- $\hat{V^\pi}(s;w) = \textbf{x}(s)^\top w$

MC policy evaluation으로 구한 VFA의 parameter $w_{\text{MC}}$는 MSVE를 최소화하는 것으로 수렴한다.

$$
\text{MSVE}(w_{\text{MC}}) = \underset{w}{\text{min }}\sum_{s\in\mathcal{S}}d(s)(V^\pi(s) - \hat{V^\pi}(s;w))^2
$$

Linear VFA이므로 analytic solution을 얻을 수 있다.
(Matrix 크기가 크면, 오래 걸린다.)

> MC는 Markov assumptions이 아니다.

$$
\underset{w}{\text{argmin }}\sum_{i=1}^N(G(s_i)-\textbf{x}(s_i)^\top w)^2
$$

- 위 식을 미분해서 0이되는 $w = (X^\top X)^{-1}X^\top G$이다.
- $G$는 $N$ 개 state의 returns이다.
- $X$는 $N$ 개 state에 관한 feature matrix이다.

#  TD(0) VFA

Tabular 환경에서 temporal difference는 bootstrapping과 sampling을 통해 $V^\pi$를 approximate하고 update한다.

$$
V^\pi(s) = V^\pi(s) + \alpha(r + \gamma V^\pi(s^\prime) - V^\pi(s))
$$

VFA에서 TD(0)이 MC와 다른 점은 $G_t$ 대신 $r + \gamma V^\pi(s^\prime)$을 target으로 사용한다 supervised learning이라는 것이다.
따라서 mean squared error는 다음과 같다.

$$
J(w) = \mathbb{E}_\pi[(r_j + \gamma\hat{V^\pi}(s_{j+1}, w) - \hat{V^\pi}(s_j;w))^2]
$$

이를 최소화하는 algorithm은 아래와 같다.

![](5.png)
_Algorithm 2: Temporal Difference Linear Value Function Approximation for Policy Evaluation_

Target $r + \gamma V^\pi(s^\prime)$이 biased 추정값 이지만, global optimum에 가깝게 수렴한다.

## Convergence Guarantees for TD(0) VFA

$$
\text{MSVE}(w_{\text{TD}}) \leq \frac{1}{1-\gamma}\underset{w}{\text{min }}\sum_{s\in\mathcal{S}}d(s)(V^\pi(s) - \hat{V^\pi}(s;w))^2
$$

- $\frac{1}{1-\gamma}$는 TD의 bootstrapping로부터 전파된 error이다.
  - Bootstrapping을 사용하기 때문에 bias가 생긴다.

# Control using VFA

State value function 대신 state-action value function을 고려해보자.

$$
Q^\pi \approx \hat{Q^\pi}(s,a;w)
$$

VFA를 사용해 policy evaluation을하고 $\epsilon$-greedy로 policy를 update한다.
계속 policy가 변경되기 때문에 더이상 stationary distribution이 아니다.

> Function approximation, bootstrapping, sampling, off-policy learning을 포함하고 있어 학습이 unstable하다.

## State-Action Value function approximation with an Oracle

True state-action value $Q^\pi(s,a)$와 그것의 근사값 사이의 mean-squared error는 다음과 같다.

$$
J(w) = \mathbb{E}_\pi[(Q^\pi(s,a) - \hat{Q^\pi}(s,a;w))^2]
$$

State와 action을 나타내는 features를 $\textbf{x}(s,a)$라 하고, linear VFA를 사용한다고 하자.
SGD를 사용한 parameter $w$ update는 다음과 같다.

$$
\begin{split}
w &\leftarrow w - \alpha\nabla_wJ(w)
& w - \alpha(Q_\pi(s,a) - \hat{Q^\pi}(s,a;w))\textbf{x}(s,a)
\end{split}
$$

MC 방법에서는 위 식에서 target을 return $G_t$로 대체한다.

$$
w - \alpha(G_t - \hat{Q^\pi}(s,a;w))\textbf{x}(s,a)
$$

SARSA에서는 target을 TD target으로 대체한다.

$$
w - \alpha(r + \gamma\hat{Q^\pi}(s^\prime, a^\prime; w) - \hat{Q^\pi}(s,a;w))\textbf{x}(s,a)
$$

Q-learning에서는 max TD target으로 대체한다.

$$
w - \alpha(r + \gamma\underset{a^\prime}{\text{max }}\hat{Q^\pi}(s^\prime, a^\prime; w) - \hat{Q^\pi}(s,a;w))\textbf{x}(s,a)
$$

## Convergence of Control Methods with VFA

VFA를 사용하는 TD는 objective funciton의 gradient를 따르지 않는다. (Chapter 11에서 자세히 살펴보자.)
이전 강의에서 Bellman backups는 contraction임을 보여줘 특정 point로의 수렴성을 증명했다.
하지만, VFA를 사용하면 Bellman backups는 expansion할 수 있다.
아래의 Baird 예시를 통해 살펴보자.
- Bellman operator $B$: $\|\|BV-B^\prime\|\|_\infty \leq \|\|V-V^\prime\|\|$
- VFA는 $BV$를 다른 space로 projection하는 것과 같다. ($BV \rightarrow PBV$)
  - P 때문에 더 이상 value function space에 있지 않을 수 있다.

![](7.png)
_Challenges of Off Policy Control: Baird Example_

Data를 생성하는 behavior policy $\mu$와 update 하려는 target policy $\pi$가 있다고 하자.
$\mu$로 sampling한 data(s,a,r,s$^\prime$)에서 $\pi(s) \neq a$인 action이 있으면 data를 버리고 update하지 않는다.
즉, dashed action이면 data를 버린다. 
문제는 behavior policy와 target policy의 data의 분포가 다르기 때문에 TD에서는 diverge할 수 있다는 점이다.

> Function approximation, bootstrapping, off-policy가 동시에 있는 경우 수렴하기 어렵다.

![](6.png)
_Table 1: Summary of convergence of Control Methods with VFA. (Yes) means the result chatters around near-optimal value function._

Chapter 11에서 converge하는 TD-style algorithms을 다룬다.
그리고 nonlinear VFA에 대한 convergence도 연구되고 있다.
중요한 점은 algorithm의 수렴 여부가 아니라 어떤 point로 수렴하는지이다.

> Objective function과 feature representation이 중요하다.

# References
1. [YouTube, CS234 \| Winter 2019 \| Lecture 5 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=buptHUzDKcE&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=5
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html