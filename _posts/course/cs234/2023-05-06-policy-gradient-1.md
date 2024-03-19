---
title: "[CS234] 8. Policy Gradient I"
categories: [Course, CS234]
tags: [Reinforcement Learning, Policy Gradient]
img_path: /assets/img/posts/course/cs234/policy_gradient_1/
author: gshan
math: true
---

임시노트 나중에 정리 예정.

# Introduction to Policy Search

VFA로 state value와 action-state value를 추정하였고, 이를 통해 SARSA나 Q-learning으로 최적의 policy를 찾는다.
그리고 효율을 위해서 optimal policy의 value-function을 찾은 뒤, 그것을 활용해 좋은 policy를 얻는 imitation learning을 사용했다.

이번 강의에서는 아래와 같이 policy-based 방식을 사용해 policy를 직접 parameterize하는 법을 알아보자.
- Policy-based 방식은 가장 높은 value function $V^\pi$을 가지는 policy를 직접적으로 찾는 것이다.
- Parameter $\theta$로 만든 policy로 얻은 value 값이 $V^\pi$


$$
\pi_\theta(a|s) = \mathbb{P}[a|s;\theta]
$$

Look-up table로 표현되는 deterministic한 policy가 아니라 parameterize된 stochasitc policy를 고려하는 것이다.
좋은 policy를 찾기 위해서는 2 가지가 고려되어야 한다.
1. Good policy parameterization
: our function approximation and state/action representations must be expressive enough
2. Effective search
: we must be able to find good parameters for our policy function approximation

![](1.png)

RL은 model-free에서 policy-based, value-based에 따라 위의 그림과 같이 나뉜다.
- Value Based
: Learn value function, implicit policy (e.g. $\epsilon$-greedy).
- Policy Based
: No value function, learn policy,
- Actor-Critic
: learn value function/policy. 자세한 내용은 나중에 살펴보자.
  - Actor가 policy이고 critic이 value이다.

Value-based RL보다 policy-based RL이 가지는 몇 가지 장/단점은 아래와 같다.

Advantages
: - Better convergence properties. (local optima에 수렴을 보장, value-based는 보장 못함.) 이는 model-free/model-based 그리고 computation 방식에 따라 논쟁이 일어날 수 있다. Chapter 13에서 자세히 살펴본다.
- Effectiveness in high-dimensional or continuous action spaces, e.g. robotics. Continuous action spaces위한 방법 중 하나(Gaussian policy)를 아래에서 다룬다.
- Ability to learn stochastic policies. (아래에서 살펴본다.)

Value-based RL은 일반적으로 추정된 value가 가장 높도록 만드는 deterministic한 policy를 얻는다. 
이는 stochastic한 policy를 얻지 못하게 막는다.

> Tabular setting에서 policy는 deterministic하고 optimal한 policy가 있지만, real-world에서는 그렇지 않은 경우가 더 많기에 stochastic policy 중요하다.

Disadvantages
: - They typically converge to locally rather than globally optimal policies, since they rely on gradient descent.
- Evaluating a policy is typically data inefficient and high variance.
 
대부분의 policy-based RL은 global optima에 수렴하지 않고, local optima에 수렴하는 경우가 많다.
또, 일반적으로 policy-based RL은 SGD를 사용하기 위해 많은 sample을 필요로한다. (data inefficient)

# Stochastic policies

그렇다면 왜 stochastic policy가 필요할까?

간단한 예시로 가위 바위 보를 agent를 학습한다고 하자.

Deterministic한 최적의 policy는 다음과 같다.
처음에는 3개 중 아무거나 고정으로 낸다고 가정하자.
그 다음부터는 이전에 상대방이 낸 것을 이기기 위한 action을 선택하게 된다.
예를 들어 상대/<span class="text-color-blue">RL</span> sequence는 '가위/<span class="text-color-blue">가위</span> → 보/<span class="text-color-blue">주먹</span> → 보/<span class="text-color-blue">가위</span> → 주먹/<span class="text-color-blue">가위</span> → 가위/<span class="text-color-blue">주먹</span> → ... '가 될 것이다.
만약 상대가 RL 방법론을 알아차리면 agent 승률은 0%가 된다.
최적의 policy는 그냥 uniform하게 3개 중 하나를 선택하는 것이다.

또 다른 예시로는 Aliased gridworld가 있다.

![](2.png)
_Figure 1: In this partially observable gridworld environment, the agent cannot distinguish between the gray states._

돈을 찾는 것이 목표이고, 해골에 도달하면 죽는다고 하자.
또한 해골이나 돈에서 시작되는 경우는 없으며 agent의 action은 '상, 하, 좌, 우'이고 오직 벽의 유무로만 action을 선택한다고 하자.
- Grid의 색은 agent가 알 수 없다.

만약 deterministic이라면 위쪽과 왼쪽에 벽이 있으면 오른쪽으로 이동하고, 위쪽에만 벽이 있으면 아래로 이동하고, 위쪽과 오른쪽에 벽이 있으면 왼쪽으로 이동하게 된다.
하지만, 회색 grid가 문제이다.
만약 회색 grid에서 "왼쪽" policy를 정한다면, 왼쪽 위 흰 grid 또는 그것에 인접한 회색 grid에서 시작하는 경우에는 목표(돈)에 도달하지 못하게 될 것이다.
"오른쪽" policy로 정해도 마찬가지로 목표에 도달하지 못하는 상황이 발생한다.

![](3.png)
_Figure 3: A stochastic policy which moves E or W with equal probability in the gray states will reach the goal in a few time steps with high probability._

따라서 회색 grid에서는 좌/우를 uniform하게 선택하는 stochastic policy가 필요하다.
Value-based RL은 stochastic policy를 얻을 수 없지만, policy-based RL은 optimal stochastic policy을 학습할 수 있다.

# Policy optimization

Parameter $\theta$로 policy $\pi_\theta(a\|s)$를 측정할 때, 최적화를 위해 그것의 quality를 측정할 수 있어야 한다.
즉, objective function을 정의해야 한다.
Episodic environments(terminal state 까지)의 경우 가장 자연스러운 방법은 policy의 start state의 expected value인 start value를 측정하는 것이다.

$$
J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[v_1]
$$

Continuing environments(terminal state 없음)에서는 policy의 평균값인 average value를 사용할 수 있다.

$$
J_{avV}(\theta) = \sum_sd^{\pi_\theta}(s)V^{\pi_\theta}(s)
$$

- $d^{\pi_\theta}(s)$는 policy $\pi_\theta$에서 유도된 Markov chain의 stationary 분포이다.
  - Stationary 분포는 이전 상태와 무관하게 고정된 분포를 의미한다. 즉, 해당 policy에서 state s일 확률은 고정되어있다는 뜻이다.

또는, time-step 마다 평균 reward를 구한다.

$$
J_{avR}(\theta) = \sum_sd^{\pi_\theta}(s)\sum_a\pi_\theta(s,a)R(s,a)
$$

본 강의에서는 간단하게 대부분 episodic한 경우만 다루지만, 쉽게 non-episodic(continuing / infinite horizon)인 경우로 확장시킬 수 있다.
또한 discount $\gamma = 1$인 경우에 대해 살펴볼 것인데 이것도 쉽게 일반적인 $\gamma$로 확장시킬 수 있다.

## Optimization methods

위에서 정의한 objective function으로 policy-based RL을 optimization 문제로 다룬다.
- 가장 높은 $V^{\pi_\theta}$를 갖는 parameter 찾기.

Gradient-free optimization methods가 있지만 최근에는 일반적으로 gradient-based methods를 사용한다.

Gradient-free 방식은 objective function의 gradient를 사용할 필요가 없어 policy parameterization이 미분 불가능해도 된다.
즉, 어떠한 policy parameterization에도 적용시킬 수 있다.
그리고 쉽게 병력화할 수 있다.
Gradient-free 방식은 baseline으로 사용하기 좋고, 때로는 잘 작동한다.
하지만, 일반적으로 reward의 temporal structure를 무시하기 때문에 매우 sample inefficient하다.
- Update는 전체 episode의 총 reward만 고려하고, trajectory에서 각 state에 대한 서로 다른 reward로 나누지 않는다.

Gradient-free 방식은 여기까지만 알아보고 Gradient-based 방식을 자세히 살펴보자.

# Policy Gradient

Episodic MDP에서 $V(\theta) = V^{\pi_\theta}$를 $\theta$에 대해 maximization하길 원하는 objective function이라고 하자.
그러면 policy gradient algorithm을 활용해  local maximum 값을 찾을 수 있다.

![](4.png)

## Computing the gradient

![](5.png)

- $u_k$는 $k \in [1,n]$번째 component가 1인 one-hot vector이다.

각 dimension마다 gradient를 구할 때, 미분 불가능한 지점에서도 gradient를 구하기 위해 finite differences(유한 차분법)을 사용한다.
Finite differences란 말 그대로 (f(x+b) - f(x+a)) / (b - a)에서 b - a가 0이 아닌 값으로 구한 미분 계수를 의미한다.
미분할 수 없는 정책에 대해 작동한다는 이점이 있고, AIBO robot gait에서 잘 작동한 방법이다.
하지만, policy graident를 구하기 위해 총 n 번의 evaluation을 해야 하므로 비효율적이다.
그리고 true policy gradient가 아니기 때문에 일반적으로 noisy를 포함하고 있다.

### Analytic gradients

이제 analytical하게 policy gradient를 계산하자.
오직 episodic에 대해서만 가능하다.

여기서는 finite differences가 아니기 때문에 policy $\pi_\theta$가 미분 가능해야 하고, policy gradient $\nabla_\theta \pi_\theta(s,a)$를 계산할 수 있다고 가정한다.


Objective function $V(\theta)$를 한 episode의 expected rewards라고 하자.

![](6.png)

- $P(\tau;\theta)$는 policy $\pi_\theta$로 생성된 trajectories에 대한 probability이다.
- $R(\tau)$는 한 trajectory의 rewards 총합이다.
- Goal은 위의 식을 최대화하는 policy parameters $\theta$를 찾는 것이다.

이 objective function은 위에서 살펴보았던 discount $\gamma = 1$을 가지는 start value $J_1(\theta)$와 동일하다.
만약 수학적으로 policy gradient $\nabla_\theta \pi_\theta(a\|s)$를 계사할 수 있다면, $\theta$에 대해 objective function의 gradient를 아래와 같이 계산할 수 있다.

![](7.png)

식을 이렇게 바꾸는 것에는 2가지 이유가 있다.
- $\tau \sim \pi_\theta$의 expectation $\mathbb{E}[\dots]$ 형태로 gradient를 얻을 수 있고, 이 값은 MC와 유사한 방법으로 approximate(empirical estimate)할 수 있다.

![](8.png)

- Dynamics model이 필요하지 않고, $\nabla_\theta\text{ log }P(\tau^{(i)};\theta)$를 계산하는 것이 $P(\tau^{(i)};\theta)$를 직접 계산하는 것보다 쉽다.

![](9.png)

> 위의 식을 직관적으로 말해보자면, reward가 높은 trajectory가 있으면 그것을 더 높은 확률로 생성할 수 있는 policy를 찾는 것이다.

$\nabla_\theta \text{ log } \pi_\theta(a_t\|s_t)$는 score function이다.
이를 합쳐서 policy gradient를 구하면 아래와 같다.

![](10.png)


$V(\theta)$를 최대화 해야 하므로 gradient ascent방법을 사용해야 한다.

$$
V(\theta) \leftarrow V(\theta) + \nabla_\theta V(\theta)
$$

Parameter $\theta$로 생성된 trajectories의 평균 reward가 커지도록 parameter를 update한다.
즉, 현재 trajectory의 특정 state에서 action A를 취하는게 reward가 높으면, A를 취하는 확률을 높이는 식으로 parameter update한다.
이 방법은 reward function이 discontinuous, unknown이거나 state space가 discrete해도 유효하다.
Gradient ascent로 $\pi_\theta$를 optimize하는 자세한 algorithm은 다음 강의에서 살펴보자.

## The policy gradient theorem

Likelihood ratio approach를 아래와 같이 policy gradient theorem으로 episodic뿐만 아니라 continuous environments에도 일반화할 수 있다.

![](11.png)


강의에선 자세히 다루지 않고 있다.
식으로 살펴보면 total episode reward $R(\tau)$가 Q-value로 대체하였다.
간단히 episodic environment가 continuous로 일반화될 수 있는 사실만 알고 넘어가자.

아래에서 temporal structure를 이용하면 해당 theorem과 비슷하게 $R(\tau)$를 future returns $G_t$로 대체되는 것을 확인할 수 있다.

## Using temporal structure of rewards for the policy gradient

![](12.png)

앞에서 살펴본 likelihood ratio score function policy gradient는 MC와 유사하게 unbiased이지만 매우 noisy하다. (variance가 크다.)
Variance를 줄이기 위한 방법으로 temporal structure과 baseline 등이 있다.
다음 강의에서 더 다양한 tricks를 살펴보자.

MC의 variance를 줄이기 위해 bootstrapping을 사용했다.
여기서도 유사하게 $R(\tau)$ 대신 covariate를 가지는 어떤 것으로 대체하여 variance를 줄인다.
먼저 temporal process를 활용하는 방법을 살펴보자.

$R(\tau)$는 아래와 같이 trajectory 동안 얻은 reward의 총합으로 나타낼 수 있다. ($\gamma = 1$)

$$
R(\tau) = \sum_{t=1}^{T-1}R(s_t, a_t) = \sum_{t=1}^{T-1}r_t
$$

그리고 single reward term $r_{t^\prime}$에 대해 다음과 같은 식을 유도할 수 있다.

![](13.png)

이들을 모두 활용해 아래와 같이 나타낼 수 있다.

![](14.png)

식 12에서 13으로 가는 것은 식을 풀어서 살펴보면 이해하기 쉽다.
예시로 3 step($T=2$)를 살펴보자.

Eq. 12는 아래와 같이 풀어 쓸 수 있다.

![](15.png)

이를 regrouping하면 아래와 같다.

![](16.png)

Main idea는 특정 time step $t$에서 policy의 선택이 $t$ 이후 단계에서 받는 reward에만 영향을 미치고 이전 단계에서 받은 reward에는 영향을 미치지 않는다는 것이다.

즉, time step $t$에서 parameter를 update할 때 이전의 reward가 가지고 있는 noise를 포함하지 않으므로 variance가 약간 줄어든다.

이들을 모두 한번에 표현하면 아래와 같다.

![](17.png)

![](12.png)

두 식을 비교해보면 Reward가 time과 관련 없이 평균적인 값을 사용했지만, temporal structure를 이용해 time이 reward와 관련 있도록 만듦으로써 variance를 약간 줄인다.

> 어쨌든 단순히 전개해서 바꾼 것인데 왜 $V(\theta)$의 variance가 줄어들까...?  
> True 값이 아니라 평균 값으로 추정한 것이기 때문에 시간을 고려하면 조금이라도 variance가 낮아진다는 걸까?


# REINFORCE: A MC policy gradient algorithm

REINFORCE는 RL policy gradient algorithms 중 가장 흔한 것이다.

![](18.png)

바로 위에서 본 것과 같은 형태이다.

# Differentiable Policy Classes

그렇다면 이제 필요한 것은 $\text{log }\pi_\theta(a\|s)$를 미분해야 한다.
대부분 policy classes로 softmax, gaussian, neural network를 선택한다.

## Discrete action space: softmax policy

Discrete action spaces에서는 보통 softmax function을 사용한다.
Linear combination으로 state s에서 action a를 고를 확률을 나타낸다고 하자.

$$
\phi(s,a)^\top \theta
$$

- $\phi$: state, action features
- $\theta$: weights

![](19.png)

그러면 score function은 아래와 같다.

![](20.png)

##  Continuous action space: Gaussian policy

Continuous action spaces에서는 보통 Gaussian policy를 사용한다.
평균이 state feature의 linearcombination $\mu(s) = \phi(s)^\top \theta$이고 variance $\sigma^2$는 고정되어 있다고 가정하자.
- Variance도 parameterize될 수 있다. (VAE처럼)

Policy 즉, action은 Gaussian $a \sim \mathcal{N}(\mu(s), \sigma^2)$에서 sampling된다.

Score function은 아래와 같다.

![](21.png)



# References
1. [YouTube, CS234 \| Winter 2019 \| Lecture 8 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=8LEuyYXGQjU&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=8
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html
