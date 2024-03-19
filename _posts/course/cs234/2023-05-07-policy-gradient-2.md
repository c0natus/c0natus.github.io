---
title: "[CS234] 9. Policy Gradient II"
categories: [Course, CS234]
tags: [Reinforcement Learning, Policy Gradient]
img_path: /assets/img/posts/course/cs234/policy_gradient_2/
author: gshan
math: true
---

임시노트 나중에 정리 예정.

이번 강의에서는 policy-based와 Actor-Critic를 살펴본다.
또한 매우 큰 state space에서도 작동할길 원하는 경우에 대해 주로 이야기할 예정이다.
- Actor-Critiec은 policy와 value function이 explicit하게 parameterize되어 있는 것이다.

특히, value-based에 비해 가지는 단점인 data inefficient와 high variance를 완화하는 방법에 대해 얘기할 것이다.

# Desired Properties of a Policy Gradient RL Algorithm

![](1.png)

Gradient로 policy를 평가하고 업데이트하는 과정에서 monotonic improvement가 필요하다.

$$
V^{\pi_1} \leq V^{\pi_2} \leq V^{\pi_3} \leq \cdots
$$

Monotonic improvement를 원하는 이유는 convergence를 보장하고, 실제로 deploy를 결정할 때 도움되기 때문이다.

대부분 value-based method는 value를 추정한 값이기 때문에 monotonic improvement 보장하지 않음.
- Value-based가 monotonic improvement를 보장하는 경우도 있다.

# Recap: MC policy gradient

![](2.png)

지난 강의에서 아주 간단한 policy gradient method를 살펴보았다.

![](3.png)

그리고 더 좋은 gradient의 추정값: less noisy(variance)을 위한 방법인 MC policy gradient algorithms을 살펴보았다.
이는 monotonic improvement를 위한 방법이기도 하다.

> Lecture 3에서 first-visit MC는 unbiased이지만 높은 variance를 가진다고 했다.
> 그리고 every-visit MC는 biased이지만 낮은 variance를 가져 MSE의 성능은 first-visit보다 더 낫다고 했다.
> 하지만 이것도 부족해 bootstrapping(Markov property)을 사용한 TD(Temporal difference)로 SARSA나 Q-learning으로 evaluation을 했다.

MC policy gradient도 temporal 정보를 활용해 아주 약간 variance를 낮췄지만, 여전히 높은 variance를 가진다.
정확히 말하자면, multiple episodes간 returns $G^{(i)}_t$는 높은 variance를 가진다.

따라서 이번 강의에서는 variance를 줄이는 Baseline과 MC의 alternatives에 대해 살펴볼 것이다.

# Baseline

![](4.png)

각 $G^{(i)}_t$에 baseline $b(s)$를 빼준다.
Baseline은 action에 따라 달라지지 않는 한 모든 function이 될 수 있다.

![](5.png)

Baseline으로는 optimal에 근사하는 expected return을 선택한다.
$G_t$가 $b(s_t)$보다 높으면, 그 차이에 비례하게 $a_t$를 고를 확률을 높이는 방향으로 parameter가 update된다.

Normalization 느낌으로 $G_t$의 크기만큼 update하는 것이 아니라 baseline과 차이(residual)만큼 update하므로 variance가 줄어든다.
그리고 그 차이를 advantage $A_t$라고 한다.

Baseline으로 상수값을 줄 수 있지만, 그것은 항상 일정한 값이기 때문에 variance를 줄일 수 없다.
따라서 좋은 baseline을 선택해야 한다.


$$
\hat{A}_t = (G^{(i)}_t - b(s_t))
$$

그리고 이렇게 baseline을 사용할 수 있는 이유는 gradient 계산에 bias를 주지 않기 때문이다.
즉, gradient estimator가 여전히 unbias이다.

$$
\mathbb{E}_\tau[b(s_t)\nabla_\theta \text{ log }\pi_\theta(a_t\|s_t)] = 0
$$

![](6.png)

# Vanilla policy gradient

Baseline을 사용해 'vanilla' policy gradient algorithm을 소개한다.
Baseline function이 parameter $w$를 가진다고 가정한다.

![](7.png)

## Alternatives to target: $R(\tau)$

'Vanilla' policy gradient에 약간의 변형을 줄 수 있다.
- Reward에 MC($G_t$), TD($r_t + \gamma V(s_{t+1})$) 사용 등.
  - MC와 TD의 trade-off 있음. TD를 사용하면 bias를 증가시키지만 variance를 낮춤.
  - Value function을 알 수 있다면, TD 사용 가능.
- Baseline을 VFA로 대체 등.

Baseline의 자연스러운 선택 중 하나는 state value function $b(s_t) = V(s_t)$이다.
또한 target $R(\tau^{(i)})$는 MC나 TD를 통해 구한 value($V$ or $Q$)의 추정값으로 대체될 수 있다.

따라서 만약 $Q$의 추정값으로 target을 대체하고, baseline을 $V$의 추정값으로 대체하면, advantage function을 정의할 수 있다. 
$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$
하지만 true state value를 알 수 없으므로 VFA를 사용한다.
즉, parameter $w$로 추정 $\hat{V}(s_t;w)$한다.
- Policy와 value function이 parameterize되어 있으므로 Action-Critic이라고 볼 수 있다.

Monte-Carlo trajectory samples로 $w$와 $\theta$를 동시에 학습할 수 있다.

Algorithm 2를 살펴보면 gradient를 각각 계산하고 있다.
즉, 효율이 떨어진다.
그래서 batch data를 활용한다.
(surrogate function)

![](8.png)

$\nabla_\theta L(\theta)$를 계산해서 한번에 gradient를 구한다.

![](9.png)

그리고 loss function에 baseline function을 추가해 $\theta$와 $w$에 관해 $L(\theta, w)$의 gradient를 구해 SGD update를 수행할 수 있다.

### N-step estimators

Reward의 추정값으로 MC나 TD 둘 중 하나만 선택해야 하는 것은 아니다.
둘을 혼합해서 사용할 수 있다.

![](10.png)

- $\hat{A}^{(1)}_t$은 TD(0)로 low variance, high bias를 가진다.
- $\hat{A}^{(\text{inf})}_t$은 MC로 bias는 없지만 high variance를 가진다.
- 적당한 k를 골라 적당한 bias와 variance를 가지게 한다.

지금까지 gradient를 추정하기 위해 $R(\tau)$와 baseline으로 선택할 수 있는 options을 살펴보았다.

이제 gradient로 parameter를 update하는 부분을 살펴보자.

# Updating the Policy Parameters Given the Gradient

RL은 supervised learning에 비해 monotonic improvement가 중요하다.
- Supervised learning의 경우 IID를 가정하고 있기 때문에, 성능이 떨어지더라도 다음 업데이트에서 회복 가능하다.
- RL의 경우 성능이 떨어질 경우, 성능이 떨어진 policy로 trajectories를 생성하고 업데이트를 진행한다. 그래서 회복하는 것이 쉽지 않다.

그렇기 때문에 RL에서는 step size를 잘 골라야 한다.
- Step size가 작으면 속도가 느리고, 크면 policy가 나빠지게 될 가능성이 커진다.

## Line search

쉽게 간단히 말하면 여러 step size를 시도해서 가장 성능이 좋은 것을 선택한다.
간단하지만 효율적이지 않다. 그리고 gradient가 안 좋은 경우 수렴 속도가 느려 진다.
다음주 Trust Region Policy Optimization이 이부분을 완화하는 알고리즘이다.

자세한 동작 방식은 아래와 같다. [참고][3]{:target="_blank"}

![](11.png)

$$
\begin{split}

l &= f(x) + t \nabla f(x)^\top\triangle x \\
l^\prime &= f(x) + \alpha t \nabla f(x)^\top\triangle x \\
\end{split}
$$

$f(x + t\triangle x)$는 $f$에서 $\triangle x$ 방향으로 $t$만큼 이동했을 때를 의미한다.
$l$은 point $x$에서 접선의 방향(gradient)로 $t$만큼 이동했을 때를 의미한다.
두 그래프를 살펴보면 $l$이 항상 $f$보다 아래에 있으므로 어떤 $t$가 적당한지 알 수 없다.

Backtracking line search에서는 $l^\prime$을 사용한다.
$l^\prime$은 접선의 기울기에 $\alpha$를 곱한 방향으로 한 step 간 경우를 의미한다.
$l^\prime$은 $f$와 항상 교차하므로 한 step 간 지점에서 $f$가 $l^\prime$보다 아래에 있으면 적당히 잘 갔다고 판단한다.
너무 많이 간 경우에는 되돌아 오기 위해 $t$를 줄이고 $f$가 $l^\prime$보다 아래에 오게 한다.

## Conservative Policy Iteration

![](12.png)

우리가 해야 할 것(objective function $V(\tilde\theta)$)은 value function을 최대화하는 policy parameter를 찾는 것이다.

$$
\tau \sim \pi_{\theta_i}\\
\underset{\theta_{i+1}}{\text{argmax }}V^{\pi_{\theta_{i+1}}}(\tau)
$$

현재 policy로 생성한 trajectories를 통해 가장 높은 state value 값을 갖도록 하는 policy를 예측해야 한다.
이는 off-policy problem으로 생각할 수 있다.

> 우리는 업데이트된 policy가 이전보다 좋다는 monotonic policy를 보장해야 한다.
> 그렇기 위해선 새로운 정책으로 trajectories를 생성하고 state value 값을 측정해야 한다.
> 하지만 이는 비효율적이다.
> 그래서 업데이트된 정책의 state value 값의 lower bound를 구하게 되는데 이를 conservative PI라 하는 것 같다.
> 자세한 것은 다음 강의에서 계속된다.

Conservative PI는 objective function을 advantage로 표현하는 것부터 시작한다.

![](13.png)

- Advantage에 $\pi$가 있는 것은 state value 측정을 $\pi$ policy로 한 것이기 때문이다.
  - Advantage는 위에서 살펴봤듯이 reward를 TD로 측정하고, basline을 state value로 측정한 것이다.
- Action은 $\pi^\prime$로 하는데 그 action으로 선택된 state의 value 값은 $\pi$로 측정한다.

위를 통해 아래와 같이 표현할 수 있다.

![](14.png)

Lemma 4.1.를 통해 $V(\tilde\theta)$를 $V(\theta)$와 advantage를 통해 알 수 있다.
즉, $\tilde\theta$로 trajectories를 sampling하지 않고, $\tilde\theta$의 state value 값을 구하는 것이다.
이는 효율적으로 업데이트된 정책이 monotonic improvement한지 파악할 수 있다.

그렇지만 업데이트된 정책에 대한 stationary(discounted visitation frequencies) $\mu_{\tilde\pi}(s)$를 구할 수 없다는 문제점이 있다.

이를 해결하기 위해 local approximation을 사용한다.

![](15.png)

Stationary를 현재 policy로 측정한 것으로 대체한다.
현재 poliy로 trajectories를 sampling했으므로 계산 가능하다.
Local approximation 자체론 큰 의미가 없고, conservative PI에서 새로운 lower bound를 구할 때 사용된다.

그리고 $L_\pi(\pi) = V(\theta)$이다.
- 이전 policy로 advantage를 구하면 기존에 $V(s_t)$를 추정하기 위해 사용된 state의 value 값을 그대로 사용 $V(s_t) = r_t + \gamma V(s_{t+1})$ 하기 때문에 advantage가 0이된다.
- 업데이트된 policy로 advantage를 구하면 action이 달라지기 때문에 다음 state도 달라져 advantage 값이 달라진다. $V(s_{t+1}) \rightarrow V(s^\prime_{t+1})$

이제 conservative PI에 대한 내용이다.
위의 내용을 모두 합쳐 새로운 policy의 state value의 lower bound를 구할 수 있다.

먼저 현재 policy와 업데이트된 policy를 혼합해서 새로운 policy를 만든다.

![](16.png)

그러면 아래와 같이 새로운 policy value의 lower bound를 얻을 수 있다.

![](17.png)

오늘은 이정도까지만 알아두고 자세한 것은 다음 강의에서 살펴보자.


# References
1. [YouTube, CS234 \| Winter 2019 \| Lecture 9 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}
3. [모두를 위한 컨벡스 최적화: 06-02-02 Backtracking line search][3]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=E-_ecpD5PkE&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=9
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html
[3]: https://convex-optimization-for-all.github.io/contents/chapter06/2021/03/20/06_02_02_backtracking_line_search/