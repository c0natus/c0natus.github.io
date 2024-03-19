---
title: "[CS234] 10. Policy Gradient III"
categories: [Course, CS234]
tags: [Reinforcement Learning, Policy Gradient]
img_path: /assets/img/posts/course/cs234/policy_gradient_3/
author: gshan
math: true
---

임시노트 나중에 정리 예정.

## Recap

먼저 강의에서 대충 설명한다.
깊게 살펴보지 말고, 그냥 이런 것도 있구나를 알고 가면 될 것 같다.

지난 시간에 이어서 policy-based RL을 살펴본다.
Policy-based RL은 주어진 환경에서 좋은 결정을 내리는 parameterized policy를 학습하는 것이다.
DNN과 softmax로 policy를 represent할 수 있고, gradient methods로 parameter를 업데이트한다.

Value based agent에 비해 환경에 숨겨진 정보가 있거나, 환경 자체가 변하는 경우에도 유연하게 대처할 수 있다.
Policy의 손실 함수를 정의할 수 없기 때문에 가장 큰 value function을 가진 policy를 찾는다.

이전 강의에서 [Vanilla policy gradient 알고리즘][3]{:target="_blank"}에서 variance를 줄여 더 좋은 estimator를 얻는 방법을 살펴보았다.
- Critic을 통해 value function($V$ or $Q$)를 계산해 TD 적용.

Monotonic improvement론, 현재 policy의 [advantage로 업데이트된 policy의 value값을 나타내는 방법][4]{:target="_blank"}을 알아 보았다.
이번 강의에서는 gradient method 방법에서 monotonic improvements를 보장하는 방법에 대해 더 자세히 살펴 볼 것이다.
- $\theta$로 $\theta^\prime$의 value를 예측하는 문제로, off-policy learning이다.

## Local approximation

![](1.png)

위 식에서 업데이트된 policy를 기반으로 만들어진 state distribution $\mu_{\tilde{\pi}}$를 모른다.
이를 알려면 업데이트된 policy를 실제로 실행시켜봐야 한다.

![](2.png)

그래서 local approximation을 사용해 새로운 objective function $L$을 얻는다.

> 직관적으로 말하자면, 매우 작은 step-size라면 둘의 변화량이 크지 않을 것이기 때문에 근사식이 가능하다.

이렇게 할 수 있는 이유는 $L_\pi(\tilde\pi)$와 $V(\tilde\theta)$의 taylor first approximation의 값이 같기 때문이다.

> $L_\pi(\pi)$를 증가시키는 $L_\pi(\tilde\pi)$로의 정책 변화가 step-size가 충분히 작다면, $V(\theta) \leq V(\tilde\theta)$라는 뜻이다.  
> 하지만 문제점은 step-size가 얼마나 작아야 하는 지 모르고, 너무 작으면 학습 속도가 떨어진다는 점이다.

Local approximation을 사용한 objective function을 최적화하는 것이 실제로 value function을 monotonic improvement하게 만드는지 모른다.
그래서 local approximation을 통해 구한 새로운 policy의 value 값의 lower bound를 최적화한다. (Conservative PI)

## Conservative Policy Iteration

Surrogate objective를 최적화하면서 얻을 수 있는 new policy의 quality의 lower bound를 찾아보자.

![](3.png)

먼저, 현재 policy와 업데이트된 policy를 섞어 새로운 policy를 만드는 것을 고려해보자.
Policy의 업데이트를 conservative(보수적)으로 한다.
- Conservative
: $\alpha$ 값이라는 비율을 도입하고, step-size를 줄이면서 업데이트 한다.

![](4.png)

그러면 mixture policy의 lower bound를 보장할 수 있다.


그렇지만, mixture policy가 아닌 일반적인 stochastic policy의 potential performance에 대한 lower bound를 얻어야 한다.

## Lower Bound in General Stochastic Policies

![](5.png)
- $\epsilon$은 계산하기 어렵다. (특히 continuous할 때)
- 그래서 아래에서 더 쉽게 계산할 수 있는 것으로 대체한다.

어떤 특수한 objective function $D^{\text{max}}_\text{TV}$를 사용하면 일반적인 stochastic policy의 lower bound를 얻을 수 있다.
이는 distance of total variation를 maximization하는 것이다.
Distance of total variation의 정의는 아래와 같다.

$$
D_\text{TV}(\pi_1(\cdot|s), \pi_2(\cdot|s)) = \underset{\text{a}}{\text{max}} (\pi_1(\cdot|s) - \pi_2(\cdot|s))
$$

이를 maximization한다는 것은 모든 states에 대해, 두 policies가 가지는 가장 큰 차이를 내는 action의 확률값을 구하겠다는 의미이다.
일반적으로 이를 계산하는 것은 어렵다.

![](6.png)

그렇기 때문에 위위와 같이, total variation의 제곱이 KL divergence보다 크거나 작다는 사실을 사용한다.

## Guaranteed Improvement

그럼 이제 이것으로 monotonic improvement을 보장할 수 있는 방법을 알아보자.

![](7.png)

업데이트된 policy와 현재 policy의 lower bound를 비교해 monotonic improvement를 알 수 있다.
이는 Minorization-Maximization (MM) algorithm의 한 유형이다.

> 모든 state의 value 값이 이전보다 크거나 같다...

## Trust Regions

Lower bound를 구할 때 $\epsilon$을 계산하는 것이 힘들었다.
Trust region policy gradient algorithm을 이용해 partical하게 lower bound를 구해보자.

![](8.png)

KL term 앞에 곱해져 있는 것을 C(상수)로 치환한다.
C는 penalty coefficient이다.
만약 penalty coefficient를 theorem과 같이 $4\epsilon\gamma/(1-\gamma)^2$을 사용하면, 현재 policy와 매우 동떨어진 action을 탐색할 수 없기 때문에 매우 작은 step size가 선택될 것이다.
- Gradient는 현재 policy의 value 값과 가장 가까우면서도 좋은 estimate이다.
- 하지만, 수렴 속도가 느리기 때문에 실용적으로는 좋지 않다.

그래서 실용적으로 좀 더 큰 step-size를 가지게 할 수 있는 한 방법은 업데이트된 정책과 현재 정책 사이의 KL term에 constraint를 주는 것이다.
이를 trust region constraint on step sizes라고 한다.

![](9.png)

Lower bound에서 KL term을 빼는 것보단, $\delta$를 통해 현재 정책과 업데이트된 정책의 거리를 제약한다.
이를 trust region이라고 부른다.
Trust region은 학습을 할 때 수렴의 방향성을 벗어나지 않는 영역이라고 할 수 있다.
Trust region이 너무 커서 업데이트된 정책과 현재 정책의 KL term이 커지는 것을 용인하면, 최적 정책으로의 수렴은 보장할 수 없다.

## From Theory to Practice

![](10.png)

먼저 stationary(discounted visitation frequencies) $\mu_\text{old}$를 바로 구할 수 없다.
State가 continuous, infinite일 수 있으므로 sampling하고 re-weight한다.

![](11.png)

두번째로 action이 continuous할 수 있으므로 이전과 마찬가지로 sampling을 통해 summation을 추정한다.
이때, 현재 policy로 sampling한 action $a \sim q$을 사용해야 하므로 importance sampling을 활용한다.

![](12.png)

마지막으로 advantage를 Q 값으로 바꾸면 된다.
해당 substitution은 위에서 살펴 봤던 optimization 문제의 solution을 바꾸지 않는다.

![](13.png)

최종적으로 우리가 최적화해야 하는 sampling 기반의 objective를 얻을 수 있다.

## Selecting the Sampling Policy

![](14.png)

마지막으로 sampling을 하는 방법이다.
Standard approach는 초기 상태 분포로부터 초기 상태를 설정하고 현재 policy로 trajectory를 얻는 것이다.
현재 policy에 따라 시뮬레이션을 하기 때문에 behavior policy q는 현재 policy가 된다.
이 trajectory를 바탕으로 모든 state-action의 Q를 구하게 된다.
그러나 이는 하나의 roll out이기 때문에 추정값의 분산이 크다는 단점이 존재한다.
Vine은 이러한 단점을 보완하기 위해 고안되었다.

일단 standard와 똑같이 초기 상태 분포로부터 초기 상태를 설정하고, 현재 policy에 여러 개의 trajectories를 생한다.
그리고 이 trajectories로부터 N개의 상태를 고른다.
즉, $S_1 ~ S_N$을 고르고 이 각각의 상태에 대해 behavior policy q에 따른 k개의 행동을 취한다.
한 개의 roll out을 분석하는게 아니라 한 state에서 가지를 여러개 쳐서 그에 대한 roll out을 분석한다.
이렇게 하면 분산이 줄어든다.

## References

1. [YouTube, CS234 \| Winter 2019 \| Lecture 10 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}
3. RL Study PPT file by [minjin][5]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=o_i5F1zGPLs&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=10
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html
[3]: https://c0natus.github.io/posts/policy-gradient-2/#vanilla-policy-gradient
[4]: https://c0natus.github.io/posts/policy-gradient-2/#conservative-policy-iteration
[5]: https://velog.io/@lm_minjin