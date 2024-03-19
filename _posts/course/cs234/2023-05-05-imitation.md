---
title: "[CS234] 7. Imitation Learning"
categories: [Course, CS234]
tags: [Reinforcement Learning, Imitation Learning]
img_path: /assets/img/posts/course/cs234/imitation/
author: gshan
math: true
---

임시노트 나중에 정리 예정.

RL algorithm에서 해결해야 하는 것은 optimization, delayed consequences, exploration, generalization이 있다.
이전 강의에서 SARSA, Q-learning 등 으로 최적의 policy를 찾는 optimization을 배웠고, VFA로 generalization하는 법을 배웠다.

이번 강의에서는 이들을 효율적으로 하기 위한 방법을 살펴본다.

일반적은 MDP에서는 좋은 policy를 학습하기 위해 많은 samples이 필요한데, 일반적으로 충분한 sample을 모을 수 없다.
예를 들어, 자율주행 자동차의 경우 사고를 무수히 많이 겪어야 운전이 제대로 될 수 있을 것이다.
즉, reward function을 얻기 어렵고 많은 exploration이 필요하다.

효율적인 exploration을 위해 문제의 구조나 추가적인 지식을 사용하는 방법 등이 있을 것이고, 이번 강의에서는 human(expert)의 behavior를 활용해 학습하는 과정(imitation learning)을 살펴 볼 것이다.

# Imitation Learning

이때가진 sparse한 rewards로 부터 policy를 학습했다.
이 방식은 많은 sample이 필요하므로 좋은 policy를 얻지 못할 것이다.
Reward function을 manually design해도 되지만 오랜 시간이 소요된다.
따라서 imitation learning을 사용한다.

Imitation learning이란 demonstration을 통해 학습하는 것이다.
즉, expert(teacher)로부터 demonstration trajectory를 만들어 implicit하게 제공된 reward로 policy를 학습한다.
Demonstration trajectory와 유사한 behavior를 생성하도록 reward를 특정화하는 것보다 expert가 선호되는 behavior를 demonstrate하는 것이 더 쉬울 때 Imitation learning이 유용하다.

> 한마디로 reward나 policy를 특정화해 (demonstration trajectory 없이) expoert와 유사한 behavior를 생성하도록 만드는 것보다, expert 도움을 받는 것이 더 쉬울 때 유용하다.  
> 예를 들어, 여러 번 사고를 내서 운전을 잘 하게 만드는 것보단, 운전을 잘 하는 사람들 데려와 직접 운전시켜 agent가 그것을 모방하도록 학습시키는 것이 더 효율적이다.

Imitation learning의 input은 다음과 같다.
- State, action space  
- Transition model $P(s^\prime\|s,a)$  
- No reward function $R$  
- Set of one or more teacher's demonstrations $(s_0, a_0, s_1, a_1, \dots)$ where actions are drawn from the teacher's policy $\pi^*$

Imitation learning의 방법으로 다음과 같은 것들이 있다.
- Behavioral Cloning
: Supervised learning을 통해 expert의 policy를 직접 배운다.
- Inverse RL
: Demonstration을 통해 reward function $R$을 얻는다.
- Apprenticeship Learning via Inverse RL
: Inverse RL로 얻은 $R$로 좋은 policy를 생성한다.


# Behavioral Cloning

Policy를 supervised learning으로 학습한다.
먼저 policy class를 설정하고, expert trajectory의 state를 input으로 action을 output으로 두고 agent를 학습시킨다. 
하지만, compounding error를 가진다는 문제가 있다.

![](1.png)

Supervised learning에서 i.i.d. 가정이 표준이기 때문에 temporal structure를 무시한다. 즉, error들이 서로 독립니다.

![](2.png)

하지만 RL의 state space는 i.i.d. 가 아니다. 그렇기 때문에 RL 관점에서 error는 서로 독립적인 것이 아니라 compounding된 것이다. 즉, episode 동안 error가 축적된다. 따라서 training data에서 발생한 오차가 계속해서 누적되어 증폭해 정책의 성능을 저하시킨다.
특히, expert가 겪지 못한 state를 만드는 action이 선택 ($\epsilon$-greedy 등) 되었을 때, error는 RL quadratically 증가한다.
이를 방지하는 방법으로 DAGGER(Data Aggregation)이 있다.

## DAGGER: Dataset Aggregation

![](3.png)
_Algorithm 1: DAGGER_

아주 간단한다. 만약 trajectories에 없는 state가 생기면 expert에게 어떤 action을 취할 지 알려달라는 것이다.
해당 방법의 문제는 당연히 expert가 해당 state에 대한 label을 제공할 수 있어야 한다는 것이다. (때때로 real time으로)
이는 사실상 불가능에 가깝워, 다른 방법보다는 연구가 되지 않고 있다.

# Inverse Reinforcement Learning (IRL)

Inverse optimal control이라고도 불리는 IRL은 expert demonstration에 기반해 policy를 직접적으로 학습하는 것이 아니라, reward function을 추론한다.

> reward function을 통해 최적의 policy를 학습하는 것이 아니라 최적의 policy를 통해 reward function을 추론한다.


물론 teacher의 policy가 optimal하다는 전제가 있어야 한다.
Teacher의 policy가 optimal이고 data가 충분히 존재한다고 했을 때, teacher의 policy를 설명할 수 있는 reward function은 여러 개가 존재할 수 있다는 문제가 있다.
간단하게 예를 들어, 모든 reward = 0이라고 한다면 해당 reward function은 어떠한 policy도 설명할 수 있게 된다. 모든 policy가 optimal policy가 된다.

> 모든 reward = 0이라고 해도 expert의 policy는 optimal이고, 모든 reward=1이라고 해도 expert의 policy가 optimal이므로 reward function이 unique하지 않다.

이를 해결하는 방법으로 Linear Feature Reward IRL이 있다.

## Linear Feature Reward Inverse RL

Reward가 feature의 linear combination으로 represent할 수 있다고 하자.

$$
R(s) = w^\top x(s)
$$

- $w \in \mathbb{R}^n, x:S \rightarrow \mathbb{R}^n$

여기서 $w$를 demonstration을 통해 찾아야 한다.
Policy $\pi$에 대한 value function은 다음과 같다.

![](4.png)

- 모든 정책에 대해 $w$는 같으므로 expectation 밖으로 나온다.
- $\mu(\pi\|s_0=s) \in \mathbb{R}^n$은 policy $\pi$에서의 state features $x(s)$의 discounted weighted frequency이다.
- 즉, 학습시킨 weight vector $w$에 policy $\pi$ 하에 평균적으로 자주 등장하는 state feature의 값을 곱해주다는 의미이다.

![](5.png)
- $R^*$: optimal reward function

해당 reward function으로 가장 큰 value 값을 가지는 것이 optimal policy가 된다.
그러므로 expert’s의 demonstrations이 optimal이라면, 위의 식을 만족하고 다음과 같은 $w^*$를 찾을 수 있다.

![](6.png)

따라서 expert policy가 다른 policies보다 더 좋은 성능을 가지게 하는 reward function의 parameter를 찾을 수 있다.
그러면 자주 등장하는 state feature의 reward가 왜 높아야 할까?
직관적으로 optimal하다고 전제했으니까, 자주 보이는 state는 높은 reward를 가져야 한다.

> 왜냐면 optimal한 action을 통해 얻은 state이므로 그 state의 value 값은 높다.

유사하게 거의 보이지 않는 state는 일반적으로 낮은 reward 값을 가진다.
자율 주행 자동차를 생각해보면, 왼쪽으로 코너를 돌 때, 차선을 잘 맞춰서 도는 일이 빈번할 것이므로 그러한 state는 당연히 높은 reward값을 가질 것이다.
반면, 왼쪽으로 코너를 돌 때 급하게 왼쪽으로 핸들을 트는 일은 거의 없을 것이므로, 그 state의 경우는 reward가 낮게 측정되는 것이 일반적이다.
이런 방식을 사용하면, 모든 경우가 같은 reward 값을 가지게 되는 일이 없어지므로, 제기되었던 문제를 어느 정도 해결한다.

# Apprenticeship Learning

Apprenticeship Learning은 좋은 policy를 생성하는데 IRL로 복원된 reward을 사용하는 것이다.
즉, expert policy 이외의 좋은 policy를 찾는 것이다.
만약 expert policy와 비슷한 value 값을 가지는 policy를 찾으면 optiaml policy 수준의 policy를 찾아냈다고 할 수 있다.

![](7.png)

그러면 $\|\|w\|\|_\infty \leq 1$인 모든 $w$에 대해 Cauchy-Schwartz inequality에 의해 다음을 만족한다.

![](8.png)

이렇게 하면, 원래 reward function이 뭐였느냐에 관계 없이, 학습으로 도출된 reward function을 사용하더라도 충분히 optimal policy에 가까운 policy를 얻어낼 수 있다.

![](9.png)
_Algorithm 2: Apprenticeship Learning via Linear Feature IRL_

이것을 algorithm으로 나타내면 위와 같다.
그런데 이제 거의 안쓰인다.

# Ambiguity

Optimal policy를 생성할 수 있는 reward function들이 많다.
지금까지 이를 완화하기 위한 algorithm을 살펴보았지만, 완벽히 해결하지는 못한다.
또한 reward function을 구했더라도 그 reward function에 최적인 policy도 여러 개가 있을 수 있다.
그러면 그러한 것들 중 어떤 것을 골라야 하는가? 바로 그 문제이다.
이러한 문제를 해결하기 위해 많은 연구가 활발히 진행되고 있다.
주요 논문으로는 Maximum Entropy IRL과 Generative adversarial imitation learning이 있다.

- Maximum Entropy IRL


# References
1. [YouTube, CS234 \| Winter 2019 \| Lecture 7 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=V7CY68zH6ps&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=7
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html