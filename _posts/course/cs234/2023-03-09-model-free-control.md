---
title: "[CS234] 4. Model Free Control"
categories: [Course, CS234]
tags: [Reinforcement Learning, SARSA, Q-learning]
img_path: /assets/img/posts/course/cs234/model_free_control/
author: gshan
math: true
---

임시 노트 나중에 정리 예정

World에 대한 model(reward/dynamic model of environment)이 없을 때 MDP.
즉 action이 추가됨 -> action decision.

Learning from experience. -> learning model.

이전 강의에선 정해진 policy가 얼마나 좋은지 evaluation함.
이걸 다르게 생각하면 좋은 policy를 학습할 수 있다.

Model이 주어졌을 땐, Q function argmax action 찾기, value function max action 찾기 했다.

MDP model is unknown but can be smapled: recsys  
MDP model is known but it is computationally infeasible to use directly, except through smapling: 바둑

이번 강의에선 On/off-policy learning에 대해서 다룬다.

On-policy learning
: Direct experience  
Learn to estimate and evaluate a policy from experience obtained from following that policy

Off-policy learning
: Learn to estimate and evaluate a policy using experience gathered from following a different policy.

# Recap

![](1.jpg)

Model이 주어졌을 때, 즉 P를 알 때.

Monotonic policy improvement(Bellman backup)로 converge 보장.
- finite states, action을 가질 때. -> finite policies($$ \|A\|^{\|S\|} $$)를 가져야하기 때문.

![](2.jpg)

Model-free policy evaluation은 이전 강의에서 했음.

지금은 line 5를 model-free로 해야 함.
Model를 estimate하는 방법도 있지만 오늘 강의에선 model-free에 집중 함.

![](3.jpg)

$$ Q^\pi(s,a)=R(s,a)+\gamma\sum_{s^\prime \in S}P(s^\prime\|s,a)V^\pi(s^\prime) $$를 model-free way로 해야 함.
Model-free policy iteration으로 $$ Q^\pi(s,a) $$를 얻고 $$ \pi $$를 update함.


# MC for On Policy $Q$ Evaluation

![](4.jpg)

On policy에서 Monte Carlo(MC)로 $Q^\pi$를 evalution한다.
Value evalution에서의 MC와 유사하지만 action에서 차이가 있다.

Policy $\pi$로 episode를 sampling한다.
그 다음 모든 time step $t$의 return $G_{i, t}(s,a)$를 계산한다.
Value evalution과 같이 first 또는 every visit으로 $Q^\pi(s,a)$ 추정치를 update한다.

![](5.jpg)

주어진 추정치로 policy를 update한다.

![](6.jpg)

요약 하면, $Q$를 추정하고 추정된 $Q$로 $\pi$를 update한다.

여기에 몇가지 주의 사항이 있다.
1. Policy $\pi$가 deterministic이거나 positive probability을 가지는 action $a$가 없다면 argmax를 계산할 수 없다.
	- Probability가 negative일수가...?
	- Policy가 deterministic이라도 sampling된 histroy는 다를 수 있다. 왜냐하면 state가 stochastic이기 때문. $P(s^\prime\|s,a_)$
	- Policy가 dterministic이면 안 되는 이유는 특정한 stata에 대해 하나를 제외하고 다른 action을 볼 수 없기 때문이다. 즉, exploration이 안 된다. 그렇기 때문에 argmax를 할 수 없다.
2. Dynamic model을 몰라 $Q^\pi$를 측정하는 것이기 때문에, model-base같이 line 5가 실제로 policy를 monotonically improve하는지 명확하지 않다.
	- 그래서 실제로 improve하는 experience(interleaving of policy)가 필요하다.
	- 그리고 exploration도 추가적으로 생각해야 한다.

## QnA

- 강의에선 가능한 모든 action을 미리 알고 있다고 가정한다.
- Policy는 world가 markov하다고 가정한다. -> MDP에선 world가 markov한다.

![](7.jpg)

MRP vs MDP

# Policy Evaluation with Exploration

$Q^\pi$ 추정치로 정말 improve가 되는지 확인하는 가장 직관적인 방법은 policy $\pi$를 따르는 모든 $(s,a)$쌍을 시도해 policy improvement가 monotonic operator가 되도록 하는 충분히 좋은 $Q^\pi$를 보장한다.

그렇다면 가능한 모든 $(s,a)$를 방문했다고 어떻게 확신할까?
$\epsilon$-greedy policies를 활용한다.

> 근사치를 구하는 것 같다.

# $\epsilon$-greedy policies

새로운 action에 대한 exploration과 현재의 지식에 의한 exploitation의 균형을 맞추는 것이다. (by random)

Trajectory에 다양성을 더욱 많이 추가하는 것이다.

Finite action $\|A\|$가 있다고 하자.

그러면 state-action value $Q(s,a)$에 관한 $\epsilon$-greedy policy는 다음과 같다.

$$
\pi(a|s) =
	\begin{cases}
		\underset{a}{\text{argmax }}Q(s,a) &\text{ w(with) prob. } 1 - \epsilon &\rightarrow \text{ best action} \\
		a &\text{ w prob. } \frac{\epsilon}{|A|} &\rightarrow \text{ one of all actions}
	\end{cases}
$$

$1 - \epsilon$ 확률로 추정 $Q$에 따라 최적의 action을 선택한다.

그리고 $\frac{\epsilon}{|A|}$ 확률로 actions 중 하나를 선택한다.
- argmax action도 포함.

![](8.jpg)

화성 탐사를 예로 살펴보자.

위와 같이 모든 state에 대해 action $a_1$과 $a_2$d에 대한 rewards가 있다.
$\gamma = 1$로 설정한다.

현재 policy는 모든 state s에 대해 $\pi(s) = a_1$이고 $\epsilon = 0.5$라고 하자.

여기서 $\epsilon$-greedy 없이 sampling하면 trajectory에서 action은 모두 $a_1$일 것이다.
하지만 중간에 $a_2$가 있다.

이를 통해 update된 정책은 그림과 같이 $s_2$에서는 $a_2$를 선택하게 된다.

Fisrt visit이 아니라 TD를 사용한 것은 SARSA라고 부른다.

![](9.jpg)

$\epsilon$-greedy가 monotonic improvement라는 증명이다.
- $Q^{\pi_{i+1}} \geq V^{\pi_i}$

![](10.jpg)

Policy가 update되었다. 따라서 policy improvement theorem에 의해 $V^{\pi_{i+1}} \geq V^{\pi_i}$이다. 
- $V^{\pi_{i+1}}$는 update된 policy로 평가하는 것.

이는 전적으로 주의 사항 2번째를 해결해주는 것이다.

뭐 어쨌든 exploration이 필요한 이유는 $Q$가 정확한 값이 아니라 추정치이기 때문이다.
Model이 주어지면 action을 했을 때의 $Q$가 정확히 얻어지지만, model이 없으면 그렇지 않다.

##  Greedy in the limit of exploration (GLIE)

![](11.jpg)

Monotonic improvement가 증명되었다. 그러면 수렴을 위해선 어떤게 필요할까?
1. 바로 episode가 무한하면 모든 $(s,a)$ 쌍들을 무한이 방문된다.
2. 그로 인해 behavior policy가 greedy policy로 수렴한다는 것이다.

- Behavior policy는 사용한 policy이고 greedy policy는 argmax $Q$의 policy이다.
- Greedy policy는 주어진 $Q$ function을 argmax하는 policy
- $Q$ function의 argmax action을 가져가니 greedy policy라고 표현한 것 같다.

> 이렇게 보면 $\epsilon$-greedy도 설명이 왜 그런 이름이 붙었는지 알 것 같다.

실현하는 방법 중 간단한 것은 $\epsilon$를 시간이 지날수록 decay 즉 0으로 보내는 것. $\epsilon_i = \frac{1}{i}$
- 어쨌든 $\epsilon$만큼 random으로 action을 고르기 때문에 첫번째 조건을 만족한다.
- 그리고 $\epsilon_i$가 작아지기 때문에 두번째 조건도 만족한다.

즉, 수렴의 충분조건이다.
- 조건 만족 -> 수렴.

이 조건들 아래에 즉, GLIE일 때 MC와 TD를 이용해 optimal policy/value의 수렴을 보일 것이다.

# MC Online Control

![](12.jpg)

MC로 $Q$를 추정하고, $\epsilon$-greedy로 policy $\pi$를 update한다.
그러면 sampling 즉, policy는 반드시 argmax $Q$를 따르지 않게 된다.

![](13.jpg)

GLIE를 만족하는 MC control은 optimal $Q$로 수렴한다.

# TD for Control (SARSA)

![](14.jpg)

$V$ 평가에서 TD는 every-visit MC에서 episode가 끝나기 전 sampling되어 있는 history를 기반으로, 현재 정해진 policy로 인해 다음 state로 가게될 경우의 value(bootstraping)를 사용해 현재 policy의 $V$를 평가.
- MC + DP

![](15.jpg)

마찬가지로 $Q$를 평가할 때 TD로 평가하고 $\epsilon$-greedy로 update한다.
이를 SARSA라고 한다. 
이름따라 (state, action, reward, next state, next action) tuple로 evaluation 및 improve한다.
- 현재 policy에 따라 action을 선택하고 reward와 netxt state를 관찰한다.  
그리고 현재 policy에 따라 next action을 선택한다.

보통 on policy이다.

![](16.jpg)

MDP의 finite state/action에서 SARSA가 converge하기 위해선 위와 같은 2가지 조건이 필요하다.

1. GLIE여야 한다. 
즉, 시간이 지날수록 greedy에 더 weight를 준다.  
하지만 GLIE를 만족하기 어려운 domain이 있다.
갑자기 헬리콥터 날개가 부러지면 greedy로 다음 action을 선택할 수 없다.
이것은 episode문제일 수도 있다.
따라서 100개 헬리콥터의 episode를 고려할 수 있다.
즉, data를 모으는 것이다.
이와 같은 문제 그리고 더 좋은 exploration을 하는 방법을 다음에 알아보겠다.
2. step size $\alpha$가 1이면 SARSA는 converge하지 않는다.
왜냐하면 과거의 정보를 모두 잊기 때문이다.
$\alpha=0$이면 update를 하지 않는다.
일반적으로 $\alpha$는 수렴을 보장할 만큼 천천히 감소해야 한다.  
일반적인 예로 $\alpha_t = \frac{1}{t}$가 있다.  
하지만 경험적으로 2번 조건은 사용하지 않는다.
수렴의 충분조건이기 때문에.
2번을 만족하지 않아도 수렴할 수 있다.

앞에서 first visit MC는 GLIE만 만족하면 수렴이 보장되었다.
차이점은 first visit MC는 unbiased method라는 것이다.

# Q-Learning

![](17.jpg)

현재 policy에서 정책을 가져오는 것이 아니라 argmax로 가져온다.
따라서 off-policy algorithm으로 볼 수 있다.

> 많은 negative reward가 있는 domain이라면, SARSA가 처음에 잘 동작할 수 있다.  
> Q-learning은 optimism을 더 자주 하는데, 이것이 exploration 측면에선 더 좋다. 
하지만 negative reward에 고통받는다.
왜냐하면, penalty를 학습하지 못하기 때문이다.

![](18.jpg)

안전한 길은 SARSA, optimal 길은 Q-learning
-> domain에 따라 알고리즘 선택.

![](19.jpg)

Q-learning 알고리즘은 위와 같다.
Next action을 선택할 필요가 없다.
그리고 해당 state에 대한 policy만 update한다.

Q-learning은 MC보다 훨씬 느리다.
지금 한 행동이 나중에 엄청 큰 reward를 얻더라도 Q-learning으론 학습하지 못한다.

![](20.jpg)

Q-learning에선 epsilon-policy에 따라 next action a-t+1을 고르는 것이 아니라 argmax로 고른다. 그래서 optimal Q를 얻을 때엔 GLIE의 두 번째 조건을 고려하지 않아도 된다. (epsilon이 decay될 필요 없다.)

SARSA에선 epsilon-policy에 따라 next action a-t+1을 고르기 때문에 policy가 수렴해야 한다. 


특정 정해진 policy에 대해 $Q$의 수렴은 첫번째와 같다.
정해진 policy이므로 GLIE는 필요없다.

이제 policy의 수렴을 위해선 GLIE가 필요하다.
추정된 $Q에 exploration이 필요하다.

# Maximization Bias

![](21.jpg)

## Double Learning

Maximization bias를 다루기 위한 방법으로 double learning이 있다.

Q-learning을 2개 쓰는 double q-learning에선 unbiased Q estimator 2개에서 1개는 value evalution에 1개는 policy improvement(decision making)에 사용한다.
이것이 unbiased estimator를 가능하게 한다...

> 하나를 쓰면 biased 되니까 2개로 나누다.
> 하나는 value evaluation 잘되게 하나는 policy improvement 잘하게... 그냥 직관적으로 설명하면.

![](22.jpg)

위는 알고리즘인다.

![](23.jpg)

위의 것은 결과이다.
때때로 성능에 차이가 많이 난다.

Q-learning은 maximization bias에 고통받는다.

Dobule은 도움이 된다.
overhead는 적다.

# References
1. [YouTube, CS234 \| Winter 2019 \| Lecture 4 \| Emma Brunskill][1]{:target="_blank"}
2. [CS234: Reinforcement Learning Winter 2019][2]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=j080VBVGkfQ&list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u&index=4
[2]: https://web.stanford.edu/class/cs234/CS234Win2019/index.html