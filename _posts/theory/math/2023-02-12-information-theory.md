---
title: "Information Theory"
categories: [Theory, Math]
tags: [Information Theory, Entropy, KL Divergence, Information Bottleneck]
author: gshan
math: true
---

# Ⅰ. Information

자신의 생각을 다른 사람에게 전하는 방법으로는 여러 가지가 있다.
제일 기본적인 언어부터 시작해 그림, 조각, 소리, 이메일 등을 예로 들 수 있다.
심지어 동물들도 소리를 통해 communication을 한다.
이것들의 차이점은 무엇일까?
아니 그것보다 더 중요한 것은, 공통점은 무엇일까?
이 모든 것들은 정보 (information)의 다른 형태라고 말할 수 있다.
정보는 아주 간단하게 말해, 하나가 다른 하나에게 영향을 미치도록 하는 것이다.

# Ⅱ. Information Theory

Information theory는 이러한 정보의 정량화, 저장 및 전달에 대한 과학적연구이다.
정보의 정량화 즉, 정보량을 측정하기 위해선 공통의 단위를 사용해야 한다. 
단위는 정보를 표현하기 위해 필요한 최소 질문 개수에서 기원했다.
예를 들어, 동전을 10번 던진 결과는 총 10번의 '앞면 인가요?'라는 질문을 통해 파악할 수 있다.
이처럼 어떤 message든 이진수로 간단하게 나타낼 수 있기 때문에, information theory에서는 정보를 이진수로 나타내고 정보량을 bit로 나타낸다.
질문에 대한 긍정을 1, 부정을 0이라고 가정한 뒤 아래 예시를 살펴보자.

<span class="text-color-bold">**6개의 알파벳 전달.**</span>
'A 인가요?' 등의 질문이 계속 될 수 있지만, 이것은 최소의 질문이 아니다.
최적의 질문은 후보 알파벳을 반으로 줄이는 것이다.
알파벳 F를 찾는다고 하자.
1. 첫 질문은 '해당 문자가 N보다 사전 순으로 앞에 있는가?'가 될 수 있고 답변은 1이 된다.
2. 유사하게, 두 번째 질문은 '해당 문자가 G보다 앞에 있는가?'
3. '해당 문자가 D보다 앞에 있는가?'
4. '해당 문자가 D인가?'
5. '해당 문자가 E인가?'

그러면 모든 질문에 대한 답변은 11000이 된다.
즉, F라는 정보를 11000으로 표현할 수 있고 총 5bit가 필요하다.
알파벳 하나를 찾기 위해 질문은 4 또는 5번만 하면 된다.
평균적으로 $\text{log }_2 26 \approx 4.7$번 질문을 통해 하나의 알파벳을 알 수 있고,
6개의 알파벳을 알려면 $6 * 4.7 = 28.2$번 질문을 해야 한다.
따라서 6개의 알파벳의 평균적인 정보량은 28.2 bits라고 할 수 있다.

이런 추상적인 개념을 처음 정립한 사람이 Ralph Hartley (랠프 하틀리)이다.

$$
H = n \text{ log }s
$$

- $H$: information
- $n$: number of symbols
- $s$: number of different symbols available

이전에 살펴본 예시에서 $n = 6, s=26$이므로 $H=6 \times \text{ log }26 \approx 4.7$이다.
물론 예시처럼 모든 알파벳의 확률이 동일하지 않을 수 있고, 그러면 정보량(최소 질문 개수)이 달라진다.

이제 일반적인 상황으로 생각해보자.
확률 분포 (probability distribution) $p(\mathcal{X}=x)$에서 random variable $\mathcal{X}$의 $x$에 대한 information은 무엇일까?
단순히 생각해보면 $x$의 확률이 높을수록 적은 질문이 필요할 것 같다.
그리고 그것이 맞다.

극단적인 예로 빨강색 사탕 98개, 파랑색 사탕 1개, 초록색 사탕 1개가 있고 1개의 사탕을 무작위로 뽑았다고 하자.
우리는 사탕의 분포를 알고 있으므로, 어떤 색의 사탕인지 알기 위한 평균 질문 횟수를 낮추기 위해선 '빨강색 사탕인가요?'라는 질문은 제일 먼저 해야 한다.
따라서 빨강색 사탕인지 알기 위해선 1번의 질문만 필요하고, 빨강색 사탕은 다른 사탕보다 적은 정보량을 가지게 된다.

위의 내용을 바탕으로 information $\mathcal{I}$를 수학적으로 아래와 같이 표현한다.

$$
\mathcal{I}(x) \triangleq -\text{ log }p(x)
$$

Information을 최소 질문 횟수 외에도 $x$에 대한 놀라움(surprise) 또는 불확실성(uncertainty)이라고 말한다.
확률이 낮은 $x$ 사건이 '실제로 일어날까?'라는 불확실성은 매우 높을 것이고, 그것이 실제로 발생하면 매우 놀랄 것이다.

[Entropy][5]{:target="_blank"}는 사건 $x$가 아닌 random variable $\mathcal{X}$에 대한 불확실성(uncertainty)를 측정하는 것과 관련있다.
즉, 사건 $x$가 가지는 정보량이 아닌 dataset $\mathcal{D} = (x_1, \cdots, x_n)$의 정보량을 뜻한다.


# Ⅲ. Machine Learning

Machine learning (ML)은 기본적으로 information processing에 관한 것이다.
ML에선, 어떤 information이든 궁극적으로 한 beliefs의 set이 다른 beliefs set으로 update되는 규모를 정량화하는 방법을 필요로 한다.
그것이 KL divergence이다.
Entropy와 mutual information은 KL divergence의 특수한 경우라고 할 수 있다.

Information theory를 활용한 application으로 data compression (or source coding)과 error correction (or channel coding)이 있다.
Data compression은 정보를 저장할 때, data의 중복성을 제거하여 무손실 방식(예: ZIP 파일) 또는 손실 방식(예: MP3 파일)으로 보다 간결하게 표현한다.
Error correction은 전화, 위성과 같은 noisy한 채널을 통해 정보가 전송될 때 error에 robust한 방식으로 data를 인코딩한다.

Data compression과 error correction 방법은 모두 data의 정확한 probabilistic model에 의존하다는 것으로 밝혀졌다.
Compression의 경우, sender가 가장 자주 발생하는 data vector에 더 짧은 codewords를 할당하여 공간을 절약할 수 있는 probabilistic model이 필요하다.
Correction의 경우, receiver가 전달 받은 noisy message를 possible messages에 대한 prior distribution과 결합하여 가장 가능성이 높은 source message를 추론할 수 있는 probabilistic model이 필요하다.

> prior distribution은 $p(\theta)$, 가장 가능성이 높은 source message를 구하는 것은 $p(\text{input and label}\|\theta)$로 maximum likelihood를 구하는 것으로 생각할 수 있다.

Probabilistic machine learning과 information theory는 서로에게 유용하다.
실제로 Bayesian machine learning은 uncertainty (불확실성)을 representing하고 줄이는 것과 관련이 있으며 근본적으로 정보에 관한 것이다.
이 방향과 관련있는 개념이 information bottleneck이다.

# References

1. [Khan Academy, What is information theory][1]{:target="_blank"}
2. [Khan Academy Labs, Measuring information, YouTube][2]{:target="_blank"}
2. [Wiki - Information theory][3]{:target="_blank"}
3. [Kevin P. Murphy. (2023). Probabilistic Machine Learning: Advanced Topics, MIT Press][4]{:target="_blank"}

[1]: https://www.khanacademy.org/computing/computer-science/informationtheory/info-theory/v/intro-information-theory
[2]: https://www.youtube.com/watch?v=PtmzfpV6CDE
[3]: https://en.wikipedia.org/wiki/Information_theory
[4]: https://probml.github.io/pml-book/book2.html
[5]: https://c0natus.github.io/posts/entropy/
