---
title: "Batch Normalization"
categories: [Theory, AI]
tags: [Batch Normalization]
img_path: /assets/img/posts/theory/ai/batch_norm/
author: gshan
math: true
---

ResNet에 있는 batch normalization에 대해 알아보자. Batch normalization은 internal covariate shift 문제를 완화하기 위해 사용된다. 먼저 internal covariate shift가 뭔지 알아보자.

# Ⅰ. Internal Covariate Shift

Covariate가 뭘까? Covariate(공변량)은 하나의 변수로서 종속변수(output data)에 대하여 독립변수들(input data, weights, etc.)이 공통적으로 함께 공유하는 변량(잡음)을 의미한다.

![](1.jpg)

DNN에서 작은 변화여도 weight에 가중되어 쌓이면 결과(hidden node, output node)가 많이 달라지는데 이러한 현상을 internal covariate shift라고 한다.

![](2.jpg)

그렇다면 이게 왜 문제가 될까? 이것은 위의 그림과 같이, 이전 train dataset과 현재 train dataset의 분포가 차이가 나서 training이 잘 안되거나, input data에 대해 normalization을 하지 않아 training이 잘 안되는 것과 유사하다.

이런 것을 해결하기 위해 initialization을 신중하게 하거나 small learning rate를 사용할 수 있지만 신중한 initialization을 하기 어렵고 small learning rate는 속도를 저하시킨다는 문제점이 있다.

따라서 이러한 internal covariate shift를 해결하기 위해서 batch normalization을 사용한다. 

# Ⅱ. Batch Normalization

![](3.jpg)

Batch norm은 neural network와 hyperparameter의 상관관계를 줄여 hyperparameter 탐색을 더 쉽게해준다. 그리고 deeper한 neural network도 쉽게 training(빠르게 수렴)될 수 있도록 해준다.
- input data에 대해 normalization하는 것과 유사한 효과이다.

Batch norm의 수식은 아래와 같다.

$$
\begin{align*}
\mu_{\mathcal{B}} &\leftarrow \frac{1}{m}\sum_{i=1}^mz_i \\
\sigma_{\mathcal{B}}^2 &\leftarrow \frac{1}{m}\sum_{i=1}^m(z_i - \mu_{\mathcal{B}})^2 \\
\hat{z_i} &\leftarrow \frac{z_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}
\end{align*}
$$

- $m$은 batch size.
- $z_i$는 batch norm을 적용하지 않은 activation function의 입력값.
- $\hat{z_i}$은 batch norm을 적용한 activation function의 입력값.
- $\epsilon$은 분모가 0이 되는 상황을 방지.

Activation function의 출력값($a_i$)에 대해 batch norm을 적용할 수도 있지만, 입력값($z_i$)에 대한 batch norm이 실제론 더 많이 사용된다.

Batch norm이후 모든 $z^{(l)}$은 평균이 0이고 분산이 1인 정규분포를 따르게 된다. 이것은 결코 좋지 못한 선택이다. 즉, $z^{(l)}$ 마다 분포가 조금씩 달라져야 한다. (scale and shift)

$$
\tilde{z_i} \leftarrow \gamma\hat{z_i} + \beta
$$

- $\gamma$와 $\beta$는 learnable parameter이다.   
    만약 $\gamma$가 $\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}$이고 $\beta$가 $\mu_{\mathcal{B}}$라면 $\tilde{z_i} = z_i$가 된다.

![](4.jpg)

ReLU 입장에서 모두 $N(0, 1)$을 따르게 되면 activation이 0이 되는 경우가 자주 발생한다. 이를 조금씩 이동시켜 activation이 활성화되도록 해야 한다.


![](5.jpg)

Sigmoide의 경우 거의 linear한 구간의 input 값만 가지게 되어서 비선형성이 사라지게 된다. 따라서 scale과 shift를 통해 비선형성을 살려줘야 한다.

# Ⅲ. Real Reason Why Batch Norm Works

[Santurkar et al., How Does Batch Normalization Help Optimization?, NeurIPS 2018][5]{:target="_blank"}에 따르면 batch norm이 효과적인 주요 이유는 loss landscape를 smooth하게 만들기 때문이라고 한다. Loss landscape가 smooth해지면 다음과 같은 효과가 있다.
- Loss gradient가 predictable하고 stable해진다.
- Flat regions이나 local minimum으로 가는 위험 없이 큰 learning rate를 사용할 수 있다.
- 훨씬 더 빠르게 training할 수 있고 hyper-parameter 선택에 민감하지 않게 된다.

![](6.jpg)

Loss landscape는 parameter에 따른 손실 값의 변화를 나타낸다. Nerual network를 훈련하는 것은 loss landscape를 미끄러져 내려가는 과정이라고 해석할 수 있다.

Loss landscape를 smooth하게 만드는 algorithm은 batch norm과 비슷한 효과를 가져왔다.



# References
1. [YouTube: Normalizing Activations in a Network (C2W3L04)][1]{:target="_blank"}
2. [YouTube: PR-021: Batch Normalization (language: korean)][2]{:target="_blank"}
3. [Tistory: 10. Batch Normalization(배치 정규화)][3]{:target="_blank"}
4. [Naver Blog: 공변량(Covariate)][4]{:target="_blank"}
5. [Alsemy: Loss landscape를 여행하는 연구자를 위한 안내서][6]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=tNIpEZLv_eg
[2]: https://www.youtube.com/watch?v=TDx8iZHwFtM
[3]: https://89douner.tistory.com/44?category=868069
[4]: https://m.blog.naver.com/imchaehong/10033930690
[5]: https://proceedings.neurips.cc/paper/2018/file/905056c1ac1dad141560467e0a99e1cf-Paper.pdf
[6]: https://www.alsemy.com/post/loss-landscape%EB%A5%BC-%EC%97%AC%ED%96%89%ED%95%98%EB%8A%94-%EC%97%B0%EA%B5%AC%EC%9E%90%EB%A5%BC-%EC%9C%84%ED%95%9C-%EC%95%88%EB%82%B4%EC%84%9C