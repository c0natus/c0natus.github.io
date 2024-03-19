---
title: "Autoencoder"
categories: [Theory, AI]
tags: [Autoencoder, Manifold Learning, Deep Learning]
img_path: /assets/img/posts/theory/ai/autoencoder/
author: gshan
math: true
---

GAN을 만든 Ian Goodfellow가 정리한 generative model의 taxonomy(분류)는 다음과 같다.

![](1.jpg)
_Figure 1: Taxonomy of generative model [출처][3]_

이 표를 보면 variational autoencoder(VAE)는 maximum likelihood density estimation(MLE)과 관련이 있는 것을 볼 수 있다.

Autoencoder는 4가지 keywords를 가진다.
- Unsupervised learning, manifold learning, generative model learning, ML density estimation

Autoencoder의 학습 방법은 unsupervised learning이고 loss는 negative maximum likelihood로 해석된다.
그리고 학습된 autoencoder에서 encoder는 차원 축소 역할(manifold learning)을 수행하며, decoder는 생성 모델의 역할(generative model learning)을 수행한다.

이번 post에서는 MLE와 VAE의 관련성을 파악하기 위한 사전 지식을 살펴볼 것이다.

# Ⅰ. Revisit Deep Neural Networks

DNN을 학습할 때 사용되는 loss function은 다양한 각도로 해석할 수 있다.
그 중 loss function을 MLE로 해석할 수 있다는 점을 천천히 이해해보자.

우선 기본적으로 DNN에서 사용하는 loss function은 2가지 가정(assumption)을 만족해야 한다.
1. Total loss of DNN over training samples is the sum of loss for each training sample.
2. Loss for each training example is a function of final output of DNN.

Trainng의 목표는 loss function이 줄어드는 방향으로 model의 parameter($\theta$)를 update하는 것이다.
일일이 high dimension인 $\theta$를 조금씩 바꾸는 것은 불가능하다.
그러면 어떻게 update해야 할까?
가장 간단한 방법은 gradient descent를 사용하는 것이다.
Gradient descent를 사용하면 왜 loss function이 감소되는 지는 강의를 통해 수식을 확인해보면 좋을 것 같다.

![](2.jpg)
_Figure 2: Backpropagation algorithm_

Gradient descent 방법으로 DNN의 각 layer를 update하는 것으로 backpropagation algorithm을 사용한다.

## ⅰ. Backpropagation View

Loss function을 MLE로 해석하기 전에 backpropagation 관점으로 MSE와 cross entropy의 차이점을 알아보자.

<span class="text-color-bold"> **Mean Square Error / Quadratic loss** </span>

![](3.jpg)
_Figure 3: MSE and Error signal_

Input으로 1을 주면 output으로 0이 나오는 model을 만들 것이다.

Loss function $C$ 으로 mean square error(MES)를 사용하고, activation function으로 sigmoid를 사용한다.
$\delta = \frac{\partial}{\partial z}C$를 error signal이라고 했을 때, $\delta = (a-y)\sigma^\prime(z)$가 된다.
$\delta$로 $w, b$를 update하는데 $x = 1$이므로 $w \leftarrow w - \eta\delta$, $b \leftarrow b - \eta\delta$가 된다.

![](4.jpg)
_Figure 4: Training speed with different initial parameter value and MSE_

초기 $w, b$의 값에 따른 output 값을 비교한 그림이다.
이 처럼 학습 속도에 초기 값이 영향을 준다.
그렇다면 그 이유가 무엇일까?
바로 activation function으로 사용된 sigmoid 함수의 특성 때문이다.

![](5.jpg)
_Figure 5: Sigmoid and its derivative_

Sigmoid의 입력값 $z$에 따른 sigmoid function 값과 그것의 미분값(derivative) 보여준다.
보시다시피 $w=2, b=2$일 때 sigmoid 미분값이 매우 작다.
그리고 그림의 왼쪽 수식처럼 $\delta$에 sigmoid 미분값 $\sigma^\prime(z)$가 곱해지기 때문에, 미분값이 작으면 model의 parameter가 매우 조금씩 update되어 버린다.
Figure 3의 우측 부분을 보면 이를 확인할 수 있다.

이처럼 MSE loss function과 sigmoid activation function을 사용하면 gradient vanishing 문제가 생긴다.

<span class="text-color-bold"> **Cross Entropy** </span>

![](6.jpg)
_Figure 6: Cross Entropy and Error signal_

Cross entropy는 마지막 layer L의 error signal에 activation function의 미분값이 곱해지지 않는다.
즉, fig. 2의 1번 식에서 미분값이 곱해지지 않는다.
그렇지만 다른 layer에 대해서는 fig. 2의 2번 식과 같이 미분값이 곱해지므로 layer가 많아지면 결국 gradient vanishing 문제가 발생한다. 

![](7.jpg)
_Figure 7: Training speed with different initial parameter value and cross entropy_

어쨌든 미분값이 한번이라도 덜 곱해지기 때문에, cross entropy는 MSE보다 초기값에 robust하다고 볼 수 있다.

## ⅱ. MLE View

![](8.jpg)
_Figure 8: Interpretation as a stochastic model_

MLE 관점에서는 model이 특정한 확률 분포(Gaussian 등) $p$를 따른다고 가정한다.
그러면 model의 output $f_\theta(x)$은 그 확률 분포의 parameter(평균, 표준편차 등)가 되고, 이 parameter $f_\theta(x)$는 해당 확률 분포에서 ground truth $y$의 likelihood를 최대화하는 것이 되어야 한다.
Likelihood란 쉽게 말해 probability mass function (pmf), probability destiny function (pdf)의 값이다.

> 실제로 model이 동작하는 방식이 아니라 단순히 직관적인 해석에 불가하다.

아주 간단한 예를 들어, model이 표준편차가 같은 Gaussian 분포를 따른다고 가정하자.
그러면 model의 output은 기댓값이 된다.
위 그림에서 ground truth $y$에 대해 $\theta_1$의 likelihood 값이 $\theta_2$의 likelihood 값보다 좋다.
이는 $\theta_1$을 parameter로 가지는 model이 더 좋은 Gaussian의 기댓값을 출력한다고 할 수 있다.

즉, 모든 training data의 ground truth에 대해 가장 높은 likelihood 값을 가지는 확률 분포의 parameter(평균, 표준편차 등)을 출력하도록 model의 parameter($\theta$)를 학습시켜야 한다.
위의 예에서 최적의 $\theta$는 $y$와 같은 평균을 출력할 때이다.

$$
\begin{split}
\theta^* &= \underset{\theta}{\text{argmin}}-\text{log}(p(y|f_\theta(x)))\\
-\text{log}(p(y|f_\theta(x))) &= -\sum_i^{|\mathcal{D}|}-\text{log}(p(y_i|f_\theta(x_i)))
\end{split}
$$

![](9.jpg)
_Figure 9: Loss function with different probability distribution._

확률 분포로 다양한 것을 사용할 수 있지만, DNN에 적용하기가 수학적으로 어렵기 때문에 Gaussian과 Bernoulli 분포를 많이 사용한다.
이 분포들로 loss function $-\text{log}(p(y_i|f_\theta(x_i)))$을 풀어쓰면, 그림과 같이 Gaussian은 MSE, Bernoulli는 cross entropy와 같은 형태를 띄게 된다.

이 관점에서는 $y$가 continuous한 값을 가지면 MSE loss를, discrete한 값을 가지면 cross entropy loss를 사용하는 것이 낫다.

![](10.jpg)
_Figure 10: Yoshua Bengio's slide [출처][4]{:target="_blank"}_

Yoshua Bengio의 slide에선 위의 내용이 slide 한장으로 요약되어 있다.

# Ⅱ. Manifold Learning

![](11.jpg)
_Figure 11: Manifold_

Manifold hypothesis라는 것이 있다.
- Natural data in high dimensional spaces concentrates close to lower dimensional manifolds.
- Probability density decreases very rapidly when moving away from the supporting manifold.

![](12.jpg){: w="500"}
_Figure 12: Manifold example - 200 x 200 image._

간단히 200x200 image로 예를 들어보자.
200x200 RGB pixel로 총 10^96329개의 image를 표현할 수 있다.
Uniform하게 sampling하면 nosiy한 image만 출력된다.
만약 manifold 가정이 틀렸다면, data는 균등하게 분포되어 있으므로 의미있는 image가 출력되어야 할 것이다.
하지만 그렇지 않다.
즉, 얼굴이나 글씨 등 의미있는 image의 분포는 특정한 manifold 상에 모여있다고 할 수 있다.
당연히 얼굴을 표현하는 image끼리는 글씨를 표현하는 image보다 manifold 상에서 더 가깝게 위치할 것이다.

![](13.jpg)
_Figure 13: Relation between images in manifolds._

그래서 특정 domain에 대해 manifold를 잘 찾았다면, 비슷한 사진들 간의 관계성을 파악할 수 있게 된다.
그리고 적절한 확률 분포를 찾았다면, 그 manifold 상에서 sampling을 통해 이전에 볼 수 없었던 image를 얻게 된다.

## ⅰ. Dimensionality Reduction

![](14.jpg)
_Figure 14: Taxonomy of dimensionality reduction_

위의 사진은 기존의 dimensionality reduction의 분류표이다.
Autoencoder와 다르게 Isomap, LLE 등 기존의 방법들은 Euclidean space 상에서 거리 정보를 활용한다.
하지만, Euclidean에서 가깝다고 실제로 가까운게 아니다.
즉, 아래 그림처럼 manifold에서 가까운게 아니다.

![](15.jpg)
_Figure 15: Distance metric - Euclidean and Manifold._

# Ⅲ. Autoencoders

![](16.jpg)
_Figure 16: Autoencoder structure._

Encoder는 고차원 입력 데이터를 압축해서 저차원인 latent vector $z$로 표현한다.
이때, $z$를 알 수 없으므로 unsupervised learning으로 model을 학습해야 했다.
Autoencoder는 여기에 decoder를 붙여서 unsupervised learning을 self-supervised learning으로 바꾸었다.
Autoencoder의 목적은 압축인데 unsupervised 방식으로 해결할 수 없어 decoder를 붙여 self-supervised 방식으로 훈련을 시킨 것이다.
그 결과 encoder는 최소한 학습 데이터는 latent vector로 잘 표현할 수 있고, decoder는 최소한 학습 데이터를 잘 생성해낼 수 있게 된다.
즉, 최소한의 성능이 보장된다.

> GAN은 최소 성능 보장이 되지 않아 학습이 다소 어렵다.
> 그렇지만, 생성된 data의 diversity는 더 높다.

## ⅰ. DAE(Denosing AutoEncoder)

![](17.jpg)
_Figure 17: DAE structure._

DAE는 입력 $x$를 그대로 사용하는 것이 아니라 약간의 noise를 추가한 $\tilde{x}$를 사용한다.
$\tilde{x}$는 사람이 보기에 $x$와 의미적으로 같아야 한다.
한마디로 manifold에서의 값 $\tilde{z} = z$라는 의미이다.
Output $y$는 noise가 없는 $x$와 같도록 학습된다.
즉, noise가 없어진다.
그렇기 때문에 denosing이라고 부른다.

# References
1. [YouTube - 오토인코더의 모든 것 - 1/3][1]{:target="_blank"}
2. [Slideshare - 오토인코더의 모든 것][2]{:target="_blank"}

[1]: https://www.youtube.com/watch?v=o_peo6U7IRM
[2]: https://www.slideshare.net/NaverEngineering/ss-96581209
[3]: https://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf
[4]: https://videolectures.net/kdd2014_bengio_deep_learning