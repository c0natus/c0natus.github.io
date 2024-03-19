---
title: "Derivative of Softmax Cross Entropy"
categories: [Theory, Math]
tags: [Cross Entropy, Softmax, Derivative]
author: gshan
math: true
---

[knowledge distillation][1]{:target="_blank"} 논문을 보던 중 cross-entropy의 gradient를 구하는 과정이 있어 직접 해보고자 한다.

# Ⅰ. Notation

$$
\begin{align*}
p_i &= \frac{\textrm{exp}(v_i/T)}{\sum_j\textrm{exp}(v_j/T)} \\
q_i &= \frac{\textrm{exp}(z_i/T)}{\sum_j\textrm{exp}(z_j/T)}
\end{align*}
$$

- $p_i$: cumbersome model(teacher model)의 probability.
- $v_i$: cumbersome model(teacher model)의 logit.
- $q_i$: distilled model(student model)의 probability.
- $z_i$: distilled model(student model)의 logit.
- $T$: temperature, 일반적으로 1이다.

$$
\textrm{CE Loss} = -\sum_{i=1}^{C}p_i\textrm{log}(q_i)
$$

- $p_i$: true probability, KD를 하지 않을 때는 $y_i$가 쓰인다.
- $q_i$: training model의 softmax probability.

## Ⅱ. Derivative 

$$
\begin{align*}
\frac{\partial\textrm{CE}}{\partial z_i} 
  &= \frac{\partial\textrm{CE}}{\partial q_j}\frac{\partial q_j}{\partial z_i} \\\\
\frac{\partial\textrm{CE}}{\partial q_j} 
  &= \sum_{j=1}^C-\frac{p_j}{q_j} \\\\
\frac{\partial q_j}{\partial z_i} 
  &= \frac{\partial \frac{\textrm{exp}(z_j/T)}{\sum_j\textrm {exp}(z_j/T)}}{\partial z_i}\\\\
&= \begin{cases} 
  \frac{1}{T} \frac{\textrm{exp}(z_i/T)\sum_j\textrm {exp}(z_j/T) - \textrm{exp}(z_i/T)\textrm{exp}(z_i/T)}{(\sum_j\textrm{exp}(z_j/T))^2} 
  = \frac{1}{T}(q_i - (q_i)^2)
  &\textrm{if } i=j \\\\
  \frac{1}{T} \frac{0 - \textrm{exp}(z_j/T)\textrm{exp}(z_i/T)}{(\sum_j\textrm{exp}(z_j/T))^2}
  = \frac{1}{T}(-q_jq_i)
  &\textrm{otherwise}
  \end{cases}\\\\
\frac{\partial\textrm{CE}}{\partial z_i} 
  &= \sum_{j=1}^C\bigg(-\frac{p_j}{q_j}\frac{\partial p_j}{\partial z_i}\bigg)\\
  &= \frac{1}{T}\bigg(-\frac{p_i}{q_i}\big(q_i - (q_i)^2\big) + \sum_{j \neq i}\frac{p_j}{q_j} q_jq_i \bigg)\\
  &= \frac{1}{T}\bigg(-p_i + \sum_{j = 1}^C\frac{p_j}{q_j} q_jq_i \bigg)\\
  &= \frac{1}{T}\bigg(-p_i + q_i\sum_{j=1}^{C} p_j \bigg)\\
  &= \frac{1}{T}\big(-p_i + q_i \big) \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \because \sum_{j=1}^{C} p_j = 1\\\\
\therefore \frac{\partial\textrm{CE}}{\partial z_i} 
  &= \frac{1}{T}\big(-p_i + q_i \big)
\end{align*}
$$

[1]: https://arxiv.org/pdf/1503.02531.pdf
