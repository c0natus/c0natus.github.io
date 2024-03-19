---
title: "Window: CUDA, cuDNN, PyTorch, Tensorflow, Keras 설치"
categories: [ETC]
tags: [CUDA, cuDNN, PyTorch, Tensorflow, Keras]
author: gshan
img_path: /assets/img/posts/etc/2022-08-01-window-cuda-pytorch-tensorflow/
---

# Ⅰ. CUDA, cuDNN 설치

먼저 그래픽 카드의 종류는 드라이브가 설치되어 있다면 아래 그림과 같이 확인할 수 있다.

```bash
wmic path win32_VideoController get name
```

![](1.jpg)

그래픽 카드의 [CUDA compute capability][1]{:target="_blank"}를 먼저 확인해보자. GeForce RTX 2070은 7.5이다.

![](2.jpg)

[위키][2]{:target="_blank"}에서 7.5가 사용할 수 있는 CUDA SDK 버전을 보니 CUDA 10, 11버전을 사용해야 했다.

![](3.jpg)

CUDA 사용가능한 버전을 봤으면 이제 PyTorch에서 지원하는 버전을 살펴보자. [공식 홈페이지][3]{:target="_blank"}에 가니 11.6 버전에 관한 install command가 있다. 11.7 버전은 아직 stable하지 않은 것 같다. 

![](4.jpg)

그래서 11.7대신 11.6 버전을 설치하자. [여기][4]{:target="_blank"}서 다운받으면 된다.

![](5.jpg)

![](6.jpg)

나는 11.6.2를 설치해주었다. 이때 환경변수가 자동으로 등록되므로 설치 경로는 바꾸지 않는게 좋다.

![](7.jpg)

그리고 [cuDNN][5]{:target="_blank"}도 설치한 CUDA의 버전에 맞게 선택하자. (다운로드 받기 전 회원가입과 설문이 있다.)

![](8.jpg)

zip 파일을 다운로드 받았으면 압축을 풀고 bin, include 등을 아래의 CUDA 폴더에 있는 bin, include 등에 옮기면 끝이다.

> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6


nvcc —version과 nvidia-smi를 통해 잘 설치되었는지 확인하자.

```bash
nvcc --version

nvidia-smi
```

![](9.jpg)

# Ⅱ. PyTorch 설치

이제 PyTorch를 설치하자. [공식 홈페이지][3]{:target="_blank"}에 들어가면 어떤 명령어를 쳐야하는 지 알려준다. 자신의 OS와 CUDA 버전에 맞는 것을 고르면 된다.

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```

이제 torch에서 cuda가 사용가능한지 확인하면 끝이다.

```python
import torch
print(torch.cuda.is_available())
```

![](10.jpg)

# Ⅲ. Tensorflow 설치

성능적인 면에서 [pip보단 conda로 설치][6]{:target="_blank"}하는 것이 더 낫다. conda 공식 문서에서 [tensorflow를 설치][7]{:target="_blank"}하는 명령어가 잘 나와 있다. 만약 CPU에서 tensorflow를 돌리지 않는다면 이 단계를 생략하고 tenworflow-gpu를 설치하러 가자. 그러면 CPU와 GPU 버전 두개가 한번에 설치된다.

이때 주의할 점은 python 3.9 이하 버전을 사용해야 한다. python 3.10은 설치가 안 된다.

## ⅰ. Tensorflow: CPU

```bash
conda install -c conda-forge tensorflow -y
```

설치를 한 후 tensorflow에서 사용하고 있는 device_type을 확인해보자.

```python
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```

![](11.jpg)

그러면 위 그림과 같이 CPU만 잡힐 것이다. 이때 아래와 같은 경고가 있을 수 있는데 CPU를 사용한다면 성능을 올리기 위해 [해결][8]{:target="_blank"}해야 하지만, GPU를 사용한다면 무시하면 된다. 왜 이런 경고가 뜨는지 알고 싶다면 [여기][9]{:target="_blank"}를 참고하자.

> 2022-08-17 14:58:46.894867: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

## ⅱ. Tensorflow: CPU+GPU

GPU를 사용하고 싶을 때 tenworflow-gpu를 설치해야 한다. 이것도 conda로 [설치][10]{:target="_blank"}하자.

```bash
conda install -c anaconda tensorflow-gpu -y
```

![](12.jpg)

GPU가 잘 잡히는 것을 볼 수 있다.

# Ⅳ. Keras 설치

Keras도 마찬가지 이다. CPU와 GPU버전을 설치할 수 있다. Keras는 tensorflow 기반이기 때문에 tensorflow-gpu를 설치 했다면 tensorflow-gpu + keras와 tensorflow-gpu + keras-gpu는 [차이가 없는 것 같다][11]{:target="_blank"}고 한다.

```bash
conda install -c conda-forge keras -y
```

```bash
conda install -c anaconda keras-gpu -y
```

[1]: https://developer.nvidia.com/cuda-gpus
[2]: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
[3]: https://pytorch.org/
[4]: https://developer.nvidia.com/cuda-toolkit-archive
[5]: https://developer.nvidia.com/cudnn
[6]: https://antilibrary.org/2378
[7]: https://anaconda.org/conda-forge/tensorflow
[8]: https://stackoverflow.com/questions/41293077/how-to-compile-tensorflow-with-sse4-2-and-avx-instructions
[9]: https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
[10]: https://anaconda.org/anaconda/tensorflow-gpu
[11]: https://stackoverflow.com/questions/52988311/what-is-the-difference-between-keras-and-keras-gpu