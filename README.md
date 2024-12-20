# 2024-2학기 기계학습과응용 프로젝트 : CIFAR-100 이미지 분류

DenseNet 모델을 사용하여 CIFAR-100 이미지 분류를 구현했습니다.

CIFAR-100 데이터셋으로 모델을 학습시키고 저장하여, 저장된 모델을 불러와 새로운 이미지를 분류하는 기능까지 구현했습니다.

## 저장소의 파일 구성

1. **`train_cifar100_model.py`**:
   - CIFAR-100 데이터셋을 사용하여 DenseNet 기반 이미지 분류 모델을 학습시키는 코드가 포함되어 있습니다.
   - 데이터 전처리, 모델 정의, 학습 과정을 수행하며, 학습된 모델을 `cifar100_densenet_model.h5` 파일로 저장합니다.

2. **`predict_cifar100_image.py`**:
   - 학습된 모델 `cifar100_densenet_model.h5`를 로드하여 새로운 이미지를 분류하는 코드가 포함되어 있습니다.
   - 사용자가 입력한 이미지를 CIFAR-100 클래스 중 하나로 분류합니다.

3. **`cifar100_densenet_model.h5`**:
   - 학습된 DenseNet 모델 파일로, `predict_cifar100_image.py`에서 이미지를 분류하는 데 사용됩니다.
   - 모델 파일이 깃허브 업로드 용량을 초과해, 아래의 구글 드라이브 링크를 통해 다운로드 할 수 있습니다.
   - https://drive.google.com/file/d/1V9xzP4QhjVelZh7tDHn1UnDO7E51oOBl/view?usp=sharing
  
## 사용 예시

![example](https://github.com/user-attachments/assets/9eb3866e-3107-405f-a93f-593a93cb0b12)


## 사용 방법

### 1. 모델 학습

모델을 재학습하거나 학습 과정을 수정하려면:

- `train_cifar100_model.py`를 실행하세요.
- 학습된 모델은 `cifar100_densenet_model.h5` 파일로 저장됩니다.

```bash
python train_cifar100_model.py
```

### 2. 새로운 이미지 분류

학습된 모델을 사용하여 새로운 이미지를 분류하려면:

- `predict_cifar100_image.py` 스크립트를 사용하세요.
- 스크립트에서 테스트 이미지 경로를 수정하거나 명령줄 인수로 전달하세요.

예제 실행 명령:

```bash
python predict_cifar100_image.py --image_path path/to/your/image.jpg
```

스크립트는 예측된 클래스와 신뢰도를 출력합니다.

## 요구 사항

다음 명령어를 사용하여 필요한 Python 라이브러리를 설치하세요:

```bash
pip install -r requirements.txt
```

### 주요 사용 라이브러리

- `tensorflow`
- `numpy`
- `matplotlib`

## 결과

모델은 CIFAR-100 데이터셋에서 다음과 같은 성능을 보입니다:

- **학습 정확도:** 에포크가 진행됨에 따라 꾸준히 향상되어 90% 이상에 도달합니다.
- **검증 정확도:** 50-60% 수준에서 안정화됩니다.

다음 그래프는 학습 정확도와 검증 정확도를 시각화한 것입니다:

![accuracy](https://github.com/user-attachments/assets/b9adf5a0-2003-4942-a64d-e26f70a2226e)


## 작성자

- 박성웅


