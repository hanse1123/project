import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 데이터 로드 및 전처리
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # 정규화
y_train, y_test = to_categorical(y_train, 100), to_categorical(y_test, 100)  # 원-핫 인코딩

# 사전 학습된 DenseNet121 모델 로드
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# 모델 구성
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(100, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(x_train, y_train,
                    validation_data=(x_test, y_test),
                    epochs=20,
                    batch_size=64)


# 학습 정확도와 검증 정확도 플롯
def plot_accuracy(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


plot_accuracy(history)

# 모델 저장
model.save("cifar100_densenet_model.h5")
print("모델이 저장되었습니다: cifar100_densenet_model.h5")


# 새로운 이미지 분류
def predict_image(model_path, image_path):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import load_img, img_to_array

    # 모델 로드
    model = load_model(model_path)
    print("모델이 로드되었습니다.")

    # 이미지 로드 및 전처리
    img = load_img(image_path, target_size=(32, 32))  # CIFAR-100의 입력 크기로 조정
    img_array = img_to_array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    # 예측
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    print(f"예측된 클래스: {predicted_class}, 확률: {confidence:.4f}")
    return predicted_class, confidence