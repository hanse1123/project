import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# CIFAR-100 클래스 이름 (CIFAR-100 데이터셋 클래스 레이블)
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider',
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# 저장된 모델을 로드하여 이미지를 분류하는 함수
def predict_image(model_path, image_path):
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

    # 클래스 이름과 확률 반환
    predicted_label = cifar100_classes[predicted_class]
    print(f"예측된 클래스: {predicted_label}, 확률: {confidence:.4f}")
    return predicted_label, confidence

predict_image("C:\\Users\\psw95\\OneDrive\\바탕 화면\\mlproject_psw\\cifar100_densenet_model.h5", "C:\\Users\\psw95\\OneDrive\\바탕 화면\\모델 예측\\123.webp")
