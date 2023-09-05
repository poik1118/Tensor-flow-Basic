# 텐서플로 2.0 시작하기 : 초보자용
# https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ko

# Tensorflow, Keras 설치와 활용한 예제
# 1. 사전에 빌드한 데이터세트를 로드합니다.
# 2. 이미지를 분류하는 신경망 머신 러닝 모델을 빌드합니다.
# 3. 이 신경망을 훈련합니다.
# 4. 모델의 정확도를 평가합니다.

# pip install tensorflow
# pip install keras
# pip install tensorrt

import tensorflow as tf
import keras
import numpy as np

#print("TensorFlow spec :", tf.__spec__)

# MNIST Dataset Load
mnist = tf.keras.datasets.mnist

# 샘플 데이터를 정수에서 부동 소수점의 숫자로 변환
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 빌드 / 훈련에 사용할 옵티마이저와 손실함수 선택
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

predictions = model(x_train[:1]).numpy()
print("\nPrediction Array :", predictions)

print("\nSoftMax Array :", tf.nn.softmax(predictions).numpy())

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 훈련되지 않은 모델은 무작위로 계산하므로 초기 손실은 -tf.math.log(1/10) ~= 2.3에 근접 해야한다.
print(-tf.math.log(1/10)) # x의 자연 로그를 요소별로 계산 / y = log e (1/10)

print("\n초기 손실 값 :", loss_fn(y_train[:1], predictions).numpy())

# 훈련 전 Keras의 model.compile로 모델을 구성하고 컴파일.
print("\nmodel.compile")
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# 모델 훈련 및 평가 ( model.fit )
print("\nmodel.fit")
model.fit(x_train, y_train, epochs=5)

# 모델 성능 확인 ( model.evaluate )
print("\nmodel.evaluate")
model.evaluate(x_test, y_test, verbose=2)

# 훈련된 모델이 확률을 반환하도록 다음과 같이 모델을 래핑하며, 여기에 소프트맥스를 첨부할 수 있다.
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
print("\nProbability Model Array :", probability_model(x_test[:5]))