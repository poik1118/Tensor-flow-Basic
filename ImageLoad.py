#%%
#https://www.tensorflow.org/tutorials/load_data/images?hl=ko

import numpy
import os
import PIL
import PIL.Image
import tensorflow as tf
import sys
sys.append("/opt/conda/lib/python3.10/site-packages/tensorflow_datasets")
#import tensorflow_datasets as tfds

print(tf.__version__)

# 꽃 데이터세트 다운로드
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

# 다운로드한 꽃 사진의 사본의 개수
image_count = len(list(data_dir.glob('*/*.jpg')))
print('사진 개수 :', image_count)

# 꽃 클래스의 roses디렉터리 사진 출력 / jupyter #%% 셀 실행
roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[0]))

roses = list(data_dir.glob('roses/*'))
PIL.Image.open(str(roses[1]))

# Keras유틸로 데이터 로드
# 로드를 위해 매개변수 정의
batch_size = 32
img_height = 180
img_width = 180

# 모델을 개발할 때 검증 분할을 사용하는 것이 좋음.
# 이미지의 80%를 훈련에 사용, 20%를 유효성 검사에 사용.
print("훈련 데이터세트")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

print("검증 데이터세트")
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

"""
!import tensorflow_datasets as tfds Error!
class_name = train_ds.class_name
print(class_name)

class_name = val_ds.class_name
print(class_name)
"""

# 데이터 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(i)
        #plt.title(class_name[label[i]])
        plt.axis("off")

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)    # 32개 이미지 묶음의 180 X 180형상이며 3은 색상채널(RGB)
    print(labels_batch.shape)   # 32개 이미지에 해당하는 레이블
    break

# 데이터 표준화
# RGB채널 값은 0~255범위인데, 신경망에 이상적이지 않아 일반적으로 입력 값을 작게 만들어야함.
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# 현재 픽셀의 값은 [0, 1]이다.
print(numpy.min(first_image), numpy.max(first_image))

# 픽셀값을 [-1, 1]로 조정하려면
#tf.keras.layers.Rescaling(1./127.5, offset=-1)

# 성능향상을 위햔 데이터세트 구성
# 버퍼링된 프리패치를 사용해 디스크에서 데이터를 생성할 수 있도록 한다.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 모델 훈련
# 순차 모델은 각각에 최대 풀링레이어가 있는 세개의 컨볼루션 블록으로 구성된다.
num_classes = 5

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
# 각 훈련 epoch에 대한 훈련 및 검증 정확도
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
# 훈련 루프 사용자 지정
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# 미세 제어를 위한 tf.data이용
list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle=False)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

for f in list_ds.take(5):
    print("File Name", f.numpy())

# 파일의 트리구조를 이용해 class_name 목록을 컴파일.
class_names = numpy.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))
print("Class Name", class_names)

# 데이터세트를 학습 및 검증 세트로 분할
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)

# 데이터세트의 길이 출력
print(tf.data.experimental.cardinality(train_ds).numpy())
print(tf.data.experimental.cardinality(val_ds).numpy())

# 파일 경로를 ( img, label ) 쌍으로 변환하는 함수
def get_label(file_path):
    # 경로 요소의 목록을 경로로 변환
    parts = tf.strings.split(file_path, os.path.sep)
    # 마지막 두 번째는 클래스의 디렉토리이다.
    one_hot = parts[-2] == class_names
    # 레이블을 인코딩하는 정수
    return tf.argmax(one_hot)

def decode_img(img):
    # 압축된 문자열을 3D uint8 tensor로 변환
    img = tf.io.decode_jpeg(img, channels=3)
    # 원하는 크기로 이미지 크기 조절
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path):
    label = get_label(file_path)
    # 파일에서 미가공 데이터를 문자열로 로드
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label

# 'num_parallel_calls'로 여러 이미지를 병렬로 load/processing 하도록 설정함.
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print("Image shape :", image.numpy().shape)
    print("Label :", label.nupy())

# 성능을 이한 데이터세트 구성 / 빨리 배치 처리해 사용 할 수 있도록 최적화
def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)

# 데이터 시각화
image_batch, labels_batch = next(iter(train_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i+1)
    plt.imshow(image_batch[i].numpy().astype("uint8"))
    label = labels_batch[i]
    plt.title(class_names[label])
    plt.axis("off")

# 데이터 계속 훈련시키기
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)
#%%
# TensorFlow 데이터세트 다운로드
#import tensorflow_datasets as tfds
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%]'],
    with_info=True,
    as_supervised=True,
)

# 꽃 데이터세트 클래스 출력
num_classes = metadata.features['label'].num_classes
print(num_classes)

# 데이터세트에서 이미지 검색
get_label_name = metadata.features['label'].int2srt

image, label = next(iter(train_ds))
x = plt.imshow(image)
y = plt.title(get_label_name(label))

# 훈련, 검증 및 테스트 세트를 일괄 처리, 셔플 구성
train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)
# %%
