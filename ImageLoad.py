#%%
import numpy as np
import os, sys
import PIL
import PIL.Image
import tensorflow as tf
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