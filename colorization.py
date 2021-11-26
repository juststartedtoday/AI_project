#필요한 외부참조 import
from keras import layers, models, optimizers
from keras import datasets
from keras import backend
import matplotlib.pyplot as plt

#U-NET 모델 클래스
class UNET(models.Model):
    #기본 지정자
    def __init__(self, org_shape, n_ch):
        #채널수 (RGB/흑백)
        channel_index = 3 if backend.image_data_format() == 'channels_last' else 1

        # Convolution층 모델링
        def conv(x, n_f, mp_flag=True):
            x = layers.MaxPooling2D((2, 2), padding='same')(x) if mp_flag else x
            x = layers.Conv2D(n_f, (3, 3), padding='same', strides=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('tanh')(x)
            x = layers.Conv2D(n_f, (3, 3), padding='same', strides=2)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('tanh')(x)
            return x

        # DeConvolution층 모델링
        def deconv_unet(x, e, n_f):
            x = layers.UpSampling2D((2, 2))(x)
            x = layers.Concatenate(axis=channel_index)([x, e])
            x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('tanh')(x)
            x = layers.Conv2D(n_f, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('tanh')(x)
            return x

        # layers에 Shape 설정
        original = layers.Input(shape=org_shape)

        # U-NET 구조 모델링
        c1 = conv(original, 16, False)
        c2 = conv(c1, 32)
        encoded = conv(c2, 64)

        x = deconv_unet(encoded, c2, 32)
        x = deconv_unet(x, c1, 16)
        decoded = layers.Conv2D(n_ch, (3, 3), activation='sigmoid',
                                padding='same')(x)

        # Adadelta Optimizer와 mean-squared-error Loss Function로 컴파일.
        super().__init__(original, decoded)
        self.compile(optimizer='adadelta', loss='mse')

#DATA() 클래스
class DATA():

    # CIFAR-10 데이터셋 로드
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()

        # 데이터 Shape 크기 지정
        if backend.image_data_format() == 'channels_first':
            n_ch, img_rows, img_cols = x_train.shape[1:]
            input_shape = (1, img_rows, img_cols)
        else:
            img_rows, img_cols, n_ch = x_train.shape[1:]
            input_shape = (img_rows, img_cols, 1)

        # 데이터 Normalization
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # RGB 이미지를 흑백화 시키는 함수
        def RGB2Gray(img, fmt):
            if fmt == 'channels_first':
                R = img[:, 0:1]
                G = img[:, 1:2]
                B = img[:, 2:3]
            else:
                R = img[..., 0:1]
                G = img[..., 1:2]
                B = img[..., 2:3]
            return 0.299 * R + 0.587 * G + 0.114 * B

        # train셋과 test셋을 흑백화 시킨다.
        x_train_in = RGB2Gray(x_train, backend.image_data_format())
        x_test_in = RGB2Gray(x_test, backend.image_data_format())

        # parameter 지정.
        self.input_shape = input_shape
        self.x_train_in, self.x_train_out = x_train_in, x_train
        self.x_test_in, self.x_test_out = x_test_in, x_test
        self.n_ch = n_ch


# loss 그래프 출력 함수
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


# 정확도 그래프 출력 함수
def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)


# 이미지 출력 함수
def show_images(in_imgs, out_imgs, unet, sample_size=10):
    x_test_in = in_imgs[:sample_size]
    x_test_out = out_imgs[:sample_size]
    decoded_imgs = unet.predict(x_test_in, batch_size=sample_size)

    print("Before")
    print("x_test_in:", x_test_in.shape)
    print("decoded_imgs:", decoded_imgs.shape)
    print("x_test_out:", x_test_out.shape)

    if backend.image_data_format() == 'channels_first':
        x_test_out = x_test_out.swapaxes(1, 3).swapaxes(1, 2)
        decoded_imgs = decoded_imgs.swapaxes(1, 3).swapaxes(1, 2)

        x_test_in = x_test_in[:, 0, ...]
    else:
        x_test_in = x_test_in[..., 0]

    print("After")
    print("x_test_in:", x_test_in.shape)
    print("decoded_imgs:", decoded_imgs.shape)
    print("x_test_out:", x_test_out.shape)

    plt.figure(figsize=(20, 6))
    for i in range(sample_size):
        ax = plt.subplot(3, sample_size, i + 1)
        plt.imshow(x_test_in[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, sample_size, i + 1 + sample_size)
        plt.imshow(decoded_imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, sample_size, i + 1 + sample_size * 2)
        plt.imshow(x_test_out[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

#메인함수 ( epochs = 10 / batch_size = 128)로 지정.
def main(epochs=10, batch_size=128):

    # 데이터셋 설정
    data = DATA()

    # Unet 모델링
    unet = UNET(data.input_shape, data.n_ch)

    # Unet 세부 설정
    history = unet.fit(data.x_train_in, data.x_train_out,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_split=0.2,
                       verbose=2)

    # loss 그래프
    plot_loss(history)

    # 정확도 그래프
    plot_acc(history)

    # 테스트 셋의 결과값과 원본이미지 출력
    show_images(data.x_test_in, data.x_test_out, unet)

#본 파일이 '__main__'일 경우에 메인함수 실행
if __name__ == '__main__':
    main()
