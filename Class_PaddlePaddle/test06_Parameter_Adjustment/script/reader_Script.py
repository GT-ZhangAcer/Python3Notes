import paddle
from PIL import Image
import numpy as np
import pickle


def data_normal_id_img(batch_size, data_num_rate=1):
    """
    验证码Reader
    默认打乱，数据集3：1划分
    :param data_num_rate: 数据集总数比例，默认为1，设置为0.5时仅读取一半的数据
    :param batch_size: 小批次数据大小
    :return: 训练数据、测试数据
    """
    with open("./data/id_img/ocrData.txt", 'rt') as f:
        a = f.read()

    def data_reader(for_test=False):
        def reader():
            start = int(1 * data_num_rate)
            stop = int(1501 * data_num_rate)
            if for_test:
                start = int(1501 * data_num_rate)
                stop = int(2001 * data_num_rate)
            for i in range(start, stop):
                im = Image.open("./data/id_img/" + str(i) + ".jpg").convert('L')
                im = np.array(im).reshape(1, 30, 15).astype(np.float32)
                im = im / 255.0 * 2.0 - 1.0
                label = a[i - 1]
                yield im, label

        return reader

    train_batch = paddle.batch(reader=paddle.reader.shuffle(data_reader(), 1500), batch_size=batch_size)
    test_batch = paddle.batch(reader=data_reader(for_test=True), batch_size=batch_size)
    return train_batch, test_batch


def data_cifar10(batch_size, modifier_def=None):
    def unpick(file_name):
        with open(file_name, 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
        return data

    train_file_name = './data/cifar-10-batches-py/data_batch_'
    test_file_name = './data/cifar-10-batches-py/test_batch'

    def reader(for_test=False):
        def train_reader():
            for i in range(1, 6):
                file_name = train_file_name + str(i)
                data = unpick(file_name)
                for img, label in zip(data[b'data'], data[b'labels']):
                    img = np.array(img).reshape(3, 32, 32).astype('float32')
                    if modifier_def is not None:
                        img = modifier_def(img)
                    yield img, int(label)

        def test_reader():
            data = unpick(test_file_name)
            for img, label in zip(data[b'data'], data[b'labels']):
                img = np.array(img).reshape(3, 32, 32).astype('float32')
                if modifier_def is not None:
                    img = modifier_def(img)
                yield img, int(label)

        if for_test is False:
            return train_reader
        else:
            return test_reader

    train_batch = paddle.batch(reader=paddle.reader.shuffle(reader(), 1000), batch_size=batch_size)
    test_batch = paddle.batch(reader=reader(for_test=True), batch_size=batch_size)
    return train_batch, test_batch
