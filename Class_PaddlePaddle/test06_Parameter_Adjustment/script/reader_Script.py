import paddle.fluid as fluid
import paddle
from PIL import Image
import numpy as np


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

    def reader(for_test=False):
        start = int(1 * data_num_rate)
        stop = int(1501 * data_num_rate)
        if for_test:
            start = int(1501 * data_num_rate)
            stop = int(2001 * data_num_rate)
        for i in range(start=start, stop=stop):
            im = Image.open("./data/id_img/" + str(i) + ".jpg").convert('L')
            im = np.array(im).reshape(1, 30, 15).astype(np.float32)
            im = im / 255.0 * 2.0 - 1.0
            label = a[i - 1]
            yield im, label

    train_batch = paddle.batch(reader=paddle.reader.shuffle(reader(), 1500), batch_size=batch_size)
    test_batch = paddle.batch(reader=paddle.reader.shuffle(reader(for_test=True), 500), batch_size=batch_size)
    return train_batch, test_batch
