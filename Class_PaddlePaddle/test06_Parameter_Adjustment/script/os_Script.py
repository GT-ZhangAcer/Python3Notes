import os
import shutil


def mkdir(path, de=False):
    """
    判断是否路径存在并新建文件夹
    :param de: 是否删除文件夹
    :param path: 文件路径

    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        if de is True:
            shutil.rmtree(path)
            os.makedirs(path)
