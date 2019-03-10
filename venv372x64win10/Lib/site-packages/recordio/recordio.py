import ctypes
import os
import six

path = os.path.join(os.path.dirname(__file__), "librecordio.pyd")
lib = ctypes.cdll.LoadLibrary(path)


def _convert_to_bytes(obj):
    if isinstance(obj, six.text_type):
        return obj.encode()
    elif isinstance(obj, six.binary_type):
        return obj
    elif obj is None:
        return obj
    else:
        return six.b(obj)


class writer(object):
    """
    writer is a recordio writer.
    """

    def __init__(self, path):
        self.w = lib.create_recordio_writer(_convert_to_bytes(path))

    def close(self):
        lib.release_recordio_writer(self.w)
        self.w = None

    def write(self, record):
        lib.recordio_write(self.w, ctypes.c_char_p(_convert_to_bytes(record)), len(record))


class reader(object):
    """
    reader is a recordio reader.
    """

    def __init__(self, path):
        self.r = lib.create_recordio_reader(_convert_to_bytes(path))

    def close(self):
        lib.release_recordio_reader(self.r)
        self.r = None

    def read(self):
        p = ctypes.c_char_p()
        ret = ctypes.pointer(p)
        size = lib.recordio_read(self.r, ret)
        if size < 0:
            # EOF
            return None
        if size == 0:
            # empty record
            return b""

        p2 = ctypes.cast(p, ctypes.POINTER(ctypes.c_char))
        record = p2[:size]

        # memory created from C should be freed.
        lib.mem_free(ret.contents)
        return record
