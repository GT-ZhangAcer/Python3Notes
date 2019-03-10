import unittest

import recordio
import six
import six.moves.cPickle as pickle
if six.PY2:
    import md5
else:
    import hashlib

def get_md5_hexdigest(obj):
    assert obj is not None
    if six.PY2:
        return md5.new(obj).hexdigest()
    else:
        if isinstance(obj, six.string_types):
            return hashlib.md5(obj.encode()).hexdigest()
        elif isinstance(obj, six.binary_type):
            return hashlib.md5(obj).hexdigest()
        else:
            raise TypeError("obj must be unicode, str or bytes, but we got %s" % type(obj))

class TestStringMethods(unittest.TestCase):
    def test_write_read(self):
        w = recordio.writer("d:\\tmp\\record_0")
        w.write(pickle.dumps("1"))
        w.write(b"2")
        w.write(b"")
        w.close()
        w = recordio.writer("d:\\tmp\\record_1")
        w.write(b"3")
        w.write(b"4")
        w.write(b"")
        w.close()

        r = recordio.reader("d:\\tmp\\record_*")
        self.assertEqual(pickle.loads(r.read()), "1")
        self.assertEqual(r.read(), b"2")
        self.assertEqual(r.read(), b"")
        self.assertEqual(r.read(), b"3")
        self.assertEqual(r.read(), b"4")
        self.assertEqual(r.read(), b"")
        self.assertEqual(r.read(), None)
        self.assertEqual(r.read(), None)
        r.close()

    def test_binary_image(self):
        #write
        w = recordio.writer("d:\\tmp\\image_binary")
        with open(".\\images\\10045_right_512", "rb") as f:
            con = f.read()

        d1 = {
            'img': con,
            'md5': get_md5_hexdigest(con)
        }

        #pickle
        p1 = pickle.dumps(d1, pickle.HIGHEST_PROTOCOL)
        print("in python before write:", get_md5_hexdigest(p1), len(p1))

        w.write(p1)
        w.close()

        #read
        r = recordio.reader("d:\\tmp\\image_binary")
        while True:
            p2 = r.read()
            if not p2:
                break
            print("in python after  read:", get_md5_hexdigest(p2), len(p2))

            d2 = pickle.loads(p2)
            self.assertEqual(get_md5_hexdigest(d2['img']), d2['md5'])

        r.close()


if __name__ == '__main__':
    unittest.main()
