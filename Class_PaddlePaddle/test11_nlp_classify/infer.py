# Author: Acer Zhang
# Datetime:2020/5/18 13:12
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle
import paddle.fluid as fluid
import numpy as np

place = fluid.CPUPlace()
exe = fluid.Executor(place)

reviews_str = [
    'read the book forget the movie', 'this is a great movie', 'this is very bad'
]
reviews = [c.split() for c in reviews_str]

word_dict = paddle.dataset.imdb.word_dict()

UNK = word_dict['<unk>']


def reader():
    for sample in reviews:
        sample_data = [word_dict.get(word, UNK) for word in sample]
        sample_data = np.array(sample_data, dtype="int64").reshape(1, -1)
        yield sample_data


params_dirname = "./understand_sentiment_conv.inference.model"
[inferencer, feed_target_names, fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

batch_reader = fluid.io.batch(reader=reader, batch_size=128)
feeder = fluid.DataFeeder(place=place, feed_list=feed_target_names, program=inferencer)

assert feed_target_names[0] == "words"

for data in batch_reader():
    results = exe.run(inferencer,
                      feed=feeder.feed(data),
                      fetch_list=fetch_targets,
                      return_numpy=False)
    np_data = np.array(results[0])
    for i, r in enumerate(np_data):
        print("Predict probability of ", r[0], " to be positive and ", r[1],
              " to be negative for review \'", reviews_str[i], "\'")
