
from visualdl import LogWriter
from random import random

logw = LogWriter("g:/random_log", sync_cycle=10000)
with logw.mode('train') as logger:
    scalar0 = logger.scalar("scratch/scalar")

with logw.mode('test') as logger:
    scalar1 = logger.scalar("scratch/scalar")

# add scalar records.
for step in range(200):
    scalar0.add_record(step, step * 1. / 200)
    scalar1.add_record(step, 1. - step * 1. / 200)