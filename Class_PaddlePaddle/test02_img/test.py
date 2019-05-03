import paddle.fluid
from PIL import Image

im = Image.open("./data/1.jpg").convert('1')
im.show()