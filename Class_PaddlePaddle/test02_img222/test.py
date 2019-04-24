import paddle.fluid

img = paddle.dataset.image.load_image("./data/1.jpg")
print(img)
print(len(img),len(img[0]),len(img[0][0]))