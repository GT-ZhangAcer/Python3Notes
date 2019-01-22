import paddle.v2 as paddle

x = paddle.layer.data(name='x', type=paddle.data_type.dense_vector(13))
y = paddle.layer.fc(input=x, size=1, param_attr=paddle.attr.Param(name="fc.w"))
params = paddle.parameters.create(y)
print params["fc.w"].shape
