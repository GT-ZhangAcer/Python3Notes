def func(b):
    for i in b:
        yield i


b = [1, 2, 3]

for i in func(b):
    i += 1
    print(i)
