from PIL import Image

id_img=Image.open("G:/111.jpg")
print(id_img.format,id_img.size,id_img.mode)
ascii_char = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")
length = len(ascii_char)
(width,height) = id_img.size
id_img = id_img.resize((int(width*0.9),int(height*0.5)))
def convert1(id_img):
    id_img = id_img.convert("L")
    txt = ""
    for i in range(id_img.size[1]):
        for j in range(id_img.size[0]):
            gray = id_img.getpixel((j, i))
            unit = 256.0 / length
            txt += ascii_char[int(gray / unit)]
        txt += '\n'
    return  txt

txt = convert1(id_img)
f = open(r"G:/2.txt","w")
f.write(txt) 
f.close()

