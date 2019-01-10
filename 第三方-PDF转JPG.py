from wand.image import Image
 
# 将pdf文件转为jpg图片文件
# ./PDF_FILE_NAME 为pdf文件路径和名称
image_pdf = Image(filename=r'G:\Temp_PDF\test.pdf',resolution=300)
image_jpeg = image_pdf.convert('jpg')
 
# wand已经将PDF中所有的独立页面都转成了独立的二进制图像对象。我们可以遍历这个大对象，并把它们加入到req_image序列中去。
req_image = []
for img in image_jpeg.sequence:
    img_page = Image(image=img)
    req_image.append(img_page.make_blob('jpg'))
 
# 遍历req_image,保存为图片文件
i = 0
for img in req_image:
    ff = open("G:\\Temp_PDF\\" + 'a' + str(i)+'.jpg','wb')
    ff.write(img)
    ff.close()
    print("转换成功第%s张图片"%(i+1))
    i += 1
