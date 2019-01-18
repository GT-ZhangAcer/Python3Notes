from wand.image import Image as Image1
from PIL import Image, ImageDraw, ImageFont
import os
import shutil

PathFile="C:/Temp_PDF_by_GT"
#欢迎界面
print("欢迎使用一键做图工具\n如果使用过程中有异常情况请联系QQ：1029550448\n-----------------------------")

#检索目录
L=[]
global PDF_number
PDF_number=1
for root, dirs, files in os.walk(PathFile): 
    
    
    for file in files:  
        if os.path.splitext(file)[1] == '.pdf': 
            L.append(os.path.join(file))
            print(str(PDF_number) + "、 " + str(L[PDF_number-1]))
            PDF_number+=1
print("共发现" + str(PDF_number-1) +"个可以做制作的PDF文件\n请选择需要制作的PDF序号\n-----------------------------")
PDF_FineNum=input()
PDF_FineNum0=int(PDF_FineNum)-1
PDF_Fine=str(L[PDF_FineNum0])
if PDF_FineNum0<100:
    print("你已经成功选择" + str(L[PDF_FineNum0]) +"文件\n-----------------------------\n现在开始制作---\n当前状态：第一步[共四步] 转换PDF文件")

"""第一步-转换PDF """
   
# 将pdf文件转为jpg图片文件
# ./PDF_FILE_NAME 为pdf文件路径和名称
image_pdf = Image1(filename="C:/Temp_PDF_by_GT/"+ PDF_Fine ,resolution=300)
image_jpeg = image_pdf.convert('jpg')
 
# wand已经将PDF中所有的独立页面都转成了独立的二进制图像对象。我们可以遍历这个大对象，并把它们加入到req_image序列中去。
req_image = []
for img in image_jpeg.sequence:
    img_page = Image1(image=img)
    req_image.append(img_page.make_blob('jpeg'))
 
# 遍历req_image,保存为图片文件
PDFi = 0
for img in req_image:
    ff = open("C:/Temp_PDF_by_GT/" + 'a' + str(PDFi)+'.jpeg','wb')
    ff.write(img)
    ff.close()
    PDFi += 1
    print("转换成功第%s张图片"%(PDFi))
    

"""第二步-获取空白部分与内容的首坐标"""

print("-----------------------------\n当前状态：第二步[共四步] 获取空白部分与内容的首坐标")

#旋转图像获取另一端坐标
def Tran (Img_Path,Angle):#Angle为角度
    Jpg_img=Image.open(Img_Path)
    Jpg_img.rotate(Angle).save(Img_Path)
    return direction
#翻转图像获取另一端坐标
def Tran2 (Img_Path,direction):#Angle为角度 direction中0为水平 1为垂直
    Jpg_img=Image.open(Img_Path)
    if direction==0:
        Jpg_img.transpose(Image.FLIP_LEFT_RIGHT).save(Img_Path)
    else:
        Jpg_img.transpose(Image.FLIP_TOP_BOTTOM).save(Img_Path)
    return direction

Pix_coordinate2=[10,10]
#获取空白部分与内容的首坐标

Apply_coordinateTemp=[]
def cutjpg(Img_Path,direction,Q):#direction为方向，0代表原图，1代表已经旋转180度,2代表上下翻转 Q=7#精确度（1-10之间最好）
    Jpg_img=Image.open(Img_Path)
    Jpg_img = Jpg_img.convert("L")
    (width,height) = Jpg_img.size
    global Apply_coordinateTemp
    
    s=1#环境变量
    i=0
    j=0
    Pix_coordinate=[10,10]
    if (i<width or j<height):#定位上半部分白色部分大小
        while (s<3 and i<width):
            j=0
            while s<=1:#双重循环进行定位
                if j==height-1: #检索完毕J 跳出循环执行下一步
                    break
                tempPixJ=Jpg_img.getpixel((i,j))#获取颜色并赋值 
                if (tempPixJ==0):#判断颜色是否为白,记录坐标
                    Pix_coordinate[0]=i
                    s=2
                    break
                j+=1;
            tempPixI=Jpg_img.getpixel((i,j))
            if (tempPixI==0 and tempPixJ==0):
                Pix_coordinate[1]=j
                break
            i+=1
    #精确获取坐标
    j=0
    while (s==2 and (i+Q<width and j<height)):
        tempPixJ=Jpg_img.getpixel((i+Q,j))
        if (tempPixJ==0):#判断颜色是否为白,记录坐标
            Pix_coordinate[1]=j
            break
        j+=1
    #Pix_coordinate中 0为Width坐标 1为Hight坐标    
    if direction==0:
        Pix_coordinate1=Pix_coordinate
        print("已确定第一次Width:" + str(Pix_coordinate[0]) + "   Height:" + str(Pix_coordinate[1]))
        Apply_coordinateTemp.append(Pix_coordinate[0])
        Apply_coordinateTemp.append(Pix_coordinate[1])
    if direction==1:
        print("第二坐标为" + str(Pix_coordinate))
        Apply_coordinateTemp.append(Pix_coordinate[0])
    if direction==2:
        print("第三坐标为" + str(Pix_coordinate))
        Apply_coordinateTemp.append(Pix_coordinate[1])
        
_PDFi=0#循环的环境变量

while _PDFi<PDFi:
    Img_Path=(PathFile + "/a%s.jpeg" %_PDFi)
    direction=0
    cutjpg(Img_Path,direction,8)#第一次输出
    Tran(Img_Path,180)
    direction+=1
    cutjpg(Img_Path,direction,7)#第二次输出
    direction+=1
    Tran2(Img_Path,0)
    cutjpg(Img_Path,direction,9)#第三次输出
    Tran2(Img_Path,1)
    _PDFi+=1
#经过多次数据统计 H最终值-10最合理
print("获取坐标成功！")
print("最终坐标值为" + str(Apply_coordinateTemp))
"""第三步 切割图片并拼接正文以及下方推广区域"""

print("-----------------------------\n当前状态：第三步[共四步] 切割图片并拼接正文以及下方推广区域")

#切割图片
Apply_coordinate=Apply_coordinateTemp#每四个为一组，一组对应一个图的 上双坐标和下双坐标[左width 上height 右--下--]

def Cutting_img(PathFile,Apply_coordinate,Img_num):#上半部分切割 Apply_coordinate为应用坐标列表
    Jpg_img=Image.open(PathFile + "/a" + str(Img_num[0]-1) + ".jpeg")
    (width,heigth)=Jpg_img.size
    
    Apply_coordinate[0+(Img_num[0]-1)*4]-=45 #增加切割空白
    if Img_num[0]==1:
        Apply_coordinate[1+(Img_num[0]-1)*4]-=45 #保留切割空白仅第一页 
    box=(Apply_coordinate[0],Apply_coordinate[1],width,heigth)#左上右下
    Make1Img=Jpg_img.crop(box)
    Make1Img.save(PathFile + "/b_" + str(Img_num[0]-1) + ".jpeg")
        


def Cutting_img2(PathFile,Apply_coordinate,Img_num):#下半部分切割
    Jpg_img=Image.open(PathFile + "/b_" + str(Img_num[0]-1) + ".jpeg")
    (width,heigth)=Jpg_img.size
    if Img_num[0]==Img_num[1]:#保留切割空白仅最后一页
        Apply_coordinate[3+(Img_num[0]-1)*4]=heigth-Apply_coordinate[3+(Img_num[0]-1)*4]+45
    else:
        Apply_coordinate[3+(Img_num[0]-1)*4]=heigth-Apply_coordinate[3+(Img_num[0]-1)*4]
    Apply_coordinate[2+(Img_num[0]-1)*4]=width-Apply_coordinate[2+(Img_num[0]-1)*4]+45
    box=(0,0,Apply_coordinate[2+(Img_num[0]-1)*4],Apply_coordinate[3+(Img_num[0]-1)*4])#左上右下
    Make1Img=Jpg_img.crop(box)
    Make1Img.save(PathFile + "/c_" + str(Img_num[0]-1) + ".jpeg")
    
Img_num=[0,0]
Img_num[1]=2 #图片数量 1为最大值 0为当前值
Img_num[0]=1
if Img_num[1]>1:
    while Img_num[0]<=Img_num[1]:
        Cutting_img(PathFile,Apply_coordinate,Img_num)
        Cutting_img2(PathFile,Apply_coordinate,Img_num)
        Img_num[0]+=1

Img_num[0]=1 #重新声明数量，上方参数已经发生变动！
#合并图片
def Copy_Img(PathFile,Img_num):
    global Tempwidth#定义全局变量
    global Tempheight
    Tempwidth=0 
    Tempheight=0
    while Img_num[0]<=Img_num[1]:#获取生成图最大长度
        Jpg_img=Image.open(PathFile + "/c_" + str(Img_num[0]-1) + ".jpeg")
        (width,heigth)=Jpg_img.size
        Tempheight+=heigth
        Img_num[0]+=1
    Make1Img=Image.new('RGBA',(1819,Tempheight+417))
    Make1Img.save(PathFile + "/D" + ".png")
    Img_num[0]=1
    """#添加水印1
    WaterMarkNum=(Tempheight//2200) #向下取整，决定需要几个水印 这里减去那个水印 2200这个值可以更改
    hightsmall=Tempheight#控制小图水印
    print("需要添加的水印个数为" + str(WaterMarkNum))
    """
    #拼接图片
    Tempheight=0
    Img_num[0]=1#重新声明数量，上方参数已经发生变动！
    while Img_num[0]<=Img_num[1]:
        Jpg_img=Image.open(PathFile + "/c_" + str(Img_num[0]-1) + ".jpeg")
        (width,heigth)=Jpg_img.size
        box=(0,Tempheight,width,Tempheight+heigth)
        Tempheight+=heigth
        Make1Img.paste(Jpg_img,box) #粘贴到指定区域
        Make1Img.save(PathFile + "/E" + ".png")
        Img_num[0]+=1
    if Img_num[0]==Img_num[1]+1:
        Jpg_img=Image.open(PathFile + "/SysIMG/DownJPG.jpg").convert('RGBA')#打开并更换模式为RGBA
        box=(Tempwidth,Tempheight,Tempwidth+1869,Tempheight+417)
        Make1Img.paste(Jpg_img,box) #粘贴到指定区域
        Make1Img.save(PathFile + "/E.png")
    """#添加水印2
    
    s=1#水印的环境变量
    if WaterMarkNum!=0:
        Jpg_img=Image.open(PathFile + "/SysIMG/Watermark.png")
        Make1Img=Image.open(PathFile + "/E" + ".png")
        while s<=WaterMarkNum:
            box=(0,1950*s,1819,1950*s+529)
            Make1Img.paste(Jpg_img,box)
            s+=1
    
    if (WaterMarkNum==0 and hightsmall>=600):
        Jpg_img=Image.open(PathFile + "/SysIMG/Watermark0.png").convert('RGBA')
        Make1Img=Image.open(PathFile + "/E" + ".png")
        box=(45,hightsmall/2-250,1774,hightsmall/2+250)
        Make1Img.paste(Jpg_img,box)
   
    Make1Img.save(PathFile + "/F" + ".png")    
    
    Waetmark_Z_img=Image.open(PathFile + "/F.png")#对水印进行修正
    (waterW,waterH)=(1819,hightsmall)
    wateri=0
    waterj=0
    while wateri<waterW:
        while waterj<waterH:
            Color=Waetmark_Z_img.getpixel((wateri,waterj))
            if Color==(255,255,255,255):
                Color=(255,255,255,0)
            waterj+=1
        wateri+=1
    Waetmark_Z_img.save(PathFile + "/F.png")
   """
    
Copy_Img(PathFile,Img_num)
print("拼接完成！")

"""第四步 合并标题并制作成图"""
print("-----------------------------\n当前状态：第四步[共四步] 合并标题并制作成图")
global Tsize #声明全局变量-文字列表1x3 0为行数 1为第一行文字 2为第二行文字
Tsize=[0," "," "]
LNum=999


def ahhhh():#标题行数输入
    global LNum
    HelloTitle="-----------------------------\n接下来该写入标题了，这部分需要你进行输入 \n第一个问题：你需要单行还是双行标题呢？（输入1或2）\n超过20字标题选择双行,低于20字禁用双行！\n请输入1或2_\n-----------------------------"
    text0=input(HelloTitle)
    HelloTitle2="你需要单行还是双行标题呢？（输入1或2）"
    if text0=="1":
        print("你选择的是1行标题\n-----------------------------")
        LNum=1
        Tsize[0]=1
        Temptext=input("请输入第一行标题_")
        Tsize[1]=Temptext
        
    elif text0=="2":
        print("你选择的是2行标题\n-----------------------------")
        Tsize[0]=2
        Temptext=input("请输入第一行标题_")
        Tsize[1]=Temptext
        Temptext=input("请输入第二行标题_")
        Tsize[2]=Temptext
        LNum=2
        
    else :
        if LNum==998:
            print("我生气了，走了，哼！")
        else:
            print("你闹哪样？我看不懂你的操作了！\n算了，再给你一次机会\n......")
            ahhhh()
def Queding():#确定标题输入正确
    print("-----------------------------\n你输入的标题为_"+Tsize[1]+Tsize[2])
    s=input("确定为该标题请输入yes，重新输入请随便输入_")
    if s=="yes":
        #下一个函数
        print("通过")
    else:
        ahhhh()            
##########
ahhhh()
Queding()
##########
#制作标题
textsize=[0,0,0]
textsize[0]=Tsize[0]
textsize[1]=len(Tsize[1])
textsize[2]=len(Tsize[2])+len(Tsize[1])
def text(PathFile,textsize):
    if textsize[0]==1:
        if textsize[1]<=10:
            height=150
            fontsize=120
        elif textsize[1]<=15:
            height=120
            fontsize=100
        elif textsize[1]<25:
            height=105
            fontsize=85
    else:
        if textsize[2]<=30:
            height=240
            fontsize=90
        elif textsize[2]<=40:
            height=180
            fontsize=70
    if textsize[0]==1:
        img0 = Image.new(mode="RGB",size=(1700,height),color=(2,72,134))
        img = ImageDraw.Draw(img0)
        img.text((0, 0), Tsize[1], (255,255,255), font=ImageFont.truetype(PathFile + "/SysFile/Yahei.ttf", fontsize))
        img = ImageDraw.Draw(img0)
        img0.save(PathFile + "/F.png","PNG")
        img0.close
    else:
        img0 = Image.new(mode="RGB",size=(1700,height),color=(2,72,134))
        img = ImageDraw.Draw(img0)
        img.text((0, 0), Tsize[1], (255,255,255), font=ImageFont.truetype(PathFile + "/SysFile/Yahei.ttf", fontsize))
        img.text((0,fontsize+30), Tsize[2], (255,255,255), font=ImageFont.truetype(PathFile + "/SysFile/Yahei.ttf", fontsize))
        img = ImageDraw.Draw(img0)
        img0.save(PathFile + "/F.png","PNG")
        img0.close
    #合并上方
    img = Image.new(mode="RGB",size=(1819,height+270),color=(2,72,134))#做底图
    Jpg_img=Image.open(PathFile + "/F.png")
    (width,height)=Jpg_img.size
    Jpg_watermark=Image.open(PathFile + "/SysIMG/UpJPG.jpg")
    box0=(0,0,1840,179)
    img.paste(Jpg_watermark,box0)
    box1=(45,200,45+width,200+height)
    img.paste(Jpg_img,box1)
    img.save(PathFile + "/GT.png","PNG")
#########
text(PathFile,textsize)
#########
#制作成图
Num0=Image.open(PathFile + "/GT.png")
Num1=Image.open(PathFile + "/E.png")
(Num0W,Num0H)=Num0.size
(Num1W,Num1H)=Num1.size
FinalING = Image.new(mode="RGB",size=(1819,Num0H+Num1H-20))
FinalING.paste(Num0,(0,0,Num0W,Num0H))
FinalING.paste(Num1,(0,Num0H-20,Num0W,Num0H+Num1H-20))
FinalING.save(PathFile + "/制作完毕.jpg")
FinalING.show()
shutil.copyfile(PathFile + "/制作完毕.jpg", "C:/微博长图文件夹/制作完毕.jpg")
