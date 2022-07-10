import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

fp=open("report.txt","w")

#1
calc=10*4-5
fp.write("+++1+++\n1. 10 * 4 - 5 = %d\n" % calc)
#2
fp.write("2. 10/3: quo = {:d}, mod = {:d}\n".format(10//3,10%3))
fp.write("10/3=%s, 10/3.0=%s\n" % (str(10/3),str(10/3.0)))

if calc%2:
    fp.write("+++2+++\n{:d} is odd\n".format(calc))
else:
    fp.write("+++2+++\n{:d} is even\n".format(calc))

#3
array=np.zeros((10),int)
for i in range(1,10):
    array[i]=array[i-1]+i
fp.write("+++3+++\nsequence of differences:\n")
for el in array:
    fp.write(str(el)+", ")

#4 & 5
for m in range(2):
    fp.write("\n+++%d+++\n" % (4+m))
    if m:
        array=[chr(i) for i in range(ord("a"),ord("k")+1) ]
    else:
        array=np.arange(11)

    fp.write("1.\n")
    for index in range(len(array)):
        fp.write(" {:2d}. {}\n".format(index,array[index]))
    fp.write("2. elements from 3rd to 5th are:\n ")
    array3to6=array[3:6]
    for el in array3to6:
        fp.write(str(el)+", ")
    fp.write("\n  And last element is {}\n".format(array[-1]))

    array_append=[]
    for i in range(11):
        if m:
            array_append.append(chr(ord("a")+i))
        else:
            array_append.append(i)

    fp.write("3. appended array is as below:\n")
    for el in array_append:
        fp.write(str(el)+", ")

    if m:
        array_append.pop(7)
    else:
        array_append.remove(array_append[7])
    fp.write("\nafter removing 8th elements:\n")
    for el in array_append:
        fp.write(str(el)+", ")

#6
f0=0
f1=1
for n in range(2,51):
    f2=f0+f1
    f0=f1
    f1=f2
fp.write("\n+++6+++\n50th number of finabocci is %d\n" % f1)

#7
a=np.zeros((3,3))
fp.write("+++7+++\n1. shape of a is {}\n".format(a.shape))
b=a/2
c=a+b
fp.write("a + a/2 = {}\n".format(c))
a=np.array([1,3,2,4,5])
fp.write("2. 3rd and later of a is {}\n".format(a[2:]))
a=a[np.where(a>3)[0]]
fp.write("3. elements larger than 3 in a is {}\n".format(a))
a=np.ones((3,3))
fp.write("4.\nBefore of a is {}\n".format(a))
a=np.where(a==1,2,a)
fp.write("After of a is {}\n".format(a))
a=np.arange(1,10).reshape(3,-1)
fp.write("array is {}\n".format(a))
fp.write("2nd column of array is {}\n".format(a[1,:]))

#8
flag=0
fp.write("+++8+++\n1.\n")
if flag:
    lena_bmp=Image.open(".\\data\\lenna.bmp")#Pillow
    ndarray_lena=np.array(lena_bmp)
    lena_info=ndarray_lena.shape
    fp.write("By pillow\n")

    lena_rgb=cv2.cvtColor(ndarray_lena,cv2.COLOR_BGR2RGB)#OpenCVはRGBの順番が逆
else:
    lena_bmp=cv2.imread(".\\data\\lenna.bmp")#OpenCV
    ndarray_lena=np.array(lena_bmp)
    lena_info=ndarray_lena.shape
    fp.write("By OpenCV\n")

    lena_rgb=ndarray_lena

fp.write("lena's width : %d\n" % lena_info[0])
fp.write("lena's height : %d\n" % lena_info[1])
fp.write("lena's channel number : %d\n" % lena_info[2])
#8.
cv2.imwrite(".\\8_2_OpenCV_lena_rgb.png",lena_rgb)

#3
lena_cut=lena_rgb[100:200,80:200]#高さ⇒幅の順番
cv2.imwrite(".\\8_3_lena_rgb_part.png",lena_cut)

#matplotlib
Dpi=200
fig=plt.figure(dpi=Dpi,figsize=(lena_info[0]//Dpi,lena_info[1]//Dpi))
plt.imshow(cv2.cvtColor(lena_rgb,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.axis("tight")
fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
plt.savefig(".\\8_2_matplotlib_lena_matplt.png")

#4 & 5:resize
lena_re=cv2.resize(lena_rgb,(2*lena_info[0]//3,2*lena_info[1]//3))
cv2.imwrite(".\\data\\8_4_lenna.bmp",lena_re)
#6:movie
movie=cv2.VideoCapture(".\\data\\human_video.avi")
fp.write("5.\nType: " + str(type(movie)))
fp.write("\nEnable: " + str(movie.isOpened()))
movie_w=movie.get(cv2.CAP_PROP_FRAME_WIDTH)
movie_h=movie.get(cv2.CAP_PROP_FRAME_HEIGHT)
fp.write("\nWidth: " + str(movie_w))
fp.write("\nHeight: " + str(movie_h))
fp.write("\nFPS: " + str(movie.get(cv2.CAP_PROP_FPS)))
fp.write("\nFrame Num: " + str(movie.get(cv2.CAP_PROP_FRAME_COUNT)))
fp.write("\nTotal time: " + str(movie.get(cv2.CAP_PROP_FRAME_COUNT)/movie.get(cv2.CAP_PROP_FPS)))

while (True):#display
    ret, frame = movie.read()
    if ret:
        frame_re=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
        cv2.imshow("fname",frame_re)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        movie.set(cv2.CAP_PROP_POS_FRAMES,0)

cv2.destroyWindow("fname")

###4-3
#1
milk=cv2.imread(".\\data\\milkdrop.bmp")
ndarray_milk=np.array(milk)
cv2.imwrite(".\\1_milk_rewrite.bmp",ndarray_milk)
#2
milk_gray=cv2.cvtColor(milk,cv2.COLOR_BGR2GRAY)
cv2.imwrite(".\\2_milkdrop_gray.bmp",milk_gray)
th, milk_gray_bin=cv2.threshold(milk_gray,0,255,cv2.THRESH_OTSU)
cv2.imwrite(".\\2_milkdrop_gray_bin.bmp",milk_gray_bin)
fp.write("\n+++4.3+++\n2:{:1f}".format(th))
fp.close()
#3
contours, hierarchy = cv2.findContours(milk_gray_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

Size=np.zeros((len(contours)))
for i in range(len(contours)):
    np_cont=np.squeeze(np.array(contours[i]))
    if np_cont.ndim>1:
        Size[i]=(np.max(np_cont[:,0])-np.min(np_cont[:,0]))*(np.max(np_cont[:,1])-np.min(np_cont[:,1]))

index=np.argmax(Size)#最大値を抽出
contours=contours[index]
cv2.drawContours(milk_gray,contours,-1,color=(0,0,255),thickness=-1)
cv2.imwrite(".\\3_milk_gray_outlines.bmp",milk_gray)

mask=np.zeros_like(milk_gray)
cv2.drawContours(mask,[contours],-1,color=255,thickness=-1)
cv2.imwrite(".\\4_masking_milk.bmp",mask)

milk_masked=np.zeros_like(milk)
for m in range(3):
    milk_masked[:,:,m]=milk[:,:,m]*(mask//np.max(mask))

cv2.imwrite(".\\5_only_milk_displayed.bmp",milk_masked)