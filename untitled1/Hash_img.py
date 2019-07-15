#coding-utf-8

import cv2
import numpy as np
import os
import shutil

#存放图片目录
path = "D:\\untitled1\\photo"
#用户传输原图
imgPathSrc = "\\untitled1\\photo\\p1.jpg"
#相似比例值
simil = 0.85
#相似图片存入地址
saveFile = "C:\\Users\\EDZ\\Desktop\\test_consult\\"

def ORBImgSimilarity(imgPathSrc,imgPathCom):
    """
    :param imgPathSrc: 原图路径
    :param imgPathCom: 待比较图片路径
    :return:图片相似度
    """
    try:
        #读取图片
        imgSrc = cv2.imread(imgPathSrc,cv2.IMREAD_GRAYSCALE)
        imgCom = cv2.imread(imgPathCom,cv2.IMREAD_GRAYSCALE)

        #初始化ORB检测器
        orb = cv2.ORB_create()
        kpSrc,deSrc = orb.detectAndCompute(imgSrc,None)
        kpCom,deCom = orb.detectAndCompute(imgCom,None)

        #提取并计算特征点
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        #knn筛选结果
        matches = bf.knnMatch(deSrc,trainDescriptors=deCom,k=2)

        #查看最大匹配点数目
        good = [m for (m,n) in matches if m.distance < 0.75 * n.distance]
        similary = len(good) / len(matches)
        return similary
    except:
        return '0'

#------------------------------PHash-------------------------------------#

#缩小图片尺寸并且图片灰度化
def imgResize(imgSrc):
    """
    :param imgSrc: 传输原图
    :return:缩小成8*8的图片
    """
    newImg = cv2.resize(imgSrc,(8,8))
    #cv2.imshow("newImg:",newImg)
    #将图片灰度化处理
    imgGray = cv2.cvtColor(newImg,cv2.COLOR_RGB2GRAY)
    return imgGray

#计算图片灰度平均值
def transformImg(imgGray):
    """
    :param imgGray: 处理后的8*8图片
    :return:图片的二维数组转一维数组
    """
    #获取灰度值
    imgTemp = np.array(imgGray)
    sum = 0
    for i in range(0,8):
        for j in range(0,8):
            sum += imgTemp[i,j]
    print("灰度值总和为: ",sum)
    avg = sum / 64
    print("灰度值均值为: ",avg)
    #二维化图像转为一维化图像
    imgArr = []
    for i in range(0,8):
        for j in range(0,8):
            if imgTemp[i][j] > avg:
                imgArr.append("1")
            else:
                imgArr.append("0")
    return imgArr

def comPercent(arrA,arrB):
    """
    :param arrA:传输原图的一维数组
    :param arrB:库图的一维数组
    :return:相似点总和
    """
    sum = 0
    for i in range(0,64):
        if arrA[i] == arrB[i]:
            sum += 1
    return sum

#获取图片
files = os.listdir(path)
#print("files: ",files)

#待比较库图
for i in range(0,len(files)):
    print("待比较图片为：" + files[i])
    imgPathCom = "\\untitled1\\photo\\" + files[i]
    imgSrc = cv2.imread(imgPathSrc)
    imgCom = cv2.imread(imgPathCom)

    imgGrayA = imgResize(imgSrc) #原图
    imgArrA = transformImg(imgGrayA)

    imgGrayB = imgResize(imgCom) #待比较图片
    imgArrB = transformImg(imgGrayB)

    #计算两者之差
    sum = 0
    sum = comPercent(imgArrA,imgArrB)
    # print(imgArrA)
    # print(imgArrB)
    print(sum / 64)
    #cv2.imshow("imgGrayA:" ,imgGrayA)
    #cv2.imshow("imgGrayB:" ,imgGrayB)

    #ORB算法
    like = ORBImgSimilarity(imgPathSrc,imgPathCom)
    print("ORB_like:",like)
    like = max(float(like),float(sum / 64))
    print("图片最大相似性为：",like)
    if like > simil:
        shutil.copy(imgPathCom,saveFile + str(like) + "_" + files[i])
        #shutil.copy("C:\\Users\\EDZ\\Desktop\\test_consult\\" + files[i],"C:\\Users\\EDZ\\Desktop\\test_consult\\" + like)
    print("---------------------------------------------")
    #打印图片尺寸大小
    #print("img_gray shape:{}".format(np.shape(imgray)))
    cv2.waitKey()