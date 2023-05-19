import pandas
import cv2
import matplotlib
import math
import numpy as np
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg = None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('res', outimage)
    cv2.waitKey(1)

def data_augment(img, brightness):
    table = np.array([(i / 255.0) * brightness * 255 for i in np.arange(0, 256)]).clip(0,255)
    image = cv2.LUT(img, table)
    return image
    # 百分位法:原始参数 min=0.025， max=0.975
def percent_range(dataset, min=0, max=1):
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)

        # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
    return new_data


video = cv2.VideoCapture("D:\作品\contest2   2022-11-24 10-45-19 - 副本.mp4")

while True:
    # print('1')
    ret, img = video.read()
    if not ret:
        break

    target = cv2.imread("D:\WeChat Image_20221117164609.png") #读取数据
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #从BGR变为HSV图像，为后面分离H,S,V做准备
    h,s,v = cv2.split(img2)
    sum_saturation = np.sum(img2[:,:,1])
    area = 604*1008
    avg_saturation = sum_saturation/area
    sat_low = 0
    val_low = 140
    # 黄色掩膜构建，滤去其他颜色，只留下黄色
    lower_yellow = np.array([10, 40, 200])
    upper_yellow = np.array([60, 255, 255])
    yellow_mask = cv2.inRange(img2, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(img, img, mask=yellow_mask)
    cv2.imshow('yellow result',yellow_result)
    cv2.waitKey(1)
    # print(canny_img.shape)
    # 红色掩膜构建，留下红色
    red_source1 = cv2.normalize(img2, dst = None, alpha = 100, beta = 0, norm_type = cv2.NORM_MINMAX)
    red_source2 = data_augment(red_source1, 1.5) #调整亮度，因为红圈图像的图像亮度，对比度太低了，单纯的掩膜无法区分环境和暗红色
    lower_red = np.array([40, 45, 0])
    upper_red = np.array([180, 255, 160])
    red_mask = cv2.inRange(red_source2, lower_red, upper_red)
    red_result = cv2.bitwise_and(img, img, mask=red_mask)
    reference1 = cv2.bitwise_and(s,v)
    Gaussian_red = cv2.GaussianBlur(red_result, (13, 13), 0)
    canny_red = cv2.Canny(Gaussian_red, 180, 50)
    # cv2.imshow('canny_red', canny_red)
    # cv2.waitKey(1)
    # cv2.imshow('red_res', red_result)
    # cv2.waitKey(1)
    #合并红色黄色掩膜的过滤成果，当黄圈掩膜有数据就用黄掩膜的数据，用红掩膜的数据就用红色掩膜

    reference2 = cv2.bitwise_or(yellow_result, red_result)
    cv2.imshow('ref2', reference2)
    cv2.waitKey(1)
    temp1 = cv2.cvtColor(reference2, cv2.COLOR_RGB2HSV)
    h2, s2, v2 = cv2.split(temp1)
    reference3 = cv2.bitwise_and(s2, v2)
    # temp_rgb = cv2.cvtColor(reference2, cv2.COLOR_HSV2RGB)
    # gray = cv2.cvtColor(reference2, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)
    # img3 = cv2.bitwise_and(reference1, gray)
    # cv2.imshow('res',reference3)
    # cv2.waitKey(0)
    # print(canny_img.shape)
    #目标环的图像处理，使用BFMacher输出匹配的点

    Gaussian_img = cv2.GaussianBlur(reference3, (13,13), 0)
    canny_img = cv2.Canny(Gaussian_img, 180, 50)
    cv2.imshow('canny_img',canny_img)
    cv2.waitKey(1)
    # print(canny_img.shape)、


    canny_img = cv2.GaussianBlur(canny_img, (9,9), 0)
    circles = cv2.HoughCircles(canny_img, method=cv2.HOUGH_GRADIENT_ALT, dp=1.5, param1=150, param2=0.5, minDist=100, minRadius=20, maxRadius=100)
    try:
        if not circles:
            continue
    except:
        pass

    print(circles)
#过滤数据

    match_x = []
    for i in range(len(circles[0])):
        match_x.append(circles[0][i][0])
        pass
    match_x1 = np.array(match_x)
    match_x2 = percent_range(match_x1)

    match_y = []
    for i in range(len(circles[0])):
        match_y.append(circles[0][i][1])
        pass
    match_y1 = np.array(match_y) #转化成array 才可以使用percentage的数据过滤
    match_y2 = percent_range(match_y1)
    # print("before try")
    print(match_x1, match_y1)
    try:
        avg_x = sum(match_x1)/len(match_x1)
        avg_y = sum(match_y1)/len(match_y1)
    except:
        continue
    # print("?!?!?!??!??!?!")
    avg_point = np.array([[avg_x+1, avg_y+1], [avg_x-1,avg_y-1], [avg_x+1,avg_y-1],[avg_x - 1, avg_y + 1]], dtype=np.int32)
#画点
    draw_img = img.copy()
    avg_point.reshape((-1,1,2))
    cv2.polylines(draw_img, [avg_point], True, (255,0,255), 5)

    print(avg_x,end=' ')
    print(avg_y)

    cv2.imshow('img', draw_img)
    cv2.waitKey(1)


    # print(center)





    # contours, hierarchy = cv2.findContours(, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # res = cv2.matchTemplate(img, target0, cv2.TM_SQDIFF_NORMED)
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    # x1 = min_loc[0]
    # x2 = min_loc[1]
    #
    # cut = img[x2:x2+200,x1:x1+200]
    # target_set1 = x2+100
    # target_set2 = x1+100
    # print(target_set1,end=' ')
    # print(target_set2)
    # draw_img = v1.copy()
    # contours = list(contours)
    # right_contour = []
    # for i in range(len(contours)):
    #
    #     area = cv2.contourArea(contours[i])
    #     if 100 < area < 1000:
    #         right_contour.append(contours[i])
    #         pass
    # print(right_contour)
    # res = cv2.drawContours(draw_img, right_contour, 1, (255,0,255), 50)