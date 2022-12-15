# def myPower(x,n):
#     if n == 0:
#         return 1
#     else:
#         partial = myPower(x, n-1)
#         result = partial * x
#         return result
# a = myPower(2, 10)
#
# print(a)

# def binarySum(L, start, stop):
#     if start >= stop:
#         return 0
#     elif start == stop - 1:
#         return L[start]
#     else:
#         mid = (start+stop)//2
#         # global n = n+1
#         print(n, mid)
#         return binarySum(L, start, mid) + binarySum(L, mid, stop)
# def main():
#     L = [1,2,3,4,5,6,7]
#     print(binarySum(L,0,len(L)))
# n = 0
# main()
# # s = ListStack()
# import queue
# def change(n, result, a):
#     if result == 0 :
#         return 1
#     else:
#         change(n,n//2,a)
#         remain = result % 2
#         a.enquene
#
# a = ListQuene()
import pandas
import cv2
import matplotlib
import math
import numpy as np
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg = None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('res', outimage)
    cv2.waitKey(1)


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


video = cv2.VideoCapture("C:/Users/Sirius/Desktop/contest2   2022-11-24 10-45-19.avi")

while True:
    print('1')
    ret, img = video.read()
    if not ret:
        break

    target = cv2.imread("D:\WeChat Image_20221117164609.png")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img2)
    sum_saturation = np.sum(img2[:,:,1])
    area = 604*1008
    avg_saturation = sum_saturation/area
    sat_low = 0
    val_low = 140
    # Yellow
    lower_yellow = np.array([10, sat_low, val_low])
    upper_yellow = np.array([60, 255, 255])
    yellow_mask = cv2.inRange(img2, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(img, img, mask=yellow_mask)
    # cv2.imshow('res',yellow_result)
    # cv2.waitKey(0)
    # print(canny_img.shape)
    # Red
    lower_red = np.array([0, sat_low, val_low])
    upper_red = np.array([50, 255, 255])
    red_mask = cv2.inRange(img2, lower_red, upper_red)
    red_result = cv2.bitwise_and(img, img, mask=red_mask)
    reference1 = cv2.bitwise_and(s,v)

    reference2 = cv2.bitwise_or(yellow_result,red_result)
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

    target1 = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    h1, s1, v1 = cv2.split(target1)
    target2 = cv2.bitwise_and(s1, v1)
    target3 = cv2.copyMakeBorder(target2, 432-99, 432-99, 960-159, 960-58, cv2.BORDER_CONSTANT, value = 0) #扩大到一样的大小
    # print(target3.shape)
    Gaussian_img = cv2.GaussianBlur(reference3, (15,15), 0)
    canny_img = cv2.Canny(Gaussian_img, 220, 90)
    # cv2.imshow('res',canny_img)
    # cv2.waitKey(0)
    # print(canny_img.shape)

    Gaussian_tar = cv2.GaussianBlur(target3, (9,9), 0)
    canny_tar = cv2.Canny(Gaussian_tar, 230, 120)
    # cv2.imshow('res', canny_tar)
    # cv2.waitKey(0)

    orb = cv2.ORB_create()
    kp_img = orb.detect(canny_img)
    kp_tar = orb.detect(canny_tar)
    kp_img, des_img = orb.compute(canny_img, kp_img)
    kp_tar, des_tar = orb.compute(canny_tar, kp_tar)
    outimg1 = cv2.drawKeypoints(canny_img, keypoints = kp_img, outImage = None)
    outimg2 = cv2.drawKeypoints(canny_tar, keypoints = kp_tar, outImage = None)
    # cv2.imshow('res',outimg1)
    # cv2.waitKey(0)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des_img, des_tar)
        # print(matches)
    try:
        min_distance = matches[0].distance
        max_distance = matches[0].distance
    except:
        continue
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    good_match = []
    for x in matches :
        if x.distance <= max(2*min_distance, 30):
            good_match.append(x)

    draw_match(canny_img, canny_tar, kp_img, kp_tar, good_match)
    center = []
    total = 0
    list_match = []
    for mat in good_match:
        img_idx = mat.queryIdx
        (x,y) = kp_img[img_idx].pt
        list_match.append((x,y))

    match_x = []
    for i in range(len(list_match)):
        match_x.append(list_match[i][0])
        pass
    match_x1 = np.array(match_x)
    match_x2 = percent_range(match_x1)

    match_y = []
    for i in range(len(list_match)):
        match_y.append(list_match[i][1])
        pass
    match_y1 = np.array(match_y) #转化成array 才可以使用percentage的数据过滤
    match_y2 = percent_range(match_y1)
    try:
        avg_x = sum(match_x2)/len(match_x2)
        avg_y = sum(match_y2)/len(match_y2)
    except:
        continue
    avg_point = np.array([[avg_x+1, avg_y+1], [avg_x-1,avg_y-1], [avg_x+1,avg_y-1],[avg_x - 1, avg_y + 1]], dtype=np.int32)

    draw_img = img.copy()
    avg_point.reshape((-1,1,2))
    cv2.polylines(draw_img, [avg_point], True, (255,0,255), 5)

    print(avg_x,end=' ')
    print(avg_y)

    cv2.imshow('img', draw_img)
    cv2.waitKey(0)



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


