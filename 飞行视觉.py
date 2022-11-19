import cv2
import math
import numpy as np
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv2.drawMatches(img1, kp1, img2, kp2, match, outImg = None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('res', outimage)
    cv2.waitKey(0)


# 百分位法:原始参数 min=0.025， max=0.975
def percent_range(dataset, min=0, max=0.85):
    range_max = np.percentile(dataset, max * 100)
    range_min = -np.percentile(-dataset, (1 - min) * 100)

    # 剔除前20%和后80%的数据
    new_data = []
    for value in dataset:
        if value < range_max and value > range_min:
            new_data.append(value)
    return new_data
img=cv2.imread("D:\WeChat Image_20221119224242.png")
target = cv2.imread("D:\WeChat Image_20221117164609.png")

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(img2)
img3 = cv2.bitwise_and(s,v)

target1 = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
h1, s1, v1 = cv2.split(target1)
target2 = cv2.bitwise_and(s1, v1)
target3 = cv2.copyMakeBorder(target2, 432-99, 432-99, 960-159, 960-58, cv2.BORDER_CONSTANT, value = 0)
# print(target3.shape)
Gaussian_img = cv2.GaussianBlur(img3, (11,11), 0)
canny_img = cv2.Canny(Gaussian_img, 220, 90)
# cv2.imshow('res',canny_img)
# cv2.waitKey(0)
# print(canny_img.shape)

Gaussian_tar = cv2.GaussianBlur(target3, (7,7), 0)
canny_tar = cv2.Canny(Gaussian_tar, 230, 120)
# cv2.imshow('res', canny_tar)
# cv2.waitKey(0)

orb = cv2.ORB_create()
kp_img = orb.detect(canny_img)
kp_tar = orb.detect(canny_tar)
kp_img, des_img = orb.compute(canny_img, kp_img)
kp_tar, des_tar = orb.compute(canny_tar, kp_tar)
# outimg1 = cv2.drawKeypoints(canny_img, keypoints = kp_img, outImage = None)
# outimg2 = cv2.drawKeypoints(canny_tar, keypoints = kp_tar, outImage = None)
# cv2.imshow('res',outimg2)
# cv2.waitKey(0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.match(des_img, des_tar)
# print(matches)
min_distance = matches[0].distance
max_distance = matches[0].distance
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
match_y1 = np.array(match_y)
match_y2 = percent_range(match_y1)


avg_x = sum(match_x2)/len(match_x2)
avg_y = sum(match_y2)/len(match_y2)

# diff_x = []
# avg_x = np.average(match_x1)
# for m in range(len(match_x1)):
#     diff_x.append(abs(match_x1[m] - avg_x))
#     avg_diff_x = np.average(diff_x)
#     if abs(match_x1[m] - avg_x) >= avg_diff_x:
#         match_x1[m] = match_x1[m] - 100000000 * diff_x[m]
#         pass
#
# diff_y = []
# avg_y = np.average(match_y1)
# for n in range(len(match_y1)):
#     diff_y.append(abs(match_y1[n] - avg_y))
#     avg_diff_y = np.average(diff_y)
#     if abs(match_y1[n] - avg_y) >= avg_diff_y:
#         match_x1[n] = match_y1[n] - 5 * diff_y[n]
#         pass

print(avg_x,end=' ')
print(avg_y)
# cv2.imshow('img', img)
# cv2.waitKey(0)


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
