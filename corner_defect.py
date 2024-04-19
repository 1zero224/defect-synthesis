import cv2
import numpy as np
import json
import os
import math
from tqdm import tqdm


def find_corners_of_approx_parallel_lines(lines, angle_tolerance):
    parallel_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance or abs(angle - 180) < angle_tolerance or abs(angle + 90) < angle_tolerance:
                parallel_lines.append(line)
    vertical_points = []
    horizontal_points = []

    for line in parallel_lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < angle_tolerance or abs(angle - 90) < angle_tolerance or abs(angle - 180) < angle_tolerance or abs(angle + 90) < angle_tolerance:
            if abs(angle) < angle_tolerance or abs(angle - 180) < angle_tolerance:
                horizontal_points.append(line[0][0:2])
                horizontal_points.append(line[0][2:])
            elif abs(angle - 90) < angle_tolerance or abs(angle + 90) < angle_tolerance:
                vertical_points.append(line[0][0:2])
                vertical_points.append(line[0][2:])

    return vertical_points,horizontal_points


def compute_intersection(line1, line2):
    # 解析线段
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    # 计算线段的交点
    d = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    # 如果两线段平行，则没有交点
    if d == 0:
        assert False, "线段平行"
        
    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / d
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / d

    return (px, py)


def find_missing_corner(defect_img,thresh_min,thresh_max,defect_quadrant):
    gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, thresh_min, thresh_max)
    # 使用霍夫线变换找到线段
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=5, minLineLength=5, maxLineGap=5)
    vertical_points ,horizontal_points=find_corners_of_approx_parallel_lines(lines, angle_tolerance=10)

    height, width = defect_img.shape[:2]
    if defect_quadrant=="top_left" :
        vertical_point=min(vertical_points, key=lambda point: point[0]**2 + (height-point[1])**2)
        horizontal_point=min(horizontal_points, key=lambda point: (width-point[0])**2 + point[1]**2)
    elif defect_quadrant=="top_right" :
        vertical_point=min(vertical_points, key=lambda point: (width-point[0])**2 + (height-point[1])**2)
        horizontal_point=min(horizontal_points, key=lambda point: point[0]**2 + point[1]**2)
    elif defect_quadrant=="bottom_left" :
        vertical_point=min(vertical_points, key=lambda point: point[0]**2 + point[1 ]**2)
        horizontal_point=min(horizontal_points, key=lambda point: (width-point[0])**2 + (height-point[1])**2)
    else:
        vertical_point=min(vertical_points, key=lambda point: (width-point[0])**2 + point[1]**2)
        horizontal_point=min(horizontal_points, key=lambda point: point[0]**2 + (height-point[1])**2)

    return vertical_point,horizontal_point


def find_corner(image, quadrant="top_left, top_right, bottom_left, bottom_right"):
    # 将图片转为灰度图
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)

    # 寻找连通域
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_img)
    
    # 找到最大连通域的索引
    max_area = 0
    max_label = -1
    for i in range(1, stats.shape[0]):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > max_area:
            max_area = area
            max_label = i
    
    # 构建最大连通域的掩码
    max_connected_mask = np.zeros_like(binary_img)
    max_connected_mask[labels == max_label] = 255

    contours, _ = cv2.findContours(max_connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    def rect_area(contour):
        rect = cv2.minAreaRect(contour)
        return rect[1][0] * rect[1][1]
    max_contour = max(contours, key=rect_area)

    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
    num_kp=400
    while True:
        orb = cv2.ORB_create(nfeatures=num_kp)
        keypoints, _ = orb.detectAndCompute(mask, None)

        height, width = gray_img.shape[:2]
        quadrant_keypoints = [[], [], [], []]  # 初始化各区域关键点列表为空列表
        for kp in keypoints:
            x, y = kp.pt
            if x < width / 2 and y < height / 2:
                quadrant_keypoints[0].append(kp)  # 左上区域
            elif x >= width / 2 and y < height / 2:
                quadrant_keypoints[1].append(kp)  # 右上区域
            elif x < width / 2 and y >= height / 2:
                quadrant_keypoints[2].append(kp)  # 左下区域
            else:
                quadrant_keypoints[3].append(kp)  # 右下区域
        
        if quadrant=="top_left, top_right, bottom_left, bottom_right":
            if quadrant_keypoints[0]!=[] and quadrant_keypoints[1]!=[] and quadrant_keypoints[2]!=[] and quadrant_keypoints[3]!=[]:
                break
            else:
                num_kp+=50
            if num_kp>500: 
                assert False, "找不到四个角点"
        else:
            break

    corner_points, corner_point = [], []
    if "top_left" in quadrant:
        min_distance = float('inf')
        for kp in quadrant_keypoints[0]:
            distance = np.linalg.norm(np.array(kp.pt))
            if distance < min_distance:
                min_distance = distance
                corner_point=kp.pt
        corner_points.append(corner_point)
    if "top_right" in quadrant:
        min_distance = float('inf')
        for kp in quadrant_keypoints[1]:
            distance = np.linalg.norm(np.array(kp.pt) - np.array([width - 1, 0]))
            if distance < min_distance:
                min_distance = distance
                corner_point=kp.pt
        corner_points.append(corner_point)
    if "bottom_left" in quadrant:
        min_distance = float('inf')
        for kp in quadrant_keypoints[2]:
            distance = np.linalg.norm(np.array(kp.pt) - np.array([0, height - 1]))
            if distance < min_distance:
                min_distance = distance
                corner_point=kp.pt
        corner_points.append(corner_point)
    if "bottom_right" in quadrant:
        min_distance = float('inf')
        for kp in quadrant_keypoints[3]:
            distance = np.linalg.norm(np.array(kp.pt) - np.array([width - 1, height - 1]))
            if distance < min_distance:
                min_distance = distance
                corner_point=kp.pt
        corner_points.append(corner_point)

    if(len(quadrant)<20): return corner_points[0]
    else: return corner_points
    

def find_center_point_defect2(base_img,defect_img,defect_quadrant,thresh_min,thresh_max):
    vertical_point,horizontal_point=find_missing_corner(defect_img,thresh_min,thresh_max,defect_quadrant)
    height, width = defect_img.shape[:2]
    if defect_quadrant=="top_left" :
        quadrant_corner=[width,height,horizontal_point[1],-horizontal_point[1],-vertical_point[0],width/2,height/2]
    elif defect_quadrant=="top_right" :
        quadrant_corner=[-width,height,horizontal_point[1],-horizontal_point[1],width-vertical_point[0],-width/2,height/2]
    elif defect_quadrant=="bottom_left" :
        quadrant_corner=[width,-height,-height+horizontal_point[1],height-horizontal_point[1],-vertical_point[0],width/2,-height/2]
    else:
        quadrant_corner=[-width,-height,-height+horizontal_point[1],height-horizontal_point[1],width-vertical_point[0],-width/2,-height/2]

    base_corner=find_corner(base_img, defect_quadrant)
    base_img_gray=cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    edge_img=base_img_gray[int(base_corner[1]-80):int(base_corner[1]+80),int(base_corner[0]+quadrant_corner[0]-80):int(base_corner[0]+quadrant_corner[0]+80)]

    edge=cv2.Canny(edge_img, thresh_min, thresh_max)
    edge_y=80
    for y in range(0,160):
        if edge[y,80]==255: 
            edge_y=y
            break
    base_corner=(base_corner[0],base_corner[1]-80+edge_y+quadrant_corner[3])
    edge_img=base_img[int(base_corner[1]+quadrant_corner[1]-80):int(base_corner[1]+quadrant_corner[1]+80),int(base_corner[0]-80):int(base_corner[0]+80)]

    edge=cv2.Canny(edge_img, thresh_min, thresh_max)
    edge_x=80
    for x in range(0,160):
        if edge[80,x]==255: 
            edge_x=x
            break
    base_corner=(base_corner[0]-80+edge_x+quadrant_corner[4],base_corner[1])
    edge_img=base_img[int(base_corner[1]+quadrant_corner[2]-80):int(base_corner[1]+quadrant_corner[2]+80),int(base_corner[0]+quadrant_corner[0]-80):int(base_corner[0]+quadrant_corner[0]+80)]

    edge=cv2.Canny(edge_img, thresh_min, thresh_max)
    edge_y=80
    for y in range(0,160):
        if edge[y,80]==255: 
            edge_y=y
            break
    base_corner=(base_corner[0],base_corner[1]-80+edge_y)
    center_point=(int(base_corner[0]+quadrant_corner[5]),int(base_corner[1]+quadrant_corner[6]))
    return center_point


def find_bg(defect_img, base_img, center_point, defect_quadrant):
    if defect_quadrant == "top_left" or defect_quadrant == "top_right":
        if center_point[1] - defect_img.shape[0]/2 < defect_img.shape[0]:
            if defect_quadrant == "top_left":
                if center_point[0] - defect_img.shape[1]/2 < defect_img.shape[1]:
                    if center_point[0] < defect_img.shape[1]: assert False, "底图距离边缘位置过近, 不足以融合缺陷"
                    else: bg_x1, bg_x2 = 0, defect_img.shape[1]
                else:
                    bg_x2 = int(center_point[0] - defect_img.shape[1]/2)
                    bg_x1 = bg_x2 - defect_img.shape[1]
            else:
                if base_img.shape[1] - center_point[0] - defect_img.shape[1]/2 < defect_img.shape[1]: 
                    if base_img.shape[1] - center_point[0] < defect_img.shape[1]: assert False, "底图距离边缘位置过近, 不足以融合缺陷" # bg_x1, bg_x2 = base_img.shape[1] - 1 - defect_img.shape[1], base_img.shape[1] - 1
                    else: bg_x1, bg_x2 = base_img.shape[1] - 1 - defect_img.shape[1], base_img.shape[1] - 1
                else:
                    bg_x1 = int(center_point[0] + defect_img.shape[1]/2)
                    bg_x2 = bg_x1 + defect_img.shape[1]
            bg_y1 = int(center_point[1] - defect_img.shape[0]/2)
            bg_y2 = bg_y1 + defect_img.shape[0]
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 1)
        else:
            bg_y2 = int(center_point[1] - defect_img.shape[0]/2)
            bg_y1 = bg_y2 - defect_img.shape[0]
            bg_x1 = int(center_point[0] - defect_img.shape[1]/2)
            bg_x2 = bg_x1 + defect_img.shape[1] 
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 0)
    else:
        if base_img.shape[0] - center_point[1] - defect_img.shape[0]/2 < defect_img.shape[0]:
            if defect_quadrant == "bottom_left":
                if center_point[0] - defect_img.shape[1]/2 < defect_img.shape[1]:
                    if center_point[0] < defect_img.shape[1]: assert False, "底图距离边缘位置过近, 不足以融合缺陷"
                    else: bg_x1, bg_x2 = 0, defect_img.shape[1]
                else:
                    bg_x2 = int(center_point[0] - defect_img.shape[1]/2)
                    bg_x1 = bg_x2 - defect_img.shape[1]
            else:
                if base_img.shape[1] - center_point[0] - defect_img.shape[1]/2 < defect_img.shape[1]:
                    if base_img.shape[1] - center_point[0] < defect_img.shape[1]: assert False, "底图距离边缘位置过近, 不足以融合缺陷"
                    else: bg_x1, bg_x2 = base_img.shape[1] - 1 - defect_img.shape[1], base_img.shape[1] - 1
                else:
                    bg_x1 = int(center_point[0] + defect_img.shape[1]/2)
                    bg_x2 = bg_x1 + defect_img.shape[1]
            bg_y1 = int(center_point[1] - defect_img.shape[0]/2)
            bg_y2 = bg_y1 + defect_img.shape[0]
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 1)
        else:
            bg_y1 = int(center_point[1] + defect_img.shape[0]/2)
            bg_y2 = bg_y1 + defect_img.shape[0]
            bg_x1 = int(center_point[0] - defect_img.shape[1]/2)
            bg_x2 = bg_x1 + defect_img.shape[1]
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 0)
    
    return background

def find_center_point_defect1(center_point, base_img, defect_img, defect_quadrant, thresh_min, thresh_max):
    base_img_gray=cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    defect_img_gray=cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    _, defect_img_gray=cv2.threshold(defect_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge=cv2.Canny(defect_img_gray, thresh_min, thresh_max)

    if defect_quadrant=="top" or defect_quadrant=="bottom" :
        edge_p_defect=defect_img.shape[0]//2
        for p in range(0,defect_img.shape[0]-1):
            if edge[p,0]==255: 
                edge_p_defect=p
                break

        edge_img=base_img_gray[int(center_point[1]-defect_img.shape[0]//2):int(center_point[1]+defect_img.shape[0]//2),int(center_point[0]-defect_img.shape[1]//2):int(center_point[0]+defect_img.shape[1]//2)]
        _, edge_img=cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge=cv2.Canny(edge_img, thresh_min, thresh_max)
        edge_p_base=defect_img.shape[0]//2
        for p in range(0,defect_img.shape[0]-1):
            if edge[p,0]==255: 
                edge_p_base=p
                break
        center_point=(center_point[0],center_point[1]-(edge_p_defect-edge_p_base))
    else:
        edge_p_defect=defect_img.shape[1]//2
        for p in range(0,defect_img.shape[1]-1):
            if edge[0,p]==255: 
                edge_p_defect=p
                break
        
        edge_img=base_img_gray[int(center_point[1]-defect_img.shape[0]//2):int(center_point[1]+defect_img.shape[0]//2),int(center_point[0]-defect_img.shape[1]//2):int(center_point[0]+defect_img.shape[1]//2)]
        _, edge_img=cv2.threshold(edge_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge=cv2.Canny(edge_img, thresh_min, thresh_max)
        edge_p_base=defect_img.shape[1]//2
        for p in range(0,defect_img.shape[1]-1):
            if edge[0,p]==255: 
                edge_p_base=p
                break
        center_point=(center_point[0]-(edge_p_defect-edge_p_base),center_point[1])

    return center_point


def point_to_segment_dist(point, segment_start, segment_end):
    point = np.array(point)
    segment_start = np.array(segment_start)
    segment_end = np.array(segment_end)

    v = segment_end - segment_start
    w = point - segment_start
    
    proj = np.dot(w, v) / np.linalg.norm(v)**2 * v
    
    # 判断投影点是否在线段上
    if np.linalg.norm(proj) <= np.linalg.norm(v) and np.dot(proj, v) >= 0:
        # 投影点在线段上
        dist = np.linalg.norm(proj - w)
    else:
        # 投影点不在线段上
        dist = min(np.linalg.norm(w), np.linalg.norm(v - w))
    
    return dist   


def edge_crop(image, quadrant):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if quadrant == "top": start_y, start_x, end_y, end_x= binary_img.shape[0]-1, 0, binary_img.shape[0], binary_img.shape[1]
    elif quadrant == "bottom": start_y, start_x, end_y, end_x= 0, 0, 1, binary_img.shape[1]
    elif quadrant == "left": start_y, start_x, end_y, end_x= 0, binary_img.shape[1]-1, binary_img.shape[0], binary_img.shape[1]
    else: start_y, start_x, end_y, end_x= 0, 0, binary_img.shape[0], 1
    
    crop_y, crop_x = None, None
    for y in range(start_y, end_y):
        for x in range(start_x, end_x):
            if binary_img[y, x] == 0:
                crop_x, crop_y = x, y
                break
    
    if crop_y is None and crop_x is None: return image
    elif quadrant == "top" or quadrant == "bottom": 
        crop_y = min(crop_y, binary_img.shape[0]-crop_y)
        return image[crop_y: , :]
    else: 
        crop_x = min(crop_x, binary_img.shape[1]-crop_x)
        return image[:, crop_x:]


def defect1(base_img, base_img_name, defect_img, defect_img_data):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    defect_img_corner= find_corner(defect_img)
    mid_x, mid_y = ((defect_img_corner[0][0]+defect_img_corner[1][0])/2+(defect_img_corner[2][0]+defect_img_corner[3][0])/2)/2, ((defect_img_corner[0][1]+defect_img_corner[2][1])/2+(defect_img_corner[1][1]+defect_img_corner[3][1])/2)/2
    defect_quadrant = []
    if dx1>defect_img_corner[0][0] and dx1<defect_img_corner[1][0] and dx2>defect_img_corner[0][0] and dx2<defect_img_corner[1][0] and dy1<mid_y and dy2<mid_y: defect_quadrant.append("top")
    if dx1>defect_img_corner[2][0] and dx1<defect_img_corner[3][0] and dx2>defect_img_corner[2][0] and dx2<defect_img_corner[3][0] and dy1>mid_y and dy2>mid_y: defect_quadrant.append("bottom")
    if dy1>defect_img_corner[0][1] and dy1<defect_img_corner[2][1] and dy2>defect_img_corner[0][1] and dy2<defect_img_corner[2][1] and dx1<mid_x and dx2<mid_x: defect_quadrant.append("left")
    if dy1>defect_img_corner[1][1] and dy1<defect_img_corner[3][1] and dy2>defect_img_corner[1][1] and dy2<defect_img_corner[3][1] and dx1>mid_x and dx2>mid_x: defect_quadrant.append("right")
    if len(defect_quadrant)==1: defect_quadrant=defect_quadrant[0]
    else:
        if dx1<mid_x and dy1<mid_y:
            if dy2-dy1>dx2-dx1: defect_quadrant="left"
            else: defect_quadrant="top"
        elif dx1>mid_x and dy1<mid_y:
            if dy2-dy1>dx2-dx1: defect_quadrant="right"
            else: defect_quadrant="top"
        elif dx1<mid_x and dy1>mid_y:
            if dy2-dy1>dx2-dx1: defect_quadrant="left"
            else: defect_quadrant="bottom"
        elif dx1>mid_x and dy1>mid_y:
            if dy2-dy1>dx2-dx1: defect_quadrant="right"
            else: defect_quadrant="bottom"
    
    if defect_quadrant=="top":
        exra_x, exra_y = 30, 10
        exra_x = min(30,min(dx1-defect_img_corner[0][0] if dx1-defect_img_corner[0][0]>0 else 0,defect_img_corner[1][0]-dx2 if defect_img_corner[1][0]-dx2>0 else 0))
    elif defect_quadrant=="bottom":
        exra_x, exra_y = 30, 10
        exra_x = min(30,min(dx1-defect_img_corner[2][0] if dx1-defect_img_corner[2][0]>0 else 0,defect_img_corner[3][0]-dx2 if defect_img_corner[3][0]-dx2>0 else 0))
    elif defect_quadrant=="left":
        exra_x, exra_y = 10, 30
        exra_y = min(30,min(dy1-defect_img_corner[0][1] if dy1-defect_img_corner[0][1]>0 else 0,defect_img_corner[2][1]-dy2 if defect_img_corner[2][1]-dy2>0 else 0))
    elif defect_quadrant=="right":
        exra_x, exra_y = 10, 30
        exra_y = min(30,min(dy1-defect_img_corner[1][1] if dy1-defect_img_corner[1][1]>0 else 0,defect_img_corner[3][1]-dy2 if defect_img_corner[3][1]-dy2>0 else 0))
    else: assert False, "无法确定缺陷所在区域"
    defect_img = defect_img[int(dy1-exra_y):int(dy2+exra_y), int(dx1-exra_x):int(dx2+exra_x)]
    if (defect_quadrant=="top" or defect_quadrant=="left" and dx1<defect_img_corner[0][0] and dx2>defect_img_corner[0][0] and dy1<defect_img_corner[0][1] and dy2>defect_img_corner[0][1]) or (defect_quadrant=="top" or defect_quadrant=="right" and dx1<defect_img_corner[1][0] and dx2>defect_img_corner[1][0] and dy1<defect_img_corner[1][1] and dy2>defect_img_corner[1][1]) or (defect_quadrant=="bottom" or defect_quadrant=="left" and dx1<defect_img_corner[2][0] and dx2>defect_img_corner[2][0] and dy1<defect_img_corner[2][1] and dy2>defect_img_corner[2][1]) or (defect_quadrant=="bottom" or defect_quadrant=="right" and dx1<defect_img_corner[3][0] and dx2>defect_img_corner[3][0] and dy1<defect_img_corner[3][1] and dy2>defect_img_corner[3][1]):
        defect_img = edge_crop(defect_img, defect_quadrant)

    if "_CAM1" in defect_img_data["name"]:
        thresh_min, thresh_max = 100, 130
    elif "_CAM2" in defect_img_data["name"]:
        thresh_min, thresh_max = 35, 60
    elif "_CAM3" in defect_img_data["name"]:
        thresh_min, thresh_max = 35, 70
    else:
        assert False, "图像名不符合规范！"

    base_img_corner = find_corner(base_img)
    if defect_quadrant == "top": pointA, pointB, min_dis = base_img_corner[0], base_img_corner[1], defect_img.shape[1]
    elif defect_quadrant == "bottom": pointA, pointB, min_dis = base_img_corner[2], base_img_corner[3], defect_img.shape[1]
    elif defect_quadrant == "left": pointA, pointB, min_dis = base_img_corner[0], base_img_corner[2], defect_img.shape[0]
    else: pointA, pointB, min_dis = base_img_corner[1], base_img_corner[3], defect_img.shape[0]

    line_length=np.linalg.norm(np.array(pointB)-np.array(pointA))
    random_len = np.random.uniform(min_dis, line_length-min_dis)
    center_point = (int(pointA[0] + random_len * (pointB[0] - pointA[0]) / line_length), int(pointA[1] + random_len * (pointB[1] - pointA[1]) / line_length))
    center_point = find_center_point_defect1(center_point, base_img, defect_img, defect_quadrant, thresh_min, thresh_max)

    mask_all = np.ones(defect_img.shape, dtype=np.uint8) * 255
    result = cv2.seamlessClone(defect_img, base_img, mask_all, center_point, cv2.NORMAL_CLONE)

    result_inf={
        "name": base_img_name[:base_img_name.index("_CAM")]+str(int(center_point[0]))+str(int(center_point[1]))+base_img_name[base_img_name.index("_CAM"):],
        "center_point": [center_point[0],center_point[1]],
        "base_img_name": base_img_name,
        "defect_img_name": defect_img_data["name"],
        "defect_bbox": defect_img_data["bbox"],
        "image_height": base_img.shape[0],
        "image_width": base_img.shape[1],
        "category": 1,
        "bbox": [center_point[0]-defect_img.shape[1]/2,center_point[1]-defect_img.shape[0]/2,center_point[0]+defect_img.shape[1]/2,center_point[1]+defect_img.shape[0]/2]
    }

    return result, result_inf


def defect2(base_img, base_img_name, defect_img_ord, defect_img_data, thresh_defect=35, thresh_bg=75):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    ord_height, ord_width = dy2-dy1, dx2-dx1
    extend_radius = 100
    while True:
        defect_img = defect_img_ord.copy()
        if dx1-extend_radius<0: dx1=extend_radius
        if dy1-extend_radius<0: dy1=extend_radius
        if dx2+extend_radius>defect_img.shape[1]: dx2=defect_img.shape[1]-extend_radius
        if dy2+extend_radius>defect_img.shape[0]: dy2=defect_img.shape[0]-extend_radius
        
        defect_img = defect_img[int(dy1-extend_radius):int(dy2+extend_radius), int(dx1-extend_radius):int(dx2+extend_radius)]
        mask_all = np.ones(defect_img.shape, dtype=np.uint8) * 255
        if dx1 < base_img.shape[0]/2:
            if dy1 < base_img.shape[1]/2:
                defect_quadrant = "top_left"
            else:
                defect_quadrant = "bottom_left"
        else:
            if dy1 < base_img.shape[1]/2:
                defect_quadrant = "top_right"
            else:
                defect_quadrant = "bottom_right" 
        
        if "_CAM1" in defect_img_data["name"]:
            thresh_min, thresh_max = 100, 130
            if defect_quadrant == "top_left" or defect_quadrant == "top_right": shadowlen1, shadowlen2 = 85, 0
            else: shadowlen1, shadowlen2 = 0, 0
        elif "_CAM2" in defect_img_data["name"]:
            thresh_min, thresh_max = 35, 60
            if defect_quadrant == "top_left" or defect_quadrant == "top_right": shadowlen1, shadowlen2 = 55, 0
            else: shadowlen1, shadowlen2 = 0, 0
        elif "_CAM3" in defect_img_data["name"]:
            thresh_min, thresh_max = 35, 70
            if defect_quadrant == "top_left" or defect_quadrant == "top_right": shadowlen1, shadowlen2 = 0, 0
            else: shadowlen1, shadowlen2 = 0, 0
        else:
            assert False, "图像名不符合规范！"

        center_point = find_center_point_defect2(base_img,defect_img,defect_quadrant,thresh_min,thresh_max)
        try:
            result = cv2.seamlessClone(defect_img, base_img, mask_all, center_point, cv2.NORMAL_CLONE)
            break
        except:
            if extend_radius==0: assert False, "底图不足以融合缺陷"
            extend_radius-=10

    defect_img_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    if not "_CAM3" in defect_img_data["name"]:
        background = find_bg(defect_img, base_img, center_point, defect_quadrant)
        _, mask_defect_bg_ord = cv2.threshold(defect_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask_defect_bg=mask_defect_bg_ord.copy()
        for px in range(0, defect_img.shape[1]):
            for py in range(0, defect_img.shape[0]):
                if mask_defect_bg_ord[py, px] == 255:
                    x, y = px, py
                    if defect_quadrant == "top_left":
                        if y-shadowlen1<=0: y=shadowlen1
                        if x-shadowlen2<=0: x=shadowlen2
                        mask_defect_bg[y-shadowlen1:y,x] = 255
                        mask_defect_bg[y,x-shadowlen2:x] = 255
                    elif defect_quadrant == "top_right":
                        if y-shadowlen1<=0: y=shadowlen1
                        if x+shadowlen2>=defect_img.shape[1]: x=defect_img.shape[1]-shadowlen2
                        mask_defect_bg[y-shadowlen1:y,x] = 255
                        mask_defect_bg[y,x:x+shadowlen2] = 255
                    elif defect_quadrant == "bottom_left":
                        if y+shadowlen1>=defect_img.shape[0]: y=defect_img.shape[0]-shadowlen1
                        if x-shadowlen2<=0: x=shadowlen2
                        mask_defect_bg[y:y+shadowlen1,x] = 255
                        mask_defect_bg[y,x-shadowlen2:x] = 255
                    else:
                        if y+shadowlen1>=defect_img.shape[0]: y=defect_img.shape[0]-shadowlen1
                        if x+shadowlen2>=defect_img.shape[1]: x=defect_img.shape[1]-shadowlen2
                        mask_defect_bg[y:y+shadowlen1,x] = 255
                        mask_defect_bg[y,x:x+shadowlen2] = 255
        mask_defect_bg = cv2.bitwise_not(mask_defect_bg)
        mask_defect_bg = cv2.blur(mask_defect_bg, (3, 3))
        result = cv2.seamlessClone(background, result, mask_defect_bg, center_point, cv2.NORMAL_CLONE)

    defect_point = (int(center_point[0]-defect_img.shape[1]/2), int(center_point[1]-defect_img.shape[0]/2))
    defect_img_ana = result[defect_point[1]:defect_point[1]+defect_img.shape[0], defect_point[0]:defect_point[0]+defect_img.shape[1]]
    _,mask_ana=cv2.threshold(defect_img_gray, thresh_max, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((int((max(defect_img.shape[0],defect_img.shape[1])*0.05)),int((max(defect_img.shape[0],defect_img.shape[1])*0.05))), np.uint8)
    dilated_mask = cv2.dilate(mask_ana, kernel, iterations=1)
    dilated_mask = cv2.blur(dilated_mask, (2, 2))
    result = cv2.seamlessClone(defect_img_ana, base_img, dilated_mask, center_point, cv2.NORMAL_CLONE)

    result_inf={
        "name": base_img_name[:base_img_name.index("_CAM")]+str(int(center_point[0]))+str(int(center_point[1]))+base_img_name[base_img_name.index("_CAM"):],
        "center_point": [center_point[0],center_point[1]],
        "base_img_name": base_img_name,
        "defect_img_name": defect_img_data["name"],
        "defect_bbox": defect_img_data["bbox"],
        "image_height": base_img.shape[0],
        "image_width": base_img.shape[1],
        "category": 2,
        "bbox": [center_point[0]-ord_width/2,center_point[1]-ord_height/2,center_point[0]+ord_width/2,center_point[1]+ord_height/2]
    }

    return result, result_inf


def defect3to6(base_img, base_img_name, defect_img, defect_img_data):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    minoffset_x, minoffset_y = math.ceil((30-(dx2-dx1))/2), math.ceil((30-(dy2-dy1))/2)
    if defect_img_data["category"]==3: offset_x, offset_y = max(max(5,minoffset_x),0.25*(dx2-dx1)), max(max(5,minoffset_y),0.25*(dy2-dy1))
    elif defect_img_data["category"]==4: offset_x, offset_y = max(max(5,minoffset_x),0.15*(dx2-dx1)), max(max(5,minoffset_y),0.15*(dy2-dy1))
    elif defect_img_data["category"]==5: offset_x, offset_y = max(max(5,minoffset_x),0.4*(dx2-dx1)), max(max(5,minoffset_y),0.4*(dy2-dy1))
    else: offset_x, offset_y = 5, 5
    defect_img = defect_img[int(dy1-offset_y):int(dy2+offset_y), int(dx1-offset_x):int(dx2+offset_x)]

    corners = find_corner(base_img)
    base_width = (corners[1][0]-corners[0][0]+corners[3][0]-corners[2][0])/2
    base_height = (corners[2][1]-corners[0][1]+corners[3][1]-corners[1][1])/2
    min_dis_x, min_dis_y = defect_img.shape[1]//2, defect_img.shape[0]//2
    center_point = (np.random.uniform(min_dis_x, base_width-min_dis_x),np.random.uniform(min_dis_y, base_height-min_dis_x))
    _, valid_area = cv2.threshold(cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    valid_center_img = valid_area[int(center_point[1]-defect_img.shape[0]//2):int(center_point[1]+defect_img.shape[0]//2), int(center_point[0]-defect_img.shape[1]//2):int(center_point[0]+defect_img.shape[1]//2)]
    while (not np.all(valid_center_img == 255)):
        center_point = (np.random.uniform(min_dis_x, base_width-min_dis_x),np.random.uniform(min_dis_y, base_height-min_dis_x))
        valid_center_img = valid_area[int(center_point[1]-defect_img.shape[0]//2):int(center_point[1]+defect_img.shape[0]//2), int(center_point[0]-defect_img.shape[1]//2):int(center_point[0]+defect_img.shape[1]//2)]
    center_point = (int(center_point[0]), int(center_point[1]))

    mask_all = np.ones(defect_img.shape, dtype=np.uint8) * 255
    result = cv2.seamlessClone(defect_img, base_img, mask_all, center_point, cv2.NORMAL_CLONE)

    if not "_CAM3" in defect_img_data["name"]: 
        defect_img_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
        _, defect_mask = cv2.threshold(defect_img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (np.count_nonzero(defect_mask[0, :] == 255)+np.count_nonzero(defect_mask[:, 0] == 255)+np.count_nonzero(defect_mask[defect_mask.shape[0] - 1, :] == 255)+np.count_nonzero(defect_mask[:, defect_mask.shape[1] - 1] == 255))>=(defect_mask.shape[1]-1)*2+(defect_mask.shape[0]-1)*2*1/3: defect_mask=cv2.bitwise_not(defect_mask)
        kernel = np.ones((int(max((max(defect_img.shape[0],defect_img.shape[1])*0.05),2)),int(max(max(defect_img.shape[0],defect_img.shape[1])*0.05,2))), np.uint8)
        base_bg_mask = cv2.dilate(defect_mask, kernel, iterations=1)
        base_point = (int(center_point[0]-defect_img.shape[1]/2), int(center_point[1]-defect_img.shape[0]/2))
        base_bg = base_img[base_point[1]:base_point[1]+defect_img.shape[0], base_point[0]:base_point[0]+defect_img.shape[1]]
        result = cv2.seamlessClone(base_bg, result, base_bg_mask, center_point, cv2.NORMAL_CLONE)

    base_point = (int(center_point[0]-defect_img.shape[1]/2), int(center_point[1]-defect_img.shape[0]/2))
    defect_img_ana = result[base_point[1]:base_point[1]+defect_img.shape[0], base_point[0]:base_point[0]+defect_img.shape[1]]
    result = cv2.seamlessClone(defect_img_ana, base_img, mask_all, center_point, cv2.NORMAL_CLONE)

    result_inf={
        "name": base_img_name[:base_img_name.index("_CAM")]+str(int(center_point[0]))+str(int(center_point[1]))+base_img_name[base_img_name.index("_CAM"):],
        "center_point": [center_point[0],center_point[1]],
        "base_img_name": base_img_name,
        "defect_img_name": defect_img_data["name"],
        "defect_bbox": defect_img_data["bbox"],
        "image_height": base_img.shape[0],
        "image_width": base_img.shape[1],
        "category": defect_img_data["category"],
        "bbox": [center_point[0]-defect_img.shape[1]/2,center_point[1]-defect_img.shape[0]/2,center_point[0]-defect_img.shape[1]/2+defect_img.shape[1],center_point[1]-defect_img.shape[0]/2+defect_img.shape[0]]
    }

    return result, result_inf

def main(base_imgs_path, defect_imgs_path, base_json_path, defect_json_path, error_json_path, output_json_path1, output_json_path2, output_folder, ord_json_path):
    # defect_imgs_path = 'rotated'
    # defect_imgs = os.listdir(defect_imgs_path)

    # base_imgs_path = 'rotated'
    # base_imgs = os.listdir(base_imgs_path)

    # input_json_path = "train_annos_rotated_fix.json"
    # error_json_path = "corner_defect_error.json"
    # output_json_path = "gen_train_annos.json"

    # output_folder = "output"    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_json_path1):
        with open(output_json_path1, "w") as wnfile1:
            json.dump([], wnfile1)

    if not os.path.exists(output_json_path2):
        with open(output_json_path2, "w") as wnfile2:
            json.dump([], wnfile2)

    if not os.path.exists(error_json_path):
        with open(error_json_path, "w") as errorfile:
            json.dump([], errorfile)

    with open(base_json_path, 'r') as base_f:
        base_imgs = json.load(base_f)
    base_img_names = [list(data.keys())[0] for data in base_imgs]

    with open(defect_json_path, 'r') as defect_f:
        defect_imgs_data = json.load(defect_f)

    with open(output_json_path1, "r") as rnfile1:
        gen_img_data = json.load(rnfile1)

    with open(output_json_path2, "r") as rnfile2:
        gen_img_data_with_ord = json.load(rnfile2)

    with open(ord_json_path, "r") as ordfile:
        ord_data = json.load(ordfile)

    # for base_img_name in base_imgs:
    #     base_img = cv2.imread(os.path.join(base_imgs_path, base_img_name))
    for defect_img_data, base_img_name in tqdm(zip(defect_imgs_data,base_img_names), total=167):
        defect_img_name = defect_img_data["name"]
        gen_img = None

        # try:
            # if base_img_name[base_img_name.index("_CAM"):] == defect_img_name[defect_img_name.index("_CAM"):]:
        base_img = cv2.imread(os.path.join(base_imgs_path, base_img_name))
        defect_img = cv2.imread(os.path.join(defect_imgs_path, defect_img_name))
        if base_img is None: assert False, "底图不存在！"
        if defect_img is None: assert False, "缺陷图像不存在！"

        if defect_img_data["category"] == 1 :
            gen_img, gen_img_inf=defect1(base_img, base_img_name, defect_img, defect_img_data)
            
        elif defect_img_data["category"] == 2 :
            gen_img, gen_img_inf=defect2(base_img, base_img_name, defect_img, defect_img_data)

        elif defect_img_data["category"] == 3 or defect_img_data["category"] == 4 or defect_img_data["category"] == 5 or defect_img_data["category"] == 6: 
            gen_img, gen_img_inf=defect3to6(base_img, base_img_name, defect_img, defect_img_data)
            # sub_gen_img = gen_img[int(gen_img_inf["bbox"][1]-50):int(gen_img_inf["bbox"][3]+50), int(gen_img_inf["bbox"][0]-50):int(gen_img_inf["bbox"][2]+50)]
            # sub_img = defect_img[int(gen_img_inf["defect_bbox"][1]-50):int(gen_img_inf["defect_bbox"][3]+50), int(gen_img_inf["defect_bbox"][0]-50):int(gen_img_inf["defect_bbox"][2]+50)]

        if gen_img is not None:
            cv2.imwrite(os.path.join(output_folder, gen_img_inf["name"]), gen_img)
            gen_img_data.append(gen_img_inf)
            gen_img_data_with_ord.append(gen_img_inf)
            img_datai = ord_data.copy()
            for item in img_datai:
                if item["name"] == base_img_name:
                    item["name"] = gen_img_inf["name"]
                    gen_img_data_with_ord.append(item)
            with open(output_json_path1, "w") as json_file1:
                json.dump(gen_img_data, json_file1, indent=4)
            with open(output_json_path2, "w") as json_file2:
                json.dump(gen_img_data_with_ord, json_file2, indent=4)

        # except:
        #     with open(error_json_path, "r") as r_error_file:
        #         error_data_list = json.load(r_error_file)
        #     error_data_list.append(defect_img_data)
        #     with open(error_json_path, "w") as w_error_file:
        #         json.dump(error_data_list, w_error_file, indent=4)
        #     continue

if __name__ == "__main__":
    # main("rotated", "rotated", r"choice\base_img_for41.json", "error_test.json", "error\error_defect4_CAM1.json", "gen_test.json", "gen_test_ord.json", "output\est", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for11.json", "choice\defect1_CAM1_img.json", "error\error_defect1_CAM1.json", "output\defect1\gen_defect1_CAM1.json", "output\defect1\gen_defect1_CAM1_with_ord.json", "output\defect1\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for12.json", "choice\defect1_CAM2_img.json", "error\error_defect1_CAM2.json", "output\defect1\gen_defect1_CAM2.json", "output\defect1\gen_defect1_CAM2_with_ord.json", "output\defect1\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for13.json", "choice\defect1_CAM3_img.json", "error\error_defect1_CAM3.json", "output\defect1\gen_defect1_CAM3.json", "output\defect1\gen_defect1_CAM3_with_ord.json", "output\defect1\CAM3", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for21.json", "choice\defect2_CAM1_img.json", "error\error_defect2_CAM1.json", "output\defect2\gen_defect2_CAM1.json", "output\defect2\gen_defect2_CAM1_with_ord.json", "output\defect2\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for22.json", "choice\defect2_CAM2_img.json", "error\error_defect2_CAM2.json", "output\defect2\gen_defect2_CAM2.json", "output\defect2\gen_defect2_CAM2_with_ord.json", "output\defect2\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for23.json", "choice\defect2_CAM3_img.json", "error\error_defect2_CAM3.json", "output\defect2\gen_defect2_CAM3.json", "output\defect2\gen_defect2_CAM3_with_ord.json", "output\defect2\CAM3", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for31.json", "choice\defect3_CAM1_img.json", "error\error_defect3_CAM1.json", "output\defect3\gen_defect3_CAM1.json", "output\defect3\gen_defect3_CAM1_with_ord.json", "output\defect3\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for32.json", "choice\defect3_CAM2_img.json", "error\error_defect3_CAM2.json", "output\defect3\gen_defect3_CAM2.json", "output\defect3\gen_defect3_CAM2_with_ord.json", "output\defect3\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for33.json", "choice\defect3_CAM3_img.json", "error\error_defect3_CAM3.json", "output\defect3\gen_defect3_CAM3.json", "output\defect3\gen_defect3_CAM3_with_ord.json", "output\defect3\CAM3", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for41.json", "choice\defect4_CAM1_img.json", "error\error_defect4_CAM1.json", "output\defect4\gen_defect4_CAM1.json", "output\defect4\gen_defect4_CAM1_with_ord.json", "output\defect4\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for42.json", "choice\defect4_CAM2_img.json", "error\error_defect4_CAM2.json", "output\defect4\gen_defect4_CAM2.json", "output\defect4\gen_defect4_CAM2_with_ord.json", "output\defect4\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for43.json", "choice\defect4_CAM3_img.json", "error\error_defect4_CAM3.json", "output\defect4\gen_defect4_CAM3.json", "output\defect4\gen_defect4_CAM3_with_ord.json", "output\defect4\CAM3", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for51.json", "choice\defect5_CAM1_img.json", "error\error_defect5_CAM1.json", "output\defect5\gen_defect5_CAM1.json", "output\defect5\gen_defect5_CAM1_with_ord.json", "output\defect5\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for52.json", "choice\defect5_CAM2_img.json", "error\error_defect5_CAM2.json", "output\defect5\gen_defect5_CAM2.json", "output\defect5\gen_defect5_CAM2_with_ord.json", "output\defect5\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for53.json", "choice\defect5_CAM3_img.json", "error\error_defect5_CAM3.json", "output\defect5\gen_defect5_CAM3.json", "output\defect5\gen_defect5_CAM3_with_ord.json", "output\defect5\CAM3", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for61.json", "choice\defect6_CAM1_img.json", "error\error_defect6_CAM1.json", "output\defect6\gen_defect6_CAM1.json", "output\defect6\gen_defect6_CAM1_with_ord.json", "output\defect6\CAM1", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for62.json", "choice\defect6_CAM2_img.json", "error\error_defect6_CAM2.json", "output\defect6\gen_defect6_CAM2.json", "output\defect6\gen_defect6_CAM2_with_ord.json", "output\defect6\CAM2", "train_annos_rotated_fix.json")
    main("rotated", "rotated", r"choice\base_img_for63.json", "choice\defect6_CAM3_img.json", "error\error_defect6_CAM3.json", "output\defect6\gen_defect6_CAM3.json", "output\defect6\gen_defect6_CAM3_with_ord.json", "output\defect6\CAM3", "train_annos_rotated_fix.json")