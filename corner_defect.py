import cv2
import numpy as np
import json
import os
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


def find_missing_corner(defect_img,thresh_min,thresh_max):
    gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
    #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, thresh_min, thresh_max)
    # 使用霍夫线变换找到线段
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, threshold=5, minLineLength=20, maxLineGap=5)
    vertical_points ,horizontal_points=find_corners_of_approx_parallel_lines(lines, angle_tolerance=10)
    height, width = defect_img.shape[:2]
    vertical_point=min(vertical_points, key=lambda point: min(point[1]**2,(height-point[1])**2))
    horizontal_point=min(horizontal_points, key=lambda point: min(point[0]**2,(width-point[0])**2))

    return vertical_point,horizontal_point


def find_corner(image, quadrant="top_left, top_right, bottom_left, bottom_right"):
    # 将图片转为灰度图
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.dilate(binary_img, kernel, iterations=1)
    binary_img = cv2.erode(binary_img, kernel, iterations=1)

    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    mask = np.zeros_like(binary_img)
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)
    num_kp=40
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
    vertical_point,horizontal_point=find_missing_corner(defect_img,thresh_min,thresh_max)
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
        if center_point[1] - defect_img.shape[0]//2 < defect_img.shape[0]:
            if defect_quadrant == "top_left":
                bg_x1, bg_x2 = center_point[0] - defect_img.shape[1]//2 - defect_img.shape[1], center_point[0] - defect_img.shape[1]//2
            else:
                bg_x1, bg_x2 = center_point[0] + defect_img.shape[1]//2, center_point[0] + defect_img.shape[1]//2 + defect_img.shape[1]
            bg_y1, bg_y2 = center_point[1] - defect_img.shape[0]//2, center_point[1] + defect_img.shape[0]//2
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 1)
        else:
            bg_y1, bg_y2 = center_point[1] - defect_img.shape[0]//2 - defect_img.shape[0], center_point[1] - defect_img.shape[0]//2
            bg_x1, bg_x2 = center_point[0] - defect_img.shape[1]//2, center_point[0] + defect_img.shape[1]//2
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 0)
    else:
        if base_img.shape[0] - center_point[1] - defect_img.shape[0]//2 < defect_img.shape[0]:
            if defect_quadrant == "bottom_left":
                bg_x1, bg_x2 = center_point[0] - defect_img.shape[1]//2 - defect_img.shape[1], center_point[0] - defect_img.shape[1]//2
            else:
                bg_x1, bg_x2 = center_point[0] + defect_img.shape[1]//2, center_point[0] + defect_img.shape[1]//2 + defect_img.shape[1]
            bg_y1, bg_y2 = center_point[1] - defect_img.shape[0]//2, center_point[1] + defect_img.shape[0]//2
            background = base_img[bg_y1:bg_y2, bg_x1:bg_x2]
            background = cv2.flip(background, 1)
        else:
            bg_y1, bg_y2 = center_point[1] + defect_img.shape[0]//2, center_point[1] + defect_img.shape[0]//2 + defect_img.shape[0]
            bg_x1, bg_x2 = center_point[0] - defect_img.shape[1]//2, center_point[0] + defect_img.shape[1]//2
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


def defect1(base_img, base_img_name, defect_img, defect_img_data):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    mid_x, mid_y = (dx1 + dx2) / 2, (dy1 + dy2) / 2
    defect_img_corner= find_corner(defect_img)

    dis_top1 = point_to_segment_dist((dx1, dy1), defect_img_corner[0], defect_img_corner[1])
    dis_topmid = point_to_segment_dist((mid_x, mid_y), defect_img_corner[0], defect_img_corner[1])
    dis_bottom1 = point_to_segment_dist((dx1, dy2), defect_img_corner[2], defect_img_corner[3])
    dis_bottommid = point_to_segment_dist((mid_x, mid_y), defect_img_corner[2], defect_img_corner[3])
    dis_left1 = point_to_segment_dist((dx1, dy1), defect_img_corner[0], defect_img_corner[2])
    dis_leftmid = point_to_segment_dist((mid_x, mid_y), defect_img_corner[0], defect_img_corner[2])
    dis_right1 = point_to_segment_dist((dx2, dy1), defect_img_corner[1], defect_img_corner[3])
    dis_rightmid = point_to_segment_dist((mid_x, mid_y), defect_img_corner[1], defect_img_corner[3])
    dis1 = [dis_top1, dis_bottom1, dis_left1, dis_right1]
    dis_mid = [dis_topmid, dis_bottommid, dis_leftmid, dis_rightmid]

    if min(dis1) == dis1[0] or min(dis_mid) == dis_mid[0]:
        defect_quadrant="top"
        exra_x, exra_y = 30, 10
        exra_x = min(30,min(dx1-defect_img_corner[0][0],defect_img_corner[1][0]-dx2))
    elif min(dis1) == dis1[1] or min(dis_mid) == dis_mid[1]:
        defect_quadrant="bottom"
        exra_x, exra_y = 30, 10
        exra_x = min(30,min(dx1-defect_img_corner[2][0],defect_img_corner[3][0]-dx2))
    elif min(dis1) == dis1[2] or min(dis_mid) == dis_mid[2]:
        defect_quadrant="left"
        exra_x, exra_y = 10, 30
        exra_y = min(30,min(dy1-defect_img_corner[0][1],defect_img_corner[2][1]-dy2))
    else:
        defect_quadrant="right"
        exra_x, exra_y = 10, 30
        exra_y = min(30,min(dy1-defect_img_corner[1][1],defect_img_corner[3][1]-dy2))
    defect_img = defect_img[int(dy1-exra_y):int(dy2+exra_y), int(dx1-exra_x):int(dx2+exra_x)]

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
        "bbox": [center_point[0]-defect_img.shape[1]//2,center_point[1]-defect_img.shape[0]//2,center_point[0]+defect_img.shape[1]//2,center_point[1]+defect_img.shape[0]//2]
    }

    return result, result_inf


def defect2(base_img, base_img_name, defect_img, defect_img_data, thresh_defect=35, thresh_bg=75):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    if dx1-100<0: dx1=100
    if dy1-100<0: dy1=100
    if dx2+100>defect_img.shape[1]: dx2=defect_img.shape[1]-100
    if dy2+100>defect_img.shape[0]: dy2=defect_img.shape[0]-100
    
    defect_img = defect_img[int(dy1-100):int(dy2+100), int(dx1-100):int(dx2+100)]
    mask_all = np.ones(defect_img.shape, dtype=np.uint8) * 255
    if dx1 < base_img.shape[0]//2:
        if dy1 < base_img.shape[1]//2:
            defect_quadrant = "top_left"
        else:
            defect_quadrant = "bottom_left"
    else:
        if dy1 < base_img.shape[1]//2:
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

    #defect_quadrant = "bottom_right"
    center_point = find_center_point_defect2(base_img,defect_img,defect_quadrant,thresh_min,thresh_max)
    result = cv2.seamlessClone(defect_img, base_img, mask_all, center_point, cv2.NORMAL_CLONE)

    background = find_bg(defect_img, base_img, center_point, defect_quadrant)
    defect_img_gray = cv2.cvtColor(defect_img, cv2.COLOR_BGR2GRAY)
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

    defect_point = (center_point[0]-defect_img.shape[1]//2, center_point[1]-defect_img.shape[0]//2)
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
        "bbox": [center_point[0]-defect_img.shape[1]//2,center_point[1]-defect_img.shape[0]//2,center_point[0]+defect_img.shape[1]//2,center_point[1]+defect_img.shape[0]//2]
    }

    return result, result_inf


def defect3to6(base_img, base_img_name, defect_img, defect_img_data):
    # 截取缺陷图像
    dx1, dy1, dx2, dy2 = defect_img_data['bbox']
    if defect_img_data["category"]==3: offset_x, offset_y = 0.25*(dx2-dx1), 0.25*(dy2-dy1)
    elif defect_img_data["category"]==4: offset_x, offset_y = 0.15*(dx2-dx1), 0.15*(dy2-dy1)
    elif defect_img_data["category"]==5: offset_x, offset_y = 0.4*(dx2-dx1), 0.4*(dy2-dy1)
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
        if defect_mask[0, 0] == 255 and defect_mask[0, defect_img.shape[1]-1] == 255 and defect_mask[defect_img.shape[0]-1, 0] == 255 and defect_mask[defect_img.shape[0]-1, defect_img.shape[1]-1] == 255: defect_mask=cv2.bitwise_not(defect_mask)
        kernel = np.ones((int((max(defect_img.shape[0],defect_img.shape[1])*0.05)),int((max(defect_img.shape[0],defect_img.shape[1])*0.05))), np.uint8)
        base_bg_mask = cv2.dilate(defect_mask, kernel, iterations=1)
        base_bg_mask = cv2.bitwise_not(base_bg_mask)
        base_bg_mask = cv2.blur(base_bg_mask, (2, 2))
        base_point = (int(center_point[0]-defect_img.shape[1]//2), int(center_point[1]-defect_img.shape[0]//2))
        base_bg = base_img[base_point[1]:base_point[1]+defect_img.shape[0], base_point[0]:base_point[0]+defect_img.shape[1]]
        result = cv2.seamlessClone(base_bg, result, base_bg_mask, center_point, cv2.NORMAL_CLONE)

    base_point = (int(center_point[0]-defect_img.shape[1]//2), int(center_point[1]-defect_img.shape[0]//2))
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
        "bbox": [center_point[0]-defect_img.shape[1]//2,center_point[1]-defect_img.shape[0]//2,center_point[0]-defect_img.shape[1]//2+defect_img.shape[1],center_point[1]-defect_img.shape[0]//2+defect_img.shape[0]]
    }

    return result, result_inf

if __name__ == "__main__":
    defect_imgs_path = 'defect_img'
    defect_imgs = os.listdir(defect_imgs_path)

    base_imgs_path = 'base_img'
    base_imgs = os.listdir(base_imgs_path)

    output_folder = "output"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists("gen_train_annos.json"):
        with open("gen_train_annos.json", "w") as wnfile:
            json.dump([], wnfile)

    with open("train_annos_rotated.json", 'r') as f:
        img_data = json.load(f)

    with open("gen_train_annos.json", "r") as rnfile:
        gen_img_data = json.load(rnfile)

    for base_img_name in base_imgs:
        base_img = cv2.imread(os.path.join(base_imgs_path, base_img_name))
        for defect_img_data in tqdm(img_data):
            defect_img_name = defect_img_data["name"]
            gen_img = None

            if base_img_name[base_img_name.index("_CAM"):] == defect_img_name[defect_img_name.index("_CAM"):]:
                defect_img = cv2.imread(os.path.join(defect_imgs_path, defect_img_name))

                if defect_img_data["category"] == 1 :
                    gen_img, gen_img_inf=defect1(base_img, base_img_name, defect_img, defect_img_data)
                    
                elif defect_img_data["category"] == 2 :
                    gen_img, gen_img_inf=defect2(base_img, base_img_name, defect_img, defect_img_data)

                elif defect_img_data["category"] == 3 or defect_img_data["category"] == 4 or defect_img_data["category"] == 5 or defect_img_data["category"] == 6: 
                    gen_img, gen_img_inf=defect3to6(base_img, base_img_name, defect_img, defect_img_data)

            if gen_img is not None:
                # cv2.rectangle(gen_img, (gen_img_inf["bbox"][0], gen_img_inf["bbox"][1]), (gen_img_inf["bbox"][2], gen_img_inf["bbox"][3]), (0, 0, 255), 1)
                cv2.imwrite(os.path.join(output_folder, gen_img_inf["name"]), gen_img)
                gen_img_data.append(gen_img_inf)
                img_datai = img_data.copy()
                for item in img_datai:
                    if item["name"] == base_img_name:
                        item["name"] = gen_img_inf["name"]
                        gen_img_data.append(item)
                with open("gen_train_annos.json", "w") as wnfile:
                    json.dump(gen_img_data, wnfile, indent=4)             