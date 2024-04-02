import cv2
import numpy as np
import os
import json
from tqdm import tqdm
            

def rota_angle(image,image_name):
    if "_CAM1" in image_name:
        thresh_min, thresh_max = 100, 130
    elif "_CAM2" in image_name:
        thresh_min, thresh_max = 35, 60
    elif "_CAM3" in image_name:
        thresh_min, thresh_max = 35, 70
    else:
        assert False, "图像名不符合规范！"
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_img = cv2.GaussianBlur(binary_img, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(binary_img, thresh_min, thresh_max)
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def rect_area(contour):
        rect = cv2.minAreaRect(contour)
        return rect[1][0] * rect[1][1]
    # 找到最大的轮廓
    max_contour = max(contours, key=rect_area)
    # 找到最小外接矩形
    rect = cv2.minAreaRect(max_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    if(rect[2]>45):
        rota_rad=90-rect[2]
    else:
        rota_rad=-rect[2]
    rota_rad=np.radians(rota_rad)

    return rota_rad

def main(image,image_name):
    rota_rad=rota_angle(image,image_name)
    center = np.array(image.shape[1::-1]) / 2
    diag_rad = np.arctan(image.shape[0] / image.shape[1])
    offset = np.linalg.norm(center) * np.sin(rota_rad) / np.sin(np.pi - rota_rad - diag_rad)
    point = []
    diaglen=np.linalg.norm(np.array([0,0])-np.array([image.shape[1],image.shape[0]]))
    if image.shape[1] > image.shape[0]:
        dis=image.shape[1]
        if offset < 0:
            point.append(np.array([-offset , 0]))
            point.append(np.array([image.shape[1] + offset , image.shape[0]]))
            sub_diaglen=np.linalg.norm(point[0]-point[1])
            addnp=np.array([np.cos(-rota_rad)*sub_diaglen/diaglen*dis , np.sin(-rota_rad)*sub_diaglen/diaglen*dis])
            point.append(point[0]+addnp)
            point.append(point[1]-addnp)
        else:
            point.append(np.array([image.shape[1] - offset , 0]))
            point.append(np.array([offset , image.shape[0]]))
            sub_diaglen=np.linalg.norm(point[0]-point[1])
            addnp=np.array([-np.cos(rota_rad)*sub_diaglen/diaglen*dis , np.sin(rota_rad)*sub_diaglen/diaglen*dis])
            point.append(point[0]+addnp)
            point.append(point[1]-addnp)
    else:
        dis=image.shape[0]
        if offset < 0:
            point.append(np.array([image.shape[1], -offset]))
            point.append(np.array([0, image.shape[0] + offset]))
            sub_diaglen=np.linalg.norm(point[0]-point[1])
            addnp=np.array([np.sin(-rota_rad)*sub_diaglen/diaglen*dis , -np.cos(-rota_rad)*sub_diaglen/diaglen*dis])
            point.append(point[1]+addnp)
            point.append(point[0]-addnp)
        else:
            point.append(np.array([0, offset]))
            point.append(np.array([image.shape[1], image.shape[0] - offset]))
            sub_diaglen=np.linalg.norm(point[0]-point[1])
            addnp=np.array([-np.sin(rota_rad)*sub_diaglen/diaglen*dis , -np.cos(rota_rad)*sub_diaglen/diaglen*dis])
            point.append(point[1]+addnp)
            point.append(point[0]-addnp)
    
    point = np.array(point)

    # 初始化矩形的四个角点：左上、右上、右下、左下
    rect = np.zeros((4, 2), dtype="float32")

    # 按照顺时针或逆时针排序
    # 左上点具有最小的和，右下点具有最大的和
    s = point.sum(axis=1)
    rect[0] = point[np.argmin(s)]
    rect[2] = point[np.argmax(s)]

    # 计算左下和右上点的差异
    diff = np.diff(point, axis=1)
    rect[1] = point[np.argmin(diff)]
    rect[3] = point[np.argmax(diff)]
    (tl, tr, br, bl) = rect

    # 计算新图像的宽度和高度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 新图像的目标四个角点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped, M, sub_diaglen/diaglen

if __name__ == "__main__":
    # JSON文件所在的文件夹路径
    rjson_path = "train_annos.json"
    wjson_path = "train_annos_rotated.json"

    # 图像文件所在的文件夹路径
    image_folder = "train_imgs"

    # 新建文件夹路径，用于保存筛选出的图像
    output_folder = "output_rotated2"

    # 如果新建文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 读取JSON文件
    with open(rjson_path, "r") as file1:
        rdata_list = json.load(file1)
    with open(wjson_path, "r") as file3:
        wdata_list = json.load(file3)
    wdata_list=rdata_list.copy()

    for data_num in tqdm(range(len(rdata_list))):
        data = rdata_list[data_num]
        if (not os.path.exists(os.path.join(output_folder, data["name"]))):
            image_name = data["name"]
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)

            rotated_image, M, proportion= main(image,image_name)
            for item_num in range(data_num,len(wdata_list)):
                item=wdata_list[item_num]
                if item["name"] == data["name"]:
                    defect_half_width=(item["bbox"][2]-item["bbox"][0])/2
                    defect_half_height=(item["bbox"][3]-item["bbox"][1])/2
                    defect_center = np.array([item["bbox"][0]+defect_half_width, item["bbox"][1]+defect_half_height,1])
                    new_center = np.dot(M, defect_center)
                    new_center = new_center[:2] / new_center[2]
                    new_centerx, new_centery = new_center[0], new_center[1]
                    
                    if(new_centerx-defect_half_width*proportion<0 or new_centerx+defect_half_width*proportion<0 or new_centery-defect_half_height*proportion<0 or new_centery+defect_half_height*proportion<0 ):
                        with open("error_img.json", "r") as error_file:
                            error_data_list = json.load(error_file)
                        error_data_list.append(item)
                        with open("error_img.json", "w") as error_file:
                            json.dump(error_data_list, error_file, indent=4)
                        continue

                    item["bbox"]=[new_centerx-defect_half_width*proportion,new_centery-defect_half_height*proportion,new_centerx+defect_half_width*proportion,new_centery+defect_half_height*proportion]
                    item["image_height"],item["image_width"]=rotated_image.shape[:2]
            
            cv2.imwrite(os.path.join(output_folder, image_name), rotated_image)

    with open(wjson_path, "w") as file4:
        json.dump(wdata_list, file4, indent=4)
