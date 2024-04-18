import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from PIL import Image
      

def main(image,image_name):
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

    # 寻找轮廓
    contours, _ = cv2.findContours(max_connected_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        rota_rad=rect[2]-90
    else:
        rota_rad=rect[2]

    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, rota_rad, 1)
    rotated = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    if rota_rad>0: diagonal = np.linspace([0, 0], [width - 1, height - 1], max(height, width))
    else: diagonal = np.linspace([width - 1, 0], [0, height - 1], max(height, width))

    for point in diagonal:
        x, y = point.astype(int)
        pixel_value = rotated[y, x]
        if not np.array_equal(pixel_value, [255, 255, 255]):
            break
        elif point[0]==diagonal[-1][0] and point[1]==diagonal[-1][1]:
            assert False, "未找到裁剪矩形"

    if rota_rad>0: x1, y1, x2, y2 = x, y, int(center[0]+center[0]-x), int(center[1]+center[1]-y)
    else: x1, y1, x2, y2 = int(center[0]+center[0]-x), y, x, int(center[1]+center[1]-y)

    rect = rotated.copy()
    cv2.rectangle(rect, (x1, y1), (x2, y2), (0, 255, 0), 2)
    rotated = rotated[y1:y2, x1:x2]

    rotated_mask = cv2.warpAffine(max_connected_mask, M, (width, height))
    rotated_mask = rotated_mask[y1:y2, x1:x2]
    cv2.imwrite("mask.jpg",rotated_mask)
    
    try:
        if np.any(rotated_mask[0,:]==255) or np.any(rotated_mask[rotated_mask.shape[0]-1:]==255) : assert False, "瓷砖距离边缘过近无法裁剪"
    except:
        return rotated_mask, M, (x1, y1), False
    
    return rotated, M, (x1, y1), True
    

if __name__ == "__main__":
    # JSON文件所在的文件夹路径
    rjson_path = "train_annos.json"
    wjson_path = "test.json"

    # 图像文件所在的文件夹路径
    image_folder = "error"

    # 新建文件夹路径，用于保存筛选出的图像
    output_folder = "rotated_test"

    # 如果新建文件夹不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 读取JSON文件
    with open(rjson_path, "r") as file1:
        rdata_list = json.load(file1)
    with open(wjson_path, "r") as file3:
        wdata_list = json.load(file3)
    wdata_list=rdata_list.copy()

    for filename in tqdm(os.listdir(image_folder)):
    # for data_num in tqdm(range(len(rdata_list))):
        # data = rdata_list[data_num]
        if (not os.path.exists(os.path.join(output_folder, filename))) :
            # image_name = data["name"]
            image_path = os.path.join("train_imgs", filename)
            image = cv2.imread(image_path)

            try:
                rotated_image, M, startp, flag= main(image,filename)
                if flag==False:
                    cv2.imwrite(filename, rotated_image)
                    continue
            except:
                with open(wjson_path, "r") as error_file:
                    error_data_list = json.load(error_file)
                error_data_list.append({"name":filename})
                with open(wjson_path, "w") as error_file:
                    json.dump(error_data_list, error_file, indent=4)
                continue

            try:
                for data in wdata_list:
                    if data["name"] == filename:
                        new_x1, new_y1 = np.dot(M, np.array([data["bbox"][0], data["bbox"][1], 1]))[:2] - startp
                        new_x2, new_y2 = np.dot(M, np.array([data["bbox"][2], data["bbox"][3], 1]))[:2] - startp

                        if(new_x1<0 or new_y1<0 or new_x2>rotated_image.shape[1] or new_y2>rotated_image.shape[0]):
                            with open(wjson_path, "r") as error_file:
                                error_data_list = json.load(error_file)
                            error_data_list.append(data)
                            with open(wjson_path, "w") as error_file:
                                json.dump(error_data_list, error_file, indent=4)
                            continue
                        else:
                            data["bbox"]= [new_x1, new_y1, new_x2, new_y2]
                            data["image_height"],data["image_width"]=rotated_image.shape[:2]
                            if (not os.path.exists(os.path.join(output_folder, filename))):
                                cv2.imwrite(os.path.join(output_folder, filename), rotated_image)
            except:
                with open(wjson_path, "r") as error_file:
                    error_data_list = json.load(error_file)
                error_data_list.append(data)
                with open(wjson_path, "w") as error_file:
                    json.dump(error_data_list, error_file, indent=4)
                continue

    with open(wjson_path, "w") as file4:
        json.dump(wdata_list, file4, indent=4)
