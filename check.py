import os
import json
import cv2
import numpy as np
from tqdm import tqdm

# 设置文件夹路径
root_folder = 'output'
defect_folder = 'rotated'  # 保存原始缺陷图像的文件夹
output_folder = 'cropped_images'  # 保存裁剪后的图像的文件夹

# 创建输出文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历每个 defect 文件夹
for defect_dir in os.listdir(root_folder):
    defect_path = os.path.join(root_folder, defect_dir)
    if os.path.isdir(defect_path):
        # 获取文件夹下的 JSON 文件
        json_files = [f for f in os.listdir(defect_path) if (f.endswith('.json') and (not f.endswith('ord.json')))]

        # 遍历每个 JSON 文件
        for json_file in json_files:
            # 构造文件路径
            json_path = os.path.join(defect_path, json_file)
            folder_name = os.path.splitext(json_file)[0].split('_')[-1]
            output_defect_folder = os.path.join(output_folder, defect_dir)
            output_cam_folder = os.path.join(output_defect_folder, folder_name)

            # 创建输出文件夹
            if not os.path.exists(output_defect_folder):
                os.makedirs(output_defect_folder)
            if not os.path.exists(output_cam_folder):
                os.makedirs(output_cam_folder)

            # 读取 JSON 文件
            with open(json_path, 'r') as f:
                data = json.load(f)

            for img_data in tqdm(data,leave=False):
                # 读取图像
                img_name = img_data['name']
                defect_img_name = img_data['defect_img_name']
                img_path = os.path.join(defect_path, folder_name, img_name)
                defect_img_path = os.path.join(defect_folder, defect_img_name)

                if os.path.exists(img_path) and os.path.exists(defect_img_path):
                    img = cv2.imread(img_path)
                    defect_img = cv2.imread(defect_img_path)

                    # 截取图像
                    bbox = img_data['bbox']  # [x, y, w, h]
                    defect_box = img_data['defect_bbox']  # [x, y, w, h]
                    img_cropped = img[int(bbox[1]-50):int(bbox[3]+50), int(bbox[0]-50):int(bbox[2]+50)]
                    defect_img_cropped = defect_img[int(defect_box[1]-50):int(defect_box[3]+50), int(defect_box[0]-50):int(defect_box[2]+50)]

                    # 为了匹配两个数组的尺寸,在更小的数组的维度 0 上添加填充
                    if img_cropped.shape[0] < defect_img_cropped.shape[0]:
                        img_cropped = np.pad(img_cropped, ((0, defect_img_cropped.shape[0] - img_cropped.shape[0]), (0, 0), (0, 0)), mode='constant')
                    elif img_cropped.shape[0] > defect_img_cropped.shape[0]:
                        defect_img_cropped = np.pad(defect_img_cropped, ((0, img_cropped.shape[0] - defect_img_cropped.shape[0]), (0, 0), (0, 0)), mode='constant')

                    # 合并图像
                    img_combined = np.hstack((img_cropped, defect_img_cropped))

                    # 保存图像
                    cv2.imwrite(os.path.join(output_cam_folder, img_name + '.jpg'), img_combined)