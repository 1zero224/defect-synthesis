import json
import random
import os


def get_cam_json():
    input_json_path = "train_annos_rotated_fix.json"
    defect1_CAM1_json_path = "defect_json/defect1_CAM1.json"
    defect1_CAM2_json_path = "defect_json/defect1_CAM2.json"
    defect1_CAM3_json_path = "defect_json/defect1_CAM3.json"
    defect1_CAM3_json_path = "defect_json/defect1_CAM3.json"
    defect2_CAM1_json_path = "defect_json/defect2_CAM1.json"
    defect2_CAM2_json_path = "defect_json/defect2_CAM2.json"
    defect2_CAM3_json_path = "defect_json/defect2_CAM3.json"
    defect3_CAM1_json_path = "defect_json/defect3_CAM1.json"
    defect3_CAM2_json_path = "defect_json/defect3_CAM2.json"
    defect3_CAM3_json_path = "defect_json/defect3_CAM3.json"
    defect4_CAM1_json_path = "defect_json/defect4_CAM1.json"
    defect4_CAM2_json_path = "defect_json/defect4_CAM2.json"
    defect4_CAM3_json_path = "defect_json/defect4_CAM3.json"
    defect5_CAM1_json_path = "defect_json/defect5_CAM1.json"
    defect5_CAM2_json_path = "defect_json/defect5_CAM2.json"
    defect5_CAM3_json_path = "defect_json/defect5_CAM3.json"
    defect6_CAM1_json_path = "defect_json/defect6_CAM1.json"
    defect6_CAM2_json_path = "defect_json/defect6_CAM2.json"
    defect6_CAM3_json_path = "defect_json/defect6_CAM3.json"
    defect_json=[defect1_CAM1_json_path,defect1_CAM2_json_path,defect1_CAM3_json_path,defect2_CAM1_json_path,defect2_CAM2_json_path,defect2_CAM3_json_path,defect3_CAM1_json_path,defect3_CAM2_json_path,defect3_CAM3_json_path,defect4_CAM1_json_path,defect4_CAM2_json_path,defect4_CAM3_json_path,defect5_CAM1_json_path,defect5_CAM2_json_path,defect5_CAM3_json_path,defect6_CAM1_json_path,defect6_CAM2_json_path,defect6_CAM3_json_path]

    for defejson in defect_json:
        if not os.path.exists(defejson):
            with open(defejson, "w") as f:
                json.dump([], f)

    with open(input_json_path, 'r') as f:
        img_data = json.load(f)

    with open(defect1_CAM1_json_path, 'r') as f: defect1_CAM1_data = json.load(f)
    with open(defect1_CAM2_json_path, 'r') as f: defect1_CAM2_data = json.load(f)
    with open(defect1_CAM3_json_path, 'r') as f: defect1_CAM3_data = json.load(f)
    with open(defect2_CAM1_json_path, 'r') as f: defect2_CAM1_data = json.load(f)
    with open(defect2_CAM2_json_path, 'r') as f: defect2_CAM2_data = json.load(f)
    with open(defect2_CAM3_json_path, 'r') as f: defect2_CAM3_data = json.load(f)
    with open(defect3_CAM1_json_path, 'r') as f: defect3_CAM1_data = json.load(f)
    with open(defect3_CAM2_json_path, 'r') as f: defect3_CAM2_data = json.load(f)
    with open(defect3_CAM3_json_path, 'r') as f: defect3_CAM3_data = json.load(f)
    with open(defect4_CAM1_json_path, 'r') as f: defect4_CAM1_data = json.load(f)
    with open(defect4_CAM2_json_path, 'r') as f: defect4_CAM2_data = json.load(f)
    with open(defect4_CAM3_json_path, 'r') as f: defect4_CAM3_data = json.load(f)
    with open(defect5_CAM1_json_path, 'r') as f: defect5_CAM1_data = json.load(f)
    with open(defect5_CAM2_json_path, 'r') as f: defect5_CAM2_data = json.load(f)
    with open(defect5_CAM3_json_path, 'r') as f: defect5_CAM3_data = json.load(f)
    with open(defect6_CAM1_json_path, 'r') as f: defect6_CAM1_data = json.load(f)
    with open(defect6_CAM2_json_path, 'r') as f: defect6_CAM2_data = json.load(f)
    with open(defect6_CAM3_json_path, 'r') as f: defect6_CAM3_data = json.load(f)

    for defect_img_data in img_data:
        defect_img_name = defect_img_data["name"]
        gen_img = None
        if defect_img_data["category"] == 1:
            if "_CAM1" in defect_img_name: defect1_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect1_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect1_CAM3_data.append(defect_img_data)
        elif defect_img_data["category"] == 2:
            if "_CAM1" in defect_img_name: defect2_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect2_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect2_CAM3_data.append(defect_img_data)
        elif defect_img_data["category"] == 3:
            if "_CAM1" in defect_img_name: defect3_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect3_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect3_CAM3_data.append(defect_img_data)
        elif defect_img_data["category"] == 4:
            if "_CAM1" in defect_img_name: defect4_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect4_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect4_CAM3_data.append(defect_img_data)
        elif defect_img_data["category"] == 5:
            if "_CAM1" in defect_img_name: defect5_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect5_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect5_CAM3_data.append(defect_img_data)
        elif defect_img_data["category"] == 6:
            if "_CAM1" in defect_img_name: defect6_CAM1_data.append(defect_img_data)
            elif "_CAM2" in defect_img_name: defect6_CAM2_data.append(defect_img_data)
            elif "_CAM3" in defect_img_name: defect6_CAM3_data.append(defect_img_data)
    
    with open(defect1_CAM1_json_path, 'w') as f: json.dump(defect1_CAM1_data, f, indent=4)
    with open(defect1_CAM2_json_path, 'w') as f: json.dump(defect1_CAM2_data, f, indent=4)
    with open(defect1_CAM3_json_path, 'w') as f: json.dump(defect1_CAM3_data, f, indent=4)
    with open(defect2_CAM1_json_path, 'w') as f: json.dump(defect2_CAM1_data, f, indent=4)
    with open(defect2_CAM2_json_path, 'w') as f: json.dump(defect2_CAM2_data, f, indent=4)
    with open(defect2_CAM3_json_path, 'w') as f: json.dump(defect2_CAM3_data, f, indent=4)
    with open(defect3_CAM1_json_path, 'w') as f: json.dump(defect3_CAM1_data, f, indent=4)
    with open(defect3_CAM2_json_path, 'w') as f: json.dump(defect3_CAM2_data, f, indent=4)
    with open(defect3_CAM3_json_path, 'w') as f: json.dump(defect3_CAM3_data, f, indent=4)
    with open(defect4_CAM1_json_path, 'w') as f: json.dump(defect4_CAM1_data, f, indent=4)
    with open(defect4_CAM2_json_path, 'w') as f: json.dump(defect4_CAM2_data, f, indent=4)
    with open(defect4_CAM3_json_path, 'w') as f: json.dump(defect4_CAM3_data, f, indent=4)
    with open(defect5_CAM1_json_path, 'w') as f: json.dump(defect5_CAM1_data, f, indent=4)
    with open(defect5_CAM2_json_path, 'w') as f: json.dump(defect5_CAM2_data, f, indent=4)
    with open(defect5_CAM3_json_path, 'w') as f: json.dump(defect5_CAM3_data, f, indent=4)
    with open(defect6_CAM1_json_path, 'w') as f: json.dump(defect6_CAM1_data, f, indent=4)
    with open(defect6_CAM2_json_path, 'w') as f: json.dump(defect6_CAM2_data, f, indent=4)
    with open(defect6_CAM3_json_path, 'w') as f: json.dump(defect6_CAM3_data, f, indent=4)


def get_defect_img_json():
    input_json_path = "train_annos_rotated_fix.json"

    with open(input_json_path, 'r') as f:
        img_data = json.load(f)

        # 创建一个字典用于存储结果
    result = {}

    # 遍历JSON数据
    for item in img_data:
        name = item['name']
        category = item['category']

        # 如果该名称已存在于结果字典中,则将category值添加到现有集合中
        if name in result:
            result[name].add(category)
        # 如果该名称不存在于结果字典中,则创建一个新的键值对
        else:
            result[name] = {category}

    # 将集合转换为列表
    for key, value in result.items():
        result[key] = list(value)

    # 将结果写入新的JSON文件
    with open('defect_img_category.json', 'w') as f:
        json.dump(result, f, indent=4)


def filter_defect12():
    with open('defect_img_category.json') as file:
        data = json.load(file)

    filtered_data = {key: value for key, value in data.items() if not any(val in value for val in [1, 2])}

    with open('filtered_defect_img_category.json', 'w') as file:
        json.dump(filtered_data, file, indent=4)


def divide_base(input_json_path, output_json_path1, output_json_path2, output_json_path3):
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    cam1_data = {}
    cam2_data = {}
    cam3_data = {}

    for filename, values in data.items():
        if '_CAM1' in filename:
            cam1_data[filename] = values
        elif '_CAM2' in filename:
            cam2_data[filename] = values
        elif '_CAM3' in filename:
            cam3_data[filename] = values

    if not output_json_path1 in os.listdir():
        with open(output_json_path1, 'w') as f:
            json.dump([], f)

    if not output_json_path2 in os.listdir():
        with open(output_json_path2, 'w') as f:
            json.dump([], f)

    if not output_json_path3 in os.listdir():
        with open(output_json_path3, 'w') as f:
            json.dump([], f)

    # 将分类后的数据写入三个新的 JSON 文件
    with open(output_json_path1, 'w') as f:
        json.dump(cam1_data, f, indent=4)

    with open(output_json_path2, 'w') as f:
        json.dump(cam2_data, f, indent=4)

    with open(output_json_path3, 'w') as f:
        json.dump(cam3_data, f, indent=4)


def create_error_json(defect_list):
    defect_json = []
    if 11 in defect_list:
        defect1_CAM1_json_path = "error/error_defect1_CAM1.json"
        defect_json.append(defect1_CAM1_json_path)
    if 12 in defect_list:
        defect1_CAM2_json_path = "error/error_defect1_CAM2.json"
        defect_json.append(defect1_CAM2_json_path)
    if 13 in defect_list:
        defect1_CAM3_json_path = "error/error_defect1_CAM3.json"
        defect_json.append(defect1_CAM3_json_path)
    if 21 in defect_list:
        defect2_CAM1_json_path = "error/error_defect2_CAM1.json"
        defect_json.append(defect2_CAM1_json_path)
    if 22 in defect_list:
        defect2_CAM2_json_path = "error/error_defect2_CAM2.json"
        defect_json.append(defect2_CAM2_json_path)
    if 23 in defect_list:
        defect2_CAM3_json_path = "error/error_defect2_CAM3.json"
        defect_json.append(defect2_CAM3_json_path)
    if 31 in defect_list:
        defect3_CAM1_json_path = "error/error_defect3_CAM1.json"
        defect_json.append(defect3_CAM1_json_path)
    if 32 in defect_list:
        defect3_CAM2_json_path = "error/error_defect3_CAM2.json"
        defect_json.append(defect3_CAM2_json_path)
    if 33 in defect_list:
        defect3_CAM3_json_path = "error/error_defect3_CAM3.json"
        defect_json.append(defect3_CAM3_json_path)
    if 41 in defect_list:
        defect4_CAM1_json_path = "error/error_defect4_CAM1.json"
        defect_json.append(defect4_CAM1_json_path)
    if 42 in defect_list:
        defect4_CAM2_json_path = "error/error_defect4_CAM2.json"
        defect_json.append(defect4_CAM2_json_path)
    if 43 in defect_list:
        defect4_CAM3_json_path = "error/error_defect4_CAM3.json"
        defect_json.append(defect4_CAM3_json_path)
    if 51 in defect_list:
        defect5_CAM1_json_path = "error/error_defect5_CAM1.json"
        defect_json.append(defect5_CAM1_json_path)
    if 52 in defect_list:
        defect5_CAM2_json_path = "error/error_defect5_CAM2.json"
        defect_json.append(defect5_CAM2_json_path)
    if 53 in defect_list:
        defect5_CAM3_json_path = "error/error_defect5_CAM3.json"
        defect_json.append(defect5_CAM3_json_path)
    if 61 in defect_list:
        defect6_CAM1_json_path = "error/error_defect6_CAM1.json"
        defect_json.append(defect6_CAM1_json_path)
    if 62 in defect_list:
        defect6_CAM2_json_path = "error/error_defect6_CAM2.json"
        defect_json.append(defect6_CAM2_json_path)
    if 63 in defect_list:
        defect6_CAM3_json_path = "error/error_defect6_CAM3.json"
        defect_json.append(defect6_CAM3_json_path)

    for defejson in defect_json:
        if not os.path.exists(defejson):
            with open(defejson, "w") as f:
                json.dump([], f)


def select_random_data(num_samples, file1_path, file2_path, output_file1_path, output_file2_path):
    with open(file1_path, 'r') as f:
        file1_data = json.load(f)
    with open(file2_path, 'r') as f:
        file2_data = json.load(f)

    # 从文件1中随机挑选数据
    file1_data_len = len(file1_data)
    random_file1_data = file1_data.copy()
    if num_samples > file1_data_len:
        remaining_samples = num_samples - file1_data_len
        file1_data_copy = file1_data.copy()
        while remaining_samples > 0:
            if not file1_data_copy:
                file1_data_copy = file1_data.copy()
            random_sample = random.choice(file1_data_copy)
            random_file1_data.append(random_sample)
            file1_data_copy.remove(random_sample)
            remaining_samples -= 1
    else:
        random_file1_data = random.sample(file1_data, num_samples)

    # 从文件2中随机挑选数据
    random_file2_data = []
    file2_keys = list(file2_data.keys())
    while len(random_file2_data) < num_samples:
        random_key = random.choice(file2_keys)
        if random_key not in [sample['name'] for sample in random_file1_data]:
            random_file2_data.append({random_key: file2_data[random_key]})
            file2_keys.remove(random_key)


    if not output_file1_path in os.listdir():
        with open(output_file1_path, 'w') as f:
            json.dump([], f)
    if not output_file2_path in os.listdir():
        with open(output_file2_path, 'w') as f:
            json.dump([], f)

    with open(output_file1_path, 'w') as f:
        json.dump(random_file1_data, f, indent=4)
    with open(output_file2_path, 'w') as f:
        json.dump(random_file2_data, f, indent=4)


def select_json(num, defect_list):
    if 11 in defect_list and 12 in defect_list and 13 in defect_list: sub_num1 = num // 3
    else: sub_num1 = num
    if 21 in defect_list and 22 in defect_list and 23 in defect_list: sub_num2 = num // 3
    else: sub_num2 = num
    if 31 in defect_list and 32 in defect_list and 33 in defect_list: sub_num3 = num // 3
    else: sub_num3 = num
    if 41 in defect_list and 42 in defect_list and 43 in defect_list: sub_num4 = num // 3
    else: sub_num4 = num
    if 51 in defect_list and 52 in defect_list and 53 in defect_list: sub_num5 = num // 3
    else: sub_num5 = num
    if 61 in defect_list and 62 in defect_list and 63 in defect_list: sub_num6 = num // 3
    else: sub_num6 = num

    if 11 in defect_list: select_random_data(sub_num1, "defect_json/defect1_CAM1.json", "filtered_defect_img_category_CAM1.json", "choice/defect1_CAM1_img.json", "choice/base_img_for11.json")
    if 12 in defect_list: select_random_data(sub_num1, "defect_json/defect1_CAM2.json", "filtered_defect_img_category_CAM2.json", "choice/defect1_CAM2_img.json", "choice/base_img_for12.json")
    if 13 in defect_list: select_random_data(sub_num1, "defect_json/defect1_CAM3.json", "filtered_defect_img_category_CAM3.json", "choice/defect1_CAM3_img.json", "choice/base_img_for13.json")
    if 21 in defect_list: select_random_data(sub_num2, "defect_json/defect2_CAM1.json", "filtered_defect_img_category_CAM1.json", "choice/defect2_CAM1_img.json", "choice/base_img_for21.json")
    if 22 in defect_list: select_random_data(sub_num2, "defect_json/defect2_CAM2.json", "filtered_defect_img_category_CAM2.json", "choice/defect2_CAM2_img.json", "choice/base_img_for22.json")
    if 23 in defect_list: select_random_data(sub_num2, "defect_json/defect2_CAM3.json", "filtered_defect_img_category_CAM3.json", "choice/defect2_CAM3_img.json", "choice/base_img_for23.json")
    if 31 in defect_list: select_random_data(sub_num3, "defect_json/defect3_CAM1.json", "defect_img_category_CAM1.json", "choice/defect3_CAM1_img.json", "choice/base_img_for31.json")
    if 32 in defect_list: select_random_data(sub_num3, "defect_json/defect3_CAM2.json", "defect_img_category_CAM2.json", "choice/defect3_CAM2_img.json", "choice/base_img_for32.json")
    if 33 in defect_list: select_random_data(sub_num3, "defect_json/defect3_CAM3.json", "defect_img_category_CAM3.json", "choice/defect3_CAM3_img.json", "choice/base_img_for33.json")
    if 41 in defect_list: select_random_data(sub_num4, "defect_json/defect4_CAM1.json", "defect_img_category_CAM1.json", "choice/defect4_CAM1_img.json", "choice/base_img_for41.json")
    if 42 in defect_list: select_random_data(sub_num4, "defect_json/defect4_CAM2.json", "defect_img_category_CAM2.json", "choice/defect4_CAM2_img.json", "choice/base_img_for42.json")
    if 43 in defect_list: select_random_data(sub_num4, "defect_json/defect4_CAM3.json", "defect_img_category_CAM3.json", "choice/defect4_CAM3_img.json", "choice/base_img_for43.json")
    if 51 in defect_list: select_random_data(sub_num5, "defect_json/defect5_CAM1.json", "defect_img_category_CAM1.json", "choice/defect5_CAM1_img.json", "choice/base_img_for51.json")
    if 52 in defect_list: select_random_data(sub_num5, "defect_json/defect5_CAM2.json", "defect_img_category_CAM2.json", "choice/defect5_CAM2_img.json", "choice/base_img_for52.json")
    if 53 in defect_list: select_random_data(sub_num5, "defect_json/defect5_CAM3.json", "defect_img_category_CAM3.json", "choice/defect5_CAM3_img.json", "choice/base_img_for53.json")
    if 61 in defect_list: select_random_data(sub_num6, "defect_json/defect6_CAM1.json", "defect_img_category_CAM1.json", "choice/defect6_CAM1_img.json", "choice/base_img_for61.json")
    if 62 in defect_list: select_random_data(sub_num6, "defect_json/defect6_CAM2.json", "defect_img_category_CAM2.json", "choice/defect6_CAM2_img.json", "choice/base_img_for62.json")
    if 63 in defect_list: select_random_data(sub_num6, "defect_json/defect6_CAM3.json", "defect_img_category_CAM3.json", "choice/defect6_CAM3_img.json", "choice/base_img_for63.json")


if __name__ == "__main__":
    get_cam_json()
    get_defect_img_json()
    filter_defect12()
    divide_base("filtered_defect_img_category.json", "filtered_defect_img_category_CAM1.json", "filtered_defect_img_category_CAM2.json", "filtered_defect_img_category_CAM3.json")
    divide_base("defect_img_category.json", "defect_img_category_CAM1.json", "defect_img_category_CAM2.json", "defect_img_category_CAM3.json")
    create_error_json()
    select_json()