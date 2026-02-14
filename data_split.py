# 划分数据集，原始数据集放入dataset_org中，处理后的数据会生成在dataset中
import os
import shutil
import random

# 划分比例，分别是训练集、验证集合测试集，根据自己需要改一下
split_rate = [0.7, 0.2, 0.1]
split_names = ["train", "valid", "test"]

# 是否打乱
shuffle = True

def replace_expand_name(file_name, ex_name):
    return ".".join(file_name.split(".")[:-1] + [ex_name])

if os.path.exists("dataset"):
    shutil.rmtree("dataset")
os.makedirs("dataset")
for name in split_names:
    os.makedirs(os.path.join("dataset", name, "images"))
    os.makedirs(os.path.join("dataset", name, "labels"))

image_folder = r"dataset_org\images"
label_folder = r"dataset_org\labels"

# 获取图片文件列表
image_files = os.listdir(image_folder)

# 过滤：只保留有对应标签文件的图片
valid_image_files = []
valid_label_files = []

for image_file in image_files:
    label_file = replace_expand_name(image_file, 'txt')
    label_path = os.path.join(label_folder, label_file)
    
    # 检查对应的标签文件是否存在
    if os.path.exists(label_path):
        valid_image_files.append(image_file)
        valid_label_files.append(label_file)
    else:
        print(f"警告: 找不到标签文件 {label_file}，跳过图片 {image_file}")

print(f"找到 {len(valid_image_files)} 对有效的图片-标签文件")

if shuffle:
    # 同时打乱图片和标签文件列表，保持对应关系
    combined = list(zip(valid_image_files, valid_label_files))
    random.shuffle(combined)
    valid_image_files, valid_label_files = zip(*combined)
    valid_image_files = list(valid_image_files)
    valid_label_files = list(valid_label_files)

def write_files(image_files, label_files, split):
    copied_count = 0
    for image_file, label_file in zip(image_files, label_files):
        image_src = os.path.join(image_folder, image_file)
        label_src = os.path.join(label_folder, label_file)
        image_dst = os.path.join("dataset", split, "images", image_file)
        label_dst = os.path.join("dataset", split, "labels", label_file)
        
        # 再次检查文件是否存在（虽然前面已经检查过）
        if os.path.exists(image_src) and os.path.exists(label_src):
            shutil.copy(image_src, image_dst)
            shutil.copy(label_src, label_dst)
            copied_count += 1
        else:
            print(f"错误: 文件不存在 - 图片: {image_src} 或 标签: {label_src}")
    
    print(f"{split} 数据集: 成功复制 {copied_count} 对文件")
    return copied_count

data_len = len(valid_image_files)
print(f"总有效文件数: {data_len}")

# 使用累积比例来计算分割点
cumulative_rates = [sum(split_rate[:i+1]) for i in range(len(split_rate))]

train_end = int(cumulative_rates[0] * data_len)
valid_end = int(cumulative_rates[1] * data_len)

# 划分数据集
train_count = write_files(valid_image_files[:train_end], valid_label_files[:train_end], split_names[0])
valid_count = write_files(valid_image_files[train_end:valid_end], valid_label_files[train_end:valid_end], split_names[1])
test_count = write_files(valid_image_files[valid_end:], valid_label_files[valid_end:], split_names[2])

print(f"\n数据集划分完成：")
print(f"训练集: {train_count} 个样本")
print(f"验证集: {valid_count} 个样本") 
print(f"测试集: {test_count} 个样本")
print(f"总计: {train_count + valid_count + test_count} 个样本")