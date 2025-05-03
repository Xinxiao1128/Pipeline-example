# 📦 ClearML Pipeline: Step 2 - 解压 + 图像预处理 + 上传为预处理数据集
from clearml import Task, Dataset
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from glob import glob
from PIL import Image
import zipfile

# 初始化任务
task = Task.init(project_name='Agri-Pest-Detection', task_name='Step 2 - Preprocessing')
task.execute_remotely(queue_name='default')  # ✅ 推送至远程 agent
params = task.connect({
    'dataset_task_id': '',  # ⚠️ 运行前需手动填写 step1 生成的 task_id
    'image_size': 256,
    'test_size': 0.2,
    'random_state': 42
})

# === Step 2.1: 下载上一步上传的数据集（zip格式） ===
dataset = Dataset.get(task_id=params['dataset_task_id'])
local_path = dataset.get_local_copy()

# === Step 2.2: 解压 archive.zip 到解压目录 ===
zip_path = os.path.join(local_path, 'archive.zip')
extract_dir = os.path.join(local_path, 'unzipped')

os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"✅ 解压完成，解压目录为：{extract_dir}")

# === Step 2.3: 图像预处理 ===
img_size = (params['image_size'], params['image_size'])
x_data, y_data = [], []
dirs = sorted([d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))])
label2id = {name: idx for idx, name in enumerate(dirs)}

for class_name in dirs:
    class_dir = os.path.join(extract_dir, class_name)
    for img_file in glob(os.path.join(class_dir, '*.jpg')):
        try:
            img = Image.open(img_file).convert('RGB').resize(img_size)
            x_data.append(np.array(img))
            y_data.append(label2id[class_name])
        except Exception as e:
            print(f"⚠️ 图像处理失败: {img_file}, 错误: {e}")
            continue

x_data = np.array(x_data)
y_data = np.array(y_data)

# === Step 2.4: 划分训练/测试集 ===
X_train, X_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=params['test_size'], random_state=params['random_state'])

# === Step 2.5: 保存为 .npy 并上传为新 Dataset ===
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('label2id.npy', label2id)

output_dataset = Dataset.create(dataset_name='PlantVillage-Preprocessed', dataset_project='Agri-Pest-Detection')
output_dataset.add_files('X_train.npy')
output_dataset.add_files('X_test.npy')
output_dataset.add_files('y_train.npy')
output_dataset.add_files('y_test.npy')
output_dataset.add_files('label2id.npy')
output_dataset.upload()
output_dataset.finalize()

print('✅ 图像预处理完成并已上传为新的 Dataset（用于训练）。')
