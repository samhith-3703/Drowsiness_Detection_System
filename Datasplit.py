import os
import shutil
import random


parent_dataset_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset_(DDD)'


train_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\train'
val_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\validation'
test_dir = r'D:\VS\Capstone_project\Trials\Driver_Drowsiness_Dataset\test'


for folder in [train_dir, val_dir, test_dir]:
    for subfolder in ['Drowsy', 'Non_Drowsy']:
        os.makedirs(os.path.join(folder, subfolder), exist_ok=True)


train_ratio = 0.7
val_ratio = 0.2    
test_ratio = 0.1  


def split_and_copy(source_dir, train_dir, val_dir, test_dir):
    files = os.listdir(source_dir)
    random.shuffle(files)  


    train_count = int(len(files) * train_ratio)
    val_count = int(len(files) * val_ratio)


   
    train_files = files[:train_count]
    val_files = files[train_count:train_count + val_count]
    test_files = files[train_count + val_count:]


    for file in train_files:
        shutil.copy(os.path.join(source_dir, file), train_dir)
    for file in val_files:
        shutil.copy(os.path.join(source_dir, file), val_dir)
    for file in test_files:
        shutil.copy(os.path.join(source_dir, file), test_dir)
split_and_copy(
    source_dir=os.path.join(parent_dataset_dir, 'Drowsy'),
    train_dir=os.path.join(train_dir, 'Drowsy'),
    val_dir=os.path.join(val_dir, 'Drowsy'),
    test_dir=os.path.join(test_dir, 'Drowsy')
)


split_and_copy(
    source_dir=os.path.join(parent_dataset_dir, 'Non_Drowsy'),
    train_dir=os.path.join(train_dir, 'Non_Drowsy'),
    val_dir=os.path.join(val_dir, 'Non_Drowsy'),
    test_dir=os.path.join(test_dir, 'Non_Drowsy')
)


print("Dataset successfully split into training, validation, and testing sets.")





