import os

def generate_text_files(root_directory, output_file):
    """
    遍历根目录下的所有子文件夹，为每个子文件夹中的图片生成文本记录。
    
    :param root_directory: 根目录路径
    :param output_file: 输出文本文件的路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # 遍历根目录
        for subdir in os.listdir(root_directory):
            subdir_path = os.path.join(root_directory, subdir)
            if os.path.isdir(subdir_path):
                # 遍历子文件夹中的文件
                for file in os.listdir(subdir_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 构建文件路径和标签
                        file_path = os.path.join(subdir, file)
                        label = subdir  # 假设文件夹名称即为标签
                        # 写入输出文件
                        f.write(f"{file_path} {label}\n")

# 使用示例
root_directory = '/root/autodl-tmp/food172/train'  # 替换为您的数据集路径
output_file = '/root/autodl-tmp/food172/train/train.txt'  # 输出文件名
generate_text_files(root_directory, output_file)