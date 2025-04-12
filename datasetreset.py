import os

def rename_jpg_files(directory):
    """
    遍历指定目录下的所有子文件夹，并将每个子文件夹中的.jpg文件重命名。
    每个子文件夹中的.jpg文件将按照数字顺序从1开始编号。
    """
    # 遍历指定目录下的所有子文件夹
    for subdir in os.listdir(directory):
        subdirectory = os.path.join(directory, subdir)
        if os.path.isdir(subdirectory):
            # 获取子文件夹中所有的.jpg文件
            files = [f for f in os.listdir(subdirectory) if f.endswith('.jpg')]
            print(f"Found {len(files)} .jpg files in {subdirectory}")
            # 对.jpg文件按数字顺序重命名
            for count, filename in enumerate(files, start=1):
                old_file_path = os.path.join(subdirectory, filename)
                new_file_name = f"{count:d}.jpg"  # 使用三位数格式化文件名
                new_file_path = os.path.join(subdirectory, new_file_name)
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                except Exception as e:
                    print(f"Error renaming '{old_file_path}': {e}")

# 调用函数并指定你的大文件夹路径
main_directory = '/root/autodl-tmp/mydataset/val'
rename_jpg_files(main_directory)