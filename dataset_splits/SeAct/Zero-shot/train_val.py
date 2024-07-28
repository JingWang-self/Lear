# 定义文件路径
seact_train_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct_train.txt'
seact_val_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct_val.txt'
test_file = '/root/wj/EZ_CLIP/dataset_splits/SeAct/Zero-shot/test.txt'
train_output_file = '/root/wj/EZ_CLIP/dataset_splits/SeAct/Zero-shot/train.txt'
val_output_file = '/root/wj/EZ_CLIP/dataset_splits/SeAct/Zero-shot/val.txt'

# 读取train和val文件
with open(seact_train_file, 'r') as f:
    train_paths = ['/'.join((line.strip().split('/')[-2],line.strip().split('/')[-1].split('-')[0])) for line in f.readlines()]
    print(train_paths)

with open(seact_val_file, 'r') as f:
    val_paths = ['/'.join((line.strip().split('/')[-2],line.strip().split('/')[-1].split('-')[0])) for line in f.readlines()]

# 初始化输出内容
train_output = []
val_output = []

# 读取test文件并分配到train和val
with open(test_file, 'r') as f:
    for line in f.readlines():
        path_identifier = '/'.join(line.strip().split(' ')[0].split('/')[-2:])  # 获取路径中的标识符
        if path_identifier in train_paths:
            train_output.append(line.strip())
        elif path_identifier in val_paths:
            val_output.append(line.strip())

# 写入train_output_file
with open(train_output_file, 'w') as f:
    for line in train_output:
        f.write(line + '\n')

# 写入val_output_file
with open(val_output_file, 'w') as f:
    for line in val_output:
        f.write(line + '\n')

print(f"train.txt and val.txt files have been created at {train_output_file} and {val_output_file}")
