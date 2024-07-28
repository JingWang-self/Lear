import json
import csv

# 定义文件路径
idx_to_label_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct_idx_to_label.json'
idx_mapping_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct.json'
output_csv_file = '/root/wj/EZ_CLIP/lists/seact_labels.csv'

# 读取JSON文件
with open(idx_to_label_file, 'r') as f:
    idx_to_label = json.load(f)

with open(idx_mapping_file, 'r') as f:
    idx_mapping = json.load(f)

# 生成CSV文件内容
csv_data = []

# 创建一个反向映射以便快速查找类名
idx_to_class_name = {value: key for key, value in idx_mapping.items()}

for idx, label in idx_to_label.items():
    class_name = idx_to_class_name.get(int(idx), "Unknown")
    csv_data.append([label, class_name])

# 写入CSV文件
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['id', 'name'])
    for row in csv_data:
        csvwriter.writerow(row)

print(f"CSV file has been created at {output_csv_file}")
