import json
import csv

# 定义文件路径
label_mapping_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct.json'
idx_to_label_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct_idx_to_label.json'
description_file = '/root/wj/ExACT/Dataloader/SeAct/SeAct_ds_cls.json'
output_csv_file = '/root/wj/EZ_CLIP/GPT_discription/SeAct_gpt_Class_discription_new.csv'

# 读取JSON文件
with open(label_mapping_file, 'r') as f:
    idx_mapping = json.load(f)

with open(idx_to_label_file, 'r') as f:
    idx_to_label = json.load(f)

with open(description_file, 'r') as f:
    descriptions = json.load(f)

# 生成CSV文件内容
csv_data = []

for class_name, idx in idx_mapping.items():
    if str(idx) in idx_to_label:
        label = idx_to_label[str(idx)]
        description = next((desc for desc, desc_label in descriptions.items() if desc_label == idx), "")
        csv_data.append([label, class_name, '\n\n'+description.replace(f"{class_name}: ", f"{class_name} is ", 1)])

# 写入CSV文件
with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['SNo', 'Class Name', 'GPT3 discription'])
    for row in csv_data:
        csvwriter.writerow(row)

print(f"CSV file has been created at {output_csv_file}")
