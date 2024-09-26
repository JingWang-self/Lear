#!/bin/bash

# 定义数据集列表和 few-shot 设置
datasets=("SeAct" "PAF" "DVS128Gesture" "HARDVS")
shots=(2 4 8 16)

# 创建日志目录
LOG_DIR="logs/base-to-novel"
mkdir -p $LOG_DIR


# 生成 base-to-novel 训练任务列表
generate_base_to_novel_task_list() {
    for dataset in "${datasets[@]}"; do
        CONFIG_FILE="configs/${dataset}/${dataset}_base_to_novel.yaml"
        LOG_FILE="$LOG_DIR/${dataset}_base_to_novel_training.log"
        echo "python train_base_to_novel_SAMPLE.py --config $CONFIG_FILE > $LOG_FILE 2>&1 | grep -v '^\s*%|'"
    done
}
# 并行执行 base-to-novel 训练任务
echo "Starting base-to-novel training tasks..."
generate_base_to_novel_task_list | parallel -j 1 --progress
