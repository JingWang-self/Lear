#!/bin/bash

# 定义数据集列表
datasets=("SeAct" "PAF" "DVS128Gesture" "HARDVS")

# 创建日志目录
LOG_DIR="logs/train"
mkdir -p $LOG_DIR

# 生成任务列表
generate_task_list() {
    for dataset in "${datasets[@]}"; do
        CONFIG_FILE="configs/${dataset}/${dataset}_train.yaml"
        LOG_FILE="$LOG_DIR/${dataset}_train.log"
        echo "python train__SAMPLE.py --config $CONFIG_FILE > $LOG_FILE 2>&1"
    done
}

# 测试任务是否预期运行
# generate_task_list | parallel --dry-run -j 4

# 生成任务并通过 parallel 执行
generate_task_list | parallel -j 1 --progress 'nohup {} &'# 动态设置并行任务数