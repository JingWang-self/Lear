#!/bin/bash

# 定义数据集列表和 few-shot 设置
datasets=("SeAct" "PAF" "DVS128Gesture" "HARDVS")
shots=(2 4 8 16)

# 创建日志目录
LOG_DIR="logs/test"
mkdir -p $LOG_DIR


# 生成测试任务列表
generate_test_task_list() {
    for dataset in "${datasets[@]}"; do
        CONFIG_FILE="configs/${dataset}/${dataset}_testing.yaml"
        LOG_FILE="$LOG_DIR/${dataset}_testing.log"
        echo "python test_SAMPLE.py --config $CONFIG_FILE > $LOG_FILE 2>&1"
    done
}

# 并行执行测试任务
echo "Starting testing tasks..."
generate_task_list | parallel -j 1 --progress 'nohup {} &'# 动态设置并行任务数