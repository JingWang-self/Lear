#!/bin/bash

# 定义数据集列表和 few-shot 设置
datasets=("SeAct" "PAF" "DVS128Gesture" )
shots=(2 4 8 16)

# 创建日志目录
LOG_DIR="logs/zero-shot"
mkdir -p $LOG_DIR

# 生成 zero-shot 测试任务列表
generate_zero_shot_task_list() {
    for dataset in "${datasets[@]}"; do
        CONFIG_FILE="configs/${dataset}/${dataset}_zero_shot_testing.yaml"
        LOG_FILE="$LOG_DIR/${dataset}_zero_shot_testing.log"
        echo "python test_SAMPLE.py --config $CONFIG_FILE > $LOG_FILE 2>&1"
    done
}

# 并行执行 zero-shot 测试任务
echo "Starting zero-shot testing tasks..."
generate_task_list | parallel -j 1 --progress 'nohup {} &'# 动态设置并行任务数