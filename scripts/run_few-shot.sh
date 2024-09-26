#!/bin/bash

# 定义数据集列表和 few-shot 设置
datasets=("SeAct" "PAF" "DVS128Gesture" "HARDVS")
shots=(2 4 8 16)

# 创建日志目录
LOG_DIR="logs/few-shot"
mkdir -p $LOG_DIR

# 生成任务列表
generate_task_list() {
    for dataset in "${datasets[@]}"; do
        for shot in "${shots[@]}"; do
            CONFIG_FILE="configs/few_shot/${dataset}_few_shot.yaml"
            LOG_FILE="$LOG_DIR/${dataset}_few_shot_${shot}.log"
            echo "python train_SAMPLE.py --config $CONFIG_FILE --few_shot $shot > $LOG_FILE 2>&1"
        done
    done
}
# 测试任务是否预期运行
# generate_task_list | parallel --dry-run -j 4
# 生成任务并通过 parallel 并行执行
generate_task_list | parallel --jobs 1 --progress 
