# Language-guided Event-based Action Recognition (Lear)
## Data Preparation
According to [ExACT/SeAct Dataset](https://github.com/jiazhou-garland/ExACT?tab=readme-ov-file#seact-dataset)
Support on SeAct|PAF|DVS128Gesture

## Training
```
# Train
python train.py --config configs/DVS128Gesture/DVS128Gesture_train.yaml
python train.py --config configs/PAF/PAF_train.yaml

```

## Testing
```
# Test 
python test.py --config configs/DVS128Gesture/DVS128Gesture_zero_shot_testing.yaml
python test.py --config configs/PAF/PAF_zero_shot_testing.yaml

```
## Acknowledgments
Code is based on [EZ-CLIP](https://github.com/Shahzadnit/EZ-CLIP) and [ExAct](https://github.com/jiazhou-garland/ExACT)
