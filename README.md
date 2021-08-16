# Plate Detection

## Usage

### Demo
检测 `./images` 文件夹下所有图片车牌，并写入结果到 `./result` 文件夹中
`python tflite_inference.py`

* 检测效果
![result_0.jpg](./result/1.jpg)

### Eval
使用 `ccpd_test` 文件夹中数据进行 int8 模型评估
`python eval_int8.py`
