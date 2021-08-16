from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image
import math
from anchor import anchors
from PIL import ImageDraw


class CVRSGModel():
    def __init__(self, tflite_model_path):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        assert self.input_details[0]['dtype'] == np.int8

    def input_w(self):
        return self.input_details[0]['shape'][2]

    def input_h(self):
        return self.input_details[0]['shape'][1]

    def process(self, img_resized):
        input_array = np.array(img_resized, dtype=np.float32)
        input_array -= 128.0
        input_array = np.array(input_array, dtype=np.int8)
        input_array = np.reshape(input_array, self.input_details[0]['shape'])
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_array)
        self.interpreter.invoke()

        class_preds = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        box_encodings = self.interpreter.get_tensor(
            self.output_details[1]['index'])
        return class_preds, box_encodings


def decode_box_encoding(box_encoding, anchor):

    decodeBox = np.zeros([4])
    y_scale = 10.0
    x_scale = 10.0
    h_scale = 5.0
    w_scale = 5.0

    ycenter = box_encoding[0] / y_scale * anchor[2] + anchor[0]
    xcenter = box_encoding[1] / x_scale * anchor[3] + anchor[1]
    half_h = 0.5 * math.exp((box_encoding[2] / h_scale)) * anchor[2]
    half_w = 0.5 * math.exp((box_encoding[3] / w_scale)) * anchor[3]

    decodeBox[0] = (ycenter - half_h)   # ymin
    decodeBox[1] = (xcenter - half_w)   # xmin
    decodeBox[2] = (ycenter + half_h)   # ymax
    decodeBox[3] = (xcenter + half_w)   # xmax

    ymin = (ycenter - half_h)   # ymin
    xmin = (xcenter - half_w)   # xmin
    ymax = (ycenter + half_h)   # ymax
    xmax = (xcenter + half_w)   # xmax

    return decodeBox, xmin, ymin, xmax, ymax


def get_bounding_box(image_name, xratio=1, yratio=1):
    kp = []
    kp_serial = image_name.split('-')[3].split('_')

    for i in range(0, 4):
        tmp = kp_serial[i].split('&')
        kp.append(int(tmp[0]))
        kp.append(int(tmp[1]))

    # 左上角点
    tlx = min(kp[2], kp[4]) * xratio
    tly = min(kp[5], kp[7]) * yratio

    # 右下角点
    brx = max(kp[0], kp[6]) * xratio
    bry = max(kp[1], kp[3]) * yratio

    return (tlx, tly, brx, bry)


def compute_iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter

    iou = inter / union
    return iou


if __name__ == '__main__':
    tflite_model_path = 'tflite_model/ssdlite_mbv2.tflite'
    image_dir = 'ccpd_test/'

    threshold = 0.8
    num_great_than_threshold = 0

    model = CVRSGModel(tflite_model_path)

    image_paths = [f for f in Path(image_dir).iterdir()]
    image_paths.sort(key=lambda f: f.stem, reverse=True)


    for image_path in image_paths:
        image = Image.open(image_path).crop(
            (0, 0, 720, 720)).resize((model.input_w(), model.input_h()))

        xratio = image.width / 720.0
        yratio = image.height / 720.0

        ground_true_bbox = get_bounding_box(image_path.name, xratio, yratio)

        class_preds, box_encodings = model.process(image)
        box_idx = np.argmax(class_preds[0][:, 1])

        box = 0.03998513147234917 * (box_encodings[0, box_idx] - 14)
        decode, xmin, ymin, xmax, ymax = decode_box_encoding(
            box, anchors[box_idx])

        detected_bbox = [xmin*image.width, ymin*image.width,
                         xmax*image.height, ymax*image.height]

        # image_draw = ImageDraw.Draw(image)
        # image_draw.rectangle(xy=ground_true_bbox, width=5, outline=(255, 0, 0))
        # image_draw.rectangle(xy=detected_bbox, width=5, outline=(0, 255, 0))

        # image.show()

        iou = compute_iou(ground_true_bbox, detected_bbox)
        print('iou: ', iou)

        if (iou > threshold):
            num_great_than_threshold += 1

    print('Total images: ', len(image_paths))
    print('Number greater than threshold: ', num_great_than_threshold)
