import tensorflow as tf
import numpy as np
from PIL import Image
from PIL import ImageDraw
import math
from pathlib import Path


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


if __name__ == '__main__':
    tflite_model_path = 'tflite_model/detection/ssdlite_mbv2_160_0.125/detection.tflite'
    anchors_path = 'tflite_model/detection/ssdlite_mbv2_160_0.125/anchors.txt'

    image_dir = './images/'
    image_dir = '/home/tang/plate_detection/DataFromWeb/'

    model = CVRSGModel(tflite_model_path)
    anchors = np.loadtxt(anchors_path)

    image_paths = [f for f in Path(image_dir).iterdir()]
    image_paths.sort(key=lambda f: f.stem, reverse=True)

    for image_path in image_paths:
        image = Image.open(image_path).resize(
            (model.input_w(), model.input_h()))

        class_preds, box_encodings = model.process(image)
        box_idx = np.argmax(class_preds[0][:, 1])

        box = 0.047391898930072784 * (box_encodings[0, box_idx] + 4)
        # box = 0.03998513147234917 * (box_encodings[0, box_idx] - 14)
        decode, xmin, ymin, xmax, ymax = decode_box_encoding(
            box, anchors[box_idx])

        xmin = int(xmin*image.width)
        ymin = int(ymin*image.width)
        xmax = int(xmax*image.height)
        ymax = int(ymax*image.height)

        image_draw = ImageDraw.Draw(image)
        image_draw.rectangle(xy=[xmin, ymin, xmax, ymax],
                             width=2, outline=(203, 67, 53))

        image.show()
