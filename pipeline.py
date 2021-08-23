import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from pathlib import Path

CHAR_DICT = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '<Anhui>': 10, '<Beijing>': 11, '<Chongqing>': 12, '<Fujian>': 13, '<Gansu>': 14, '<Guangdong>': 15, '<Guangxi>': 16, '<Guizhou>': 17, '<Hainan>': 18, '<Hebei>': 19, '<Heilongjiang>': 20, '<Henan>': 21, '<Hubei>': 22, '<Hunan>': 23, '<InnerMongolia>': 24, '<Jiangsu>': 25, '<Jiangxi>': 26, '<Jilin>': 27, '<Liaoning>': 28,
             '<Ningxia>': 29, '<Qinghai>': 30, '<Shaanxi>': 31, '<Shandong>': 32, '<Shanghai>': 33, '<Shanxi>': 34, '<Sichuan>': 35, '<Tianjin>': 36, '<Tibet>': 37, '<Xinjiang>': 38, '<Yunnan>': 39, '<Zhejiang>': 40, 'A': 41, 'B': 42, 'C': 43, 'D': 44, 'E': 45, 'F': 46, 'G': 47, 'H': 48, 'J': 49, 'K': 50, 'L': 51, 'M': 52, 'N': 53, 'P': 54, 'Q': 55, 'R': 56, 'S': 57, 'T': 58, 'U': 59, 'V': 60, 'W': 61, 'X': 62, 'Y': 63, 'Z': 64, '_': 65}


class CVRSGModel(object):
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


class DetectionModel(CVRSGModel):
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


class RecognitionModel(CVRSGModel):
    def process(self, img_resized):
        input_array = np.array(img_resized, dtype=np.float32)
        input_array -= 128.0
        input_array = np.array(input_array, dtype=np.int8)
        input_array = np.reshape(input_array, self.input_details[0]['shape'])
        self.interpreter.set_tensor(
            self.input_details[0]['index'], input_array)
        self.interpreter.invoke()

        out_tensor = self.interpreter.get_tensor(
            self.output_details[0]['index'])
        # print(out_tensor)
        return out_tensor


def decode_box_encoding(box_encoding, anchor):

    y_scale = 10.0
    x_scale = 10.0
    h_scale = 5.0
    w_scale = 5.0

    ycenter = box_encoding[0] / y_scale * anchor[2] + anchor[0]
    xcenter = box_encoding[1] / x_scale * anchor[3] + anchor[1]
    half_h = 0.5 * math.exp((box_encoding[2] / h_scale)) * anchor[2]
    half_w = 0.5 * math.exp((box_encoding[3] / w_scale)) * anchor[3]

    ymin = (ycenter - half_h)   # ymin
    xmin = (xcenter - half_w)   # xmin
    ymax = (ycenter + half_h)   # ymax
    xmax = (xcenter + half_w)   # xmax

    return xmin, ymin, xmax, ymax


def infer_rcgn(rcgn_model, plt_img_resized):
    out_tensor = rcgn_model.process(plt_img_resized)  # 1*1*16*66
    # Get the result
    out_char_codes = [np.argmax(out_tensor[0][0][i])
                      for i in range(out_tensor.shape[2])]
    # print(out_char_codes)
    out_str = ''
    prev_char = None
    no_character_code = len(CHAR_DICT) - 1
    for i, char_code in enumerate(out_char_codes):
        if char_code == no_character_code or char_code == prev_char:
            prev_char = char_code
            continue
        prev_char = char_code
        for k, v in CHAR_DICT.items():
            if char_code == v:
                out_str += k
                continue
    return out_str


if __name__ == '__main__':
    detection_model_path = 'tflite_model/detection/ssdlite_mbv2_160_0.125/detection.tflite'
    anchors_path = 'tflite_model/detection/ssdlite_mbv2_160_0.125/anchors.txt'

    rcgn_model_path = 'tflite_model/recognition/rcgn_int8.tflite'

    image_dir = './images/'
    # image_dir = '/home/tang/plate_detection/DataFromWeb/'

    detection_model = DetectionModel(detection_model_path)
    anchors = np.loadtxt(anchors_path)

    rcgn_model = RecognitionModel(rcgn_model_path)

    image_paths = [f for f in Path(image_dir).iterdir()]
    image_paths.sort(key=lambda f: f.stem, reverse=True)

    for image_path in image_paths:
        original_image = Image.open(image_path)
        resized_image = original_image.resize(
            (detection_model.input_w(), detection_model.input_h()))

        class_preds, box_encodings = detection_model.process(resized_image)
        box_idx = np.argmax(class_preds[0][:, 1])

        box = 0.047391898930072784 * (box_encodings[0, box_idx] + 4)
        # box = 0.03998513147234917 * (box_encodings[0, box_idx] - 14)
        xmin, ymin, xmax, ymax = decode_box_encoding(box, anchors[box_idx])

        bbox = (int(xmin*original_image.width), int(ymin*original_image.height),
                int(xmax*original_image.width), int(ymax*original_image.height))

        region = original_image.crop(bbox)
        region = region.resize((rcgn_model.input_w(), rcgn_model.input_h()))

        rcgn_str = infer_rcgn(rcgn_model, region)

        image_draw = ImageDraw.Draw(original_image)
        image_draw.rectangle(xy=bbox,
                             width=2,
                             outline=(203, 67, 53))

        ft = ImageFont.truetype("FiraCode-Regular.ttf", 20)
        image_draw.text(xy=(0, 0), text=rcgn_str, font=ft, fill=(255, 0, 0))

        original_image.show()
