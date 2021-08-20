import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    default=None,
                    type=str,
                    required=True,
                    help='The dataset dir for representative dataset')

parser.add_argument('--model',
                    default=None,
                    type=str,
                    required=True,
                    help='The path for tensorflow pb model')

parser.add_argument('--output',
                    default=None,
                    type=str,
                    required=True,
                    help='The output path for tflite model')

parser.add_argument('--input_h',
                    default=None,
                    type=int,
                    required=True,
                    help='The height for the input model')

parser.add_argument('--input_w',
                    default=None,
                    type=int,
                    required=True,
                    help='The width for the input model')

args = parser.parse_args()

IMAGE_DIR = args.dataset
MODEL_PATH = args.model
OUTPUT_PATH = args.output
NORM_H = args.input_h
NORM_W = args.input_w


image_paths = [f for f in Path(IMAGE_DIR).iterdir()]
image_paths.sort(key=lambda f: f.stem, reverse=True)


def representative_dataset_gen():
    dataset = []
    for image_path in image_paths:
        image = Image.open(image_path).crop(
            (0, 0, 720, 720)).resize((NORM_H, NORM_W))
        image = np.array(image, dtype=np.float32)
        image -= 127.0
        image /= 128.0
        dataset.append(image)

    num = len(dataset)
    dataset = np.array(dataset)

    images = tf.data.Dataset.from_tensor_slices(dataset).batch(1)

    for img in images.take(num):
        yield [img]


input_arrays = ['normalized_input_image_tensor']
output_arrays = ['raw_outputs/class_predictions',
                 'raw_outputs/box_encodings']
input_shapes = {'normalized_input_image_tensor': [1, NORM_H, NORM_W, 3]}

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(MODEL_PATH, input_arrays,
                                                                output_arrays, input_shapes)
converter.allow_custom_ops = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

converter.representative_dataset = representative_dataset_gen

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_quant = converter.convert()
with open(OUTPUT_PATH, 'wb') as tflite_file:
    tflite_file.write(tflite_model_quant)
