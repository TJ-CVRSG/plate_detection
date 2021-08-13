import argparse
from pathlib import Path
from PIL import Image
import tensorflow as tf
import io
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--data",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data dir. Should contain all the images")

parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The output dir for resized images")

args = parser.parse_args()

IMAGE_DIR = Path(args.data)
INPUT_SIZE_H = 320
INPUT_SIZE_W = 320

OUTPUT_DIR = Path(args.output_dir)
if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def spilt_train_val(image_names, val_ratio=0.15):
    np.random.shuffle(image_names)
    train_keys, validation_keys = (
        image_names[int(len(image_names) * val_ratio):],
        image_names[: int(len(image_names) * val_ratio)],
    )
    return train_keys, validation_keys


def create_tf_example(info):
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(info["height"]),
        'image/width': int64_feature(info["width"]),
        'image/filename': bytes_feature(info["filename"]),
        'image/source_id': bytes_feature(info["source_id"]),
        'image/encoded': bytes_feature(info["encoded"]),
        'image/format': bytes_feature(info["format"]),
        'image/object/bbox/xmin': float_list_feature(info["xmin"]),
        'image/object/bbox/xmax': float_list_feature(info["xmax"]),
        'image/object/bbox/ymin': float_list_feature(info["ymin"]),
        'image/object/bbox/ymax': float_list_feature(info["ymax"]),
        'image/object/class/text': bytes_list_feature(info["text"]),
        'image/object/class/label': int64_list_feature(info["label"]),
    }))
    return tf_example

# 获取 bounding-box 信息
# 输入：image path
# 返回：
#   bounding box


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


def write_tf_record(writer, img_files):

    for img_file in img_files:
        image = Image.open(img_file).crop(
            (0, 0, 720, 720)).resize((INPUT_SIZE_W, INPUT_SIZE_H))
        # 获取boundingbox的左上角、右下角
        xratio = INPUT_SIZE_W / 720.0
        yratio = INPUT_SIZE_H / 720.0

        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            encoded_jpg = output.getvalue()

        bounding_box = get_bounding_box(img_file.name, xratio, yratio)

        if (bounding_box[0] > INPUT_SIZE_W or bounding_box[1] > INPUT_SIZE_H or bounding_box[2] > INPUT_SIZE_W or bounding_box[3] > INPUT_SIZE_H):
            continue

        # 图片信息
        info = {
            'height': image.height,
            'width': image.width,
            'filename': img_file.name.encode('utf8'),
            'source_id': img_file.name.encode('utf8'),
            'encoded': encoded_jpg,
            'format': b'jpg',
            'xmin': [bounding_box[0]],
            'xmax': [bounding_box[2]],
            'ymin': [bounding_box[1]],
            'ymax': [bounding_box[3]],
            'text': [b'plate'],
            'label': [1],
        }

        example = create_tf_example(info)
        writer.write(example.SerializeToString())


def main():

    # 加载图片信息
    img_files = [f for f in IMAGE_DIR.iterdir()]
    img_files.sort(key=lambda f: f.stem, reverse=True)  # 排序，防止顺序错乱、数据和标签不对应
    train_files, val_files = spilt_train_val(img_files)

    with tf.io.TFRecordWriter(str(OUTPUT_DIR.joinpath("train.tfrecord"))) as writer:
        write_tf_record(writer, train_files)

    with tf.io.TFRecordWriter(str(OUTPUT_DIR.joinpath("val.tfrecord"))) as writer:
        write_tf_record(writer, val_files)


if __name__ == "__main__":
    main()
