import argparse
import datetime
import json
from pathlib import Path
from PIL import Image

from multiprocessing import pool
from pycococreatortools import pycococreatortools

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

INFO = {
    "description": "CCPD Dataset in COCO Format",
    "url": "",
    "version": "0.1.0",
    "year": 2021,
    "contributor": "CVRSG",
    # 显示此刻时间，格式：'2019-04-30 02:17:49.040415'
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "ALL RIGHTS RESERVED",
        "url": ""
    }
]

# 初始化类别（背景）
CATEGORIES = [
    {
        'id': 1,
        'name': 'license plate',
        'supercategory': 'shape',
    },
    {
        'id': 2,
        'name': 'background',
        'supercategory': 'shape',
    }
]

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


def main():
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    # 初始化id（以后依次加一）
    image_id = 1
    annotation_id = 1

    # 加载图片信息
    img_files = [f for f in IMAGE_DIR.iterdir()]
    img_files.sort(key=lambda f: f.stem, reverse=True)  # 排序，防止顺序错乱、数据和标签不对应

    myPool = pool.Pool(processes=4)  # 并行化处理

    for img_file in img_files:
        # 写入图片信息（id、图片名、图片大小）,其中id从1开始
        image = Image.open(img_file).crop((0, 0, 720, 720)).resize((INPUT_SIZE_W, INPUT_SIZE_H))
        # 获取boundingbox的左上角、右下角
        xratio = INPUT_SIZE_W / 720.0
        yratio = INPUT_SIZE_H / 720.0

        image.save(OUTPUT_DIR.joinpath(img_file.name), bitmap_format="jpg")

        bounding_box = get_bounding_box(img_file.name, xratio, yratio)

        if (bounding_box[0] > INPUT_SIZE_W or bounding_box[1] > INPUT_SIZE_H or bounding_box[2] > INPUT_SIZE_W or bounding_box[3] > INPUT_SIZE_H):      
            continue

        # 图片信息
        img_info = pycococreatortools.create_image_info(image_id, img_file.name, image.size)  
        # 存储图片信息（id、图片名、大小）
        coco_output['images'].append(img_info)

        class_id = 1  # id 为数字形式，如 1,此时是list形式，后续需要转换 # 指定为1，因为只有”是车牌“这一类

        # 显示日志
        print(bounding_box)

        annotation_info = pycococreatortools.mask_create_annotation_info(1, image_id, 0, class_id, image.size, bounding_box, None)
        coco_output['annotations'].append(annotation_info)

        image_id += 1

    myPool.close()
    myPool.join()

    # 保存成json格式
    print("[INFO] Storing annotations json file...")
    output_json = Path(f'ccpd_annotations.json')
    with output_json.open('w', encoding='utf-8') as f:
        json.dump(coco_output, f)
    print("[INFO] Annotations JSON file saved in：", str(output_json))


if __name__ == "__main__":
    main()
