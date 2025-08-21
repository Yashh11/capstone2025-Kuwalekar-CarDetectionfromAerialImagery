import os
import glob
import tensorflow as tf
from object_detection.utils import dataset_util
import xml.etree.ElementTree as ET
import argparse

def xml_to_dict(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    data = {
        'filename': root.find('filename').text,
        'size': {
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text)
        },
        'objects': []
    }

    for member in root.findall('object'):
        obj = {
            'name': member.find('name').text,
            'xmin': int(member.find('bndbox/xmin').text),
            'ymin': int(member.find('bndbox/ymin').text),
            'xmax': int(member.find('bndbox/xmax').text),
            'ymax': int(member.find('bndbox/ymax').text)
        }
        data['objects'].append(obj)

    return data

def create_tf_example(data, image_dir, label_map_dict):
    image_path = os.path.join(image_dir, data['filename'])
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        encoded_image = fid.read()

    filename = data['filename'].encode('utf8')
    width = data['size']['width']
    height = data['size']['height']

    xmins = [obj['xmin'] / width for obj in data['objects']]
    xmaxs = [obj['xmax'] / width for obj in data['objects']]
    ymins = [obj['ymin'] / height for obj in data['objects']]
    ymaxs = [obj['ymax'] / height for obj in data['objects']]
    classes_text = [obj['name'].encode('utf8') for obj in data['objects']]
    classes = [label_map_dict[obj['name']] for obj in data['objects']]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def load_label_map(label_map_path):
    label_map_dict = {}
    with open(label_map_path, 'r') as f:
        for line in f:
            if "id:" in line:
                idx = int(line.strip().split(" ")[-1])
            if "name:" in line:
                name = line.strip().split(" ")[-1].strip('\'"')
                label_map_dict[name] = idx
    return label_map_dict

def main(args):
    writer = tf.io.TFRecordWriter(args.output_path)
    label_map_dict = load_label_map(args.label_map)

    for xml_file in glob.glob(os.path.join(args.data_dir, '*.xml')):
        data = xml_to_dict(xml_file)
        tf_example = create_tf_example(data, args.data_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print(f"TFRecord written to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Path to folder containing .xml and .jpg files')
    parser.add_argument('--label_map', help='Path to label_map.pbtxt file')
    parser.add_argument('--output_path', help='Path to output TFRecord file')
    args = parser.parse_args()
    main(args)
