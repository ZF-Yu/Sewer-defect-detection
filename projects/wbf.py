# -*- coding: utf-8 -*-

import json
from ensemble_boxes import weighted_boxes_fusion

def get_all_img_in_json(json_path):

    with open(json_path, 'r') as load_f:
        json_data = json.load(load_f)

    all_images = []
    for i, box in enumerate(json_data):
        name = box['name']
        if name not in all_images:
            all_images.append(name)
    return all_images

def ensemble_wbf(save_path, th_nmsiou, th_score, conf_type, weights=None, imageid2size=None):
    save_folders = ['wbf\\cbswinl_faster.bbox.json',
                     'wbf\\cbswinl_cascade.bbox.json']

    print(save_folders)
    model_nums = len(save_folders)
    final_result = []
    id2annos = [{} for _ in range(model_nums)]
    model_weights = [1 for _ in range(model_nums)]
    for i, json_file in enumerate(save_folders):
        if weights is not None:
            model_weights[i] = weights[os.path.basename(json_file)]
        with open(json_file, 'r') as f:
            result = json.load(f)
        for box in result:
            img_id = box['image_id']

            if img_id not in id2annos[i]:
                id2annos[i][img_id] = []
            id2annos[i][img_id].append(box)
    iou_thr = th_nmsiou
    skip_box_thr = th_score
    for id, _ in id2annos[0].items():
        scores_list = [[] for _ in range(model_nums)]
        boxes_list = [[] for _ in range(model_nums)]
        labels_list = [[] for _ in range(model_nums)]
        img_id = id
        img_size = imageid2size[img_id]
        img_w, img_h = img_size

        for j in range(model_nums):
            if id in id2annos[j]:
                for anno in id2annos[j][id]:
                    box = anno['bbox']
                    xmin = box[0]
                    ymin = box[1]
                    width = box[2]
                    height = box[3]
                    xmax = xmin + width
                    ymax = ymin + height

                    xmax = xmax / img_w
                    xmin = xmin / img_w
                    ymin = ymin / img_h
                    ymax = ymax / img_h
                    confidence = anno["score"]
                    label_class = anno["category_id"]
                    scores_list[j].append(confidence)
                    boxes_list[j].append([xmin, ymin, xmax, ymax])
                    labels_list[j].append(label_class)

        boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                      iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i]
            score = float(scores[i])
            label = int(labels[i])
            x1 = round(float(x1*img_w), 4)
            y1 = round(float(y1*img_h), 4)
            x2 = round(float(x2*img_w), 4)
            y2 = round(float(y2*img_h), 4)
            final_result.append({'image_id': id, "bbox": [x1, y1, x2-x1, y2-y1], "score": score, "category_id": label})
    with open(save_path, 'w') as fp:
        json.dump(final_result, fp)

    return final_result


if __name__ == '__main__':
    save_path = 'wbf_result.json'

    raw_testa_json = r'annotations\instances_val2017.json'
    with open(raw_testa_json, 'r') as f:
        infos = json.load(f)
    imageid2size = {}
    for info in infos['images']:
        imageid2size[info['id']] = [info['width'], info['height']]

    weights = None

    final_result = ensemble_wbf(save_path, th_nmsiou=0.7, th_score=0.00001, conf_type='max', weights=weights, imageid2size=imageid2size)




