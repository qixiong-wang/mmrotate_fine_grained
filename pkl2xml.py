import os
import re
import pdb
import time
import tqdm
import torch
import pickle
import math
import shutil
import tempfile
import numpy as np
from PIL import Image
from functools import partial
from mmcv.ops import nms_rotated
from collections import defaultdict
from xml.dom.minidom import Document
from mmengine.utils import track_iter_progress, track_parallel_progress

ori_cls = ('Passenger Ship', 'Motorboat', 'Fishing Boat',
               'Tugboat', 'other-ship', 'Engineering Ship', 'Liquid Cargo Ship',
               'Dry Cargo Ship', 'Warship', 'Small Car', 'Bus', 'Cargo Truck',
               'Dump Truck', 'other-vehicle', 'Van', 'Trailer', 'Tractor',
               'Excavator', 'Truck Tractor', 'Boeing737', 'Boeing747',
               'Boeing777', 'Boeing787', 'ARJ21', 'C919', 'A220', 'A321',
               'A330', 'A350', 'other-airplane', 'Baseball Field', 'Basketball Court',
               'Football Field', 'Tennis Court', 'Roundabout', 'Intersection', 'Bridge')
    
CLASSES = ('Passenger', 'Motorboat', 'Fishing',#2
               'Tugboat', 'other-ship', 'Engineering', 'Liquid',#6
               'Dry', 'Warship', 'Small', 'Bus',#10
               'Cargo', 'Dump', 'other-vehicle', 'Van',#14
               'Trailer', 'Tractor', 'Excavator', 'Truck',#18
               'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',#22
               'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',#28
               'other-airplane', 'Baseball', 'Basketball',#31
               'Football', 'Tennis', 'Roundabout', 'Intersection', 'Bridge')
    

def create_xml(img_id, in_dicts, out_path):
    doc = Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)
    source_list = {'filename':img_id+'.tif', 'origin': 'GF2/GF3'}
    node_source = doc.createElement('source')
    for source in source_list:
        node_name = doc.createElement(source)
        node_name.appendChild(doc.createTextNode(source_list[source]))
        node_source.appendChild(node_name)
    root.appendChild(node_source)

    research_list = {'version': '1.0', 'provider': 'FAIR1M', 'author': 'Cyber',
                    'pluginname': 'FAIR1M', 'pluginclass': 'object detection', 'time': '2021-07-21'}
    node_research = doc.createElement('research')
    for research in research_list:
        node_name = doc.createElement(research)
        node_name.appendChild(doc.createTextNode(research_list[research]))
        node_research.appendChild(node_name)
    root.appendChild(node_research)

    img = Image.open(os.path.join('/home/ningwy/FAIR1M1.0/test/images/images', img_id+'.tif'))
    size_list = {'width': str(img.size[0]), 'height': str(img.size[1]), 'depth': '3'}
    node_size = doc.createElement('size')
    for size in size_list:
        node_name = doc.createElement(size)
        node_name.appendChild(doc.createTextNode(size_list[size]))
        node_size.appendChild(node_name)
    root.appendChild(node_size)

    node_objects = doc.createElement('objects')
    for cls_name in in_dicts.keys():

        for i in range(len(in_dicts[cls_name])):

            node_object = doc.createElement('object')
            object_fore_list = {'coordinate': 'pixel', 'type': 'rectangle', 'description': 'None'}
            for object_fore in object_fore_list:
                node_name = doc.createElement(object_fore)
                node_name.appendChild(doc.createTextNode(object_fore_list[object_fore]))
                node_object.appendChild(node_name)

            node_possible_result = doc.createElement('possibleresult')
            node_name = doc.createElement('name')
            node_name.appendChild(doc.createTextNode(cls_name))
            node_possible_result.appendChild(node_name)

            node_probability = doc.createElement('probability')
            node_probability.appendChild(doc.createTextNode(str(in_dicts[cls_name][i][8])))
            node_possible_result.appendChild(node_probability)

            node_object.appendChild(node_possible_result)

            node_points = doc.createElement('points')

            for j in range(4):
                node_point = doc.createElement('point')

                text = '{},{}'.format(in_dicts[cls_name][i][int(0+2*j)], in_dicts[cls_name][i][int(1+2*j)])
                node_point.appendChild(doc.createTextNode(text))
                node_points.appendChild(node_point)
                
            node_point = doc.createElement('point')
            text = '{},{}'.format(in_dicts[cls_name][i][0], in_dicts[cls_name][i][1])
            node_point.appendChild(doc.createTextNode(text))
            node_points.appendChild(node_point)
            node_object.appendChild(node_points)
            node_objects.appendChild(node_object)

    root.appendChild(node_objects)

    # 开始写xml文档
    filename = os.path.join(out_path, img_id + '.xml')
    fp = open(filename, 'w', encoding='utf-8')
    doc.writexml(fp, indent='', addindent='\t', newl='\n', encoding="utf-8")
    fp.close()

def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))

def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.

    Args:
        coordinate (ndarray): shape(n, 9).

    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates

def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys

def _results2submission(id_list, dets_list, out_folder=None):
    """Generate the submission of full images.

    Args:
        id_list (list): Id of images.
        dets_list (list): Detection results of per class.
        out_folder (str, optional): Folder of submission.
    """
    alias_dict = {}
    for i in range(len(CLASSES)):
        alias_dict.update({CLASSES[i]: ori_cls[i]})

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:  # add
        shutil.rmtree(out_folder)
        os.makedirs(out_folder)

    for img_id, dets_per_cls in tqdm.tqdm(zip(id_list, dets_list), total=len(id_list)):#zip(id_list, dets_list):
        result_dict={}
        for cls_name, dets in zip(CLASSES, dets_per_cls):
            if dets.size == 0:
                continue
            bboxes = obb2poly_np_le90(dets)
            result_dict.update({alias_dict[cls_name]:bboxes})

        create_xml(img_id, result_dict, out_folder)

    return None

def _merge_func(info, CLASSES, iou_thr):
    """Merging patch bboxes into full image.

    Args:
        CLASSES (list): Label category.
        iou_thr (float): Threshold of IoU.
    """
    img_id, label_dets = info
    label_dets = np.concatenate(label_dets, axis=0)

    labels, dets = label_dets[:, 0], label_dets[:, 1:]

    big_img_results = []
    for i in range(len(CLASSES)):
        if len(dets[labels == i]) == 0:
            big_img_results.append(dets[labels == i])
        else:
            try:
                cls_dets = torch.from_numpy(dets[labels == i]).cuda()
            except:  # noqa: E722
                cls_dets = torch.from_numpy(dets[labels == i])
            nms_dets, keep_inds = nms_rotated(cls_dets[:, :5], cls_dets[:, -1],
                                              iou_thr)
            big_img_results.append(nms_dets.cpu().numpy())
    return img_id, big_img_results

def merge_det(results, nproc=4):
    """Merging patch bboxes into full image.

    Args:
        results (list): Testing results of the dataset.
        nproc (int): number of process. Default: 4.
    """
    collector = defaultdict(list)
    for idx in range(len(results)):
        result = results[idx]
        img_id = result['img_id']
        splitname = img_id.split('__')
        oriname = splitname[0]
        pattern1 = re.compile(r'__\d+___\d+')
        x_y = re.findall(pattern1, img_id)
        x_y_2 = re.findall(r'\d+', x_y[0])
        x, y = int(x_y_2[0]), int(x_y_2[1])
        new_result = []
        bboxes, scores = result['pred_instances']['bboxes'].numpy(), result['pred_instances']['scores'].numpy().reshape(-1, 1)
        ori_bboxes = bboxes.copy()
        ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array([x, y], dtype=np.float32)
        labels = result['pred_instances']['labels'].numpy().reshape(-1, 1)
        new_result.append(np.concatenate([labels, ori_bboxes, scores], axis=1))

        new_result = np.concatenate(new_result, axis=0)
        collector[oriname].append(new_result)

    merge_func = partial(_merge_func, CLASSES=CLASSES, iou_thr=0.1)
    if nproc <= 1:
        print('Single processing')
        merged_results = track_iter_progress(
            (map(merge_func, collector.items()), len(collector)))
    else:
        print('Multiple processing')
        merged_results = track_parallel_progress(
            merge_func, list(collector.items()), nproc)

    return zip(*merged_results)

def format_results(results, submission_dir, nproc=4, **kwargs):
    """Format the results to submission text (standard format for DOTA
    evaluation).

    Args:
        results (list): Testing results of the dataset.
        submission_dir (str, optional): The folder that contains submission
            files. If not specified, a temp folder will be created.
            Default: None.
        nproc (int, optional): number of process.

    Returns:
        tuple:

            - result_files (dict): a dict containing the json filepaths
            - tmp_dir (str): the temporal directory created for saving \
                json files when submission_dir is not specified.
    """
    nproc = min(nproc, os.cpu_count())
    assert isinstance(results, list), 'results must be a list'
    # assert len(results) == len(self), (
    #     f'The length of results is not equal to '
    #     f'the dataset len: {len(results)} != {len(self)}')
    if submission_dir is None:
        submission_dir = tempfile.TemporaryDirectory()
    else:
        tmp_dir = None

    print('\nMerging patch bboxes into full image!!!')
    start_time = time.time()
    id_list, dets_list = merge_det(results, nproc)
    stop_time = time.time()
    print(f'Used time: {(stop_time - start_time):.1f} s')

    result_files = _results2submission(id_list, dets_list, submission_dir)

    return result_files, tmp_dir

def pkl_xml(pkl_path, out_path, thr):

    with open(pkl_path, "rb") as fp_data:
        pkl_file=pickle.load(fp_data)
        format_results(pkl_file, out_path)
        # scores=re['pred_instances']['scores'].tolist()
        # bboxes=[box for i,box in enumerate(re['pred_instances']['bboxes'].tolist()) if scores[i]>thr]
        # labels=[label for i,label in enumerate(re['pred_instances']['labels'].tolist()) if scores[i]>thr]
        # scores=[score for score in scores if score>thr]
            
        # img_id=re['img_id']
        # ori_shape=re ['ori_shape']
        # create_xml(img_id, ori_shape, bboxes, scores, labels, out_path)


if __name__ == '__main__':
    pkl_path='/home/ningwy/pycharmprojects/mmrotate_fine_grained/work_dirs/rotated_rtmdet_l-3x-dota_dataaug/out_epoch18.pkl'
    out_path='/home/ningwy/pycharmprojects/mmrotate_fine_grained/work_dirs/rotated_rtmdet_l-3x-dota_dataaug/test'
    pkl_xml(pkl_path, out_path, 0.1)