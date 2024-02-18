import time
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from yolov8.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer)
from yolov8.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from counter.draw_counter import draw_up_down_counter
import argparse
import platform
import shutil
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import cv2
import os
from PIL import Image
from pylab import *
from matplotlib.pyplot import ginput, ion, ioff

sys.path.insert(0, './yolov8')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def bbox_rel(image_width, image_height, *xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def Estimated_speed(locations, fps, width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index = []
    work_IDs_prev_index = []
    work_locations = []
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        speed.append(
            math.sqrt((work_locations[i][0] - work_prev_locations[i][0]) ** 2 +
                      (work_locations[i][1] - work_prev_locations[i][1]) ** 2) *
            width[work_locations[i][3]] / (work_locations[i][4]) * fps / 5 * 3.6 * 2)
    for i in range(len(speed)):
        speed[i] = [round(speed[i], 1), work_locations[i][2]]
    return speed

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    capture = cv2.VideoCapture(source)
    frame_fature = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    width = [0, 0.2, 1.85, 0.5, 0, 2.3, 0, 2.5]
    time_person = 3
    locations = []
    speed = []
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


    def cv_show(p, im0):
        height, width = im0.shape[:2]
        a = 1200 / width
        size = (1200, int(height * a))

        img_resize = cv2.resize(im0, size, interpolation=cv2.INTER_AREA)
        cv2.imshow(p, img_resize)
        cv2.waitKey(1)  # 1 millisecond

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # ---------------------------测速------------------------------------------#
                bbox_xywh = []
                confs = []
                classes = []
                img_h, img_w, _ = im0.shape

                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append([cls.item()])
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)

                outputs = deepsort.update(xywhs, confss, im0, classes)
                box_centers = []
                for i, each_box in enumerate(outputs):
                    # 求得每个框的中心点
                    if each_box[5] == 1 or each_box[5] == 2 or each_box[5] == 3 or each_box[5] == 5 or each_box[5] == 7:
                        box_centers.append([(each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2, each_box[4],
                                        each_box[5], each_box[2] - each_box[0]])
                location = box_centers

                if len(location) != 0:
                    with open('track.txt', 'a+') as track_record:
                        track_record.write('frame:%s\n' % str(frame_idx))
                        for j in range(len(location)):
                            track_record.write('id:%s,x:%s,y:%s\n' % (str(location[j][2]), str(location[j][0]), str(location[j][1])))
                    print('done!')
                locations.append(location)
                print(len(locations))

                if len(locations) == 5:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width)
                    with open('speed.txt', 'a+') as speed_record:
                        for sp in speed:
                            speed_record.write('id:%s %skm/h\n' % (str(sp[1]), str(sp[0])))
                    locations = []

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    conf2 = float(f'{conf:.2f}')
                    if conf2 > 0.40:

                        if names[int(cls)] == 'bicycle' or names[int(cls)] == 'car' or names[int(cls)] == 'motorcycle' or names[int(cls)] == 'bus' or names[int(cls)] == 'truck':
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, speed, outputs, time_person, label=label, color=[0, 0, 255], line_thickness=3,
                                             name=names[int(cls)])



            #  Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')


            if view_img:
                cv_show(str(p), im0)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov8s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='TEST1.mp4', help='source')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.01, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results',default=True)
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
