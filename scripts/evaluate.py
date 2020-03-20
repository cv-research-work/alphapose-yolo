"""Script for single-gpu/multi-gpu demo."""

import argparse
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

import json
import matplotlib.pyplot as plt

import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects

from PIL import Image

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap,get_func_heatmap_to_coord
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
import pickle

from alphapose.utils.metrics import DataLogger, calc_accuracy, evaluate_mAP

num_gpu = torch.cuda.device_count()

"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')

parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")

parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video', action='store_true', default=False)

args = parser.parse_args()
cfg = update_config(args.cfg)
args.save_img = True
args.debug = True
if platform.system() == 'Windows':
    args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)

heatmap_to_coord = get_func_heatmap_to_coord(cfg)

# Load pose model
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

print(f'Loading pose model from {args.checkpoint}...')
pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))

pose_model.to(args.device)
pose_model.eval()

batchSize = args.posebatch
print("Pose batchsize",batchSize)


def load_annotations(filename):
    with open(filename, 'rb') as fid:
        items, labels = pickle.load(fid)
    return (items,labels)

def visualize(imgdir, visdir, tracked, cmap,labels):
    print("Start visualization...\n")
    index = 0

    for img_dict in tqdm(tracked):
        imgpath = img_dict["image_id"]
        
        img = Image.open(imgpath)

        label_dict = labels[index]
        bbox = label_dict["bbox"]
        index = index + 1
        width, height = img.size
        fig = plt.figure(figsize=(width/10,height/10),dpi=10)
        plt.imshow(img)
        pose = np.array(img_dict['keypoints']).reshape(-1,3)[:,:3]
        print(pose)
        if np.mean(pose[:,2]) <1 :
            alpha_ratio = 1.0
        else:
            alpha_ratio = 5.0

        mpii_part_names = ["Top of the head",
            'Highest point on the back',
            "Left eye","Right eye",
            "Jaw",
            "Base of trunk","Left elbow",
            "Right elbow","Bottom of the belly","Left knee","Right knee",
            "Widest point of left ear",
            "Widest point of right ear",
            "Base of tail",
            "Bottom of the right backfoot",
            "Bottom of the right front foot",
            "Bottom of the left front foot",
            "Bottom of the left backfoot",
            "Base of right tusk",
            "Base of left tusk"
            ]
        colors = ['m','r', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'b', 
        'b', 'r', 'r','r','r','r','r','b']
        pairs = [[6,7],[8,9],[12,14],[13,15]]
        text_colors = ['blue', 'orange', 
        'green', 'red', 
        'purple', 'brown', 
        'pink', 'gray', 
        'olive', 'cyan',
        'white','gray','ivory','black',
        'khaki','chocolate','peru','palegreen','rosybrown','midnightblue'
        ]
        print("-----")
        for i in range(len(text_colors)):
            print(f"{mpii_part_names[i]}:{text_colors[i]}")
        print("-----")
        texts_list = []
        for idx_c, color in enumerate(colors):
            plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', 
                    color=text_colors[idx_c], ms=80/alpha_ratio*np.mean(pose[idx_c,2]), markerfacecolor=text_colors[idx_c])
            name = mpii_part_names[idx_c]
            texts_list.append((name,np.mean(pose[idx_c,2])))
      
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        
        plt.axis('off')
        
        ax = plt.gca()
        # Display the image
        ax.imshow(img)
        
        # Create a Rectangle patch
        box_w = x2 - x1
        box_h = y2 - y1
        bounding_box = patches.Rectangle((x1, y1), x2, y2, linewidth=2, edgecolor="r", facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bounding_box)

        ax.set_xlim([0,width])
        ax.set_ylim([height,0])
        
        
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        if not os.path.exists(visdir): 
            os.mkdir(visdir)
        
        image_name = os.path.basename(imgpath)
        image_name = image_name.split(".")[0]
        
        fig.savefig(os.path.join(visdir,image_name+".png"), pad_inches = 0.0, bbox_inches=extent, dpi=13)
        
        plt.close()
        
    return


def main():
    ### Get Dataset
    with torch.no_grad():
        ## build dataset
        test_dataset = builder.build_dataset(cfg.DATASET.TEST, 
        preset_cfg=cfg.DATA_PRESET, train=False)

        
        eval_joints = test_dataset.EVAL_JOINTS
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batchSize, shuffle=False, num_workers=20, drop_last=False)

        kpt_json = []
        
        
        for inps, labels, label_masks, img_ids, bboxes in tqdm(test_loader, dynamic_ncols=True):
            start_time = getTime()
            

            if isinstance(inps, list):
                inps = [inp.cuda for inp in inps]
            else:
                inps = inps.cuda()

            output = pose_model(inps)

            pred = output.cpu().data.numpy()
            assert pred.ndim == 4

            pred = pred[:,eval_joints,:,:]
            for i in range(output.shape[0]):
                bbox = bboxes[i].tolist()
                pose_coords, pose_scores = heatmap_to_coord(pred[i][test_dataset.EVAL_JOINTS], bbox)
                ## concatenate - pose coordinates and pose scores
                keypoints = np.concatenate((pose_coords, pose_scores), axis=1)

                ## keypoints
                keypoints = keypoints.reshape(-1).tolist()

                data = dict()
                data['bbox'] = bboxes[i].tolist()
                data['image_id'] = img_ids[i]
                data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
                data['category_id'] = 22
                data['keypoints'] = keypoints

                kpt_json.append(data)

    with open(os.path.join(args.outputpath, 'test_gt_kpt.json'), 'w') as fid:
        json.dump(kpt_json, fid)
	
    ## Visualization
    images, annotations = load_annotations("test_annot_keypoint.pkl")
    ## visualize
    cmap = plt.cm.get_cmap("hsv", 1)
    visualize(cfg.DATASET.TEST,args.outputpath,kpt_json,cmap,annotations)
 
    
main()