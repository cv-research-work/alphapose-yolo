import os
import sys
import string
from xml.sax.saxutils import escape
import argparse
import  cv2

## 
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torchvision.datasets import ImageFolder

from torchvision import transforms

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.pPose_nms import write_json
from alphapose.utils.transforms import flip, flip_heatmap,get_func_heatmap_to_coord
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter
import pickle

import numpy as np
import json
from alphapose.utils.metrics import DataLogger, calc_accuracy, evaluate_mAP
import json
import matplotlib.pyplot as plt


import matplotlib.patches as patches
import matplotlib.patheffects as PathEffects

from PIL import Image

num_gpu = torch.cuda.device_count()


## Image folder
"""----------------------------- Demo options -----------------------------"""

parser = argparse.ArgumentParser(description="AlphaPose")

parser.add_argument("--image_folder", type=str, default="1001", help="path to dataset")
parser.add_argument("--csv_file", type=str, default="test.csv", help="path to dataset")
parser.add_argument('--cfg', type=str, required=True,help='experiment configure file name')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--checkpoint', type=str, required=True,help='checkpoint file name')
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

opt = parser.parse_args()

cfg = update_config(opt.cfg)

foldername = opt.image_folder
csv_file = opt.csv_file

## read bboxes
bboxes_dict = []

with open(csv_file,"r") as f:
    for cnt,line in enumerate(f):
        line = line.split(",")
        classname = line[1]

        if classname not in ["elephant"]:
            continue       
        
        x1,y1,x2,y2 = float(line[3]),float(line[4]),float(line[5]),float(line[6])
        bboxes_dict.append([x1,y1,x2,y2])

opt.device = torch.device("cuda:" + str(opt.gpus[0]) if int(opt.gpus[0]) >= 0 else "cpu")
heatmap_to_coord = get_func_heatmap_to_coord(cfg)

# Load pose model
pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
print(f'Loading pose model from {opt.checkpoint}...')
pose_model.load_state_dict(torch.load(opt.checkpoint, map_location=opt.device))
pose_model.to(opt.device)
pose_model.eval()

batchSize = 1
print("Pose batchsize",batchSize)


def visualize(imgdir, visdir, tracked, cmap):
    print("Start visualization...\n")
    index = 0
    for img_dict in tqdm(tracked):
        imgpath = img_dict["image_id"]
        img = Image.open(imgpath)

        bbox = img_dict["bbox"]
        index = index + 1
        width, height = img.size
        fig = plt.figure(figsize=(width/10,height/10),dpi=10)

        plt.imshow(img)

        pose = np.array(img_dict['keypoints']).reshape(-1,3)[:,:3]
        
        if np.mean(pose[:,2]) <1 :
            alpha_ratio = 1.0
        else:
            alpha_ratio = 5.0

        mpii_part_names = ["Top of the head",
            'Highest point on the back',
            "Left eye","Right eye",
            "Jaw","Eyebrow of the face",
            "Base of trunk","End of trunk","Left elbow",
            "Right elbow","Bottom of the belly","Left knee","Right knee",
            "Top of left ear","Top of right ear","Widest point of left ear",
            "Widest point of right ear","Bottom of left ear","Bottom of right ear",
            "Top of the left shoulder","Top of the right shoulder",
            "Top of the shoulder","Base of tail",
            "End of tail", "Bottom of the right backfoot",
            "Bottom of the right front foot",
            "Bottom of the left front foot",
            "Bottom of the left backfoot",
            "Base of right tusk","End of right tusk",
            "Base of left tusk","End of left tusk"
            ]
        colors = ['m','r', 'b', 'b', 'r', 'r', 'b', 'b', 'r', 'r', 'm', 'b', 'b', 'r', 'r','r','r','r','r','b','b','b',
        'r','r','m','m','m','m','b','b','b','b']
        pairs = [[6,7],[8,9],[12,14],[13,15],[28,29],[30,31],[22,23]]

        texts_list = []
        for idx_c, color in enumerate(colors):
            plt.plot(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height), marker='o', 
                    color=color, ms=80/alpha_ratio*np.mean(pose[idx_c,2]), markerfacecolor=(1, 1, 0, 0.7/alpha_ratio*pose[idx_c,2]))
            name = mpii_part_names[idx_c]
            txt = plt.text(np.clip(pose[idx_c,0],0,width), np.clip(pose[idx_c,1],0,height),idx_c,color="black")
            txt.set_path_effects([PathEffects.withStroke(linewidth=5, foreground='w')])
            texts_list.append((name,np.mean(pose[idx_c,2])))
      
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        


        plt.axis('off')
        
        ax = plt.gca()
        # Display the image
        ax.imshow(img)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
        
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
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    data_transform = transforms.Compose([
        transforms.Resize((opt.img_size,opt.img_size),interpolation=Image.NEAREST),
        transforms.ToTensor()
       
    ])

    with torch.no_grad():

        ## build dataset
        eval_joints =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
    14, 15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

        test_dataset = ImageFolder(opt.image_folder,transform=data_transform)
        test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu)

        
        kpt_json = []
        imgs = [(i[0]) for i in (test_dataset.imgs)]
        
        for batch_i, (img_paths, inps) in enumerate(test_loader):
            with torch.no_grad():                
                start_time = getTime()
                img_paths = img_paths.cuda()
                output = pose_model(img_paths)
                pred = output.cpu().data.numpy()

                assert pred.ndim == 4

                pred = pred[:,eval_joints,:,:]

                for i in range(output.shape[0]):
                    bbox = bboxes_dict[i]
                    pose_coords, pose_scores = heatmap_to_coord(pred[i][eval_joints], bbox)

                    ## concatenate - pose coordinates and pose scores
                    keypoints = np.concatenate((pose_coords, pose_scores), axis=1)

                    ## keypoints
                    keypoints = keypoints.reshape(-1).tolist()

                    data = dict()
                    data['bbox'] = bbox
                    data['image_id'] = imgs[batch_i]
                    data['score'] = float(np.mean(pose_scores) + np.max(pose_scores))
                    data['category_id'] = 22
                    data['keypoints'] = keypoints

                    kpt_json.append(data)

    with open('test_gt_kpt.json', 'w') as fid:
        json.dump(kpt_json, fid)
	
    ## Visualization
    #images, annotations = load_annotations("test_annot_keypoint.pkl")
    ## visualize
    cmap = plt.cm.get_cmap("hsv", 1)
    visualize(opt.image_folder,".",kpt_json,cmap)
 
    
main()