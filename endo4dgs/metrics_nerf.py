#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
# from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
# import lpips
from utils.image_utils import lpips_score, ssim
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import numpy as np
import faulthandler
faulthandler.enable()
import statistics as stat

root_nerf = {"nerf": "/suedata1/Free/xkangyu/mycodes/endonerf/logs/endonerf",
             "surf": "/suedata1/Free/xkangyu/data/endosurf/logs/endosurf"}

TYPE = "nerf"
# TYPE = "surf"

def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

def readImages(renders_dir, gt_dir, masks_dir):
    renders = []
    gts = []
    image_names = []
    masks = []
    file_name = args.model_paths[0].split("/")[-1]
    img_path = os.path.join(root_nerf[TYPE], "base-endonerf-"+file_name, "demo/iter_00100000/test_2d")
    
    for fname in os.listdir(gt_dir):
        render = np.array(Image.open(os.path.join(img_path, fname.split(".")[0][-3:]+"_rgb_vr.png")))
        gt = np.array(Image.open(gt_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        image_names.append(fname)
    
    return renders, gts, masks, image_names


def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    with torch.no_grad():
        for scene_dir in model_paths:
            # try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            
            output_dir = os.path.join(root_nerf[TYPE], "base-endonerf-"+scene_dir.split("/")[-1])

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                masks_dir = method_dir / "masks"
                
                renders, gts, masks, image_names = readImages(renders_dir, gt_dir, masks_dir)

                ssims = []
                psnrs = []
                lpipss = []
                                
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render, gt, mask = renders[idx], gts[idx], masks[idx]
                    
                    render = render * mask
                    gt = gt * mask
                    psnrs.append(psnr(render, gt).cpu().numpy()[0].tolist())
                    ssims.append(ssim(render, gt).cpu().numpy().tolist())
                    lpipss.append(lpips_score(render, gt).cpu().flatten().numpy()[0].tolist())
                
                print(torch.tensor(ssims).shape)

                print("Scene: ", scene_dir,  "SSIM : {:>12.7f}, std: {}".format(torch.tensor(ssims).mean(), stat.stdev(ssims)))
                print("Scene: ", scene_dir,  "PSNR : {:>12.7f}, std: {}".format(torch.tensor(psnrs).mean(), stat.stdev(psnrs)))
                print("Scene: ", scene_dir,  "LPIPS: {:>12.7f}, std: {}".format(torch.tensor(lpipss).mean(), stat.stdev(lpipss)))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item(),
                                                        "SSIM_std": stat.stdev(ssims),
                                                        "PSNR_std": stat.stdev(psnrs),
                                                        "lpips_std": stat.stdev(lpipss)})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(output_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(output_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        # except:
        #     print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)
