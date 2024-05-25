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

import os
from argparse import ArgumentParser

# tanks_and_temples_scenes = [
#         'Barn', 'Caterpillar', 'Courthouse', 'Ignatius',
#         'Meetingroom', 'Truck'
# ]
tanks_and_temples_scenes = [
    'Truck'
]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="output/tnt")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(tanks_and_temples_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument("--tnt_image", required=True, type=str)
    parser.add_argument('--tnt_point', required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1"
    for scene in tanks_and_temples_scenes:
        source = args.tnt_image + "/" + scene
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in tanks_and_temples_scenes:
        all_sources.append(args.tnt_image + "/" + scene)

    common_args = " --quiet --eval --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
        print("python metrics.py -m " + scenes_string)
        os.system("python metrics.py -m " + scenes_string)
        cmd_eval = f"python eval_tnt/run.py --dataset-dir {args.tnt_image}/{scene} --traj-path {args.tnt_point}/{scene}/{scene}_COLMAP_SfM.log --ply-path {args.output_path}/{scene}/train/ours_30000/fuse_post.ply"
        print(cmd_eval)
        os.system(cmd_eval)