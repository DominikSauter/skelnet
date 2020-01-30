#!/usr/bin/python

import os
import shutil
import click
from PIL import Image


@click.command()
@click.option('--preds', default="")
def main(preds):
    dataset_point_path = 'dataset/point/'
    splitted_paths = preds.split(" ")
    for split_path in splitted_paths:
        assert os.path.isdir(os.path.join(dataset_point_path, split_path))

    print("Starting pipeline step 1/3 (possible merging)..")
    # if multiple prediction sets shall be merged
    if len(splitted_paths) > 1:
        print("MERGING: " + str(splitted_paths.tostr))
        # create dir paths, etc.
        out_merged_path = "merged"
        for p in splitted_paths:
            out_merged_path += '_' + p.replace("_","")
        merged_full_path = os.path.join(dataset_point_path, out_merged_path)
        shutil.rmtree(merged_full_path, ignore_errors=True)
        if not os.path.isdir(merged_full_path):
            os.makedirs(merged_full_path)
        in_path_img2pt = merged_full_path

        # load and gather all images to be merged
        images_list = []
        for path in splitted_paths:
            pred_dir_path = os.path.join(dataset_point_path, path)
            pred_images = []
            for img_path in sorted(os.listdir(pred_dir_path)):
                pred_images.append((img_path, Image.open(os.path.join(pred_dir_path, img_path)), 'r'))
            images_list.append(pred_images)

        # merge images and store result to file
        for i in range(219):
            merged_img = Image.new('1', (256, 256))
            merged_img_immat = merged_img.load()
            for j in range(len(images_list)):
                img_immat = images_list[j][i][1].load()
                for k in range(256):
                    for l in range(256):
                        merged_img_immat[(k, l)] += img_immat[(k, l)]
            file_name = images_list[0][i][0]
            print(file_name)
            merged_img.save(os.path.join(merged_full_path, file_name))

    # if only one prediciton set shall be used without any merging
    elif len(splitted_paths) == 1 and not splitted_paths[0] == "":
        print("NO MERGING, USING: " + str(splitted_paths))
        path = preds
        in_path_img2pt = os.path.join(dataset_point_path, path)
    else:
        raise ValueError("No predictions given!")
    print("Pipeline step 1/3 (possible merging) done.")

    # get points clouds from skeleton images
    print("Starting pipeline step 2/3 (img2pt)..")
    print("INPUT PATH: " + in_path_img2pt)
    os.system("python3 img2pt.py --in_path=" + in_path_img2pt)
    print("Pipeline step 2/3 (img2pt) done.")

    # calculate MST from generated point clouds and create submission file
    print("Starting pipeline step 3/3 (skelimg2adjlist)..")
    os.system("python3 skel_img_to_adj_list.py ")
    print("Pipeline step 3/3 (skelimg2adjlist) done.")
    print("USED PREDICTIONS FROM: " + in_path_img2pt)
    print("Pipeline finished, check your submission!")


if __name__ == '__main__':
    main()
