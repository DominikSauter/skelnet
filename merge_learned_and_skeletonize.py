from PIL import Image
import os
import shutil
import numpy as np


def main():
    in_learned_skeletons_path = "dataset/point/skelnet_black_skel_test_black_epoch252"
    in_learned_black_skeletons_test_normal_path = "dataset/point/skelnet_black_skel_test_normal_epoch252"
    in_skeletonize_path = "dataset/point/skeletonize"
    in_skeletonize_lee_path = "dataset/point/skeletonize_lee"
    in_medial_axis_skeletons_path = "dataset/point/medial_axis_skeletons"
    in_thinned_skeletons_path = "dataset/point/thinned_skeletons"
    out_merged_path = "dataset/point/merged"

    # delete out dirs and recreate    
    shutil.rmtree(out_merged_path, ignore_errors=True)
    if not os.path.isdir(out_merged_path):
        os.makedirs(out_merged_path)

    for learned_img, learned_black_skel_test_normal, skeletonize_img, skeletonize_lee_img, medial_axis_skeleton_img, thinned_skeleton_img in zip(os.listdir(in_learned_skeletons_path), os.listdir(in_learned_black_skeletons_test_normal_path), os.listdir(in_skeletonize_path), os.listdir(in_skeletonize_lee_path), os.listdir(in_medial_axis_skeletons_path), os.listdir(in_thinned_skeletons_path)):
        print(learned_img)
        l_img = Image.open(os.path.join(in_learned_skeletons_path, learned_img), 'r')
        lbn_img = Image.open(os.path.join(in_learned_black_skeletons_test_normal_path, learned_black_skel_test_normal), 'r')
        s_img = Image.open(os.path.join(in_skeletonize_path, skeletonize_img), 'r')
        sl_img = Image.open(os.path.join(in_skeletonize_lee_path, skeletonize_lee_img), 'r')
        m_img = Image.open(os.path.join(in_medial_axis_skeletons_path, medial_axis_skeleton_img), 'r')
        t_img = Image.open(os.path.join(in_thinned_skeletons_path, thinned_skeleton_img), 'r')

        l_img_immat = l_img.load()
        lbn_img_immat = lbn_img.load()
        s_img_immat = s_img.load()
        sl_img_immat = sl_img.load()
        m_img_immat = m_img.load()
        t_img_immat = t_img.load()
        merged_img = Image.new('1', (256, 256))
        merged_img_immat = merged_img.load()
        for i in range (256):
            for j in range(256):
                merged_img_immat[(i, j)] = (
                        l_img_immat[(i, j)] +
                        lbn_img_immat[(i, j)] +
                        s_img_immat[(i, j)] +
                        sl_img_immat[(i, j)]
                        #m_img_immat[(i, j)] +
                        #t_img_immat[(i, j)]
                        )
        merged_img.save(os.path.join(out_merged_path, learned_img))

        #Image.fromarray(np.array(merged_img), 'L').show()
        

if __name__ == '__main__':
    main()

