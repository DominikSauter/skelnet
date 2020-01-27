import os
import shutil
import json


def main():
    # scores to add
    #skeletonize
    #skeletonize_lee
    #medial_axis
    #thinned
    #skelnetblacktestnormal
    #skelnet_pretrained
    #skelnet_pretrained_erosion
    #merged_skelnetpre_skeletonize_skeletonizelee
    #skelnetblacktestblack
    in_test_scores_path = "dataset/point/selective_ensemble/test_scores/"
    out_test_images_path = "dataset/point/selective_ensemble/"

    scores_dicts = []
    for scores_file in os.listdir(in_test_scores_path):
        if os.path.isfile(os.path.join(in_test_scores_path, scores_file)):
            with open(os.path.join(in_test_scores_path, scores_file), 'r') as file:
                scores = json.load(file)
            scores_dicts.append((scores, scores_file))
    
    max_scores = []
    for key in scores_dicts[0][0].keys():
        max_score = (-1., None, None)
        for scores_dict in scores_dicts:
            score = scores_dict[0][key]
            if score > max_score[0]:
                max_score = (score, key, scores_dict[1][:-5])
        max_scores.append(max_score)    
        print('max_score: ' + str(max_score))
   
    avg_score = sum(i for i, _, _ in max_scores) / len(max_scores)
    print('avg_score: ' + str(avg_score))
    

    # delete out dirs and recreate    
    selected_images_path = os.path.join(out_test_images_path, 'selected_images')
    selected_images_details_path = os.path.join(out_test_images_path, 'selected_images_details')
    shutil.rmtree(selected_images_path, ignore_errors=True)
    shutil.rmtree(selected_images_details_path, ignore_errors=True)
    if not os.path.isdir(selected_images_path):
        os.makedirs(selected_images_path)
    if not os.path.isdir(selected_images_details_path):
        os.makedirs(selected_images_details_path)
    
    
    for max_score in max_scores:
        src_path = os.path.join('dataset/point/', max_score[2], max_score[1] + '_ip.png')
        dst_selected_path = os.path.join(out_test_images_path, 'selected_images')
        dst_selected_details_path = os.path.join(out_test_images_path, 'selected_images_details', str(round(max_score[0], 3)) + '_' + max_score[1]  + '_' + max_score[2] + '.png')
        
        shutil.copy(src_path, dst_selected_path)
        shutil.copy(src_path, dst_selected_details_path)


if __name__ == '__main__':
    main()

