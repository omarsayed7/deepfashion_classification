import os, sys, time, json, argparse
from pathlib import Path 
from tqdm import tqdm 
import pandas as pd 
from collections import Counter

main_data_dir = Path("../../data/classification_data")
images_dir = Path(main_data_dir / "img")
raw_annotations_dir = Path(main_data_dir / "raw_annotations")
raw_annotations_file = Path(raw_annotations_dir / "list_category_img.txt")

def get_key(val, my_dict):
    '''
    Helper function to get the key of a value in a dictionary
    '''
    for key, value in my_dict.items():
        if val == value:
            return key
    return "key doesn't exist"


def build_data(raw_annotations_path: Path, save_path: Path, max_num_images: int, sampling_threshold: int):
    print("[INFO] Starting to build fashion data")
    img_path_list = []
    labels_id_list = []
    labels_str_list = []
    f = open(raw_annotations_path, "r")
    for line in f:
        if ".jpg" in line:
            #this is image and label
            parse_line = line.split(".jpg")
            image_path = parse_line[0]+".jpg".strip()
            label_id = parse_line[-1].replace(" ", "").replace("\n","").strip()
            label_id = int(label_id)
            if label_id == 47:
                label_id = 38 
            if label_id == 48:
                label_id = 45
            label_str = image_path.split("/")[1].split("_")[-1].strip()
            img_path_list.append(image_path)
            labels_id_list.append((label_id - 1))
            labels_str_list.append(label_str)

    class_distribution = dict(Counter(labels_str_list))
    sampled_classes = []
    for key, value in class_distribution.items():
        if value > sampling_threshold:
            sampled_classes.append(key)
    print(f"[INFO] Number of sampled classes {len(sampled_classes)}")
    sampled_img_path_list = []
    sampled_labels_id_list = []
    sampled_labels_str_list = []

    #sample from every class number of images 
    classes_num_images = {}
    label_mapping = {}
    start_mapping = 0
    for i, label_str in enumerate(set(labels_str_list)):
        if label_str in sampled_classes:
            classes_num_images[label_str] = 0 
            #build the label mapping from id to str
            label_mapping[start_mapping] = label_str
            start_mapping += 1
    print(label_mapping)
    for i in range(len(labels_id_list)):
        label_str = labels_str_list[i]
        if label_str in sampled_classes: 
            if classes_num_images[label_str] < max_num_images:
                sampled_img_path_list.append(img_path_list[i])
                sampled_labels_id_list.append(get_key(label_str, label_mapping))
                sampled_labels_str_list.append(labels_str_list[i])
                classes_num_images[label_str] += 1
    print(f"[INFO] Distribution of the sampled classes {Counter(sampled_labels_str_list)}")
    print(f"[INFO] Total number of images {len(sampled_img_path_list)}")
    assert len(sampled_img_path_list) == len(sampled_labels_id_list) == len(sampled_labels_str_list)

    fashion_df = pd.DataFrame(columns=["img_path", "label_id", "label_str"])
    fashion_df["img_path"] = sampled_img_path_list
    fashion_df["label_id"] = sampled_labels_id_list
    fashion_df["label_str"] = sampled_labels_str_list
    fashion_df.to_csv(f"{save_path}/fashion_data.csv", index=False)

    with open(f'{save_path}/mapping.json', 'w') as fp:
        json.dump(label_mapping, fp)

    print("[INFO] Finished parsing the fashion dataset and saved to {dataframe_path}/fashion_data.csv")



if __name__ == "__main__":
    '''
    Main function, used to parse the arguments and call the main function
    '''
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-maximum_num_images', '--maximum_num_images', type= int, help= '', default = 1000)
    parser.add_argument('-sampling_threshold', '--sampling_threshold', type= int, help= '', default = 500)
    args = parser.parse_args()

    build_data(raw_annotations_file, main_data_dir, args.maximum_num_images, args.sampling_threshold)
    