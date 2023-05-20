import os, sys, time, json
from pathlib import Path 
from tqdm import tqdm 
import pandas as pd 

main_data_dir = Path("../../data/classification_data")
images_dir = Path(main_data_dir / "img")
raw_annotations_dir = Path(main_data_dir / "raw_annotations")
raw_annotations_file = Path(raw_annotations_dir / "list_category_img.txt")


SEED = 42

def build_data(raw_annotations_path: Path, save_path:Path):
    print("[INFO] Starting to build fashion data")
    img_path_list = []
    labels_id_list = []
    labels_str_list = []
    label_mapping = {}
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
            label_mapping[label_id] = label_str
            img_path_list.append(image_path)
            labels_id_list.append(label_id)
            labels_str_list.append(label_str)

    assert len(img_path_list) == len(labels_id_list) == len(labels_str_list)
    fashion_df = pd.DataFrame(columns=["img_path", "label_id", "label_str"])
    fashion_df["img_path"] = img_path_list
    fashion_df["label_id"] = labels_id_list
    fashion_df["label_str"] = labels_str_list
    
    fashion_df.to_csv(f"{save_path}/fashion_data.csv", index=False)
    with open(f'{save_path}/mapping.json', 'w') as fp:
        json.dump(label_mapping, fp)
    print("[INFO] Finished parsing the fashion dataset and saved to {dataframe_path}/fashion_data.csv")

build_data(raw_annotations_file, main_data_dir)
    