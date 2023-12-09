import os
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

HOME = os.getcwd()
print("HOME:", HOME)

CHECKPOINT_PATH = os.path.join(HOME, "checkpoints", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

INPUTDIR = os.path.join(HOME, "Images_61")
OUTPUTDIR = os.path.join(HOME, sys.argv[1])

for IMAGE_NAME in tqdm(os.listdir(INPUTDIR), desc="Inferring"):
    IMAGE_PATH = os.path.join(INPUTDIR, IMAGE_NAME)

    OUTPUTPATH = os.path.join(OUTPUTDIR, IMAGE_NAME.split(".")[0])
    os.makedirs(OUTPUTPATH, exist_ok=True)

    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    sam_result = mask_generator.generate(image_rgb)

    # print(sam_result[0].keys())

    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

    detections = sv.Detections.from_sam(sam_result=sam_result)

    annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

    # sv.plot_images_grid(
    #     images=[image_bgr, annotated_image],
    #     grid_size=(1, 2),
    #     titles=['source image', 'segmented image']
    # )

    masks = [
        mask['segmentation']
        for mask
        in sorted(sam_result, key=lambda x: x['area'], reverse=True)
    ]

    # sv.plot_images_grid(
    #     images=masks,
    #     grid_size=(8, int(len(masks) / 8)+1),
    #     size=(16, 16)
    # )

    for i, mask in enumerate(masks):
        plt.imsave(os.path.join(OUTPUTPATH, f"{i}.png"), mask, cmap="gray")

    

