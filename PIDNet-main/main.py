from models import pidnet
from configs import config

import torch
import cv2


def main():

    device = torch.device('cuda')
    model = pidnet.get_seg_model(config, imgnet_pretrained=True)
    model.eval()
    model.to(device)

    img = cv2.imread("../Images_61/0001.png")
    resized = torch.from_numpy(cv2.resize(img, (2048, 1024))).swapaxes(1,2).swapaxes(0,1)[None, :, :, :].to(device).float()

    with torch.no_grad():
        seg = model(resized)
        print(seg[0].shape)

    # cv2.imshow(f"{img.shape}", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow(f"{resized.shape}", resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()