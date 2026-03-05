import os
import os.path as osp
import argparse
import torch
import numpy as np
import cv2
import torch.nn.functional as F
from models.AGNet import AGNet

parser = argparse.ArgumentParser(description="Inference Only")
parser.add_argument('--input_dir', type=str, default="/path/to/data", help='input image folder')
parser.add_argument('--pretrained_model', type=str, default="/path/to/pretained", help='path to model')
parser.add_argument('--save_dir', type=str, default='./result', help='output folder')
parser.add_argument('--no_cuda', action='store_true', default=False)


def load_image(path, device):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(device)

def save_hdr(pred, save_path):
    pred = pred.clamp(0, 1)
    pred = pred[0].permute(1, 2, 0).cpu().numpy()
    pred_uint16 = (pred * 65535).astype(np.uint16)
    pred_uint16 = pred_uint16[..., ::-1]  # RGB → BGR
    cv2.imwrite(save_path, pred_uint16)

def main():
    args = parser.parse_args()

    print(">>>>>>>>> Start Inference >>>>>>>>>")
    print("Load weights from:", args.pretrained_model)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = AGNet().to(device)
    checkpoint = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    os.makedirs(args.save_dir, exist_ok=True)

    image_list = os.listdir(args.input_dir)

    for idx, name in enumerate(image_list):
        img_path = osp.join(args.input_dir, name)

        if not name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        print(f"[{idx}] Processing: {name}")

        with torch.no_grad():
            input_img = load_image(img_path, device)
            pred = model(input_img)

        base_name = osp.splitext(name)[0]  # 去掉原后缀
        save_name = base_name + ".png"  # 强制改成 png
        save_path = osp.join(args.save_dir, save_name)

        save_hdr(pred, save_path)

    print(">>>>>>>>> Finish Inference >>>>>>>>>")

if __name__ == '__main__':
    main()