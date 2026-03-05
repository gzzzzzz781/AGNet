import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Batch μ-law Tone Mapping for 16-bit HDR PNG images")
    parser.add_argument("--input_dir", type=str, default="./result", help="Path to input folder containing 16-bit PNG images")
    parser.add_argument("--output_dir", type=str, default="./visual_result", help="Path to save tone-mapped images")
    parser.add_argument("--mu", type=float, default=100.0, help="Mu parameter for mu-law tone mapping (default: 100)")

    return parser.parse_args()

def mu_tonemap(hdr, mu=10.0):
    return np.log(1.0 + mu * hdr) / np.log(1.0 + mu)


def process_folder(args):
    os.makedirs(args.output_dir, exist_ok=True)
    files = [f for f in os.listdir(args.input_dir)
             if f.lower().endswith(".png")]
    files.sort()

    for fname in tqdm(files, desc="Tone Mapping"):
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"Failed to read {fname}")
            continue
        if img.dtype != np.uint16:
            print(f"Warning: {fname} is not uint16, skip")
            continue

        img = img.astype(np.float32) / 65535.0
        img_tm = mu_tonemap(img, mu=args.mu)
        img_tm_8bit = np.clip(img_tm * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite(output_path, img_tm_8bit)


if __name__ == "__main__":
    args = get_args()
    process_folder(args)
    print("Done.")
