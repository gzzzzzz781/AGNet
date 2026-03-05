import os
import glob
import shutil
import random
import argparse
import cv2


def get_croped_data_per_scene(scene_dir, patch_size=480, stride=240):
    folder_name = os.path.basename(scene_dir)
    ldr_file_path = os.path.join(scene_dir, f'{folder_name}_medium.png')
    label_path = os.path.join(scene_dir, f'{folder_name}_gt.png')

    ldr_0 = cv2.imread(ldr_file_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

    crop_data = []
    h, w, _ = label.shape
    for x in range(w):
        for y in range(h):
            if x * stride + patch_size <= w and y * stride + patch_size <= h:
                crop_ldr = ldr_0[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_label = label[y * stride:y * stride + patch_size, x * stride:x * stride + patch_size]
                crop_sample = {
                    'ldr': crop_ldr,
                    'label': crop_label,
                    }
                crop_data.append(crop_sample)
    print("{} samples of scene {}.".format(len(crop_data), scene_dir))
    return crop_data

def rotate_sample(data_sample, mode=0):
    if mode == 0:
        flag = cv2.ROTATE_90_CLOCKWISE
    elif mode == 1:
        flag = cv2.ROTATE_90_COUNTERCLOCKWISE
    rotate_ldr = cv2.rotate(data_sample['ldr'], flag)
    rotate_label = cv2.rotate(data_sample['label'], flag)
    return {
        'ldr_': rotate_ldr,
        'label': rotate_label
        }

def flip_sample(data_sample, mode=0):

    flip_ldr = cv2.flip(data_sample['ldr'], mode)
    flip_label = cv2.flip(data_sample['label'], mode)
    return {
        'ldr': flip_ldr,
        'label': flip_label
        }

def save_sample(data_sample, save_root, id):
    save_path = os.path.join(save_root, id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, '0.png'), data_sample['ldr'])
    cv2.imwrite(os.path.join(save_path, 'label.png'), data_sample['label'])

def main():
    parser = argparse.ArgumentParser(description='Prepare cropped data',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="/path/to/data")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--aug", action='store_true', default=False)

    args = parser.parse_args()

    full_size_training_data_path = os.path.join(args.data_root, 'Training')
    #保存patch图片的路径
    cropped_training_data_path = os.path.join(args.data_root, 'NTIRE_training_crop{}_stride{}'.format(str(args.patch_size), str(args.stride)))
    if not os.path.exists(cropped_training_data_path):
        os.makedirs(cropped_training_data_path)

    global counter
    counter = 0
    all_scenes = sorted(glob.glob(os.path.join(full_size_training_data_path, '*')))
    for scene in all_scenes:
        print("==>Process scene: {}".format(scene))
        scene_dir = os.path.join(args.data_root, scene)
        croped_data = get_croped_data_per_scene(scene_dir, patch_size=args.patch_size, stride=args.stride)
        for data in croped_data:
            save_sample(data, cropped_training_data_path, str(counter).zfill(6))
            counter += 1

            if args.aug:
                # sample_1 = flip_sample(data, 0)
                # save_sample(sample_1, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                sample_2 = flip_sample(data, 1)
                save_sample(sample_2, cropped_training_data_path, str(counter).zfill(6))
                counter += 1

                # sample_3 = flip_sample(data, 0)
                # sample_3 = flip_sample(sample_3, 1)
                # save_sample(sample_3, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_4 = rotate_sample(data, 0)
                # save_sample(sample_4, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_5 = rotate_sample(data, 0)
                # sample_5 = flip_sample(sample_5, 0)
                # save_sample(sample_5, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_6 = rotate_sample(data, 0)
                # sample_6 = flip_sample(sample_6, 1)
                # save_sample(sample_6, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_7 = rotate_sample(data, 0)
                # sample_7 = flip_sample(sample_7, 0)
                # sample_7 = flip_sample(sample_7, 1)
                # save_sample(sample_7, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_8 = rotate_sample(data, 0)
                # sample_8 = rotate_sample(sample_8, 0)
                # save_sample(sample_8, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_9 = rotate_sample(data, 0)
                # sample_9 = rotate_sample(sample_9, 0)
                # sample_9 = flip_sample(sample_9, 0)
                # save_sample(sample_9, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_10 = rotate_sample(data, 0)
                # sample_10 = rotate_sample(sample_10, 0)
                # sample_10 = flip_sample(sample_10, 1)
                # save_sample(sample_10, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_11 = rotate_sample(data, 0)
                # sample_11 = rotate_sample(sample_11, 0)
                # sample_11 = flip_sample(sample_11, 0)
                # sample_11 = flip_sample(sample_11, 1)
                # save_sample(sample_11, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_12 = rotate_sample(data, 1)
                # save_sample(sample_12, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_13 = rotate_sample(data, 1)
                # sample_13 = flip_sample(sample_13, 0)
                # save_sample(sample_13, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_14 = rotate_sample(data, 1)
                # sample_14 = flip_sample(sample_14, 1)
                # save_sample(sample_14, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

                # sample_15 = rotate_sample(data, 1)
                # sample_15 = flip_sample(sample_15, 0)
                # sample_15 = flip_sample(sample_15, 1)
                # save_sample(sample_15, cropped_training_data_path, str(counter).zfill(6))
                # counter += 1

if __name__ == '__main__':
    main()