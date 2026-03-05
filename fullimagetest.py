import os.path as osp
import argparse
import math
from torch.utils.data import DataLoader
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from dataset.dataset import Validation_Dataset
from utils.utils import *
from torch import nn
import torch.nn.functional as F
from models.AGNet import AGNet

parser = argparse.ArgumentParser(description="Test Setting")
parser.add_argument("--dataset_dir", type=str, default="/path/to/data",help='dataset directory')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--pretrained_model', type=str, default="/path/to/pretrained")
parser.add_argument('--save_results', action='store_true', default=True)
parser.add_argument('--save_dir', type=str, default="./result")

def main():
    args = parser.parse_args()

    print(">>>>>>>>> Start Testing >>>>>>>>>")
    print("Load weights from: ", args.pretrained_model)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = AGNet().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device)['state_dict'])
    model.eval()

    datasets = Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=128)
    dataloader = DataLoader(dataset=datasets, batch_size=1, num_workers=1, shuffle=False)
    psnr_l = AverageMeter()
    ssim_l = AverageMeter()
    psnr_mu = AverageMeter()
    ssim_mu = AverageMeter()
    pu_psnr = AverageMeter()
    pu_ssim = AverageMeter()

    for idx, img_dataset in enumerate(dataloader):
        print(idx)
        with torch.no_grad():
            batch_ldr, label = img_dataset['input'].to(device), img_dataset['label'].to(device)

            pred = model(batch_ldr)

            pred_img = torch.squeeze(pred.detach().cpu()).numpy().astype(np.float32)
            label = torch.squeeze(label.detach().cpu()).numpy().astype(np.float32)

        # psnr-l and psnr-\mu
        scene_psnr_l = peak_signal_noise_ratio(label, pred_img, data_range=1.0)
        label_mu = range_compressor(label)
        pred_img_mu = range_compressor(pred_img)
        scene_psnr_mu = peak_signal_noise_ratio(label_mu, pred_img_mu, data_range=1.0)

        # ssim-l
        pred_img = np.clip(pred_img * 255.0, 0., 255.).transpose(1, 2, 0)
        label = np.clip(label * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_l = calculate_ssim(pred_img, label)

        # ssim-\mu
        pred_img_mu = np.clip(pred_img_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        label_mu = np.clip(label_mu * 255.0, 0., 255.).transpose(1, 2, 0)
        scene_ssim_mu = calculate_ssim(pred_img_mu, label_mu)

        # pu21
        pu21_pred = pred_img / 255.0 * 10000
        pu21_label = label / 255.0 * 10000
        scene_pu_psnr = pu21_psnr(pu21_pred, pu21_label)
        scene_pu_ssim = pu21_ssim(pu21_pred, pu21_label)

        psnr_l.update(scene_psnr_l)
        ssim_l.update(scene_ssim_l)
        psnr_mu.update(scene_psnr_mu)
        ssim_mu.update(scene_ssim_mu)
        pu_psnr.update(scene_pu_psnr)
        pu_ssim.update(scene_pu_ssim)

        if args.save_results:
            if not osp.exists(args.save_dir):
                os.makedirs(args.save_dir)
            pred = pred.clamp(0, 1)
            pred = pred.cpu()[0]
            pred = pred.permute(1, 2, 0)
            pred_uint16 = (pred.numpy() * 65535).astype(np.uint16)
            pred_uint16 = pred_uint16[..., ::-1]
            cv2.imwrite(os.path.join(args.save_dir, '{}_pred.png'.format(idx)), pred_uint16)

    print("Average PSNR_mu: {:.4f}  SSIM_mu: {:.4f}".format(psnr_mu.avg, ssim_mu.avg))
    print("Average PSNR_l: {:.4f}  SSIM_l: {:.4f}".format(psnr_l.avg, ssim_l.avg))
    print("Average PU_PSNR: {:.4f}  PU_SSIM: {:.4f}".format(pu_psnr.avg, pu_ssim.avg))
    print(">>>>>>>>> Finish Testing >>>>>>>>>")

if __name__ == '__main__':
    main()




