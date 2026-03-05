import numpy as np
import os, glob
import cv2
import math
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
from skimage.metrics import peak_signal_noise_ratio
from torchmetrics.functional import structural_similarity_index_measure

def list_all_files_sorted(folder_name, extension=""):
    return sorted(glob.glob(os.path.join(folder_name, "*" + extension)))

def read_images(file_name):
    img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB) / 255.0
    return img

def read_label(file_name):
    label = cv2.cvtColor(cv2.imread(file_name, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 65535.0
    return label

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def range_compressor_cuda(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)

def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)

def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (psnr/Img.shape[0])

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.5 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def init_parameters(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

class PU21:
    def __init__(self, type='banding_glare'):
        self.L_min = 0.005
        self.L_max = 10000.
        self.epsilon = 1e-6

        if type == 'banding':
            self.par = [1.070275272, 0.4088273932, 0.153224308,
                        0.2520326168, 1.063512885, 1.14115047, 521.4527484]
        elif type == 'banding_glare':
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05,
                        0.9062562627, 0.09150303166, 0.9099517204, 596.3148142]
        elif type == 'peaks':
            self.par = [1.043882782, 0.6459495343, 0.3194584211,
                        0.374025247, 1.114783422, 1.095360363, 384.9217577]
        elif type == 'peaks_glare':
            self.par = [816.885024, 1479.463946, 0.001253215609,
                        0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
        else:
            raise ValueError("Unknown PU21 type")

    def encode(self, hdr):

        p = self.par
        if isinstance(hdr, torch.Tensor):
            Y = torch.clamp(hdr, self.L_min, self.L_max)

            V = p[6] * torch.pow(
                (p[0] + p[1] * torch.pow(Y, p[3])) /
                (1 + p[2] * torch.pow(Y, p[3])),
                p[4] - p[5]
            )
            return V

        elif isinstance(hdr, np.ndarray):
            Y = np.clip(hdr, self.L_min, self.L_max)

            V = p[6] * np.power(
                (p[0] + p[1] * np.power(Y, p[3])) /
                (1 + p[2] * np.power(Y, p[3])),
                p[4] - p[5]
            )
            return V

        else:
            raise TypeError("Input must be torch.Tensor or numpy.ndarray")

    def decode(self, V):
        p = self.par

        if isinstance(V, torch.Tensor):
            V_p = torch.pow(torch.clamp(V / p[6] + p[5], min=0), 1 / p[4])
            Y = torch.pow(
                torch.clamp((V_p - p[0]) / (p[1] - p[2] * V_p), min=0),
                1 / p[3]
            )
            return Y

        elif isinstance(V, np.ndarray):
            V_p = np.power(np.maximum(V / p[6] + p[5], 0), 1 / p[4])
            Y = np.power(
                np.maximum((V_p - p[0]) / (p[1] - p[2] * V_p), 0),
                1 / p[3]
            )
            return Y

        else:
            raise TypeError("Input must be torch.Tensor or numpy.ndarray")

def pu21_psnr(pred, gt, pu_type='banding_glare'):
    pu = PU21(type=pu_type)

    pred_pu = pu.encode(pred)
    gt_pu   = pu.encode(gt)

    # ===== 分别判断类型 =====
    if isinstance(pred_pu, np.ndarray):
        pred_pu = torch.from_numpy(pred_pu).float()

    if isinstance(gt_pu, np.ndarray):
        gt_pu = torch.from_numpy(gt_pu).float()

    # ===== 保证 device 一致 =====
    device = pred.device
    pred_pu = pred_pu.to(device)
    gt_pu   = gt_pu.to(device)

    # ===== 计算 PSNR =====
    mse = torch.mean((pred_pu - gt_pu) ** 2)
    mse = torch.clamp(mse, min=1e-8)

    max_val = torch.max(gt_pu)
    psnr = 20 * torch.log10(max_val) - 10 * torch.log10(mse)

    return psnr.item()


def pu21_ssim(pred, gt):
    pu = PU21()

    pred_pu = pu.encode(pred)
    gt_pu   = pu.encode(gt)

    # numpy → torch
    if isinstance(pred_pu, np.ndarray):
        pred_pu = torch.from_numpy(pred_pu).float()
    if isinstance(gt_pu, np.ndarray):
        gt_pu = torch.from_numpy(gt_pu).float()

    device = pred.device
    pred_pu = pred_pu.to(device)
    gt_pu   = gt_pu.to(device)

    # ===== 关键：统一维度 =====

    # 如果是 HWC → CHW
    if pred_pu.ndim == 3:
        pred_pu = pred_pu.permute(2, 0, 1)
        gt_pu   = gt_pu.permute(2, 0, 1)

    # 加 batch 维度
    if pred_pu.ndim == 3:
        pred_pu = pred_pu.unsqueeze(0)
        gt_pu   = gt_pu.unsqueeze(0)

    # 最终必须是 BCHW
    assert pred_pu.ndim == 4

    data_range = torch.max(gt_pu) - torch.min(gt_pu)

    ssim = structural_similarity_index_measure(
        pred_pu,
        gt_pu,
        data_range=data_range
    )

    return ssim.item()





