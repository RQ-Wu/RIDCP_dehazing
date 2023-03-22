import argparse
import cv2
import glob
import os
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from basicsr.utils.download_util import load_file_from_url


def main():
    """Inference demo for FeMaSR 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-w', '--weight', type=str, default=None, help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('--use_weight', action="store_true")
    parser.add_argument('--alpha', type=float, default=1.0, help='value of alpha')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=1500, help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    weight_path = args.weight
    
    # set up the model
    sr_model = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], LQ_stage=True, use_weight=args.use_weight, weight_alpha=args.alpha).to(device)
    sr_model.load_state_dict(torch.load(weight_path)['params'], strict=False)
    sr_model.eval()
    
    os.makedirs(args.output, exist_ok=True)
    if os.path.isfile(args.input):
        paths = [args.input]
    else:
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

    pbar = tqdm(total=len(paths), unit='image')
    for idx, path in enumerate(paths):
        img_name = os.path.basename(path)
        save_path = os.path.join(args.output, f'{img_name}')
        pbar.set_description(f'Test {img_name}')
        
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.max() > 255.0:
            img = img / 255.0
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = args.max_size ** 2 
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output, _ = sr_model.test(img_tensor)
        else:
            down_img = torch.nn.UpsamplingBilinear2d((h//2, w//2))(img_tensor)
            output, _ = sr_model.test(down_img)
            output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        output_img = tensor2img(output)

        imwrite(output_img, save_path)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
