import os
import glob
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from skimage import io
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import U2NET
from model import U2NETP

from data_loader import RescaleT
from data_loader import ToTensorLab
from data_loader import SalObjDataset

import numpy as np
# normalize the predicted SOD probability map
def norm_pred(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi) / (ma-mi)

    return dn


def save_output(image_name, predict, d_dir):
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np * 255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1, len(bbb)):
        imidx = imidx + "." + bbb[i]
    imo.save(os.path.join(d_dir, imidx+'.jpg'))


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="U2NET Segmentation Test", add_help=add_help)

    # File parameters
    parser.add_argument("--data-path", default='/content/data', type=str, help="path to input images folder")
    parser.add_argument("--save-path", default='output', type=str, help="path for output masks folder")
    parser.add_argument("--model-path", default='weights/u2net_full_model.pth', type=str, help="path to models")

    parser.add_argument("--model", default='u2net', type=str, help="model name: u2net or u2netp")
    parser.add_argument("--worker", default=2, type=int, help="number of workers")

    # Pre-processing parameters
    parser.add_argument("--resize", default=320, type=int, help="rescale size (int or tuple (h, w))")

    return parser


def main():
    args = get_args_parser().parse_args()

    # --------- 1. get image path and name ---------
    model_name = args.model

    image_dir = args.data_path
    prediction_dir = args.save_path
    model_path = args.model_path

    img_name_list = sorted(glob.glob(os.path.join(image_dir, '*.*')))[:]
    # print(img_name_list)

    # --------- 2. dataloader ---------
    # 1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list,
                                        lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(args.resize),
                                                                      ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.worker)

    # --------- 3. model define ---------
    if model_name == 'u2netp':
        print("...load U2NETP---4.7 MB")
        net = U2NETP(3, 1)
    else:
        print("...load U2NET---173.6 MB")
        net = U2NET(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path)['model'])
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu')['model'])
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in tqdm(enumerate(test_salobj_dataloader)):

        # print("inferring:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = norm_pred(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7


if __name__ == "__main__":
    main()
