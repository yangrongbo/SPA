import argparse
import torch
import numpy as np
import torchvision.transforms as transforms
import utils
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--s_model1', type=str, default='tf_inception_v3', help='Surrogate model to use.')
parser.add_argument('--s_model2', type=str, default='tf_inception_v4', help='Surrogate model to use.')
parser.add_argument('--s_model3', type=str, default='tf_inc_res_v2', help='Surrogate model to use.')
parser.add_argument('--s_model4', type=str, default='tf_resnet_v2_101', help='Surrogate model to use.')
parser.add_argument('--t_model', type=str, nargs='+',
                    default=['tf_inception_v3', 'tf_inception_v4', 'tf_inc_res_v2', 'tf_resnet_v2_101',
                             'tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2'],
                    help='Target model to use.')
args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
output_dir = './Outputs_ENS_NI/'
input_dir = './dataset/images'
input_csv = './dataset/images.csv'
data_transform = transforms.Compose(
    [transforms.Resize(299), transforms.ToTensor()]
)
Input = utils.ImageNet(input_dir, input_csv, data_transform)
batch_size = 20
data_loader = DataLoader(Input, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

models_path = './models/'
s_model1 = utils.get_model(args.s_model1, models_path).eval().to(device)
s_model2 = utils.get_model(args.s_model2, models_path).eval().to(device)
s_model3 = utils.get_model(args.s_model3, models_path).eval().to(device)
s_model4 = utils.get_model(args.s_model4, models_path).eval().to(device)
t_model = []
for s in range(len(args.t_model)):
    t_model.append(utils.get_model(args.t_model[s], models_path).eval().to(device))

num_iter = 10
epsilon = 16.0 / 255
alpha = epsilon / num_iter
success_count = [0, 0, 0, 0, 0, 0, 0]
momentum = 1.0
m = 5

for i, [x, name, y] in enumerate(data_loader):
    x = x.to(device)
    y = y.to(device)
    x_max = torch.clamp(x + epsilon, 0.0, 1.0)
    x_min = torch.clamp(x - epsilon, 0.0, 1.0)

    gt = 0
    for k in range(num_iter):
        x_nes = x + alpha * momentum * gt
        global_grad = 0
        for j in range(m):
            x_temp = x_nes / (2 ** j)
            x_temp = x_temp.detach().requires_grad_()
            out1 = s_model1(x_temp)
            out2 = s_model2(x_temp)
            out3 = s_model3(x_temp)
            out4 = s_model4(x_temp)
            logits = (out1[0] + out2[0] + out3[0] + out4[0]) / 4.
            auxlogits = (out1[1] + out2[1] + out3[1]) / 3.
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss += torch.nn.functional.cross_entropy(auxlogits, y) * 0.4
            loss.backward()
            global_grad += x_temp.grad.data
        new_grad = global_grad / (m * 1.)
        gt = momentum * gt + new_grad / new_grad.norm(1)
        x = x + alpha * torch.sign(gt)
        x = torch.clamp(x, x_min, x_max)

    adv_img_np = x.detach().cpu().numpy()
    adv_img_np = np.transpose(adv_img_np, (0, 2, 3, 1)) * 255
    utils.save_image(adv_img_np, name, output_dir)

    for r in range(len(args.t_model)):
        success_count[r] += (t_model[r](x)[0].argmax(1) != y).detach().sum().cpu()
    print('%4d :' % ((i + 1) * batch_size), [t * 1.0 / ((i + 1) * batch_size) for t in success_count])
