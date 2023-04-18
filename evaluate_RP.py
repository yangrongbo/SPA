import utils
import numpy as np
import torch
from torchvision import transforms as T
from loader import ImageNet
from torch.utils.data import DataLoader


batch_size = 10
input_csv = './dataset/images.csv'
adv_dir = './Outputs_MI_v3'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def evaluate(model_name, path):
    model = utils.get_model(model_name, path).eval().to(device)
    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    sum = 0
    for x, _, y in data_loader:
        resize = np.random.randint(310, 331)
        x = torch.nn.functional.interpolate(x, size=(resize, resize), mode='nearest')
        a = np.random.randint(0, 331 - resize)
        b = np.random.randint(0, 331 - resize)
        x = torch.nn.functional.pad(x, (a, 331 - resize - a, b, 331 - resize - b))
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            sum += (model(x)[0].argmax(1) != y).detach().sum().cpu()
    print(model_name + '  rate = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ['tf_ens3_adv_inc_v3']
    models_path = './models/'
    for model_name in model_names:
        evaluate(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    main()
