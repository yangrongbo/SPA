import torch
import torchvision
from torchvision import transforms as T
from loader import ImageNet
from torch.utils.data import DataLoader
import utils

batch_size = 10
input_csv = './dataset/images.csv'
adv_dir = './Outputs_SPACOZMI_v3'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def evaluate(model_name, path):
    model = utils.get_model(model_name, path).eval().to(device)
    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    sum = 0
    for x, _, y in data_loader:
        x_org = x.clone()
        x_min = torch.min(x)
        x_max = torch.max(x)
        x = (x - x_min) / (x_max - x_min) * 255  # 先(0,1),再(0,255)
        x = x.to(dtype=torch.uint8)
        temp = []
        for i in range(batch_size):
            temp.append(torchvision.io.decode_jpeg(torchvision.io.encode_jpeg(x[i], 5)))
        x = torch.stack(temp)  # 张量list合并为一个张量
        x = x.reshape(x_org.shape)
        x = x.to(dtype=x_org.dtype)
        x = x / 255.0 * (x_max - x_min) + x_min
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
