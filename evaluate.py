import torch
from torchvision import transforms as T
import utils
from torch.utils.data import DataLoader


batch_size = 20
input_csv = './dataset/images.csv'
adv_dir = './Outputs_COZ_v3'
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


def evaluate(model_name, path):
    model = utils.get_model(model_name, path).eval().to(device)
    Input = utils.ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]))
    data_loader = DataLoader(Input, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, labels in data_loader:
        labels = labels.to(device)
        images = images.to(device)
        with torch.no_grad():
            sum += (model(images)[0].argmax(1) != labels).detach().sum().cpu()
    print(model_name + '  rate = {:.2%}'.format(sum / 1000.0))


def main():
    model_names = ['tf_inception_v3', 'tf_inception_v4', 'tf_inc_res_v2', 'tf_resnet_v2_101', 'tf_ens3_adv_inc_v3',
                   'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']
    models_path = './models/'
    for model_name in model_names:
        evaluate(model_name, models_path)
        print("===================================================")


if __name__ == '__main__':
    main()
