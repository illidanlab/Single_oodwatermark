"""Configuration file for defining paths to data."""
import os


def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)


hostname = os.uname()[1]  # type: str
# Update your paths here.
CHECKPOINT_ROOT = './checkpoint'
if int(hostname.split('-')[-1]) >= 8:
    data_root = '/localscratch2/yushuyan/'
elif hostname.startswith('illidan'):
    data_root = '/media/Research/jyhong/data'
else:
    data_root = './data'
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)

if hostname.startswith('illidan') and int(hostname.split('-')[-1]) < 8:
    # personal config
    home_path = os.path.expanduser('~/')
    DATA_PATHS = {
        "Digits": home_path + "projects/FedBN/data",
        "DomainNet": data_root + "/DomainNet",
        # store the path list file from FedBN
        "DomainNetPathList": home_path + "projects/FedBN/data/",
        "Cifar10": data_root+"/Cifar10",
    }

    if int(hostname.split('-')[-1]) == 7:
        BDBlocker_path = '/home/jyhong/projects_ex/BackdoorBlocker/'
    else:
        raise RuntimeError(f"No pretrained models at {hostname}")
else:
    DATA_PATHS = {
        "Digits": data_root + "/Digits",
        "DomainNet": data_root + "/DomainNet",
        # store the path list file from FedBN
        "DomainNetPathList": data_root + "/DomainNet/domainnet10/",
        "Cifar10": data_root+"/Cifar10",
        "Cifar100": data_root + "/Cifar100",
        "ImageNetDS": "/localscratch2/jyhong",
        "ImageNet": "/localscratch2/jyhong/image-net-all/ILSVRC2012",
        "gtsrb": data_root + '/gtsrb',
        "stl10": data_root + '/stl10',
        "SVHN": data_root +'/SVHN',
    }
    #BDBlocker_path = '/localscratch/jyhong/projects/BackdoorBlocker/'
    BDBlocker_path = '/localscratch/yushuyan/projects/backdoorblocker/'
# repository for store the pre-trained model which you want to inject watermark
haotao_PT_model_path = BDBlocker_path + 'results/normal_training/'

