from typing import Any
from src.anomalib import TaskType
from src.anomalib.engine import Engine
from  src.anomalib.pipelines.benchmark.job import BenchmarkJob
from src.anomalib.models import LWinNN, Padim, Patchcore, SPADE, SPALWinNN
from src.anomalib.data import MVTec, Visa
import csv
import argparse
import os
from anomalib.utils.normalization import NormalizationMethod
from math import ceil
import torchvision.transforms.v2 as transforms
import torch
import yaml
from pathlib import Path
import shutil

DATASETS = ['mvtec_ad', 'visa', 'mvtec_loco']
CATEGORIES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper',
              'candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum',
              'breakfast_box','juice_bottle','pushpins','screw_bag','splicing_connectors']
BACKBONES = ['resnet18','wide_resnet50','wide_resnet101','resnet34','resnet50']
MODELS = ['padim', 'lwinnn', 'patchcore', 'spade', 'spalwinnn']
image_sizes = {"mvtec_ad": {'bottle':(1024, 1024),
                                'cable':(1024, 1024),
                                'capsule':(1024, 1024),
                                'carpet':(1024, 1024),
                                'grid':(1024, 1024),
                                'hazelnut':(1024, 1024),
                                'leather':(1024, 1024),
                                'metal_nut':(1024, 1024),
                                'pill':(1024, 1024),
                                'screw':(1024, 1024),
                                'tile':(1024, 1024),
                                'toothbrush':(1024, 1024),
                                'transistor':(1024, 1024),
                                'wood':(1024, 1024),
                                'zipper':(1024, 1024)},

                  "visa": {'candle':(1284,1168), 
                           'capsules': (1500,1000),
                           'cashew': (1274, 1176), 
                           'chewinggum': (1342,1118),
                           'fryum': (1500,1000),
                           'macaroni1': (1500, 1000),
                           'macaroni2': (1500, 1000),
                           'pcb1': (1404,1070),
                           'pcb2': (1404,1070),
                           'pcb3': (1562, 960),
                           'pcb4': (1358, 1104),
                           'pipe_fryum': (1300, 1154)},

                  "mvtec_loco": {'breakfast_box': (1600, 1280),
                                 'juice_bottle': (800, 1600),
                                 'pushpins': (1700, 1000),
                                 'screw_bag': (1600, 1100),
                                 'splicing_connectors': (1700, 850)}
    }

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', choices=DATASETS, help='Parent dataset', type=str, default='mvtec_ad')
    parser.add_argument('--category', choices=CATEGORIES, help='category to test on', type=str, default='bottle')
    parser.add_argument('--model', choices=MODELS, help='AD model', type=str, default='lwinnn')
    parser.add_argument('--window_size', help="window size for nearest neighbor search", default=7,type=int)
    parser.add_argument('--batch_size', help='batch size', default=512, type=int)
    parser.add_argument("--gpu_type", default="mps")
    parser.add_argument("--gpu_number", default=0)
    parser.add_argument("--write_scores", type=str, default="")

    parser.add_argument("--interpolation_mode", type=str, default="bilinear")
    parser.add_argument("--K", type=int, default=-1)
    parser.add_argument("--anomaly_map_detection", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument('--pooling', help="pool features", default=False, type=lambda x: (str(x).lower() == 'true'))

    args = parser.parse_args()

    if args.gpu_type != "mps":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    return args

def main():
    args = parse_arguments()
    try:
        shutil.rmtree('results')
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    roots = {"mvtec_ad":"../AdversariApple/Data/mvtec_anomaly_detection/",
             "visa": "../AdversariApple/Data/VisA_20220922"}
    print("Running dataset {} {} with model {}".format(args.dataset, args.category, args.model))
    image_size = image_sizes[args.dataset][args.category]
    H = ceil((256/max(image_size))*min(image_size))
    transform = transforms.Compose([
            transforms.Resize(size=H, max_size=max(H+1, 256), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    models = {"padim": Padim(backbone="resnet18", layers=["layer1", "layer2", "layer3"]),
              "lwinnn": LWinNN(backbone="resnet18", window_size=args.window_size, layers=["layer1", "layer2", "layer3"]),
              "patchcore": Patchcore(backbone="resnet18", layers=["layer2", "layer3"]),
              "spade": SPADE(backbone="resnet18", layers=["layer1", "layer2", "layer3"]), 
              "spalwinnn": SPALWinNN(backbone="resnet18", layers=["layer1", "layer2", "layer3"], K_im=args.K, interpolation_mode=args.interpolation_mode,
                                     anomaly_map_detection=args.anomaly_map_detection, window_size=args.window_size, pooling=args.pooling)}
    
    batch_sizes = {"padim": 8, "lwinnn": 8, "patchcore": 8, "spade": 8, "spalwinnn" : 32}
    
    num_workers = {'mps': 1, 'cuda':1}[args.gpu_type]
    if args.dataset == "mvtec_ad":
        datamodule = MVTec(root=roots[args.dataset], num_workers=num_workers,category=args.category, train_batch_size=batch_sizes[args.model], eval_batch_size=batch_sizes[args.model])
    elif args.dataset == "visa":
        datamodule = Visa(root=roots[args.dataset], num_workers=num_workers,category=args.category, train_batch_size=batch_sizes[args.model], eval_batch_size=batch_sizes[args.model])
    datamodule.prepare_data()  # Downloads the dataset if it's not in the specified `root` directory
    datamodule.setup()

    model = models[args.model]
    model._transform = transform


    
    # config_file = yaml.safe_load(Path("configs/model/{}.yaml".format(args.model)).read_text())
    # # config_file['model']['init_args']['pooling']=False
    # # print("configg", config_file)
    # config_file = {'metrics': {'pixel': 'AUPRO', 'image': 'AUROC'}}

    # print(config_file)

    # test_results = BenchmarkJob(args.gpu_type, model=model, datamodule=datamodule, seed=42, flat_cfg=config_file)
    # res = test_results.run()


    # start training
    engine = Engine(accelerator=args.gpu_type,task=TaskType.SEGMENTATION, image_metrics=["AUROC"], pixel_metrics=["AUPRO"])#, normalization=NormalizationMethod.NONE)
    engine.fit(model=model, datamodule=datamodule)

    # load best model from checkpoint before evaluating
    test_results = engine.test(
        model=model,
        datamodule=datamodule
    )

    image_AUROC = test_results[0]['image_AUROC']
    pixel_AUPRO = test_results[0]['pixel_AUPRO']

    if args.write_scores != "":
        if args.model == "spalwinnn":
            row = [args.dataset,args.category,args.model,args.interpolation_mode, args.pooling, args.K, args.window_size, args.anomaly_map_detection, image_AUROC,pixel_AUPRO]
        else:
            row = [args.dataset,args.category,args.model,image_AUROC,pixel_AUPRO]
        with open('{}'.format(args.write_scores),'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(row)
    else:
        print("Printing scores for {} of {} dataset with model {}...".format(args.category, args.dataset, args.model))
        print("Image AUROC: {}, Pixel AUPRO: {}".format(image_AUROC, pixel_AUPRO))

if __name__ == '__main__':
    main()
    torch.cuda.empty_cache()

