import pandas as pd
# import matplotlib.pyplot as plt

from matplotlib import rc
import matplotlib.pylab as plt

import csv

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

mvtec_ad = {'objects':['bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper',],
            'textures': ['carpet','grid','leather','tile','wood'],
            }

visa = {'complex_structure':['pcb1','pcb2','pcb3','pcb4'],
        'multiple_instances':['candle','capsules','macaroni1','macaroni2'],
        'single_instance':['cashew','chewinggum','fryum','pipe_fryum'],
        }

datasets = {"mvtec_ad":mvtec_ad, "visa":visa}

header = ['dataset','category_type',
          'None_im','None_pix',
          '1_im','1_pix',
          '1+2_im','1+2_pix',
          '1+2+3_im','1+2+3_pix',
          '1+2+3+4_im','1+2+3+4_pix',
          '1+2+3+4+5_im','1+2+3+4+5_pix'
]

df = pd.read_csv("experiment_1.csv")

ablations = {'None':(False, 'nearest', 50, 1, False),
          '1':(True, 'nearest', 50, 1, False),
          '1+2':(True, 'bilinear', 50, 1, False),
          '1+2+3':(True, 'nearest', -1, 1, False),
          '1+2+3+4':(True, 'nearest', -1, 5, False),
          '1+2+3+4+5':(True, 'nearest', -1, 5, True),}

rows = []
df['category_type'] = "2"
for dataset, dataset_categories in datasets.items():
    for category_type, categories in dataset_categories.items():
        df.loc[df['category'].isin(categories),'category_type'] = category_type
        category_type_df = df[df['category'].isin(categories)]

        row = {'dataset':dataset, 'category_type':category_type}
        for ablation, spec in ablations.items():
            temp = category_type_df.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection'.format(spec[0], spec[1], spec[2], spec[3], spec[4]), engine="python")
            if len(temp)>0:
                row[ablation+"_im"] = temp['image_AUROC'].mean()
                row[ablation+"_pix"]= temp['pixel_AUPRO'].mean()
            else:
                row[ablation+"_im"] = 0
                row[ablation+"_pix"]= 0

        rows.append(row)

        for category in categories:
            category_type_df = df[df['category']==category]

            row = {'dataset':dataset, 'category_type':category}
            for ablation, spec in ablations.items():
                temp = category_type_df.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection'.format(spec[0], spec[1], spec[2], spec[3], spec[4]), engine="python")
                if len(temp)>0:
                    row[ablation+"_im"] = temp['image_AUROC'].mean()
                    row[ablation+"_pix"]= temp['pixel_AUPRO'].mean()
                else:
                    row[ablation+"_im"] = 0
                    row[ablation+"_pix"]= 0

            rows.append(row)

with open('processed_experiment_1.csv','w',newline="") as f:
    w = csv.DictWriter(f,fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    f.close()

