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
          'SPADE_im','SPADE_pix',
          'Padim_im','Padim_pix',
          'Patchcore_im','Patchcore_pix',
          'LWinNN_im','LWinNN_pix',
]

df_1 = pd.read_csv("experiment_1.csv")
df_2 = pd.read_csv("experiment_2.csv")

rows = []
df_1['category_type'] = "2"
df_2['category_type'] = "2"
models = {'SPADE': df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(False, 'nearest', 50, 1, False)), 
          'Padim': df_2[df_2['model']=='padim'],
          'Patchcore': df_2[df_2['model']=='patchcore'],
          'LWinNN': df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(True, 'bilinear', -1, 5, True))}
for dataset, dataset_categories in datasets.items():
    for category_type, categories in dataset_categories.items():
        row = {'dataset':dataset, 'category_type':category_type}
        for model, model_df in models.items():
            model_df.loc[model_df['category'].isin(categories),'category_type'] = category_type
            category_type_df = model_df[model_df['category'].isin(categories)]
            if len(category_type_df)>0:
                row[model+"_im"] = category_type_df['image_AUROC'].mean()
                row[model+"_pix"]= category_type_df['pixel_AUPRO'].mean()
            else:
                row[model+"_im"] = 0
                row[model+"_pix"]= 0

        rows.append(row)

        for category in categories:
            row = {'dataset':dataset, 'category_type':category}
            for model, model_df in models.items():
                category_type_df = model_df[model_df['category']==category]
                if len(category_type_df)>0:
                    row[model+"_im"] = category_type_df['image_AUROC'].mean()
                    row[model+"_pix"]= category_type_df['pixel_AUPRO'].mean()
                else:
                    row[model+"_im"] = 0
                    row[model+"_pix"]= 0
            rows.append(row)

        

        # for category in categories:
        #     category_type_df = df[df['category']==category]

        #     row = {'dataset':dataset, 'category_type':category}
        #     for ablation, spec in ablations.items():
        #         temp = category_type_df.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(spec[0], spec[1], spec[2], spec[3], spec[4]), engine="python")
        #         if len(temp)>0:
        #             row[ablation+"_im"] = temp['image_AUROC'].mean()
        #             row[ablation+"_pix"]= temp['pixel_AUPRO'].mean()
        #         else:
        #             row[ablation+"_im"] = 0
        #             row[ablation+"_pix"]= 0

        #     rows.append(row)

with open('processed_experiment_2.csv','w',newline="") as f:
    w = csv.DictWriter(f,fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    f.close()

