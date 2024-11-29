import pandas as pd

from matplotlib import rc
import matplotlib.pylab as plt

import csv
import seaborn as sns

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

header = ['dataset','category_type','window_size','image_AUROC','pixel_AUPRO'
]

df_1 = pd.read_csv("experiment_1.csv")
df_2 = pd.read_csv("experiment_3.csv")

rows = []
df_1['category_type'] = "2"
df_2['category_type'] = "2"
szs = {1: df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(True, 'bilinear', -1, 1, True)), 
       3: df_2[df_2['window_size']==3],
       5: df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(True, 'bilinear', -1, 5, True)),
       7: df_2[df_2['window_size']==7],
       9: df_2[df_2['window_size']==9],
       11: df_2[df_2['window_size']==11],
       13: df_2[df_2['window_size']==13],
       15: df_2[df_2['window_size']==15]}
for dataset, dataset_categories in datasets.items():
    for category_type, categories in dataset_categories.items():
        for category in categories:
            for window_size, window_df in szs.items():
                row = {'dataset':dataset, 'category_type':category, 'window_size':window_size}
                category_type_df = window_df[window_df['category']==category]
                if len(category_type_df)>0:
                    row['image_AUROC'] = category_type_df['image_AUROC'].mean()
                    row['pixel_AUPRO']= category_type_df['pixel_AUPRO'].mean()
                else:
                    row['image_AUROC'] = 0
                    row['pixel_AUPRO'] = 0
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

with open('processed_experiment_3.csv','w',newline="") as f:
    w = csv.DictWriter(f,fieldnames=header)
    w.writeheader()
    w.writerows(rows)
    f.close()




rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

df = pd.read_csv("processed_experiment_4.csv")

new_labels = ['MVTec AD', 'VisA','mean']

metrics = {
    "image_AUROC": "Anomaly Detection (AUROC)",
    "pixel_AUPRO": "Anomaly localization (AUPRO)",
}

figure = plt.figure()
figure.set_figheight(5)
figure.set_figwidth(15)

def specs(x, **kwargs):
    plt.axhline(x.mean(), c='k', ls='-', lw=2.5, label="mean")

count = 1
for label, name in metrics.items():
    ax = figure.add_subplot(1,len(metrics),count)
    # ax.set_aspect(2)
    count+=1
    g = sns.lineplot(
        data=df,
        x="window_size", 
        y=label, 
        hue="dataset",
        legend=True,
        palette="Set2",
        ax=ax,
        errorbar="se",
        marker='o',
        markersize='3'
    )

    a = df.groupby('window_size')[label].mean()
    g.plot(a,c='k', ls='-', lw=2.5, label='mean')
    g.set_xlabel("Window size")
    g.set_ylabel(name)
    if label != "latency":
        ax.set_ylim([0.5,1])
    ax.legend_.set_visible(False)
    ax.set_xticks(range(1,19, 2),
                  labels=range(1,19, 2))
    
lines_labels = [ax.get_legend_handles_labels() for ax in figure.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
figure.legend(lines, new_labels, title="Dataset", ncols=len(new_labels), loc="upper center")#bbox_to_anchor=(0.8, 1))
# plt.tight_layout()
plt.savefig("scores_window_size.png")
plt.show()
