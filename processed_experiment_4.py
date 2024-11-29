import pandas as pd

from matplotlib import rc
import matplotlib.pylab as plt

import csv
import seaborn as sns

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

latency_df = pd.read_csv("experiment_4.csv")
df_1 = pd.read_csv("experiment_1.csv")
df_2 = pd.read_csv("experiment_2.csv")

rows = []
accuracy_dfs = {'spade': df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(False, 'nearest', 50, 1, False)), 
          'padim': df_2[df_2['model']=='padim'],
          'patchcore': df_2[df_2['model']=='patchcore'],
          'lwinnn': df_1.query('pooling=={} and `interpolation_mode`=="{}" and K=={} and window_size=={} and anomaly_map_detection=={}'.format(True, 'bilinear', -1, 5, True))}

header = ['dataset','category','model','image_AUROC','pixel_AUPRO', 'latency'
]
new_df = pd.DataFrame(columns=header)
for model, accuracy_df in accuracy_dfs.items():
    model_latency_df = latency_df[latency_df['model']==model]['latency']
    accuracy_df['latency'] = model_latency_df
    accuracy_df['model'] = model
    accuracy_df = accuracy_df[['dataset', 'category', 'model', 'image_AUROC', 'pixel_AUPRO', 'latency']]
    print(accuracy_df)
    new_df = pd.concat([new_df, accuracy_df])

new_df.to_csv("processed_experiment_4.csv")

new_labels = ['MVTec AD', 'VisA']

metrics = {
    "latency": "Latency",
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
