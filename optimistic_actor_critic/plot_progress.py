import sys, os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="darkgrid")


def plot_error_band(axs, x_data, y_data, min, max, data_name, title=None, colour=None, error_band=False, x_label='Epoch', plot_convergence_epoch=False):

    axs.plot(x_data, y_data, color=colour, alpha=1.0)

    if error_band:
        axs.fill_between(x_data, min, max, color=colour, alpha=0.25)

    if plot_convergence_epoch:
        thresh_value = y_data.max() * 0.95
        convergence_epoch = np.argmax(y_data > thresh_value)
        axs.plot([0, convergence_epoch], [y_data[convergence_epoch], y_data[convergence_epoch]], colour, '-', lw=1, dashes=[2, 2], label='_nolegend_') # ref line
        axs.plot([convergence_epoch, convergence_epoch], [0, y_data[convergence_epoch]], colour, '-', lw=1, dashes=[2, 2], label='_nolegend_') # ref line
        print('Convergence value: ', y_data[convergence_epoch])
        print('Convergence epoch: ', convergence_epoch)

    axs.set(xlabel=x_label, ylabel=data_name)
    axs.set_title(title)

    for item in ([axs.title]):
        item.set_fontsize(20)

    for item in ([axs.xaxis.label, axs.yaxis.label]):
        item.set_fontsize(16)

    for item in  axs.get_xticklabels() + axs.get_yticklabels():
        item.set_fontsize(12)

def plot_progress(progess_file, show_plot=True):
    fig, axs = plt.subplots(1, 2, figsize=(18,6))

    data = pd.read_csv(progess_file, sep="\t")
    data_len = len(data)

    plot_error_band(axs[0], data['Epoch'], data['AverageEpRet'],        data['MinEpRet'],     data['MaxEpRet'],     'Episode Return',          colour='r' )
    plot_error_band(axs[1], data['Epoch'], data['AverageTestEpRet'],    data['MinTestEpRet'], data['MaxTestEpRet'], 'Test Episode Return',     colour='b' )

    if show_plot:
        plt.show()
    fig.savefig(os.path.join(os.path.dirname(progess_file), 'training_curves.png'), dpi=320, pad_inches=0.01, bbox_inches='tight')

if __name__ == '__main__':
    progess_file = 'saved_models/oac_disc_CartPole-v1/oac_disc_CartPole-v1_s2/progress.txt'
    plot_progress(progess_file)
