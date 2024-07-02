
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def get_data(data,label):
    # digits = datasets.load_digits(n_class=6)
    data = data
    label = label
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(figsize=(5,5))
    # ax = plt.subplot(111)

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []

    for i in range(len(label)):
        if label[i] == 1:
            type1_x.append(data[i][0])
            type1_y.append(data[i][1])

        if label[i] == 2:
            type2_x.append(data[i][0])
            type2_y.append(data[i][1])

        if label[i] == 3:
            type3_x.append(data[i][0])
            type3_y.append(data[i][1])

        if label[i] == 4:
            type4_x.append(data[i][0])
            type4_y.append(data[i][1])

        if label[i] == 5:
            type5_x.append(data[i][0])
            type5_y.append(data[i][1])

        if label[i] == 6:
            type6_x.append(data[i][0])
            type6_y.append(data[i][1])

    type1 = plt.scatter(type1_x, type1_y, s=5, c='#2874B2')
    type2 = plt.scatter(type2_x, type2_y, s=5, c='#F70000')
    type3 = plt.scatter(type3_x, type3_y, s=5, c='#468641')
    type4 = plt.scatter(type4_x, type4_y, s=5, c='#2E3234')
    type5 = plt.scatter(type5_x, type5_y, s=5, c='#EC7C25')
    type6 = plt.scatter(type6_x, type6_y, s=5, c='#83639D')


    plt.xlim()
    plt.ylim()

    fig.savefig('1.svg', format='svg')

    plt.legend((type1, type2, type3, type4, type5,type6), (u'0.5', u'1.4', u'2.3', u'3.2', u'4.1', u'5.0'))

    return fig


def main_tsne(data,label):
    data, label, n_samples, n_features = get_data(data,label)
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label)
    # plt.show(fig)
    fig.savefig('2.svg',format='svg')
    fig.savefig('2.png', format='png')


# if __name__ == '__main__':
#     main_tsne()
