import matplotlib.pyplot as plt
import numpy
import pickle
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AffinityPropagation, AgglomerativeClustering, \
    Birch, MeanShift
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import os
from collections import defaultdict

THRESHOLD = .5
run_list = ['data1', 'data2', 'data3']
NORM = True


def norm(data):
    return (data - numpy.mean(data, axis=1, keepdims=True)) / numpy.std(data, axis=1, keepdims=True)


def export_separate_ele(labels, name, threshold=THRESHOLD):
    """
    pay attention to -1 value in labels, which means the element belongs to no cluster
    """
    lab_count = defaultdict(int)
    for lab in labels:
        lab_count[lab] += 1

    mean_label_size = numpy.mean([v for k, v in lab_count.items() if k != -1])  # ignore -1 value
    separate_labels = dict(filter(lambda e: e[0] == -1 or e[1] < mean_label_size * threshold, lab_count.items()))
    separate_labels = list(separate_labels.keys())

    buf = []
    for lab in numpy.unique(labels):
        idx = numpy.where(lab == labels)[0] + 1
        assert f"class {int(lab)}" not in [t[0] for t in buf]
        buf.append((f"class {int(lab)}", idx.tolist()))

    buf.sort(key=lambda t: len(t[1]), reverse=False)
    print(f"Export to ./export/{name}.json")
    with open(f"./export/{name}.json", 'w') as f:
        json.dump(buf, f, ensure_ascii=False)


def draw(data, labels, name):
    data, labels = numpy.array(data), numpy.array(labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    plt.title(name)

    classes = numpy.unique(labels)
    colors = [plt.cm.Spectral(e) for e in numpy.linspace(0, 1, len(classes))]
    for cls, col in zip(classes, colors):
        mask = labels == cls
        x = data[mask][:, 0]
        y = data[mask][:, 1]

        if numpy.sum(mask) > 50:
            plt.plot(x, y, 'o',
                     label=f"class {cls}",
                     markersize=1,
                     color=tuple(col))
        else:
            plt.plot(x, y, 'o',
                     markersize=1,
                     color=tuple(col))

    plt.legend()
    plt.savefig(f'./results/{name}.png', bbox_inches='tight', dpi=500)
    plt.show()

    export_separate_ele(labels, name)


def save(name, data):
    data = numpy.array(data)
    with open(f"./saves/{name}.npy", 'wb') as f:
        numpy.save(f, data)
        print(f"Save to ./saves/{name}.npy")


def load(name):
    if os.path.exists(f"./saves/{name}.npy"):
        print(f"Load from ./saves/{name}.npy")
        with open(f"./saves/{name}.npy", 'rb') as f:
            data = numpy.load(f)
        return data
    else:
        if 'data1' in name:
            with open(f"./data/837427_usr2vec_LinearCluster_passenger_256.pkl", "rb") as f:
                data = pickle.load(f)  # dict, [11106, 256]
                data = [data[i + 1] for i in range(len(data))]
        elif 'data2' in name:
            with open(f"./data/feature(1).csv", "r") as f:
                data = pd.read_csv(f)  # df, 11106, 12
                data = data.sort_values('PPID')
                data = data.drop('PPID', axis=1)
                data = numpy.array(data)
                data[:, 0] = (data[:, 0] - numpy.mean(data[:, 0])) / numpy.std(data[:, 0])
        elif 'data3' in name:
            with open(f"./data/Embed/data_map.json", "r") as f:
                data = json.load(f)
                data = [data[str(i + 1)] for i in range(len(data))]
        else:
            raise ValueError()

        if NORM:
            data = norm(data)

        data = TSNE(n_components=2, init='pca', random_state=0).fit_transform(data)
        save(name, data)
        return data


class Cluster(object):
    @staticmethod
    def fit(method, data, **kwargs):
        if method == 'kmeans':
            return KMeans(n_clusters=kwargs['n_clusters']).fit(data).labels_
        elif method == 'dbscan':
            return DBSCAN(eps=kwargs['eps'], min_samples=1).fit(data).labels_
        elif method == 'optics':
            return OPTICS(min_samples=1).fit(data).labels_
        elif method == 'affinity':
            return AffinityPropagation(random_state=5).fit(data).labels_
        elif method == 'agglomerative':
            return AgglomerativeClustering(n_clusters=kwargs['n_clusters']).fit(data).labels_
        elif method == 'birch':
            return Birch(n_clusters=kwargs['n_clusters']).fit(data).labels_
        elif method == 'shift':
            return MeanShift(bandwidth=kwargs['bandwidth']).fit(data).labels_
        else:
            raise ValueError()


if __name__ == '__main__':
    os.makedirs(f"./results", exist_ok=True)
    record = pd.read_excel(f"PASSENGER_RECORD.xlsx")  # df, 985759, 16

    for name in run_list:
        if name == 'data1':
            name = name + "-norm" if NORM else name
            data = load(name)

            for i in range(5, 10):
                draw(data, Cluster.fit('kmeans', data, n_clusters=i), name=f'{name}-tsne-kmeans{i}')
            for i in range(1, 10):
                draw(data, Cluster.fit('dbscan', data, eps=i / 2), name=f'{name}-tsne-dbscan{i / 2}')
            draw(data, Cluster.fit('optics', data), name=f'data3-tsne-optics')
            for i in range(5, 15):
                draw(data, Cluster.fit('agglomerative', data, n_clusters=i), name=f'{name}-tsne-agglomerative{i}')
            for i in range(5, 20):
                draw(data, Cluster.fit('birch', data, n_clusters=i), name=f'{name}-tsne-birch{i}')

        elif name == 'data2':
            name = name + "-norm" if NORM else name
            data = load(name)

            for i in range(2, 8):
                draw(data, Cluster.fit('kmeans', data, n_clusters=i), name=f'{name}-tsne-kmeans{i}')
            for i in range(4, 10):
                draw(data, Cluster.fit('dbscan', data, eps=i / 2), name=f'{name}-tsne-dbscan{i / 2}')
            draw(data, Cluster.fit('optics', data), name=f'data3-tsne-optics')
            for i in range(2, 8):
                draw(data, Cluster.fit('agglomerative', data, n_clusters=i), name=f'{name}-tsne-agglomerative{i}')
            for i in range(2, 8):
                draw(data, Cluster.fit('birch', data, n_clusters=i), name=f'{name}-tsne-birch{i}')

        elif name == 'data3':
            name = name + "-norm" if NORM else name
            data = load(name)

            for i in range(5, 10):
                draw(data, Cluster.fit('kmeans', data, n_clusters=i), name=f'{name}-tsne-kmeans{i}')
            for i in range(1, 10):
                draw(data, Cluster.fit('dbscan', data, eps=i / 2), name=f'{name}-tsne-dbscan{i / 2}')
            draw(data, Cluster.fit('optics', data), name=f'data3-tsne-optics')
            for i in range(5, 20):
                draw(data, Cluster.fit('agglomerative', data, n_clusters=i), name=f'{name}-tsne-agglomerative{i}')
            for i in range(5, 20):
                draw(data, Cluster.fit('birch', data, n_clusters=i), name=f'{name}-tsne-birch{i}')
