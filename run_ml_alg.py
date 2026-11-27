import h5py
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from kneed import KneeLocator
from pathlib import Path


def cluster_dbs_data(raw_data, fold=None, clust_plots=False, nclusters=None):
    #
        # 'animal': animal,
        # 'day': day,
        # 'signal': sig,
        # 'rate_spikes': rate_spikes,
        # 'frequency_values': frequency_values,
        # 'error_rate': error_rate,
        # 'var_spikes': var_spikes,
        # 'matrix_rates': matrix_rates

    # possibility to provide a custom number of clusters

    to_cluster = []

    fold = Path("C:/Users/adifranc/OneDrive - UAB - The University of Alabama at Birmingham/Finger, Adam's files - DBS_ML_FinalProject") if fold is None else Path(fold)

    if nclusters == None:
    #elbow if nclusters not provided
        wcss = []
        K = range(1, 15)

        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
            kmeans.fit(to_cluster)
            wcss.append(kmeans.inertia_)
            
        if clust_plots:
            save_clust = fold / 'clust_plots'
            save_clust.mkdir(parents=True, exist_ok=True)

            plt.plot(K, wcss, marker='o')
            plt.xlabel('Number of clusters (k)')
            plt.ylabel('WCSS (Inertia)')
            plt.title('Elbow')
            plt.savefig(save_clust / 'Elbow.png', dpi=300, bbox_inches='tight')
            plt.savefig(save_clust / 'Elbow.eps', dpi=300, bbox_inches='tight', format='eps')
            plt.close()

        #clustering selecting the best amount of clusters automatically (with KneeLocator)
        knee = KneeLocator(K, wcss, curve='convex', direction='decreasing')
        nclusters = knee.elbow

    kmeans = KMeans(n_clusters=nclusters, n_init=20)
    km = kmeans.fit(to_cluster)
    labels = km.labels_
    
    if clust_plots:
        #units per cluster
        counts = np.bincount(labels)
        plt.bar(np.array([1,2,3,4]), counts)
        plt.xlabel('Cluster')
        plt.ylabel('n of neurons')
        plt.savefig(save_clust / 'n_per_clust.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_clust / 'n_per_clust.eps', dpi=300, bbox_inches='tight', format='eps')
        plt.close()
        #PCA plot
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(to_cluster)
        plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='tab10')
        plt.title('KMeans clusters (PCA projection)')
        plt.savefig(save_clust / 'PCA.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_clust / 'PCA.eps', dpi=300, bbox_inches='tight', format='eps')
        plt.close()
        #TSNE plot
        X_2d = TSNE(n_components=2, perplexity=30).fit_transform(to_cluster)
        plt.scatter(X_2d[:,0], X_2d[:,1], c=labels, cmap='tab10')
        plt.title('t-SNE clustering view')
        plt.savefig(save_clust / 'tSNE.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_clust / 'tSNE.eps', dpi=300, bbox_inches='tight', format='eps')
        plt.close()
        #heatmap
        order = np.argsort(labels)
        plt.imshow(to_cluster[order], aspect='auto', cmap='magma')
        plt.colorbar()
        plt.title('Units sorted by cluster')
        plt.xlabel('Frequency bins')
        plt.ylabel('Units')
        plt.savefig(save_clust / 'heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_clust / 'heatmap.eps', dpi=300, bbox_inches='tight', format='eps')
        plt.close()
        #tuning curves
        plt.figure(figsize=(10, 8))
        for k in range(nclusters):
            plt.plot(kmeans.cluster_centers_[k], label=f'Cluster {k}')
        plt.legend()
        plt.xticks(rotation=60, ha="right")
        plt.xlabel('Frequency bins')
        plt.title('Cluster tuning profiles')
        plt.savefig(save_clust / 'tuning_profiles.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_clust / 'tuning_profiles.eps', dpi=300, bbox_inches='tight', format='eps')        
        plt.close()
    
    return to_cluster, km
