from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from pyclustering.cluster.kmedoids import kmedoids
import json

app = Flask(__name__)

@app.route('/')
def index():
    # 1. LOAD DATA
    data = pd.read_excel("data_kesehatan.xlsx")
    provinsi = data["Provinsi"]
    X = data.drop(columns=["Provinsi"])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

    # 2. STANDARDIZATION & PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = sum(pca.explained_variance_ratio_) * 100

    # 3. EVALUASI SILHOUETTE & ELBOW METHOD
    sil_scores_eval = []
    wcss_eval = []
    K_range = list(range(2, 8))
    for k in K_range:
        kmed_temp = kmedoids(X_pca, list(range(k)))
        kmed_temp.process()
        clusters_temp = kmed_temp.get_clusters()
        medoids_temp = kmed_temp.get_medoids()
        
        labels_temp = np.zeros(len(X_pca))
        for idx, c_temp in enumerate(clusters_temp):
            labels_temp[c_temp] = idx
        sil_scores_eval.append(float(silhouette_score(X_pca, labels_temp)))
        
        total_distance = 0
        for i, cluster in enumerate(clusters_temp):
            cluster_points = X_pca[cluster]
            medoid = X_pca[medoids_temp[i]]
            distances = np.sum(np.sqrt(np.sum((cluster_points - medoid) ** 2, axis=1)))
            total_distance += float(distances)
        wcss_eval.append(total_distance)

    # 4. FINAL K-MEDOIDS (K=4)
    best_k = 4
    initial_medoids = list(range(best_k))
    kmed = kmedoids(X_pca, initial_medoids)
    kmed.process()
    clusters = kmed.get_clusters()
    medoids = kmed.get_medoids()

    labels = np.zeros(len(X_pca))
    for idx, cluster in enumerate(clusters):
        labels[cluster] = idx

    sil_score_final = silhouette_score(X_pca, labels)
    wcss_final = wcss_eval[K_range.index(best_k)]

    # 5. PERSIAPAN DATA UNTUK DIKIRIM KE WEBSITE
    table_data = []
    for i in range(len(provinsi)):
        table_data.append({
            "id": i + 1,
            "provinsi": provinsi[i],
            "cluster": int(labels[i]) + 1,
            "is_medoid": i in medoids
        })

    colors = ['#ef4444', '#06b6d4', '#10b981', '#f59e0b']
    scatter_datasets = []
    for c in range(best_k):
        scatter_datasets.append({
            "label": f"Cluster {c+1}",
            "data": [],
            "backgroundColor": colors[c],
            "pointRadius": 8
        })
    for i, point in enumerate(X_pca):
        cluster_idx = int(labels[i])
        scatter_datasets[cluster_idx]["data"].append({"x": float(point[0]), "y": float(point[1])})

    data_cluster = data.copy()
    data_cluster["Cluster"] = labels + 1
    profil_cluster = data_cluster.groupby("Cluster")[X.columns].mean().to_dict('index')
    
    bar_labels = list(X.columns)
    bar_datasets = []
    for c in range(1, best_k + 1):
        bar_datasets.append({
            "label": f"Cluster {c}",
            "data": [float(profil_cluster[c][col]) for col in bar_labels],
            "backgroundColor": colors[c-1]
        })

    return render_template('index.html', 
                           k=best_k, 
                           sil_score=round(sil_score_final, 3),
                           wcss_score=round(wcss_final, 2),
                           var_pca=round(explained_variance, 1),
                           total_prov=len(provinsi),
                           table_data=table_data,
                           eval_labels=json.dumps([f"K={k}" for k in K_range]),
                           eval_data=json.dumps(sil_scores_eval),
                           wcss_data=json.dumps(wcss_eval),
                           scatter_datasets=json.dumps(scatter_datasets),
                           bar_labels=json.dumps(bar_labels),
                           bar_datasets=json.dumps(bar_datasets))

if __name__ == '__main__':
    app.run(debug=True)