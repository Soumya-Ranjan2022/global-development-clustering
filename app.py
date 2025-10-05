# clustering_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

st.set_page_config(layout="wide", page_title="Country Clustering App")

st.title("World Development — Clustering App")
st.markdown("Load your cleaned dataset (scaled features) or PCA-transformed data and run clustering models interactively.")

# -----------------------
# Sidebar: Data selection
# -----------------------
st.sidebar.header("Data input")

use_example = st.sidebar.checkbox("Use example files from disk (clean_df.csv / pca_data.csv)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload a CSV file (cleaned scaled data or PCA data)", type=["csv"])

if use_example:
    try:
        df_full = pd.read_csv("clean_df.csv")  # scaled/cleaned data with Country column
    except Exception:
        st.sidebar.error("Example file 'clean_df.csv' not found in working directory.")
        df_full = None
else:
    if uploaded_file is not None:
        df_full = pd.read_csv(uploaded_file)
    else:
        df_full = None

if df_full is None:
    st.info("Please upload a CSV file (cleaned scaled data or PCA data) or enable example files.")
    st.stop()

st.write("Data preview:")
st.dataframe(df_full.head())

# Detect if Country column present
has_country = 'Country' in df_full.columns

# Let user choose whether this is PCA data or feature-scaled data
data_type = st.sidebar.selectbox("Type of data in file", ["Scaled features (original)", "PCA-transformed"])

# If scaled features, we need to prepare a 2D view for plotting (PCA or first two cols)
if data_type == "Scaled features (original)":
    # allow user to choose columns to use for clustering (default: all numeric)
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
    # remove possible index-like columns
    st.sidebar.markdown("Select numeric columns to use for clustering")
    selected_features = st.sidebar.multiselect("Features", numeric_cols, default=numeric_cols)
    X = df_full[selected_features].values
    plot_x = selected_features[0] if len(selected_features) >= 1 else None
    plot_y = selected_features[1] if len(selected_features) >= 2 else None
else:
    # assume PCA data – try detect PC columns or use first two numeric columns
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
    # detect PC names like 'PC1','PC2' or 'PC_1'
    pc_candidates = [c for c in numeric_cols if ('PC' in c.upper()) or (c.lower().startswith('pc'))]
    if len(pc_candidates) >= 2:
        plot_x, plot_y = pc_candidates[0], pc_candidates[1]
    else:
        # fallback to first two numeric columns
        plot_x, plot_y = (numeric_cols[0], numeric_cols[1]) if len(numeric_cols) >= 2 else (None, None)
    selected_features = numeric_cols
    X = df_full[selected_features].values

st.sidebar.markdown("---")

# -----------------------
# Sidebar: choose model
# -----------------------
st.sidebar.header("Clustering model")
model_choice = st.sidebar.selectbox("Algorithm", ["KMeans", "DBSCAN", "Hierarchical (Agglomerative)"])

# KMeans params
if model_choice == "KMeans":
    st.sidebar.markdown("KMeans parameters")
    k_choice = st.sidebar.radio("Select k", [3, 4], index=0)
    init_choice = st.sidebar.selectbox("Init method", ["k-means++", "random"], index=0)
    max_iter = st.sidebar.number_input("Max iterations", min_value=100, max_value=2000, value=300, step=50)

# DBSCAN params
if model_choice == "DBSCAN":
    st.sidebar.markdown("DBSCAN parameters")
    eps = st.sidebar.slider("eps (neighborhood radius)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    min_samples = st.sidebar.slider("min_samples", min_value=2, max_value=50, value=5, step=1)

# Agglomerative params
if model_choice == "Hierarchical (Agglomerative)":
    st.sidebar.markdown("Agglomerative parameters")
    n_clusters_h = st.sidebar.slider("n_clusters", min_value=2, max_value=10, value=3, step=1)
    linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"], index=0)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("Run clustering")

# -----------------------
# Run clustering
# -----------------------
if run_btn:
    # Fit model and produce labels
    try:
        if model_choice == "KMeans":
            km = KMeans(n_clusters=k_choice, random_state=42, init=init_choice, max_iter=max_iter)
            labels = km.fit_predict(X)
            model_info = km
        elif model_choice == "DBSCAN":
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            model_info = db
        else:  # Hierarchical
            agg = AgglomerativeClustering(n_clusters=n_clusters_h, linkage=linkage)
            labels = agg.fit_predict(X)
            model_info = agg
    except Exception as e:
        st.error(f"Error fitting model: {e}")
        st.stop()

    # attach labels to dataframe (use new copy to avoid changing original)
    df_out = df_full.copy()
    label_col = f"{model_choice}_cluster"
    # if KMeans and user tried both k values, name accordingly
    if model_choice == "KMeans":
        label_col = f"KMeans_k{str(k_choice)}"
    df_out[label_col] = labels

    # Show cluster counts
    st.subheader("Cluster counts")
    counts = df_out[label_col].value_counts().sort_index()
    st.write(counts)

    # Compute metrics if sensible
    n_unique = len(set(labels)) - (1 if -1 in labels else 0)  # -1 is DBSCAN noise
    if n_unique >= 2:
        try:
            sil = silhouette_score(X, labels)
            dbi = davies_bouldin_score(X, labels)
        except Exception as e:
            sil, dbi = None, None
            st.warning(f"Could not compute some metrics: {e}")
    else:
        sil, dbi = None, None

    st.subheader("Clustering performance")
    st.write(f"Silhouette Score: {sil if sil is not None else 'N/A (requires >=2 clusters)'}")
    st.write(f"Davies-Bouldin Index: {dbi if dbi is not None else 'N/A (requires >=2 clusters)'}")

    # -----------------------
    # Cluster profiling (means) on original features (if available)
    # -----------------------
    st.subheader("Cluster profiling (feature means)")
    # if original dataset had Country separated and contained original features, show cluster means for numeric cols
    numeric_profile_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
    profile = df_out.groupby(label_col)[numeric_profile_cols].mean().round(4)
    st.dataframe(profile)

    # -----------------------
    # Top countries per cluster (if Country exists)
    # -----------------------
    st.subheader("Top countries per cluster (by selected feature)")
    if not has_country:
        st.warning("No 'Country' column found in the dataset — top countries list not available.")
    else:
        # let user choose a feature to examine (must be numeric)
        feat_for_top = st.selectbox("Feature used to list top countries in each cluster", numeric_profile_cols, index=0)
        # show top 10 countries per cluster
        top_n = st.slider("Number of top countries to show", min_value=3, max_value=30, value=10)
        for cl in sorted(df_out[label_col].unique()):
            st.markdown(f"**Cluster {cl}**")
            sub = df_out[df_out[label_col] == cl]
            # if DBSCAN noise cluster (-1), mention it
            if sub.shape[0] == 0:
                st.write("No points in this cluster.")
                continue
            top_countries = sub.sort_values(feat_for_top, ascending=False).loc[:, ["Country", feat_for_top]].head(top_n)
            st.table(top_countries)

    # -----------------------
    # Visualization: 2D scatter (using selected two columns or PCA first two PCs)
    # -----------------------
    st.subheader("2D cluster visualization")
    if plot_x is None or plot_y is None:
        st.info("Not enough numeric columns to make a 2D scatter plot.")
    else:
        fig = plt.scatter(df_out, x=plot_x, y=plot_y, color=label_col, hover_data=df_out.columns,
                         title=f"Clusters visualized on {plot_x} vs {plot_y}")
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Download CSV with cluster labels
    # -----------------------
    st.subheader("Download results")
    to_download = df_out.copy()
    csv = to_download.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download labeled CSV", data=csv, file_name="clustered_data.csv", mime='text/csv')

    st.success("Clustering run finished.")

# helpful tip
st.sidebar.markdown("---")
st.sidebar.info("Tip: if you used scaled (original) data and want a PCA-based scatter, you can compute PCA in a separate notebook and upload the PCA CSV here.")
