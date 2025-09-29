import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np


def data_overview(df):
    """Show target distribution, missing values, and summary statistics."""
    print("Target value distribution (normalized):")
    display(df['target'].value_counts(normalize=True))
    print("Missing values per column:")
    display(df.isna().sum().sort_values(ascending=False))
    print("Summary statistics:")
    display(df.describe())

def numeric_feature_distributions(df):
    """Plot distributions of all numeric features by target."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            plt.figure(figsize=(6,4))
            sns.histplot(data=df, x=col, hue="target", kde=False)
            plt.title(f"Distribution of {col} by target")
            plt.show()

def correlation_and_feature_importance(df):
    """Show correlations and mutual information scores for numeric features."""
    num_df = df.select_dtypes(include="number")
    print("Correlation of numeric features with target:")
    display(num_df.corr()['target'].sort_values())
    mi = mutual_info_classif(num_df.drop(columns="target"), num_df["target"], random_state=42)
    mi_scores = pd.Series(mi, index=num_df.drop(columns="target").columns, name="MI_Score").sort_values(ascending=False)
    print("Mutual information scores for numeric features:")
    display(mi_scores)
    plt.figure(figsize=(10,8))
    sns.heatmap(num_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation matrix of numeric features")
    plt.show()

def pca_analysis(df):
    """Perform PCA and show reconstruction error and explained variance for 2-14 components. Also plot 2D PCA visualization."""
    X = df.drop(columns="target").select_dtypes("number")
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    for i in range(2,15):
        pca = PCA(n_components=i, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        if i == 2:
            plt.figure(figsize=(8,6))
            plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="coolwarm", alpha=0.6, edgecolor="k")
            plt.xlabel("PCA 1")
            plt.ylabel("PCA 2")
            plt.title("PCA (2 components) visualization")
            plt.colorbar(label="target")
            plt.show()
        X_reconstructed = pca.inverse_transform(X_pca)
        X_reconstructed = scaler.inverse_transform(X_reconstructed)
        X_reconstructed = pd.DataFrame(X_reconstructed, columns=X.columns)
        reconstruction_error = np.mean((X.values - X_reconstructed.values)**2)
        print(f"For {i} components:")
        print("Reconstruction MSE:", reconstruction_error)
        print("Explained variance ratio:", pca.explained_variance_ratio_)
        print("Cumulative variance explained:", np.sum(pca.explained_variance_ratio_))

def tsne_analysis(df):
    """Perform t-SNE and plot 2D and 3D embeddings."""
    X = df.drop(columns="target").select_dtypes("number")
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
    X_embedded_3d = TSNE(n_components=3, random_state=42).fit_transform(X_scaled)
    tsne2d_df = pd.DataFrame(X_embedded, columns=["TSNE1", "TSNE2"])
    tsne2d_df["target"] = y.values
    tsne3d_df = pd.DataFrame(X_embedded_3d, columns=["TSNE1", "TSNE2", "TSNE3"])
    tsne3d_df["target"] = y.values
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=tsne2d_df, x="TSNE1", y="TSNE2", hue="target", alpha=0.7, palette="coolwarm")
    plt.title("t-SNE 2D embedding")
    plt.show()
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        tsne3d_df["TSNE1"], tsne3d_df["TSNE2"], tsne3d_df["TSNE3"],
        c=tsne3d_df["target"], cmap="coolwarm", alpha=0.7
    )
    ax.set_title("t-SNE 3D embedding")
    ax.set_xlabel("TSNE1")
    ax.set_ylabel("TSNE2")
    ax.set_zlabel("TSNE3")
    fig.colorbar(scatter, label="target")
    plt.show()
