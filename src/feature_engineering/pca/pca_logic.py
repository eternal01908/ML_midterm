# src/feature_engineering/pca/pca_logic.py
# Tích hợp code của đồng đội (PCA)
# Logic của đồng đội bạn là sử dụng sklearn.PCA.

from sklearn.decomposition import PCA
# Import tương đối 2 cấp
from .. import config 

def apply_pca(X_train, X_test, n_components=0.95):
    """
    Áp dụng PCA (từ pca.ipynb).
    n_components=0.95 nghĩa là giữ lại 95% phương sai.
    """
    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"PCA: Đã giảm từ {X_train.shape[1]} xuống {pca.n_components_} chiều.")
    return X_train_pca, X_test_pca