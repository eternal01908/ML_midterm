# src/models/knn/knn_model.py
# Tích hợp logic từ K-NN.ipynb
# Logic của đồng đội bạn là sử dụng sklearn.KNeighborsRegressor.

from sklearn.neighbors import KNeighborsRegressor
from ... import config # Import tương đối từ thư mục gốc src/

def get_knn(n_neighbors=5):
    """
    Trả về mô hình KNN Regressor (từ K-NN.ipynb).
    Sử dụng k=5 làm mặc định (như trong file của bạn).
    """
    return KNeighborsRegressor(n_neighbors=n_neighbors)