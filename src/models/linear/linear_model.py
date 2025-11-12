# src/models/linear/linear_model.py
# Tích hợp logic từ linear.ipynb
# Logic của đồng đội bạn là sử dụng sklearn.LinearRegression.

from sklearn.linear_model import LinearRegression
from ... import config # Import tương đối từ thư mục gốc src/

def get_linear_regression(): 
    """
    Trả về mô hình Linear Regression (từ linear.ipynb).
    """
    # LinearRegression không có tham số 'random_state'
    return LinearRegression()