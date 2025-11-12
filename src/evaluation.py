# src/evaluation.py
# Nhiệm vụ 3: Đánh giá mô hình (Phiên bản Hồi quy)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from . import config # Sửa import: Dùng import tương đối

def get_regression_metrics(y_true, y_pred, title="Model"):
    """
    Tính toán, in và lưu báo cáo hồi quy, vẽ biểu đồ phân tán.
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n" + "-"*30)
    print(f"KẾT QUẢ ĐÁNH GIÁ CHO: {title}")
    print(f"R-Squared (R²): {r2:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("-" * 30)

    # Vẽ và lưu biểu đồ Predicted vs Actual
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true.ravel(), y=y_pred.ravel(), alpha=0.5)
    
    # Vẽ đường 45 độ (hoàn hảo)
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # Tìm min
        np.max([plt.xlim(), plt.ylim()]),  # Tìm max
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0) # zorder=0 để nó nằm dưới
    
    plt.title(f'Predicted vs. Actual - {title}')
    plt.xlabel('Giá trị Thực tế (Actual Revenue)')
    plt.ylabel('Giá trị Dự đoán (Predicted Revenue)')
    plt.xlim(lims)
    plt.ylim(lims)
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    fig_path = os.path.join(config.FIGURES_DIR, f'scatter_{title.lower().replace(" ", "_")}.png')
    
    try:
        plt.savefig(fig_path)
        print(f"Đã lưu biểu đồ phân tán tại: {fig_path}")
    except Exception as e:
        print(f"Lỗi khi lưu hình ảnh: {e}")
        
    plt.show() # Hiển thị biểu đồ
    
    return {"R-Squared": r2, "MAE": mae, "RMSE": rmse}