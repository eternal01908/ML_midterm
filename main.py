# main.py
# File thực thi chính (Phiên bản Hồi quy - ĐÃ SỬA LỖI)

import pandas as pd
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from src.tuning import manual_grid_search_mlp

# Thêm 'src' vào đường dẫn hệ thống
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    from src import config
    from src.data_processing import load_and_preprocess_data
    from src.evaluation import get_regression_metrics 
    
    # NHIỆM VỤ CỦA BẠN:
    from src.models.custom_mlp.custom_mlp import CustomMLPRegressor
    
    # NHIỆM VỤ CỦA ĐỒNG ĐỘI:
    from src.models.baseline_models import get_knn, get_linear_regression
    # SỬA LỖI 2: Xóa import LDA
    from src.feature_engineering import apply_pca
    
except ImportError as e:
    print(f"Lỗi Import: {e}")
    print("Hãy đảm bảo bạn đã tạo file __init__.py trong các thư mục src/ và src/models/")
    sys.exit(1)

def main():
    print("="*60)
    print("BẮT ĐẦU QUY TRÌNH SO SÁNH MÔ HÌNH (HỒI QUY)")
    print("="*60)
    
    # --- 1. TẢI DỮ LIỆU (Nhiệm vụ 1) ---
    if not os.path.exists(config.PROCESSED_DATA_PATH):
        print("File dữ liệu đã xử lý không tồn tại. Đang chạy tiền xử lý...")
        data = load_and_preprocess_data()
        if data is None:
            print("LỖI: Tiền xử lý thất bại. Đang dừng...")
            return
    else:
        print(f"Đang tải dữ liệu đã xử lý từ: {config.PROCESSED_DATA_PATH}")
        data = pd.read_csv(config.PROCESSED_DATA_PATH)

    # --- 2. CHIA DỮ LIỆU VÀ SCALING ---
    print("\n--- Chuẩn bị dữ liệu cho mô hình ---")
    
    target_col = 'Revenue_(USD)' 
    
    if target_col not in data.columns:
        print(f"LỖI: Không tìm thấy cột target '{target_col}' trong file đã xử lý.")
        return
        
    X = data.drop(columns=[target_col])
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE
    )
    
    # Scale dữ liệu X
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Scale dữ liệu y (Target)
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

    print("Chuẩn bị dữ liệu hoàn tất. X và y đã được scale.")

    # --- 3. ĐÁNH GIÁ CÁC MÔ HÌNH (Nhiệm vụ 3) ---
    
    # === 3.1 ĐÁNH GIÁ CUSTOM MLP (Nhiệm vụ 2 & 3) ===
    param_grid_mlp = {
        'learning_rate': [0.01, 0.005],
        'batch_size': [32, 64],
        'hidden_layers': [(32, 16), (64, 32)]
        # 'epochs': [1000] # Giữ cố định epochs
    }
    # Chạy Grid Search (Sử dụng X_train GỐC, chưa scale)
    # Hàm tuning sẽ tự xử lý scaling bên trong K-Folds
    best_params, best_score, tuning_results_df = manual_grid_search_mlp(
        X_train_scaled, # Dùng X_train đã scale từ trước (để tiết kiệm thời gian)
        y_train_scaled, # Dùng y_train đã scale từ trước
        param_grid_mlp, 
        k_folds=3 # Dùng 3-fold cho nhanh
    )
    
    print("\n--- Báo cáo kết quả Tuning ---")
    print(tuning_results_df.sort_values(by='mean_mae'))

    # --- Huấn luyện mô hình MLP TỐT NHẤT trên toàn bộ tập Train ---
    print("\n--- Đang huấn luyện: 1. Custom MLP TỐT NHẤT (Full Features) ---")
    
    # Xây dựng lại layer_dims tốt nhất
    best_hidden = best_params.get('hidden_layers', (32, 16))
    best_layer_dims = [X_train_scaled.shape[1]] + list(best_hidden) + [1]
    
    mlp_best = CustomMLPRegressor(
        layer_dims=best_layer_dims,
        learning_rate=best_params.get('learning_rate', 0.01),
        epochs=1500, # Dùng nhiều epochs hơn cho mô hình cuối
        random_state=config.RANDOM_STATE
    )
    
    # Fit trên toàn bộ X_train_scaled, y_train_scaled
    mlp_best.fit(
        X_train_scaled, 
        y_train_scaled, 
        batch_size=best_params.get('batch_size', 64), 
        verbose=True
    )
    
    # Đánh giá trên tập Test
    y_pred_scaled_mlp = mlp_best.predict(X_test_scaled)
    y_pred_mlp = scaler_y.inverse_transform(y_pred_scaled_mlp)
    get_regression_metrics(y_test, y_pred_mlp, title="Custom_MLP_Tuned_(Full_Features)")
    

    # === 3.2 ĐÁNH GIÁ CÁC MÔ HÌNH KHÁC ===
    
    print("\n--- Đang huấn luyện: 2. KNN (Full Features) ---")
    knn = get_knn()
    knn.fit(X_train_scaled, y_train) 
    y_pred_knn = knn.predict(X_test_scaled)
    get_regression_metrics(y_test, y_pred_knn, title="KNN_(Full_Features)")

    print("\n--- Đang huấn luyện: 3. Linear Regression (Full Features) ---")
    lin_reg = get_linear_regression()
    lin_reg.fit(X_train_scaled, y_train)
    y_pred_lin_reg = lin_reg.predict(X_test_scaled)
    get_regression_metrics(y_test, y_pred_lin_reg, title="Linear_Regression_(Full_Features)")

    # === 3.3 ĐÁNH GIÁ VỚI GIẢM CHIỀU (PCA) ===
    
    print("\n--- Áp dụng PCA và chạy lại mô hình ---")
    
    # *** SỬA LỖI 1: Thêm dòng gọi apply_pca ĐẦU TIÊN ***
    # Biến trả về được đặt tên là `_raw` để phân biệt
    X_train_pca_raw, X_test_pca_raw = apply_pca(X_train_scaled, X_test_scaled, n_components=0.95)
    
    print("Scaling đầu ra của PCA bằng RobustScaler...")
    pca_scaler = RobustScaler()
    X_train_pca = pca_scaler.fit_transform(X_train_pca_raw) # Bây giờ X_train_pca_raw đã tồn tại
    X_test_pca = pca_scaler.transform(X_test_pca_raw)
    
    print("\n--- Đang huấn luyện: 4. Custom MLP (PCA) ---")
    mlp_pca = CustomMLPRegressor(
        layer_dims=[X_train_pca.shape[1], 16, 1], # Lớp input nhỏ hơn
        learning_rate=0.01, epochs=1000, random_state=config.RANDOM_STATE
    )
    mlp_pca.fit(X_train_pca, y_train_scaled, verbose=False)
    
    y_pred_scaled_mlp_pca = mlp_pca.predict(X_test_pca)
    y_pred_mlp_pca = scaler_y.inverse_transform(y_pred_scaled_mlp_pca)
    get_regression_metrics(y_test, y_pred_mlp_pca, title="Custom_MLP_(PCA)")
    
    # === 3.4 ĐÁNH GIÁ VỚI LDA (BỊ LOẠI BỎ) ===
    
    # *** SỬA LỖI 2: Xóa (hoặc comment) toàn bộ khối LDA vì nó không dùng cho hồi quy ***
    print("\n--- Bỏ qua LDA (Không áp dụng cho bài toán Hồi quy) ---")
    
    # print("\n--- Áp dụng LDA và chạy lại mô hình ---")
    # X_train_lda_raw, X_test_lda_raw = apply_lda(X_train_scaled, X_test_scaled, y_train, n_components=1)
    # print("Scaling đầu ra của LDA bằng RobustScaler...")
    # lda_scaler = RobustScaler()
    # X_train_lda = lda_scaler.fit_transform(X_train_lda_raw)
    # X_test_lda = lda_scaler.transform(X_test_lda_raw)
    # print("\n--- Đang huấn luyện: 5. Custom MLP (LDA) ---")
    # mlp_lda = CustomMLPRegressor(...)
    # mlp_lda.fit(X_train_lda, y_train_scaled, verbose=False)
    # ... (Toàn bộ khối LDA đã bị vô hiệu hóa)
    
    print("="*60)
    print("TOÀN BỘ QUY TRÌNH SO SÁNH HOÀN TẤT.")
    print("="*60)

# --- Điểm vào của chương trình ---
if __name__ == "__main__":
    main()