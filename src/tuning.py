# src/tuning.py
# Chứa hàm để thực hiện Grid Search thủ công với K-Fold Cross-Validation

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

# Import mô hình MLP của bạn
from .models.custom_mlp.custom_mlp import CustomMLPRegressor
from . import config

def manual_grid_search_mlp(X_train_full, y_train_full, param_grid, k_folds=3):
    """
    Thực hiện Grid Search thủ công cho CustomMLPRegressor sử dụng K-Fold.
    
    Args:
        X_train_full (np.array): Toàn bộ dữ liệu X_train (chưa scale)
        y_train_full (pd.Series or np.array): Toàn bộ dữ liệu y_train (chưa scale)
        param_grid (dict): Một dict chứa các siêu tham số cần thử.
        k_folds (int): Số lượng fold cho cross-validation.

    Returns:
        best_params (dict): Bộ tham số tốt nhất.
        best_score (float): Điểm MAE trung bình tốt nhất.
        results_df (pd.DataFrame): DataFrame chứa kết quả của tất cả các lần thử.
    """
    
    print("\n" + "="*50)
    print(f"BẮT ĐẦU GRID SEARCH THỦ CÔNG VỚI {k_folds}-FOLD CV")
    print("="*50)
    
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=config.RANDOM_STATE)
    grid = ParameterGrid(param_grid)
    
    results = []
    best_score = float('inf') # Vì dùng MAE (thấp hơn là tốt hơn)
    best_params = {}

    # Chuyển y_train_full sang numpy array
    y_train_full_np = y_train_full.to_numpy() if isinstance(y_train_full, pd.Series) else y_train_full
    
    total_fits = len(list(grid)) * k_folds
    fit_count = 0

    # Lặp qua mọi kết hợp tham số
    for params in grid:
        fold_scores = []
        print(f"\nĐang kiểm tra tham số: {params}")
        
        # Lặp qua từng K-Fold
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train_full)):
            fit_count += 1
            print(f"  [Fit {fit_count}/{total_fits}] Đang chạy Fold {fold_idx + 1}/{k_folds}...")
            
            # 1. Lấy dữ liệu cho fold này
            X_train_fold, X_val_fold = X_train_full[train_index], X_train_full[val_index]
            y_train_fold, y_val_fold = y_train_full_np[train_index], y_train_full_np[val_index]

            # 2. Scale (X và y) - Cực kỳ quan trọng: Fit CHỈ trên train_fold
            scaler_x = RobustScaler().fit(X_train_fold)
            X_train_fold_scaled = scaler_x.transform(X_train_fold)
            X_val_fold_scaled = scaler_x.transform(X_val_fold)
            
            scaler_y = RobustScaler().fit(y_train_fold.reshape(-1, 1))
            y_train_fold_scaled = scaler_y.transform(y_train_fold.reshape(-1, 1))
            
            # 3. Xây dựng kiến trúc (layer_dims)
            input_dim = X_train_fold_scaled.shape[1]
            hidden_dims = params.get('hidden_layers', (32, 16)) # Lấy từ grid, hoặc dùng default
            layer_dims = [input_dim] + list(hidden_dims) + [1]

            # 4. Khởi tạo và Fit mô hình
            mlp = CustomMLPRegressor(
                layer_dims=layer_dims,
                learning_rate=params.get('learning_rate', 0.01),
                batch_size=params.get('batch_size', 64), # Lấy batch_size từ params
                epochs=params.get('epochs', 1000), # Lấy epochs từ params
                random_state=config.RANDOM_STATE
            )
            
            # Fit MLP (nên có hàm fit hỗ trợ mini-batch như đã thảo luận)
            # Giả sử hàm fit của bạn hỗ trợ 'batch_size'
            if 'batch_size' in mlp.__init__.__code__.co_varnames:
                 mlp.fit(X_train_fold_scaled, y_train_fold_scaled, batch_size=params.get('batch_size', 64), verbose=False)
            else:
                 mlp.fit(X_train_fold_scaled, y_train_fold_scaled, verbose=False) # Dùng BGD nếu chưa implement mini-batch
            
            # 5. Đánh giá trên tập Validation của fold
            y_pred_scaled = mlp.predict(X_val_fold_scaled)
            # Chuyển đổi ngược về thang đo gốc
            y_pred_unscaled = scaler_y.inverse_transform(y_pred_scaled)
            
            # Tính MAE (hoặc R2 nếu muốn)
            score = mean_absolute_error(y_val_fold, y_pred_unscaled)
            fold_scores.append(score)
        
        # 6. Tính điểm trung bình cho bộ tham số này
        avg_score = np.mean(fold_scores)
        results.append({'params': params, 'mean_mae': avg_score})
        print(f"  -> MAE Trung bình: {avg_score:.4f}")
        
        # 7. Cập nhật kết quả tốt nhất
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
            print(f"  *** TÌM THẤY BỘ THAM SỐ TỐT NHẤT MỚI ***")

    print("\n" + "="*50)
    print("GRID SEARCH HOÀN TẤT")
    print(f"Điểm MAE tốt nhất: {best_score:.4f}")
    print(f"Bộ tham số tốt nhất: {best_params}")
    print("="*50)

    return best_params, best_score, pd.DataFrame(results)