# src/data_processing.py
# Nhiệm vụ 1: Tiền xử lý dữ liệu (Phiên bản Hồi quy)

import pandas as pd
import numpy as np
import kagglehub
import os
from . import config # Sửa import: Dùng import tương đối
import pandas.api.types as ptypes

def _time_string_to_seconds(time_str):
    """Hàm nội bộ: Chuyển đổi 'HH:MM:SS' hoặc 'MM:SS' thành giây."""
    if isinstance(time_str, str):
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = map(int, parts)
            return m * 60 + s
    # Trả về time_str nếu nó đã là số (hoặc NaN)
    return pd.to_numeric(time_str, errors='coerce')

def load_and_preprocess_data():
    """
    Hàm chính: Tải, làm sạch, và chuẩn bị dữ liệu cho mô hình HỒI QUY.
    """
    print("--- Bắt đầu Giai đoạn 1: Tiền xử lý Dữ liệu (Hồi quy) ---")
    
    # 1. & 2. Tải và Gộp (Concat) 3 file CSV
    # (Bỏ qua logic tải từ Kaggle, giả sử file đã có trong data/raw/)
    files_to_load = ['Table-data-2018.csv', 'Table-data-2019.csv', 'Table-data-2020.csv']
    df_list = []
    print(f"Đang tải và gộp các file từ: {config.RAW_DATA_PATH}")

    for f_name in files_to_load:
        file_path = os.path.join(config.RAW_DATA_PATH, f_name)
        if not os.path.exists(file_path):
            # Nếu file không có, thử chạy logic tải file dataset.csv
            file_path = os.path.join(config.RAW_DATA_PATH, 'dataset.csv')
            if os.path.exists(file_path):
                print(f"Đang tải file dự phòng: {file_path}")
                df = pd.read_csv(file_path)
                df_list = [df] # Đặt nó làm df duy nhất
                break # Thoát vòng lặp
            else:
                print(f"LỖI: Không tìm thấy file {f_name} hoặc dataset.csv. Dừng.")
                return None
        
        # Logic [1:-1] này khớp với cấu trúc file Kaggle gốc
        df_list.append(pd.read_csv(file_path)[1:-1])
        
    # Gộp 3 DataFrame lại nếu chúng ta không dùng file dự phòng
    if len(df_list) > 1:
        df = pd.concat(df_list).reset_index(drop=True)
    elif len(df_list) == 1:
        df = df_list[0]
    else:
        print("LỖI: Không đọc được file CSV nào.")
        return None
        
    print("Gộp file thành công.")

    # 3. Sửa tên cột
    df.columns = df.columns.str.replace(' ', '_')
    col_map = {
        'Av­er­age_views_per_view­er': 'Average_views_per_viewer',
        'Unique_view­ers': 'Unique_viewers',
        'Av­er­age_per­cent­age_viewed_(%)': 'Average_viewed_(%)',
        'Im­pres­sions': 'Impressions', 'Dis\xadlikes': 'Dislikes',
        'Sub­scribers_lost': 'Subscribers_lost', 'Sub­scribers_gained': 'Subscribers_gained',
        'Videos_pub­lished': 'Videos_published_original',
        'Videos_ad­ded': 'Videos_added_original',
        'Sub­scribers': 'Subscribers_net_change',
        'Im­pres­sions_click-through_rate_(%)': 'Click_rate_(%)',
        'Com­ments_ad­ded': 'Comments', 'Watch_time_(hours)': 'Watch_hours',
        'Av­er­age_view_dur­a­tion': 'Average_view_sec',
        'Your_es­tim­ated_rev­en­ue_(USD)': 'Revenue_(USD)',
        'Likes_(vs._dis­likes)_(%)': 'Likes_vs_Dislikes_(%)',
        'Videos_published': 'Videos_published_original', 'Videos_added': 'Videos_added_original',
        'Subscribers': 'Subscribers_net_change', 'Comments_added': 'Comments',
    }
    df = df.rename(columns=col_map)
    
    # 4. Xử lý NaN
    if 'Videos_added_original' in df.columns:
        df['Videos_added_original'] = df['Videos_added_original'].fillna(0)
    if 'Videos_published_original' in df.columns:
        df['Videos_published_original'] = df['Videos_published_original'].fillna(0)

    # 5. Chuyển đổi kiểu dữ liệu
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Kiểm tra xem cột có phải là object không TRƯỚC KHI convert
    if 'Average_view_sec' in df.columns and ptypes.is_object_dtype(df['Average_view_sec']):
        print("Phát hiện 'Average_view_sec' là string. Đang chuyển đổi sang giây...")
        df['Average_view_sec'] = df['Average_view_sec'].apply(_time_string_to_seconds)
    
    # Đảm bảo tất cả các cột số là numeric, điền NaN nếu có lỗi
    numeric_cols = df.select_dtypes(include='object').columns.drop('Date', errors='ignore')
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(0) # Điền 0 cho bất kỳ NaN nào còn sót lại

    # 6. Xử lý giá trị âm
    cols_to_clip = ['Likes', 'Dislikes', 'Views', 'Impressions', 'Comments', 'Shares', 
                    'Watch_hours', 'Unique_viewers', 'Average_view_sec']
    for col in cols_to_clip:
        if col in df.columns: 
            df[col] = df[col].clip(lower=0) 

    # 7. Kỹ thuật thuộc tính
    if 'Subscribers_net_change' in df.columns:
        df['Subs_accumulated'] = df['Subscribers_net_change'].cumsum()
    elif 'Subs_accumulated' not in df.columns:
         df['Subs_accumulated'] = 0 # Tạo cột nếu không có

    # 8. TẠO BIẾN MỤC TIÊU (Hồi quy)
    # *** BƯỚC 8 (PHÂN LOẠI) ĐÃ BỊ XÓA ***
    # Mục tiêu của chúng ta là 'Revenue_(USD)'

    # 9. Lựa chọn thuộc tính cuối cùng
    features_to_drop = [
        'Date', 
        # 'Revenue_(USD)' SẼ LÀ TARGET, KHÔNG PHẢI FEATURE
        'Subscribers_net_change', 
        'Likes_vs_Dislikes_(%)', 
        'Videos_added_original', # Bỏ theo kết quả Thí nghiệm 2
        'Videos_published_original' # Bỏ theo kết quả Thí nghiệm 2
    ]
    
    # Lấy tất cả các cột còn lại
    cols_to_keep = [col for col in df.columns if col not in features_to_drop]
    df_clean = df[cols_to_keep]
    
    # 10. Lưu dữ liệu đã xử lý
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    df_clean.to_csv(config.PROCESSED_DATA_PATH, index=False)
    
    print(f"Tiền xử lý hoàn tất. Dữ liệu sạch đã lưu tại: {config.PROCESSED_DATA_PATH}")
    print("--- Kết thúc Giai đoạn 1 ---")
    return df_clean

if __name__ == '__main__':
    # Cho phép chạy file này trực tiếp để kiểm tra
    # Cần thêm '..' vào sys.path để 'from src import config' hoạt động
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config # Import lại với đường dẫn đúng
    load_and_preprocess_data()