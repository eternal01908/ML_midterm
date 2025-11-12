# src/models/custom_mlp/custom_mlp.py
# Nhiệm vụ 2: Xây dựng MLPRegressor (Hồi quy) từ công thức toán học.

import numpy as np

# --- Các hàm kích hoạt (Activation Functions) và đạo hàm ---

# ReLU vẫn giữ nguyên
def relu(Z):
    """Hàm kích hoạt ReLU: max(0, Z)"""
    return np.maximum(0, Z)

def relu_derivative(Z):
    """Đạo hàm của ReLU: 1 nếu Z > 0, 0 nếu không"""
    return (Z > 0).astype(float)

# Lớp Output không dùng Sigmoid nữa, dùng Linear (không kích hoạt)
def linear(Z):
    """Hàm kích hoạt tuyến tính (không làm gì cả)"""
    return Z

def linear_derivative(Z):
    """Đạo hàm của hàm tuyến tính (luôn là 1)"""
    return np.ones_like(Z)

# --- Lớp MLPRegressor ---

class CustomMLPRegressor:
    """
    MLPRegressor (Multi-Layer Perceptron) cho bài toán Hồi quy.
    - Lớp ẩn (Hidden Layers): ReLU
    - Lớp đầu ra (Output Layer): Linear
    - Hàm mất mát (Loss Function): Mean Squared Error (MSE)
    - Tối ưu hóa (Optimizer): Batch Gradient Descent
    """
    
    def __init__(self, layer_dims, learning_rate=0.01, epochs=1000, batch_size=64, beta=0.9, random_state=42):
        """
        Khởi tạo MLP Regressor.
        
        Args:
            layer_dims (list): Kích thước của mỗi lớp. 
                               VD: [input_dim, 64, 32, 1]
            learning_rate (float): Tốc độ học (alpha).
            epochs (int): Số lần lặp qua toàn bộ tập dữ liệu.
            batch_size (int): Kích thước của mỗi mẻ (mini-batch).
            beta (float): Hệ số momentum (thường là 0.9).
            random_state (int): Để đảm bảo kết quả có thể tái lập.
        """
        np.random.seed(random_state)
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.beta = beta
        
        self.parameters = {} # Nơi lưu W (trọng số) và b (bias)
        self.v = {}          # Nơi lưu "vận tốc" (velocity) cho momentum
        self.costs = []      # Lưu trữ loss qua mỗi epoch
        
        # --- Khởi tạo Trọng số và Vận tốc ---
        for l in range(1, len(self.layer_dims)):
            # Khởi tạo Xavier/Glorot (tốt cho cả ReLU và Linear)
            input_dim_l_minus_1 = self.layer_dims[l-1]
            output_dim_l = self.layer_dims[l]
            
            self.parameters[f'W{l}'] = np.random.randn(input_dim_l_minus_1, output_dim_l) * np.sqrt(1. / input_dim_l_minus_1)
            self.parameters[f'b{l}'] = np.zeros((1, output_dim_l))
            
            # Khởi tạo vận tốc (velocity) cho momentum, tất cả bằng 0
            self.v[f'dW{l}'] = np.zeros_like(self.parameters[f'W{l}'])
            self.v[f'db{l}'] = np.zeros_like(self.parameters[f'b{l}'])

    def _compute_loss(self, A_last, Y):
        """Tính hàm mất mát Mean Squared Error (MSE)"""
        m = Y.shape[0]
        # Công thức MSE: 1/m * sum((A_last - Y)^2)
        # Chúng ta dùng 1/(2*m) để đạo hàm đẹp hơn (hủy số 2)
        cost = (1 / (2 * m)) * np.sum(np.square(A_last - Y))
        return np.squeeze(cost)

    def _feedforward(self, X):
        """Lan truyền tiến"""
        cache = {'A0': X}
        A = X
        L = len(self.layer_dims) - 1
        
        # Lớp ẩn (ReLU)
        for l in range(1, L):
            Z = np.dot(A, self.parameters[f'W{l}']) + self.parameters[f'b{l}']
            A = relu(Z)
            cache[f'Z{l}'] = Z
            cache[f'A{l}'] = A
        
        # Lớp output (Linear)
        ZL = np.dot(A, self.parameters[f'W{L}']) + self.parameters[f'b{L}']
        A_last = linear(ZL) # Không kích hoạt
        cache[f'Z{L}'] = ZL
        cache[f'A{L}'] = A_last
        
        return A_last, cache

    def _backpropagate(self, A_last, Y, cache):
        """Lan truyền ngược cho Hồi quy (MSE)"""
        gradients = {}
        m = Y.shape[0]
        L = len(self.layer_dims) - 1
        
        # Đảm bảo Y có cùng shape với A_last
        Y = Y.reshape(A_last.shape)

        # ---- Lớp Output (L) (Linear + MSE) ----
        # dL/dA_L = (A_L - Y)
        # dA_L/dZ_L = g'(Z_L) = 1 (vì là hàm linear)
        # dL/dZ_L = dL/dA_L * dA_L/dZ_L = (A_L - Y) * 1
        dZ_last = (A_last - Y) 
        
        A_prev = cache[f'A{L-1}']
        
        gradients[f'dW{L}'] = (1/m) * np.dot(A_prev.T, dZ_last)
        gradients[f'db{L}'] = (1/m) * np.sum(dZ_last, axis=0, keepdims=True)
        dA_prev = np.dot(dZ_last, self.parameters[f'W{L}'].T)

        # ---- Lớp Ẩn (L-1 đến 1) (ReLU) ----
        for l in reversed(range(1, L)):
            Z = cache[f'Z{l}']
            # dL/dZ_l = dA_prev * g'(Z_l)
            dZ = dA_prev * relu_derivative(Z)
            A_prev = cache[f'A{l-1}']
            
            gradients[f'dW{l}'] = (1/m) * np.dot(A_prev.T, dZ)
            gradients[f'db{l}'] = (1/m) * np.sum(dZ, axis=0, keepdims=True)
            
            if l > 1:
                dA_prev = np.dot(dZ, self.parameters[f'W{l}'].T)
            
        return gradients

    def _update_weights(self, gradients):
        """Cập nhật trọng số bằng Gradient Descent VỚI MOMENTUM"""
        for l in range(1, len(self.layer_dims)):
            # Tính vận tốc mới (v = beta*v + (1-beta)*gradient)
            self.v[f'dW{l}'] = self.beta * self.v[f'dW{l}'] + (1 - self.beta) * gradients[f'dW{l}']
            self.v[f'db{l}'] = self.beta * self.v[f'db{l}'] + (1 - self.beta) * gradients[f'db{l}']

            # Cập nhật trọng số (W = W - alpha * v)
            self.parameters[f'W{l}'] -= self.learning_rate * self.v[f'dW{l}']
            self.parameters[f'b{l}'] -= self.learning_rate * self.v[f'db{l}']
            
    def fit(self, X, Y, batch_size=None, verbose=True):
        """
        Huấn luyện mô hình - ĐÃ CẬP NHẬT LÊN MINI-BATCH SGD
        """
        if batch_size is None:
            batch_size = self.batch_size # Lấy batch_size từ __init__
            
        print(f"Bắt đầu huấn luyện Custom MLP Regressor với {self.epochs} epochs (batch_size={batch_size})...")
        self.costs = []
        m = X.shape[0] 

        for i in range(self.epochs):
            epoch_cost = 0.0
            
            # Xáo trộn dữ liệu
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            Y_shuffled = Y[permutation]

            # Lặp qua các Mini-Batch
            for j in range(0, m, batch_size):
                X_batch = X_shuffled[j : j + batch_size]
                Y_batch = Y_shuffled[j : j + batch_size]
                
                A_last, cache = self._feedforward(X_batch)
                
                # Sửa lỗi: Đảm bảo Y_batch có shape đúng cho _compute_loss
                Y_batch_reshaped = Y_batch.reshape(A_last.shape)
                cost = self._compute_loss(A_last, Y_batch_reshaped)
                epoch_cost += cost * X_batch.shape[0] 
                
                # Sửa lỗi: Đảm bảo Y_batch có shape đúng cho _backpropagate
                gradients = self._backpropagate(A_last, Y_batch_reshaped, cache)
                
                self._update_weights(gradients)
            
            avg_cost = epoch_cost / m
            self.costs.append(avg_cost)
            
            if verbose and (i % 100 == 0 or i == self.epochs - 1):
                print(f"Epoch {i+1}/{self.epochs}, Cost (MSE): {avg_cost:.6f}")
                
        if verbose:
            print("Huấn luyện hoàn tất.")

    def predict(self, X):
        """Dự đoán giá trị hồi quy"""
        A_last, _ = self._feedforward(X)
        return A_last
