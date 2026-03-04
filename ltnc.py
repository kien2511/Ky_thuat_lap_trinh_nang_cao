## import thư viện
import numpy as np
import matplotlib.pyplot as plt
import random

## Khởi tạo class KMeans
class KMeans:
    ## Khởi tạo tham số
    def __init__(self, n_clusters=3, max_iter=100, tol=0.0001):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    ## Khởi tạo tâm cụm (Initialization) và huấn luyện mô hình
    def fit(self, X):
        ## 1. Khởi tạo Tâm Cụm (Chọn ngẫu nhiên k điểm dữ liệu)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[indices]
        
        ## Vòng lặp tối đa
        for i in range(self.max_iter):
            ## 2. Gán các điểm vào các tâm cụm gần nhất
            ## Tính khoảng cách từ mỗi điểm đến từng tâm cụm
            ## Ma trận dữ liệu X: chiều (số lượng mẫu, số lượng đặc trưng)
            ## Ma trận tâm cụm (centroids): chiều (k, số lượng đặc trưng)
            ## Mục tiêu là tìm ma trận khoảng cách: chiều (số lượng mẫu, k)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            
            ## Gán mỗi điểm vào cụm gần nhất
            labels = np.argmin(distances, axis=1)
            
            ## 3. Cập nhật tâm cụm bằng trung bình cộng tọa độ các điểm thuộc cụm
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            ## Kiểm tra hội tụ (Khi các tâm cụm ổn định, dịch chuyển ít hơn ngưỡng cho phép)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                print(f"Hội tụ tại vòng lặp thứ {i}")
                self.centroids = new_centroids
                break
            
            self.centroids = new_centroids
        
        self.labels_ = labels
        return self

    ## Dự đoán dữ liệu mới bằng cách phân vào nhóm có tâm gần nhất
    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

## Hàm tự động sinh dữ liệu thực nghiệm
def generate_data(n_samples=300, centers=3, cluster_std=1.0):
    ## Logic đơn giản tạo các cụm khi không có thư viện sklearn
    data = []
    
    ## Định nghĩa tọa độ ngẫu nhiên của một số tâm cụm
    blob_centers = [
        (2, 2),
        (8, 3),
        (5, 8)
    ]
    
    ## Nếu số tượng tâm cụm yêu cầu lớn hơn số có sẵn, thì sinh ngẫu nhiên thêm
    if centers > len(blob_centers):
        blob_centers = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(centers)]
    
    X = []
    y = []
    
    samples_per_blob = n_samples // centers
    
    for idx, (cx, cy) in enumerate(blob_centers):
        ## Sinh các tọa độ điểm xung quanh mỗi tâm sử dụng phân phối Gaussian
        bx = np.random.normal(cx, cluster_std, samples_per_blob)
        by = np.random.normal(cy, cluster_std, samples_per_blob)
        
        ## Ghép tọa độ x và y thành 1 điểm
        blob_data = np.column_stack((bx, by))
        X.append(blob_data)
        y.extend([idx] * samples_per_blob)
        
    return np.vstack(X)

## Hàm chính chạy chương trình
def main():
    print("Đang tạo số liệu giả lập...")
    X = generate_data(n_samples=300, centers=3, cluster_std=0.8)

    print("Đang chạy thuật toán K-Means...")
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print("Đang vẽ biểu đồ kết quả...")
    plt.figure(figsize=(10, 6))
    
    ## Vẽ điểm dữ liệu có cùng màu khi ở chung cụm phân loại
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6, label='Điểm dữ liệu')
    
    ## Vẽ các tâm cụm lên biểu đồ dưới dạng dấu X lớn màu đỏ
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, marker='X', label='Tâm Cụm')
    
    plt.title('Kết quả phân chia bằng K-Means Clustering')
    plt.xlabel('Trục Hoành (X)')
    plt.ylabel('Trục Tung (Y)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
