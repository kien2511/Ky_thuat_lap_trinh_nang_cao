from flask import Flask, render_template, request, jsonify
import numpy as np
import matplotlib
matplotlib.use('Agg') ## Sử dụng backend dạng non-GUI (không giao diện) để render ảnh
import matplotlib.pyplot as plt
import io
import base64
from ltnc import KMeans, generate_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_kmeans():
    try:
        data = request.json
        n_samples = int(data.get('n_samples', 300))
        n_clusters = int(data.get('n_clusters', 3))
        cluster_std = float(data.get('cluster_std', 1.0))
        
        ## Giới hạn các tham số để đảm bảo an toàn cho máy chủ
        n_samples = max(10, min(n_samples, 2000))
        n_clusters = max(2, min(n_clusters, 10))

        ## Tạo dữ liệu ngẫu nhiên
        X = generate_data(n_samples=n_samples, centers=n_clusters, cluster_std=cluster_std)
        
        ## Chạy thuật toán K-Means
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(X)
        
        ## Vẽ biểu đồ
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ## Biểu đồ phân tán (scatter plot) hiển thị các điểm dữ liệu
        scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
        
        ## Biểu đồ phân tán hiển thị các tâm cụm (centroid)
        ax.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, marker='X', label='Tâm Cụm')
        
        ax.set_title('Kết quả phân chia bằng K-Means')
        ax.set_xlabel('Đặc trưng 1')
        ax.set_ylabel('Đặc trưng 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ## Lưu biểu đồ thành chuỗi base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return jsonify({'status': 'success', 'image': image_base64})
    
    except Exception as e:
        print(f"Lỗi: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
