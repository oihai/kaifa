from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings("ignore")


class BuildingMaterialEnergyDataCleaner:
    def __init__(self):
        self.scaler = StandardScaler()
        self.original_data = None
        self.cleaned_data = None

    def load_data(self, data_path_or_df):
        """加载数据"""
        if isinstance(data_path_or_df, str):
            self.original_data = pd.read_csv(data_path_or_df)
        else:
            self.original_data = data_path_or_df.copy()

        print(f"数据加载完成，形状: {self.original_data.shape}")
        print(f"列名: {list(self.original_data.columns)}")
        return self.original_data

    def check_data_quality(self):
        """检查数据质量"""
        quality_report = {}

        # 缺失值统计
        missing_data = self.original_data.isnull().sum()
        quality_report["missing_values"] = missing_data[missing_data > 0]

        # 数据类型
        quality_report["data_types"] = self.original_data.dtypes

        # 基本统计
        quality_report["basic_stats"] = self.original_data.describe()

        # 重复行
        quality_report["duplicate_rows"] = self.original_data.duplicated().sum()

        # 时间连续性检查（假设存在时间列）
        if "timestamp" in self.original_data.columns:
            time_diffs = (
                pd.to_datetime(self.original_data["timestamp"])
                .diff()
                .dt.total_seconds()
            )
            gaps = time_diffs[time_diffs > 3600]  # 超过1小时的间隔
            quality_report["time_gaps"] = len(gaps)

        return quality_report

    def handle_missing_values(self, method="interpolate", columns=None):
        """处理缺失值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        if method == "interpolate":
            # 时间序列插值
            for col in columns:
                if col in self.original_data.columns:
                    self.original_data[col] = self.original_data[col].interpolate(
                        method="linear"
                    )

        elif method == "forward_fill":
            self.original_data[columns] = self.original_data[columns].fillna(
                method="ffill"
            )

        elif method == "backward_fill":
            self.original_data[columns] = self.original_data[columns].fillna(
                method="bfill"
            )

        elif method == "mean":
            for col in columns:
                if col in self.original_data.columns:
                    self.original_data[col].fillna(
                        self.original_data[col].mean(), inplace=True
                    )

        print(f"缺失值处理完成，使用方法: {method}")
        return self.original_data

    def detect_outliers_zscore(self, threshold=3, columns=None):
        """使用Z-score方法检测异常值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        outliers = pd.DataFrame()
        for col in columns:
            if col in self.original_data.columns:
                z_scores = np.abs(stats.zscore(self.original_data[col].dropna()))
                outlier_mask = z_scores > threshold
                if len(outlier_mask) > 0:
                    outliers[col] = outlier_mask

        return outliers

    def detect_outliers_iqr(self, columns=None):
        """使用IQR方法检测异常值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        outliers = pd.DataFrame()
        for col in columns:
            if col in self.original_data.columns:
                Q1 = self.original_data[col].quantile(0.25)
                Q3 = self.original_data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (self.original_data[col] < lower_bound) | (
                    self.original_data[col] > upper_bound
                )
                outliers[col] = outlier_mask

        return outliers

    def detect_outliers_isolation_forest(self, contamination=0.1, columns=None):
        """使用孤立森林检测异常值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        # 准备数据
        data_for_model = self.original_data[columns].dropna()

        # 训练孤立森林
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_for_model)

        # 创建异常值标记
        outlier_mask = pd.Series(outlier_labels == -1, index=data_for_model.index)

        return outlier_mask

    def smooth_data(self, method="savgol", window_length=5, polyorder=2, columns=None):
        """数据平滑处理"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.original_data.columns:
                if method == "savgol":
                    # 确保window_length不超过数据长度
                    actual_window = min(window_length, len(self.original_data[col]))
                    if actual_window % 2 == 0:
                        actual_window -= 1  # Savitzky-Golay要求奇数窗口
                    if actual_window >= polyorder + 2:
                        self.original_data[col] = savgol_filter(
                            self.original_data[col], actual_window, polyorder
                        )
                elif method == "moving_average":
                    self.original_data[col] = (
                        self.original_data[col]
                        .rolling(window=window_length, center=True)
                        .mean()
                    )

        print(f"数据平滑完成，方法: {method}")
        return self.original_data

    def normalize_data(self, method="standard", columns=None):
        """数据标准化/归一化"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        if method == "standard":
            self.original_data[columns] = self.scaler.fit_transform(
                self.original_data[columns]
            )
        elif method == "minmax":
            scaler = MinMaxScaler()
            self.original_data[columns] = scaler.fit_transform(
                self.original_data[columns]
            )
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            self.original_data[columns] = scaler.fit_transform(
                self.original_data[columns]
            )

        print(f"数据标准化完成，方法: {method}")
        return self.original_data

    def create_time_features(self, timestamp_col="timestamp"):
        """创建时间特征"""
        if timestamp_col in self.original_data.columns:
            self.original_data[timestamp_col] = pd.to_datetime(
                self.original_data[timestamp_col]
            )

            # 提取时间特征
            self.original_data["hour"] = self.original_data[timestamp_col].dt.hour
            self.original_data["day_of_week"] = self.original_data[
                timestamp_col
            ].dt.dayofweek
            self.original_data["month"] = self.original_data[timestamp_col].dt.month
            self.original_data["is_weekend"] = (
                self.original_data["day_of_week"] >= 5
            ).astype(int)

            print("时间特征创建完成")

        return self.original_data

    def sliding_window_features(self, window_size=10, columns=None):
        """创建滑动窗口特征"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.original_data.columns:
                # 滑动均值
                self.original_data[f"{col}_rolling_mean_{window_size}"] = (
                    self.original_data[col].rolling(window=window_size).mean()
                )

                # 滑动标准差
                self.original_data[f"{col}_rolling_std_{window_size}"] = (
                    self.original_data[col].rolling(window=window_size).std()
                )

                # 滑动最大值
                self.original_data[f"{col}_rolling_max_{window_size}"] = (
                    self.original_data[col].rolling(window=window_size).max()
                )

                # 滑动最小值
                self.original_data[f"{col}_rolling_min_{window_size}"] = (
                    self.original_data[col].rolling(window=window_size).min()
                )

        print(f"滑动窗口特征创建完成，窗口大小: {window_size}")
        return self.original_data

    def remove_outliers(self, outlier_method="zscore", threshold=3, columns=None):
        """移除异常值"""
        global outliers
        if outlier_method == "zscore":
            outliers = self.detect_outliers_zscore(threshold=threshold, columns=columns)
        elif outlier_method == "iqr":
            outliers = self.detect_outliers_iqr(columns=columns)
        elif outlier_method == "isolation_forest":
            outlier_mask = self.detect_outliers_isolation_forest(columns=columns)
            # 这里需要进一步处理
            return self.original_data[~outlier_mask]

        # 合并所有异常值标记
        combined_outliers = outliers.any(axis=1)

        # 移除异常值
        self.original_data = self.original_data[~combined_outliers]
        print(f"异常值移除完成，方法: {outlier_method}")
        return self.original_data

    def clean_complete_pipeline(self, config=None):
        """完整的数据清洗流水线"""
        if config is None:
            config = {
                "handle_missing": True,
                "missing_method": "interpolate",
                "remove_outliers": True,
                "outlier_method": "zscore",
                "outlier_threshold": 3,
                "smooth_data": True,
                "smooth_method": "savgol",
                "normalize": True,
                "normalize_method": "standard",
                "create_time_features": True,
                "create_sliding_features": True,
                "sliding_window_size": 10,
            }

        print("开始数据清洗流水线...")

        # 1. 处理缺失值
        if config["handle_missing"]:
            self.handle_missing_values(method=config["missing_method"])

        # 2. 移除异常值
        if config["remove_outliers"]:
            self.remove_outliers(
                outlier_method=config["outlier_method"],
                threshold=config["outlier_threshold"],
            )

        # 3. 数据平滑
        if config["smooth_data"]:
            self.smooth_data(method=config["smooth_method"])

        # 4. 标准化
        if config["normalize"]:
            self.normalize_data(method=config["normalize_method"])

        # 5. 创建时间特征
        if config["create_time_features"]:
            self.create_time_features()

        # 6. 创建滑动窗口特征
        if config["create_sliding_features"]:
            self.sliding_window_features(window_size=config["sliding_window_size"])

        print("数据清洗流水线完成！")
        self.cleaned_data = self.original_data.copy()
        return self.cleaned_data

    def visualize_data_quality(self, columns=None):
        """可视化数据质量"""
        global i
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns[
                :6
            ]  # 限制为前6列

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(self.original_data[col].dropna(), bins=50, alpha=0.7)
                axes[i].set_title(f"{col} 分布")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel("频率")

        # 隐藏多余的子图
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return fig

    def get_cleaning_report(self):
        """生成清洗报告"""
        report = {
            "original_shape": self.original_data.shape,
            "missing_values_after_cleaning": self.original_data.isnull().sum().sum(),
            "columns": list(self.original_data.columns),
            "data_types": dict(self.original_data.dtypes),
            "statistical_summary": self.original_data.describe(),
        }
        return report


app = Flask(__name__)

# 全局数据清洗器实例
cleaner = BuildingMaterialEnergyDataCleaner()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "没有选择文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "没有选择文件"}), 400

    try:
        # 读取CSV文件
        df = pd.read_csv(file.stream)
        cleaner.load_data(df)

        # 获取数据质量报告
        quality_report = cleaner.check_data_quality()

        # 返回数据预览和质量报告
        preview_data = cleaner.original_data.head(10).to_dict("records")
        columns = list(cleaner.original_data.columns)

        return jsonify(
            {
                "success": True,
                "preview": preview_data,
                "columns": columns,
                "shape": cleaner.original_data.shape,
                "missing_values": (
                    quality_report["missing_values"].to_dict()
                    if not quality_report["missing_values"].empty
                    else {}
                ),
                "duplicate_rows": int(quality_report["duplicate_rows"]),
            }
        )
    except Exception as e:
        return jsonify({"error": f"文件处理错误: {str(e)}"}), 500


@app.route("/preview_data", methods=["GET"])
def preview_data():
    if cleaner.original_data is None:
        return jsonify({"error": "未上传数据"}), 400

    try:
        preview = cleaner.original_data.head(10).to_dict("records")
        columns = list(cleaner.original_data.columns)

        return jsonify(
            {
                "preview": preview,
                "columns": columns,
                "shape": cleaner.original_data.shape,
            }
        )
    except Exception as e:
        return jsonify({"error": f"获取预览数据错误: {str(e)}"}), 500


@app.route("/clean_data", methods=["POST"])
def clean_data():
    if cleaner.original_data is None:
        return jsonify({"error": "未上传数据"}), 400

    try:
        # 获取清洗配置
        config = request.json

        # 执行清洗
        cleaned_df = cleaner.clean_complete_pipeline(config)

        # 返回清洗后的预览
        preview = cleaned_df.head(10).to_dict("records")
        report = cleaner.get_cleaning_report()

        return jsonify({"success": True, "preview": preview, "report": report})
    except Exception as e:
        return jsonify({"error": f"数据清洗错误: {str(e)}"}), 500


@app.route("/visualize", methods=["POST"])
def visualize():
    if cleaner.original_data is None:
        return jsonify({"error": "未上传数据"}), 400

    try:
        # 获取要可视化的列
        data = request.json
        columns = data.get("columns", None)

        # 生成图表
        fig = cleaner.visualize_data_quality(columns)

        # 将图表转换为base64
        img = io.BytesIO()
        fig.savefig(img, format="png", bbox_inches="tight")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        plt.close(fig)  # 关闭图表以释放内存

        return jsonify({"plot_url": plot_url})
    except Exception as e:
        return jsonify({"error": f"可视化错误: {str(e)}"}), 500


@app.route("/download", methods=["GET"])
def download():
    if cleaner.cleaned_data is None:
        return jsonify({"error": "未进行数据清洗"}), 400

    try:
        # 将清洗后的数据保存到内存中的CSV
        output = io.StringIO()
        cleaner.cleaned_data.to_csv(output, index=False)
        output.seek(0)

        # 返回CSV内容
        return jsonify({"csv_content": output.getvalue()})
    except Exception as e:
        return jsonify({"error": f"下载错误: {str(e)}"}), 500


if __name__ == "__main__":
    # 创建模板目录
    os.makedirs("templates", exist_ok=True)

    # 创建HTML模板
    template_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>建筑板材生产线能耗数据清洗系统</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background-color: #fafafa;
        }
        .section-title {
            color: #3498db;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        input[type="file"], input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }
        button {
            background-color: #3498db;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
            margin-bottom: 10px;
        }
        button:hover {
            background-color: #2980b9;
        }
        button.secondary {
            background-color: #95a5a6;
        }
        button.secondary:hover {
            background-color: #7f8c8d;
        }
        button.success {
            background-color: #2ecc71;
        }
        button.success:hover {
            background-color: #27ae60;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #ecf0f1;
            font-weight: bold;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            display: none;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            display: block;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
            display: block;
        }
        .visualization {
            text-align: center;
            margin-top: 20px;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
        }
        .config-panel {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }
        .config-item {
            flex: 1;
            min-width: 200px;
        }
        .data-preview {
            max-height: 300px;
            overflow-y: auto;
        }
        @media (max-width: 768px) {
            .config-panel {
                flex-direction: column;
            }
            .container {
                padding: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>建筑板材生产线能耗数据清洗系统</h1>

        <div class="section">
            <h2 class="section-title">1. 上传数据文件</h2>
            <div class="form-group">
                <label for="fileInput">选择CSV文件:</label>
                <input type="file" id="fileInput" accept=".csv">
            </div>
            <button onclick="uploadFile()">上传文件</button>

            <div id="uploadStatus" class="status"></div>

            <div id="dataPreview" style="display:none;">
                <h3>数据预览</h3>
                <p>数据形状: <span id="dataShape"></span></p>
                <div class="data-preview">
                    <table id="previewTable">
                        <thead id="previewHeader"></thead>
                        <tbody id="previewBody"></tbody>
                    </table>
                </div>
            </div>

            <div id="qualityReport" style="display:none;">
                <h3>数据质量报告</h3>
                <p>缺失值: <span id="missingValues"></span></p>
                <p>重复行: <span id="duplicateRows"></span></p>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">2. 数据清洗配置</h2>
            <div class="config-panel">
                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="handleMissing" checked> 处理缺失值
                        </label>
                        <select id="missingMethod" style="margin-top: 5px;">
                            <option value="interpolate">线性插值</option>
                            <option value="forward_fill">前向填充</option>
                            <option value="backward_fill">后向填充</option>
                            <option value="mean">均值填充</option>
                        </select>
                    </div>
                </div>

                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="removeOutliers" checked> 移除异常值
                        </label>
                        <select id="outlierMethod" style="margin-top: 5px;">
                            <option value="zscore">Z-Score方法</option>
                            <option value="iqr">IQR方法</option>
                            <option value="isolation_forest">孤立森林</option>
                        </select>
                        <input type="number" id="outlierThreshold" value="3" min="1" max="10" step="0.1" 
                               placeholder="阈值" style="margin-top: 5px;">
                    </div>
                </div>

                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="smoothData" checked> 数据平滑
                        </label>
                        <select id="smoothMethod" style="margin-top: 5px;">
                            <option value="savgol">Savitzky-Golay滤波</option>
                            <option value="moving_average">移动平均</option>
                        </select>
                    </div>
                </div>

                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="normalizeData" checked> 数据标准化
                        </label>
                        <select id="normalizeMethod" style="margin-top: 5px;">
                            <option value="standard">标准化 (Z-score)</option>
                            <option value="minmax">最小-最大标准化</option>
                            <option value="robust">鲁棒标准化</option>
                        </select>
                    </div>
                </div>

                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="createTimeFeatures" checked> 创建时间特征
                        </label>
                    </div>
                </div>

                <div class="config-item">
                    <div class="form-group">
                        <label>
                            <input type="checkbox" id="createSlidingFeatures" checked> 创建滑动窗口特征
                        </label>
                        <input type="number" id="slidingWindowSize" value="10" min="3" max="100" 
                               placeholder="窗口大小" style="margin-top: 5px;">
                    </div>
                </div>
            </div>

            <button class="success" onclick="startCleaning()">开始清洗数据</button>

            <div id="cleaningStatus" class="status"></div>
        </div>

        <div class="section">
            <h2 class="section-title">3. 清洗结果预览</h2>
            <div id="cleanedPreview" style="display:none;">
                <p>清洗后数据形状: <span id="cleanedShape"></span></p>
                <div class="data-preview">
                    <table id="cleanedPreviewTable">
                        <thead id="cleanedPreviewHeader"></thead>
                        <tbody id="cleanedPreviewBody"></tbody>
                    </table>
                </div>
            </div>

            <button class="secondary" onclick="visualizeData()" style="display:none;" id="visualizeBtn">生成可视化图表</button>

            <div class="visualization" id="visualization" style="display:none;">
                <h3>数据分布图</h3>
                <img id="plotImage" src="" alt="数据分布图">
            </div>

            <button class="success" onclick="downloadData()" style="display:none;" id="downloadBtn">下载清洗后的数据</button>
        </div>
    </div>

    <script>
        let originalData = null;
        let cleanedData = null;

        function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (!file) {
                showStatus('uploadStatus', '请选择一个CSV文件', 'error');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    originalData = data;
                    showStatus('uploadStatus', '文件上传成功！', 'success');

                    document.getElementById('dataShape').textContent = `${data.shape[0]} 行 × ${data.shape[1]} 列`;

                    // 显示数据预览
                    displayTable('previewHeader', 'previewBody', data.preview, data.columns);
                    document.getElementById('dataPreview').style.display = 'block';

                    // 显示质量报告
                    document.getElementById('missingValues').textContent = 
                        Object.keys(data.missing_values).length > 0 ? 
                        JSON.stringify(data.missing_values) : '无缺失值';
                    document.getElementById('duplicateRows').textContent = data.duplicate_rows;
                    document.getElementById('qualityReport').style.display = 'block';
                } else {
                    showStatus('uploadStatus', data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('uploadStatus', '上传失败: ' + error.message, 'error');
            });
        }

        function startCleaning() {
            const config = {
                handle_missing: document.getElementById('handleMissing').checked,
                missing_method: document.getElementById('missingMethod').value,
                remove_outliers: document.getElementById('removeOutliers').checked,
                outlier_method: document.getElementById('outlierMethod').value,
                outlier_threshold: parseFloat(document.getElementById('outlierThreshold').value),
                smooth_data: document.getElementById('smoothData').checked,
                smooth_method: document.getElementById('smoothMethod').value,
                normalize: document.getElementById('normalizeData').checked,
                normalize_method: document.getElementById('normalizeMethod').value,
                create_time_features: document.getElementById('createTimeFeatures').checked,
                create_sliding_features: document.getElementById('createSlidingFeatures').checked,
                sliding_window_size: parseInt(document.getElementById('slidingWindowSize').value)
            };

            fetch('/clean_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    cleanedData = data;
                    showStatus('cleaningStatus', '数据清洗完成！', 'success');

                    // 显示清洗后的数据预览
                    document.getElementById('cleanedShape').textContent = 
                        `${data.report.original_shape[0]} 行 × ${data.report.original_shape[1]} 列`;
                    displayTable('cleanedPreviewHeader', 'cleanedPreviewBody', data.preview, data.report.columns);
                    document.getElementById('cleanedPreview').style.display = 'block';

                    // 显示可视化和下载按钮
                    document.getElementById('visualizeBtn').style.display = 'inline-block';
                    document.getElementById('downloadBtn').style.display = 'inline-block';
                } else {
                    showStatus('cleaningStatus', data.error, 'error');
                }
            })
            .catch(error => {
                showStatus('cleaningStatus', '清洗失败: ' + error.message, 'error');
            });
        }

        function visualizeData() {
            // 获取数值列用于可视化
            const numericColumns = cleanedData ? cleanedData.report.columns.filter(col => 
                !['timestamp', 'hour', 'day_of_week', 'month', 'is_weekend'].includes(col)
            ) : [];

            fetch('/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ columns: numericColumns.slice(0, 6) }) // 限制为前6列
            })
            .then(response => response.json())
            .then(data => {
                if (data.plot_url) {
                    document.getElementById('plotImage').src = 'data:image/png;base64,' + data.plot_url;
                    document.getElementById('visualization').style.display = 'block';
                } else {
                    alert('生成可视化图表失败: ' + data.error);
                }
            })
            .catch(error => {
                alert('可视化失败: ' + error.message);
            });
        }

        function downloadData() {
            fetch('/download')
            .then(response => response.json())
            .then(data => {
                if (data.csv_content) {
                    // 创建下载链接
                    const blob = new Blob([data.csv_content], { type: 'text/csv' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'cleaned_energy_data.csv';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } else {
                    alert('下载失败: ' + data.error);
                }
            })
            .catch(error => {
                alert('下载失败: ' + error.message);
            });
        }

        function displayTable(headerId, bodyId, data, columns) {
            const header = document.getElementById(headerId);
            const body = document.getElementById(bodyId);

            // 清空现有内容
            header.innerHTML = '';
            body.innerHTML = '';

            // 创建表头
            const headerRow = document.createElement('tr');
            columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            header.appendChild(headerRow);

            // 创建表体
            data.forEach(row => {
                const tr = document.createElement('tr');
                columns.forEach(col => {
                    const td = document.createElement('td');
                    td.textContent = row[col] !== undefined ? row[col] : '';
                    tr.appendChild(td);
                });
                body.appendChild(tr);
            });
        }

        function showStatus(elementId, message, type) {
            const element = document.getElementById(elementId);
            element.textContent = message;
            element.className = 'status ' + type;
            element.style.display = 'block';

            // 3秒后自动隐藏成功消息
            if (type === 'success') {
                setTimeout(() => {
                    element.style.display = 'none';
                }, 3000);
            }
        }
    </script>
</body>
</html>
    """

    # 写入模板文件
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(template_content)

    app.run(debug=True, host="0.0.0.0", port=5000)
