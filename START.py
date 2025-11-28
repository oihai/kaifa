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
                # 只对非空值计算z-score
                series = self.original_data[col].dropna()
                if len(series) > 1:  # 确保至少有2个值
                    z_scores = np.abs(stats.zscore(series))
                    outlier_mask = z_scores > threshold
                    outliers[col] = pd.Series(outlier_mask, index=series.index)
                else:
                    # 如果只有一个或没有非空值，创建全为False的mask
                    outliers[col] = pd.Series(
                        [False] * len(self.original_data),
                        index=self.original_data.index,
                    )

        return outliers

    def detect_outliers_iqr(self, columns=None):
        """使用IQR方法检测异常值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        outliers = pd.DataFrame()
        for col in columns:
            if col in self.original_data.columns:
                series = self.original_data[col].dropna()
                if len(series) >= 4:  # IQR需要至少4个值
                    Q1 = series.quantile(0.25)
                    Q3 = series.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                    outlier_mask = (self.original_data[col] < lower_bound) | (
                        self.original_data[col] > upper_bound
                    )
                    outliers[col] = outlier_mask
                else:
                    # 如果数据不足，创建全为False的mask
                    outliers[col] = pd.Series(
                        [False] * len(self.original_data),
                        index=self.original_data.index,
                    )

        return outliers

    def detect_outliers_isolation_forest(self, contamination=0.1, columns=None):
        """使用孤立森林检测异常值"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        # 准备数据
        data_for_model = self.original_data[columns].dropna()

        if len(data_for_model) < 2:
            # 如果数据不足，返回全为False的mask
            return pd.Series(
                [False] * len(self.original_data), index=self.original_data.index
            )

        # 训练孤立森林
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(data_for_model)

        # 创建异常值标记
        outlier_mask = pd.Series(outlier_labels == -1, index=data_for_model.index)

        # 为原始数据创建完整mask
        full_mask = pd.Series(
            [False] * len(self.original_data), index=self.original_data.index
        )
        full_mask.update(outlier_mask)

        return full_mask

    def smooth_data(self, method="savgol", window_length=5, polyorder=2, columns=None):
        """数据平滑处理"""
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns

        for col in columns:
            if col in self.original_data.columns:
                series = self.original_data[col].dropna()

                if len(series) < 3:
                    # 如果数据不足，跳过平滑
                    continue

                if method == "savgol":
                    # 确保window_length不超过数据长度
                    actual_window = min(window_length, len(series))
                    if actual_window % 2 == 0:
                        actual_window -= 1  # Savitzky-Golay要求奇数窗口
                    if actual_window >= polyorder + 2:
                        # 只对非空值进行平滑处理
                        smoothed_values = savgol_filter(
                            series, actual_window, polyorder
                        )
                        # 将平滑后的值放回原数据
                        self.original_data.loc[series.index, col] = smoothed_values
                elif method == "moving_average":
                    # 使用原始数据进行移动平均，保留NaN值
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

        # 选择非空的数值列进行标准化
        numeric_data = self.original_data[columns].dropna()

        if len(numeric_data) == 0:
            print("没有足够的数据进行标准化")
            return self.original_data

        if method == "standard":
            # 使用原始数据进行标准化
            temp_data = self.original_data[columns].copy()
            temp_data_clean = temp_data.dropna()

            if len(temp_data_clean) > 1:  # 确保有足够的数据点
                normalized_values = self.scaler.fit_transform(temp_data_clean)
                # 将标准化后的值放回原数据
                self.original_data.loc[temp_data_clean.index, columns] = (
                    normalized_values
                )
        elif method == "minmax":
            scaler = MinMaxScaler()
            temp_data = self.original_data[columns].copy()
            temp_data_clean = temp_data.dropna()

            if len(temp_data_clean) > 1:
                normalized_values = scaler.fit_transform(temp_data_clean)
                self.original_data.loc[temp_data_clean.index, columns] = (
                    normalized_values
                )
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler

            scaler = RobustScaler()
            temp_data = self.original_data[columns].copy()
            temp_data_clean = temp_data.dropna()

            if len(temp_data_clean) > 1:
                normalized_values = scaler.fit_transform(temp_data_clean)
                self.original_data.loc[temp_data_clean.index, columns] = (
                    normalized_values
                )

        print(f"数据标准化完成，方法: {method}")
        return self.original_data

    def create_time_features(self, timestamp_col="timestamp"):
        """创建时间特征"""
        if timestamp_col in self.original_data.columns:
            # 转换时间列
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
                # 确保窗口大小不超过数据长度
                actual_window = min(window_size, len(self.original_data))

                # 滑动均值
                self.original_data[f"{col}_rolling_mean_{actual_window}"] = (
                    self.original_data[col]
                    .rolling(window=actual_window, min_periods=1)
                    .mean()
                )

                # 滑动标准差
                self.original_data[f"{col}_rolling_std_{actual_window}"] = (
                    self.original_data[col]
                    .rolling(window=actual_window, min_periods=1)
                    .std()
                )

                # 滑动最大值
                self.original_data[f"{col}_rolling_max_{actual_window}"] = (
                    self.original_data[col]
                    .rolling(window=actual_window, min_periods=1)
                    .max()
                )

                # 滑动最小值
                self.original_data[f"{col}_rolling_min_{actual_window}"] = (
                    self.original_data[col]
                    .rolling(window=actual_window, min_periods=1)
                    .min()
                )

        print(f"滑动窗口特征创建完成，窗口大小: {window_size}")
        return self.original_data

    def remove_outliers(self, outlier_method="zscore", threshold=3, columns=None):
        """移除异常值"""
        if outlier_method == "zscore":
            outliers = self.detect_outliers_zscore(threshold=threshold, columns=columns)
        elif outlier_method == "iqr":
            outliers = self.detect_outliers_iqr(columns=columns)
        elif outlier_method == "isolation_forest":
            outlier_mask = self.detect_outliers_isolation_forest(columns=columns)
            # 返回非异常值的数据
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
        if columns is None:
            columns = self.original_data.select_dtypes(include=[np.number]).columns[
                :6
            ]  # 限制为前6列

        # 过滤掉不存在的列
        available_columns = [
            col for col in columns if col in self.original_data.columns
        ]

        if not available_columns:
            # 如果没有数值列，使用所有列的前几个
            available_columns = self.original_data.columns[:6]

        # 限制为最多6个列进行可视化
        available_columns = available_columns[:6]

        if not available_columns:
            # 如果仍然没有可用列，创建一个空图
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "没有可用数据进行可视化",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            return fig

        n_cols = min(3, len(available_columns))
        n_rows = (len(available_columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(available_columns):
            if i < len(axes):
                # 只对数值列进行直方图绘制
                if self.original_data[col].dtype in [
                    "int64",
                    "float64",
                    "int32",
                    "float32",
                ]:
                    # 移除NaN值后绘制
                    clean_data = self.original_data[col].dropna()
                    if len(clean_data) > 0:
                        axes[i].hist(
                            clean_data, bins=min(50, len(clean_data) // 2), alpha=0.7
                        )
                        axes[i].set_title(f"{col} 分布")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("频率")
                    else:
                        axes[i].text(
                            0.5,
                            0.5,
                            f"无数据\n{col}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[i].transAxes,
                        )
                        axes[i].set_title(f"{col} 分布")
                else:
                    # 对于非数值列，显示值计数
                    value_counts = self.original_data[col].value_counts()
                    if len(value_counts) > 0:
                        axes[i].bar(range(len(value_counts)), value_counts.values)
                        axes[i].set_title(f"{col} 计数")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("计数")
                        axes[i].set_xticks(range(len(value_counts)))
                        axes[i].set_xticklabels(
                            value_counts.index[:10], rotation=45, ha="right"
                        )
                    else:
                        axes[i].text(
                            0.5,
                            0.5,
                            f"无数据\n{col}",
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axes[i].transAxes,
                        )
                        axes[i].set_title(f"{col} 分布")

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
            "data_types": {
                k: str(v) for k, v in dict(self.original_data.dtypes).items()
            },  # 转换为字符串
            "statistical_summary": self.original_data.describe(),
        }
        return report


# 辅助函数：将数据转换为JSON兼容格式
def convert_for_json(data):
    """将包含pandas数据类型的数据转换为JSON兼容格式"""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # 将键转换为字符串以确保JSON兼容性
            str_key = str(key) if not isinstance(key, str) else key
            result[str_key] = convert_for_json(value)
        return result
    elif isinstance(data, list):
        return [convert_for_json(item) for item in data]
    elif isinstance(data, pd.DataFrame):
        # 将DataFrame转换为字典列表，处理pandas数据类型
        result = []
        for _, row in data.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None  # 将NaN转换为None
                else:
                    row_dict[col] = convert_value(val)
            result.append(row_dict)
        return result
    elif isinstance(data, pd.Series):
        # 将Series转换为字典，处理pandas数据类型
        result = {}
        for idx, val in data.items():
            if pd.isna(val):
                result[str(idx)] = None  # 将索引转为字符串
            else:
                result[str(idx)] = convert_value(val)
        return result
    elif pd.isna(data):
        return None
    else:
        return convert_value(data)


def convert_value(val):
    """转换单个值为JSON兼容格式"""
    # 完全移除可能导致isinstance错误的复杂检查
    # 直接使用基本类型转换
    if pd.isna(val):
        return None
    elif isinstance(val, (int, float, str, bool)):
        # 基本类型直接返回
        return val
    elif hasattr(val, "item"):
        # numpy标量类型
        try:
            return val.item()
        except:
            return str(val)
    elif isinstance(val, pd.Timestamp):
        # pandas时间戳
        return str(val)
    elif isinstance(val, pd.Period):
        # pandas周期
        return str(val)
    elif isinstance(val, pd.Interval):
        # pandas区间
        return str(val)
    elif isinstance(val, (np.integer, np.floating)):
        # numpy整数和浮点数
        return float(val) if isinstance(val, np.floating) else int(val)
    elif isinstance(val, np.bool_):
        # numpy布尔值
        return bool(val)
    elif isinstance(val, (np.ndarray, pd.array)):
        # numpy数组和pandas数组
        return [convert_value(item) for item in val]
    elif isinstance(val, complex):
        # 复数
        return str(val)
    else:
        # 其他类型转换为字符串
        try:
            return val
        except:
            return str(val)


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

        # 返回数据预览和质量报告（转换为JSON兼容格式）
        preview_data = convert_for_json(cleaner.original_data.head(10))
        columns = list(cleaner.original_data.columns)

        return jsonify(
            {
                "success": True,
                "preview": preview_data,
                "columns": columns,
                "shape": cleaner.original_data.shape,
                "missing_values": (
                    convert_for_json(quality_report["missing_values"])
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
        preview = convert_for_json(cleaner.original_data.head(10))
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

        # 检查是否有足够的数据进行处理
        if len(cleaner.original_data) == 0:
            return jsonify({"error": "数据已被完全移除，无法进行处理"}), 400

        # 执行清洗
        cleaned_df = cleaner.clean_complete_pipeline(config)

        # 返回清洗后的预览
        preview = convert_for_json(cleaned_df.head(10))
        report = cleaner.get_cleaning_report()
        report = convert_for_json(report)

        return jsonify({"success": True, "preview": preview, "report": report})
    except Exception as e:
        return jsonify({"error": f"数据清洗错误: {str(e)}"}), 500


@app.route("/visualize", methods=["POST"])
def visualize():
    if cleaner.original_data is None:
        return jsonify({"error": "未上传数据"}), 400

    try:
        # 检查是否有足够的数据进行可视化
        if len(cleaner.original_data) == 0:
            return jsonify({"error": "没有足够的数据进行可视化"}), 400

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
                    const value = row[col];
                    // 处理null值显示
                    td.textContent = value !== null && value !== undefined ? value : '';
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
