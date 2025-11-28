import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
        plt.show()

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


# 使用示例
def example_usage():
    """使用示例"""
    # 创建模拟数据用于演示
    np.random.seed(42)
    n_samples = 1000

    # 模拟建筑板材生产线能耗数据
    timestamps = pd.date_range(start="2023-01-01", periods=n_samples, freq="H")
    energy_consumption = (
        50 + 20 * np.sin(np.arange(n_samples) * 0.1) + np.random.normal(0, 5, n_samples)
    )
    temperature = (
        200
        + 50 * np.sin(np.arange(n_samples) * 0.05)
        + np.random.normal(0, 10, n_samples)
    )
    pressure = (
        10 + 5 * np.sin(np.arange(n_samples) * 0.08) + np.random.normal(0, 2, n_samples)
    )
    production_rate = (
        100
        + 30 * np.sin(np.arange(n_samples) * 0.12)
        + np.random.normal(0, 8, n_samples)
    )

    # 添加一些异常值
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    energy_consumption[outlier_indices] += np.random.normal(0, 50, 20)

    # 创建DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "energy_consumption": energy_consumption,
            "temperature": temperature,
            "pressure": pressure,
            "production_rate": production_rate,
        }
    )

    # 随机引入一些缺失值
    missing_indices = np.random.choice(n_samples, size=30, replace=False)
    df.loc[missing_indices[:10], "energy_consumption"] = np.nan
    df.loc[missing_indices[10:20], "temperature"] = np.nan
    df.loc[missing_indices[20:], "pressure"] = np.nan

    print("原始数据示例:")
    print(df.head(10))
    print(f"\n原始数据形状: {df.shape}")

    # 初始化清洗器
    cleaner = BuildingMaterialEnergyDataCleaner()
    cleaner.load_data(df)

    # 检查数据质量
    quality_report = cleaner.check_data_quality()
    print(f"\n数据质量报告:")
    print(f"缺失值: {quality_report['missing_values']}")
    print(f"重复行: {quality_report['duplicate_rows']}")

    # 执行完整清洗流水线
    cleaned_df = cleaner.clean_complete_pipeline()

    print(f"\n清洗后数据形状: {cleaned_df.shape}")
    print(f"剩余缺失值: {cleaned_df.isnull().sum().sum()}")

    # 生成清洗报告
    report = cleaner.get_cleaning_report()
    print(f"\n清洗报告:")
    print(f"最终数据形状: {report['original_shape']}")
    print(f"特征列: {report['columns']}")

    # 可视化
    cleaner.visualize_data_quality(["energy_consumption", "temperature", "pressure"])

    return cleaner, cleaned_df


if __name__ == "__main__":
    cleaner, cleaned_data = example_usage()
