import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

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
        if n_cols == 1 and n_rows == 1:
            axes = [axes]
        elif n_rows == 1:
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
