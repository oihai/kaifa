# utils.py
import pandas as pd
import numpy as np
from decimal import Decimal


def convert_for_json(data):
    """将任意数据转换为JSON兼容格式"""
    try:
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # 键必须是字符串
                str_key = str(key) if not isinstance(key, str) else key
                result[str_key] = convert_for_json(value)
            return result
        elif isinstance(data, (list, tuple, set)):
            return [convert_for_json(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            # DataFrame转为记录列表，处理各种数据类型
            records = []
            for _, row in data.iterrows():
                record = {}
                for col, val in row.items():
                    record[str(col)] = convert_scalar_for_json(val)
                records.append(record)
            return records
        elif isinstance(data, pd.Series):
            # Series转为字典
            result = {}
            for idx, val in data.items():
                result[str(idx)] = convert_scalar_for_json(val)
            return result
        else:
            return convert_scalar_for_json(data)
    except Exception as e:
        # 如果转换失败，返回字符串表示
        return str(data)


def convert_scalar_for_json(val):
    """转换单个标量值为JSON兼容格式"""
    try:
        # 处理空值
        if pd.isna(val) or val is None:
            return None

        # 处理numpy标量类型
        if isinstance(val, (np.integer, np.floating, np.bool_, np.str_)):
            # 转换为Python原生类型
            if isinstance(val, np.bool_):
                return bool(val)
            elif isinstance(val, np.integer):
                # 检查是否在Python int范围内
                python_int = int(val)
                if python_int == val:  # 确保转换无损
                    return python_int
                else:
                    return float(val)  # 超出范围则转为float
            elif isinstance(val, np.floating):
                # 处理特殊浮点值
                if np.isnan(val):
                    return None
                elif np.isinf(val):
                    return str(val)  # 正无穷或负无穷转为字符串
                else:
                    python_float = float(val)
                    # 检查精度是否足够
                    if abs(python_float - val) < 1e-10:
                        return python_float
                    else:
                        return str(val)  # 精度不足时转为字符串
            elif isinstance(val, np.str_):
                return str(val)

        # 处理Python原生类型
        if isinstance(val, bool):
            return val
        elif isinstance(val, (int, float)):
            # 特殊处理Python float
            if isinstance(val, float):
                if np.isnan(val):
                    return None
                elif np.isinf(val):
                    return str(val)
            return val
        elif isinstance(val, str):
            return val
        elif isinstance(val, bytes):
            return val.decode("utf-8", errors="replace")

        # 处理pandas特殊类型
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        elif isinstance(val, pd.Timedelta):
            return str(val)
        elif isinstance(val, pd.Period):
            return str(val)
        elif isinstance(val, pd.Interval):
            return {
                "left": convert_scalar_for_json(val.left),
                "right": convert_scalar_for_json(val.right),
                "closed": val.closed,
            }

        # 处理numpy数组
        if isinstance(val, np.ndarray):
            # 递归处理数组元素
            return [convert_scalar_for_json(item) for item in val.flat]

        # 处理Decimal
        if isinstance(val, Decimal):
            return float(val)

        # 处理复数
        if isinstance(val, complex):
            return {"real": val.real, "imag": val.imag}

        # 其他类型尝试转换为字符串
        return str(val)
    except Exception:
        # 最后兜底，返回字符串表示
        try:
            return str(val)
        except:
            return "<无法序列化的对象>"
