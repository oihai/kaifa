from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from data_cleaner import BuildingMaterialEnergyDataCleaner
from utils import convert_for_json

app = Flask(__name__, template_folder=".")

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
    app.run(debug=True, host="0.0.0.0", port=5000)
