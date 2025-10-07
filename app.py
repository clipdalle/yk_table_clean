"""
麦序数据清洗 Flask 应用
使用现有的 process_one_file 核心业务逻辑
"""

from flask import Flask, request, render_template, jsonify, send_file, session
import pandas as pd
import tempfile
import os
import sys
from pathlib import Path
from func_timeout import func_timeout, FunctionTimedOut
# Vercel Blob是可选的，生产环境会自动提供

from vercel_blob import blob_store as VercelBlobStore

# 添加 pipeline 模块路径
sys.path.append('pipeline')
import os 
from global_config import STRICT_DATE_FILTER
# 导入核心业务逻辑
from pipeline.clean_pipeline_v3 import process_one_file, get_date_str_from_text

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于 session


# 初始化Vercel Blob
BLOB_READ_WRITE_TOKEN = 'vercel_blob_rw_UDHA4kmifSvG3WQk_CC7V5VsRXmouv2ag9gI4EQU65DEoVR'
os.environ['BLOB_READ_WRITE_TOKEN'] = BLOB_READ_WRITE_TOKEN
import os
if VercelBlobStore and BLOB_READ_WRITE_TOKEN:
    blob_store = VercelBlobStore
else:
    print("⚠️  Vercel Blob not available, using local file storage for development")
    blob_store = None

# 配置
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'txt', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

def get_date_from_ui(selected_date):
    """
    从用户界面选择的日期获取日期字符串
    
    Args:
        selected_date: 用户选择的日期 (YYYY-MM-DD 格式)
    
    Returns:
        str: 日期字符串 (YYYYMMDD 格式)，如果格式错误则返回 None
    """
    if not selected_date:
        return None
    
    from datetime import datetime
    try:
        date_obj = datetime.strptime(selected_date, '%Y-%m-%d')
        return date_obj.strftime('%Y%m%d')
    except ValueError:
        return None

def get_date_from_file(excel_filename):
    """
    从Excel文件名提取日期字符串
    
    Args:
        excel_filename: Excel文件名
    
    Returns:
        str: 日期字符串 (YYYYMMDD 格式)，如果无法提取则返回 None
    """
    return get_date_str_from_text(excel_filename)

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        # 检查文件是否存在
        if 'name_list' not in request.files or 'excel_file' not in request.files:
            return jsonify({'error': '请上传全量名单和Excel文件'}), 400
        
        name_file = request.files['name_list']
        excel_file = request.files['excel_file']
        
        if name_file.filename == '' or excel_file.filename == '':
            return jsonify({'error': '请选择文件'}), 400
        
        if not allowed_file(name_file.filename) or not allowed_file(excel_file.filename):
            return jsonify({'error': '文件格式不支持'}), 400
        
        # 创建临时文件
        temp_dir = tempfile.mkdtemp()
        
        # 保存全量名单
        name_path = os.path.join(temp_dir, 'known_names_select.txt')
        name_content = name_file.read().decode('utf-8')
        with open(name_path, 'w', encoding='utf-8') as f:
            f.write(name_content)
        
        # 保存Excel文件
        excel_path = os.path.join(temp_dir, 'input.xlsx')
        excel_file.save(excel_path)
        
        # 生成输出路径
        output_path = os.path.join(temp_dir, 'output.xlsx')
        
        # 提取日期（现在UI日期是必填的）
        date_str_from_ui = get_date_from_ui(request.form.get('selected_date'))
        if not date_str_from_ui:
            return jsonify({'error': '请选择日期'}), 400
        
        # 获取选中的厅号（原生FormData方式）
        selected_halls = request.form.getlist('hall')
        print(f"🏛️ 选中的厅号: {selected_halls}")
        
        # 校验厅号必须选择
        if not selected_halls or len(selected_halls) == 0:
            return jsonify({'error': '请至少选择一个厅号'}), 400
        
        date_str_from_file = get_date_from_file(excel_file.filename)
        
        # 日期一致性校验
        if date_str_from_file:  # 如果文件名能提取到日期
            if date_str_from_ui != date_str_from_file:  # 两个日期不一致
                return jsonify({
                    'error': f'日期不一致！\n\n从日期控件您选择的日期是：{date_str_from_ui}\n从文件名 "{excel_file.filename}"，我们得到的日期是：{date_str_from_file}\n\n两个日期不一致，请您再检查一下'
                }), 400
        # 检测环境并设置合适的超时时间
        is_vercel = os.getenv('VERCEL') == '1'
        timeout_seconds = 120 if is_vercel else 120
        
        print(f"🌍 环境检测: {'Vercel生产环境' if is_vercel else '本地开发环境'}")
        print(f"⏱️ 超时设置: {timeout_seconds}秒")
        
        # 读取原始数据获取行数
        original_df = pd.read_excel(excel_path)
        original_rows = len(original_df)
        print(f"📊 原始数据行数: {original_rows}")
        
        # 使用 func_timeout 实现超时控制
        try:
            func_timeout(
                timeout_seconds,
                process_one_file,
                kwargs={
                    'excel_path': excel_path,
                    'output_path': output_path,
                    'date_str_from_file': date_str_from_ui,
                    'strict_date_filter': STRICT_DATE_FILTER,
                    'selected_halls': selected_halls
                }
            )
        except FunctionTimedOut:
            return jsonify({
                'error': f'处理超时（{timeout_seconds}秒），请尝试减少数据量或稍后重试'
            }), 500
        
        # 读取处理结果
        result_df = pd.read_excel(output_path)
        processed_rows = len(result_df)
        
        # 处理 NaN 值，转换为 None 以便 JSON 序列化
        result_df = result_df.fillna('')
        
        # 统一使用Base64方式（本地和生产环境都适用）
        import uuid
        import base64
        file_id = str(uuid.uuid4())
        
        # 读取文件内容并转换为base64
        with open(output_path, 'rb') as f:
            file_content = f.read()
        
        file_base64 = base64.b64encode(file_content).decode('utf-8')
        

        output_filename = Path(excel_file.filename).with_suffix(f'.{date_str_from_ui}.cleaned.xlsx').name
        
        return jsonify({
            'success': True,
            'message': f'处理完成！共处理 {processed_rows} 条记录',
            'download_url': f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{file_base64}',
            'file_id': file_id,
            'filename': output_filename,
            'stats': {
                'selected_halls': selected_halls,
                'original_rows': original_rows,
                'processed_rows': processed_rows,
                'selected_date': date_str_from_ui
            }
        })
        
    except Exception as e:
        import traceback
        import sys
        error_details = traceback.format_exc()
        
        # 强制输出到stderr，确保在Vercel dev中可见
        print(f"❌ 处理失败: {str(e)}", file=sys.stderr)
        print(f"详细错误信息:\n{error_details}", file=sys.stderr)
        
        # 同时输出到stdout
        print(f"❌ 处理失败: {str(e)}")
        print(f"详细错误信息:\n{error_details}")
        
        return jsonify({'error': f'处理失败: {error_details}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """下载处理结果文件"""
    try:
        # 从 session 中获取临时目录
        temp_dir = session.get('temp_dir')
        if not temp_dir:
            return jsonify({'error': '会话已过期，请重新上传文件'}), 404
        
        file_path = os.path.join(temp_dir, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)