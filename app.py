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
import signal
import threading
import time
# Vercel Blob是可选的，生产环境会自动提供
try:
    from vercel_blob import BlobStore
except ImportError:
    BlobStore = None

# 添加 pipeline 模块路径
sys.path.append('pipeline')

# 导入核心业务逻辑
from pipeline.clean_pipeline_v3 import process_one_file, get_date_str_from_text

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # 用于 session

# 初始化Vercel Blob
import os
if BlobStore and os.getenv('VERCEL_BLOB_TOKEN'):
    blob_store = BlobStore()
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

def process_with_timeout(excel_path, output_path, date_str_from_file, timeout_seconds=120):
    """带超时的处理函数"""
    result = {'success': False, 'error': None}
    
    def target():
        try:
            process_one_file(
                excel_path=excel_path,
                output_path=output_path,
                date_str_from_file=date_str_from_file,
                STRICT_DATE_FILTER=False
            )
            result['success'] = True
        except Exception as e:
            result['error'] = str(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        result['error'] = f'处理超时（{timeout_seconds}秒），请尝试减少数据量或稍后重试'
    
    return result

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
        
        date_str_from_file = get_date_from_file(excel_file.filename)
        
        # 日期一致性校验
        if date_str_from_file:  # 如果文件名能提取到日期
            if date_str_from_ui != date_str_from_file:  # 两个日期不一致
                return jsonify({
                    'error': f'日期不一致！\n\n从日期控件您选择的日期是：{date_str_from_ui}\n从文件名 "{excel_file.filename}"，我们得到的日期是：{date_str_from_file}\n\n两个日期不一致，请您再检查一下'
                }), 400
        
  
        # 使用带超时的处理函数（120秒超时）
        process_result = process_with_timeout(
            excel_path=excel_path,
            output_path=output_path,
            date_str_from_file=date_str_from_ui,
            timeout_seconds=120
        )
        
        if not process_result['success']:
            return jsonify({'error': process_result['error']}), 500
        
        # 读取处理结果
        result_df = pd.read_excel(output_path)
        
        # 处理 NaN 值，转换为 None 以便 JSON 序列化
        result_df = result_df.fillna('')
        
        # 根据环境选择存储方式
        if blob_store:
            # 生产环境：使用Vercel Blob
            import uuid
            file_id = str(uuid.uuid4())
            blob_filename = f'processed_{file_id}.xlsx'
            
            # 读取文件内容
            with open(output_path, 'rb') as f:
                file_content = f.read()
            
            # 上传到Vercel Blob
            blob_url = blob_store.put(blob_filename, file_content)
            
            return jsonify({
                'success': True,
                'message': f'处理完成！共处理 {len(result_df)} 条记录',
                'download_url': blob_url,
                'file_id': file_id
            })
        else:
            # 本地开发：使用本地文件存储
            import uuid
            file_id = str(uuid.uuid4())
            local_filename = f'processed_{file_id}.xlsx'
            
            # 将文件复制到static目录（如果存在）
            import shutil
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            if not os.path.exists(static_dir):
                os.makedirs(static_dir)
            
            local_path = os.path.join(static_dir, local_filename)
            shutil.copy2(output_path, local_path)
            
            return jsonify({
                'success': True,
                'message': f'处理完成！共处理 {len(result_df)} 条记录（本地开发模式）',
                'download_url': f'/static/{local_filename}',
                'file_id': file_id
            })
        
    except Exception as e:
        return jsonify({'error': f'处理失败: {str(e)}'}), 500

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