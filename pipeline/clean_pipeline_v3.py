"""
排麦人员 & 主持人员字段清洗 Pipeline
使用 Gemini API 批量解析混乱的文本，生成人工校验表
"""

import pandas as pd
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_fixed

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.llm.llm_client import _get_client
from pipeline.prompts_v3 import PROMPT_BATCH
from global_config import TEMPERATURE, BATCH_SIZE, CURRENT_MODEL, COLS_CONFIG



def get_temp_path(filename):
    import tempfile
    temp_dir = tempfile.gettempdir()
    return os.path.join(temp_dir, filename)
    
# 读取已知人名清单
with open('known_names_select.txt', 'r', encoding='utf-8') as f:
    known_names_select = f.read().splitlines()
    known_names_select = [name.strip() for name in known_names_select if name.strip()]
known_names_select = list(set(known_names_select))

print(f"✅ 已加载 {len(known_names_select)} 个已知人名")
print(f"人名清单: {known_names_select[:10]}..." if len(known_names_select) > 10 else f"人名清单: {known_names_select}")


# ========== 核心函数 ==========
import traceback 

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def parse_batch(batch_df: pd.DataFrame, assist_known_names: List[str]) -> List[Dict]:
    """
    批量解析多行数据（一次 API 调用）
    
    Args:
        batch_df: 批次数据框
    
    Returns:
        解析结果列表
    """
    # 构建批量输入数据
    batch_data = []
    for idx, row in batch_df.iterrows():
        batch_data.append({
            "行号": idx,
            "厅名中文": row['厅名中文'],
            "日期": row['日期（必填）'],
            "主持": row['主持（必填）'],
            "排麦人员": row['排麦人员（必填）']
        })
    
    # 构建 prompt
    prompt = PROMPT_BATCH.format(
        batch_data=json.dumps(batch_data, ensure_ascii=False, indent=2),
        known_names=json.dumps(assist_known_names, ensure_ascii=False, indent=2)
    )
    
    try:
        # 调用 LLM
        response = model_client.get_completion(prompt, temperature=TEMPERATURE)
        
        # 清理 markdown 代码块
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        # 解析 JSON
        results = json.loads(response)
        
        # 验证结果
        if not isinstance(results, list):
            raise ValueError("返回结果不是数组")
        
        return results
    
    except Exception as e:
        # 批量解析失败，返回默认值
        print(f"\n⚠️ 批量解析失败: {traceback.format_exc()}")
        print(f"\n原始输入数据:")
        print(json.dumps(batch_data, ensure_ascii=False, indent=2))
        print(f"\nLLM 响应:")
        print(response[:500] if 'response' in locals() else "无响应")
        
        return [
            {
                "行号": idx,
                "主持": {
                    "主持人员列表": []
                },
                "排麦": {
            "排麦人员列表": [],
            "缺席人数": 0,
                    "置信度": "low"
                },
                "错误": f"批量解析失败: {str(e)[:50]}"
            }
            for idx in batch_df.index
        ]


def further_split(names: List[str], split_char: str = '-') -> List[str]:
    res = []
    for name in names:
        res.extend(name.split(split_char))
    return res
 

def batch_parse_fields(
    df: pd.DataFrame, 
    date_str_from_file: str, 
    strict_date_filter: bool = False,
    assist_known_names: List[str] = None) -> pd.DataFrame:
    """
    批量解析排麦人员和主持人员字段（每个 batch：构造输入 -> 调用 LLM -> 解析 -> 直接写回 DataFrame）
    """
    total = len(df)
    num_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"开始批量解析 {total} 条记录...")
    print(f"批次设置: 每批 {BATCH_SIZE} 行，共 {num_batches} 批")
    print(f"预计耗时: 约 {num_batches * 0.5:.1f} 分钟")

    # 确保目标列存在
    target_columns = [
        '主持人员列表_AI解析',
        '排麦人员列表_AI解析',
        '排麦出席人数_AI解析',
        '排麦缺席人数_AI解析',
        '排麦置信度_AI解析',
        '标准化日期_AI解析',
        '日期匹配标志_AI解析',
        '主持错误_AI解析',
        '排麦错误_AI解析',
    ]
    for col in target_columns:
        if col not in df.columns:
            df[col] = ''

    total_start_ts = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total)

        batch_df = df.iloc[start_idx:end_idx]

        print(f"处理批次 {batch_idx + 1}/{num_batches} (第 {start_idx+1}-{end_idx} 行)...", end='')
        batch_start_ts = time.time()

        batch_results = parse_batch(batch_df, assist_known_names=assist_known_names)
            
        # 2) 逐行写回
        for result in batch_results:
            idx = result.get('行号')
            host_data = result['主持']
            paimai_data = result['排麦']
            error_msg = result.get('错误', '')

            host_list = further_split(host_data['主持人员列表'], split_char='-')
            paimai_list = further_split(paimai_data['排麦人员列表'], split_char='-')
            lack_num = paimai_data['缺席人数']
            conf = paimai_data['置信度']
            row_standard_date_str = result['标准化日期']

            df.at[idx, '主持人员列表_AI解析'] = '|'.join(host_list)
            df.at[idx, '排麦人员列表_AI解析'] = '|'.join(paimai_list)
            df.at[idx, '排麦出席人数_AI解析'] = len(paimai_list)
            df.at[idx, '排麦缺席人数_AI解析'] = lack_num
            df.at[idx, '排麦置信度_AI解析'] = conf
            df.at[idx, '标准化日期_AI解析'] = row_standard_date_str
            df.at[idx, '日期匹配标志_AI解析'] = (row_standard_date_str == date_str_from_file)
            df.at[idx, '主持错误_AI解析'] = error_msg
            df.at[idx, '排麦错误_AI解析'] = error_msg

            print(row_standard_date_str, date_str_from_file)

        elapsed = time.time() - batch_start_ts
        print(f" 完成，用时 {elapsed:.2f}s")



        # 限流保护
        if batch_idx < num_batches - 1:
            time.sleep(0.5)

    total_elapsed = time.time() - total_start_ts
    print(f"\n✅ 解析完成！总用时 {total_elapsed/60:.2f} 分钟（{total_elapsed:.1f} 秒）")
    
    # 可选的严格过滤：只保留匹配日期的行（用于统计/导出）
    if strict_date_filter:
        before_cnt = len(df)
        df = df[df['标准化日期_AI解析'] == date_str_from_file]

        # print(date_str_from_file)
        # df.to_excel('debug.xlsx', index=False)
        after_cnt = len(df)
        print(f"📆 严格日期过滤: 仅保留 标准化日期=={date_str_from_file} 的行 {after_cnt}/{before_cnt}")
    
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str):
    """
    保存清洗后的数据（按 COLS_CONFIG 排序列）

    Args:
        df: 包含解析结果的数据框
        output_path: 输出文件路径
    """
    # 按 COLS_CONFIG 重新排列列
    col_order = [col['col_name'] for col in COLS_CONFIG if col['col_name'] in df.columns]
    # 添加不在 COLS_CONFIG 中的列（如果有）
    remaining_cols = [col for col in df.columns if col not in col_order]
    final_col_order = col_order + remaining_cols

    df_sorted = df[final_col_order]

    # 保存为普通 Excel
    df_sorted.to_excel(output_path, index=False)

    print(f"✅ 清洗后的数据已保存: {output_path}")
    print(f"   - 原始列: {len(df_sorted.columns) - 7} 列")
    print(f"   - 新增列: 7 列（主持x1、排麦x4、错误x2）")
    print(f"   - 总列数: {len(df_sorted.columns)} 列")
    print(f"   - 列已按 CONFIG.COLS_CONFIG 排序")


def normalize_ascii_lower(s: str) -> str:
    """
    仅将英文字母 A-Z 转为小写，其他字符（含中文）保持不变
    
    Args:
        s: 输入字符串
        
    Returns:
        标准化后的字符串
    """
    if not s:
        return ''
    
    # 仅将英文字母 A-Z 转为小写，其他字符（含中文）不变
    ascii_lower_table = {ord(c): ord(c.lower()) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
    return str(s).translate(ascii_lower_table).strip()


def dudup_names(names: List[str]) -> List[str]:
    """
    去重人名（仅英文大小写不敏感；中文/其他字符保持不变）。
    - 保留第一次出现的展示名顺序
    - 规范键：将 A-Z 转小写，其它字符不变
    """
    from collections import OrderedDict

    if not names:
        return []

    norm_to_display = OrderedDict()
    for display in names:
        if display is None:
            continue
        disp = str(display).strip()
        if not disp:
            continue
        norm = normalize_ascii_lower(disp)
        if norm and norm not in norm_to_display:
            norm_to_display[norm] = disp

    return list(norm_to_display.values())


def apply_color_by_value(excel_path, value_color_mapping, output_path=None):
    """
    根据单元格值进行着色，每个配置项支持独立的匹配模式
    
    Args:
        excel_path: Excel文件路径
        value_color_mapping: 值-颜色映射列表，格式如：
            [
                {'cell_value': 'Siri', 'color_code': 'FF0000', 'mode': 'exact'},      # 精确匹配
                {'cell_value': '王摆摆', 'color_code': '00FF00', 'mode': 'contains_value'},    # 包含匹配
                {'cell_value': 'error', 'color_code': 'FFFF00', 'mode': 'exact'}      # 精确匹配
            ]
        output_path: 输出文件路径，如果为None则覆盖原文件
    """
    import openpyxl
    from openpyxl.styles import PatternFill
    
    # 如果没有指定输出路径，则覆盖原文件
    if output_path is None:
        output_path = excel_path
    
    # 加载工作簿
    wb = openpyxl.load_workbook(excel_path)
    ws = wb.active
    
    # 遍历所有单元格进行着色
    colored_count = 0
    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                cell_value_str = str(cell.value).strip()
                
                # 遍历每个配置项，使用其独立的匹配模式
                for config in value_color_mapping:
                    cell_value = config['cell_value']
                    color_code = config['color_code']
                    mode = config.get('mode', 'exact')  # 默认使用精确匹配
                    
                    # 根据匹配模式进行判断
                    should_color = False
                    if mode == 'exact':
                        # 精确匹配：str == str
                        should_color = (cell_value_str == cell_value)
                    elif mode == 'contains_value':
                        # 包含匹配：cell_value in cell_value_str
                        should_color = (cell_value in cell_value_str)
                    else:
                        raise ValueError(f"不支持的匹配模式: {mode}，请使用 'exact' 或 'contains_value'")
                    
                    # 如果匹配成功，进行着色
                    if should_color:
                        cell.fill = PatternFill(
                            start_color=color_code,
                            end_color=color_code,
                            fill_type='solid'
                        )
                        colored_count += 1
                        break  # 找到一个匹配就停止，避免重复着色
    
    # 保存文件
    wb.save(output_path)
    wb.close()
    
    print(f"✅ 已根据值着色完成")
    print(f"   - 输入文件: {excel_path}")
    print(f"   - 输出文件: {output_path}")
    print(f"   - 着色单元格数: {colored_count}")
    print(f"   - 颜色映射: {len(value_color_mapping)} 个")
    
    return colored_count



def save_cleaned_data_with_formula(df: pd.DataFrame, output_path: str):
    """
    保存清洗后的数据（带公式版本，统计表在右侧）
    
    Args:
        df: 包含解析结果的数据框
        output_path: 输出文件路径
    """
    from openpyxl import Workbook, load_workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import PatternFill, Font, Alignment
    import io

    # 按 COLS_CONFIG 重新排列列
    col_order = [col['col_name'] for col in COLS_CONFIG if col['col_name'] in df.columns]
    remaining_cols = [col for col in df.columns if col not in col_order]
    final_col_order = col_order + remaining_cols
    df_sorted = df[final_col_order]

    # 保存第一步：清洗后的数据
    path_step1 = output_path.replace('.xlsx', '.step1_clean_table.xlsx')
    df_sorted.to_excel(path_step1, index=False)
    
    # 加载工作簿
    wb = load_workbook(path_step1)
    ws = wb.active

    # 找到相关列（用于右侧统计区）
    headers = [cell.value for cell in ws[1]]
    host_list_col = None
    paimai_list_col = None
    
    for idx, header in enumerate(headers, 1):
        if header == '主持人员列表_AI解析':
            host_list_col = idx
        elif header == '排麦人员列表_AI解析':
            paimai_list_col = idx
    
    # 初始化sorted_names为空列表
    sorted_names = []
    
    if host_list_col and paimai_list_col:
        host_col_letter = get_column_letter(host_list_col)
        paimai_col_letter = get_column_letter(paimai_list_col)
        data_row_count = ws.max_row
        
        # 提取所有唯一的人名（从主持和排麦列表中）
        all_names = set()
        
        # 从主持列提取（使用竖线分隔）
        for row in range(2, data_row_count + 1):
            cell_value = ws[f'{host_col_letter}{row}'].value
            if cell_value and str(cell_value).strip():
                names = [name.strip() for name in str(cell_value).split('|')]
                all_names.update(names)
        
        # 从排麦列提取（使用竖线分隔）
        for row in range(2, data_row_count + 1):
            cell_value = ws[f'{paimai_col_letter}{row}'].value
            if cell_value and str(cell_value).strip():
                names = [name.strip() for name in str(cell_value).split('|')]
                all_names.update(names)
        
        # 按字母顺序排序
        all_names = dudup_names(all_names)
        sorted_names = sorted(all_names)
        
        # 统计表放在右侧，空两列后开始
        stats_start_col = len(headers) + 3  # 空两列
        stats_col_1 = get_column_letter(stats_start_col)
        stats_col_2 = get_column_letter(stats_start_col + 1)
        stats_col_3 = get_column_letter(stats_start_col + 2)
        
        # 设置统计表表头（带样式）
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')  # 深蓝色
        header_font = Font(bold=True, color='FFFFFF')  # 白色粗体
        
        ws[f'{stats_col_1}1'] = '人员姓名'
        ws[f'{stats_col_1}1'].fill = header_fill
        ws[f'{stats_col_1}1'].font = header_font
        ws[f'{stats_col_1}1'].alignment = Alignment(horizontal='center')
        
        ws[f'{stats_col_2}1'] = '主持次数'
        ws[f'{stats_col_2}1'].fill = header_fill
        ws[f'{stats_col_2}1'].font = header_font
        ws[f'{stats_col_2}1'].alignment = Alignment(horizontal='center')
        
        ws[f'{stats_col_3}1'] = '排麦次数'
        ws[f'{stats_col_3}1'].fill = header_fill
        ws[f'{stats_col_3}1'].font = header_font
        ws[f'{stats_col_3}1'].alignment = Alignment(horizontal='center')
        
        # 填充人名和公式
        for idx, name in enumerate(sorted_names, 2):  # 从第2行开始
            # 人员姓名
            ws[f'{stats_col_1}{idx}'] = name
            
            # 主持次数（COUNTIF 公式）
            formula_host = f'=COUNTIF({host_col_letter}:{host_col_letter},"*{name}*")'
            ws[f'{stats_col_2}{idx}'] = formula_host
            
            # 排麦次数（COUNTIF 公式）
            formula_paimai = f'=COUNTIF({paimai_col_letter}:{paimai_col_letter},"*{name}*")'
            ws[f'{stats_col_3}{idx}'] = formula_paimai
        
        # 设置列宽
        ws.column_dimensions[stats_col_1].width = 20
        ws.column_dimensions[stats_col_2].width = 15
        ws.column_dimensions[stats_col_3].width = 15

    # 保存
    path_step2 = output_path.replace('.xlsx', '.add_stats_formula.xlsx')
    wb.save(path_step2)
    wb.close()

    # 直接对最终文件进行着色（不生成额外文件）
    path_step3 = output_path.replace('.xlsx', '.step3_nowarn.output.xlsx')
    from pipeline.coloring import apply_column_colors_from_config
    apply_column_colors_from_config(path_step2, COLS_CONFIG, path_step3)

    cell_config_list = [
        {'cell_value': 'medium', 'color_code': 'FF0000','mode': 'exact'},
        {'cell_value': 'low', 'color_code': 'FF0000','mode': 'exact'},
        {'cell_value': '9', 'color_code': 'FF0000','mode': 'exact'},
    ]
    for stat_name in sorted_names:
        if normalize_ascii_lower(stat_name) not in known_names_select:
            cell_config_list.append({'cell_value': stat_name, 'color_code': 'FF0000','mode': 'contains_value'})

    apply_color_by_value(path_step3, cell_config_list, output_path=output_path)

    print(f"✅ 已保存（带公式且着色）: {output_path}")
    print(f"   - 数据列: {len(headers)} 列")
    print(f"   - 统计表: 在右侧（空2列后）")
    print(f"   - 统计人数: {len(sorted_names) if host_list_col and paimai_list_col else 0} 人")
    print(f"   - 公式会自动统计每个人的主持次数和排麦次数")
    print(f"   - 修改人员列表后，统计会自动更新")
    print(f"   - 基于 CONFIG.COLS_CONFIG 配置着色")

    #清理step1和step2中间文件
    Path(path_step1).unlink()
    Path(path_step2).unlink()
    Path(path_step3).unlink()



def generate_statistics(df: pd.DataFrame):
    """
    生成统计报告
    
    Args:
        df: 包含解析结果的数据框
    """
    print("\n" + "="*60)
    print("解析结果统计")
    print("="*60)
    
    total = len(df)
    
    # 主持字段统计
    print("\n【主持字段】")
    host_errors = (df['主持错误_AI解析'] != '').sum()
    print(f"成功解析: {total - host_errors} ({(total-host_errors)/total*100:.1f}%)")
    print(f"解析错误: {host_errors}")
    
    # 排麦字段统计
    print("\n【排麦字段】")
    paimai_high = (df['排麦置信度_AI解析'] == 'high').sum()
    paimai_medium = (df['排麦置信度_AI解析'] == 'medium').sum()
    paimai_low = (df['排麦置信度_AI解析'] == 'low').sum()
    paimai_errors = (df['排麦错误_AI解析'] != '').sum()
    
    print(f"高置信度: {paimai_high} ({paimai_high/total*100:.1f}%)")
    print(f"中置信度: {paimai_medium} ({paimai_medium/total*100:.1f}%)")
    print(f"低置信度: {paimai_low} ({paimai_low/total*100:.1f}%)")
    print(f"解析错误: {paimai_errors}")
    
    # 建议
    need_check = paimai_low + paimai_medium
    print(f"\n建议优先检查: 约 {need_check} 条记录（排麦低+中置信度）")

from pathlib import Path
# ========== 主流程 ==========
import re 
def extract_chinese(text):
    try:
        return re.sub(r'[^\u4e00-\u9fa5]', '', text)
    except:
        raise ValueError(text)



def get_hall_2_names():
    hall_2_names = {}
    for p in Path('name_list').glob('*.txt'):
        with open(p, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.read().splitlines() if line.strip()]
            hall_2_names[p.stem] = lines
    return hall_2_names
def gen_all_known_names(known_names_from_ui: List[str], selected_halls: List[str]):
    hall_2_names = get_hall_2_names()
    known_names_from_local = []
    for hall in selected_halls:
        known_names_from_local.extend(hall_2_names[hall])

    
    all_names = []
    if known_names_from_ui:
        all_names.extend(known_names_from_ui)
    all_names.extend(known_names_from_local)
    all_names = list(set(all_names))
    return all_names
 

def process_one_file(
        excel_path: str, 
        output_path: str, 
        date_str_from_file: str, 
        strict_date_filter: bool = False, 
        selected_halls: list = None,
        known_names_from_ui: List[str] = None):
    """处理单个 Excel 文件并输出最终着色文件

    Args:
        excel_path: 输入 Excel 路径
        output_path: 最终着色文件路径（必填）。
        date_str_from_file: 日期字符串
        strict_date_filter: 是否严格按日期筛选
        selected_halls: 选择的厅号列表，用于筛选数据
        known_names_from_ui: 从前端上传的已知人名列表
    """
    output_dir = Path(output_path).parent
    # 确保只在temp目录中创建目录
    import tempfile
    temp_dir = tempfile.gettempdir()
    if not str(output_dir).startswith(temp_dir):
        output_dir = Path(tempfile.mkdtemp())
        output_path = str(output_dir / Path(output_path).name)
    output_dir.mkdir(exist_ok=True)
    print("="*60)
    print("排麦人员 & 主持人员字段清洗 Pipeline")
    print("="*60)
    print(f"当前使用模型: {CURRENT_MODEL.upper()}")
    print("="*60)
    
    # 1. 读取数据
    print(f"\n📖 读取数据: {excel_path}")
    df = pd.read_excel(excel_path, sheet_name=0)
    print(excel_path, '+++++++++++++++')
    df['厅名中文'] = df['厅号（必填）'].apply(extract_chinese)
    print(f"   - 原始记录数: {len(df)}")
        
    # 筛选特定厅号（必须选择厅号）
    if not selected_halls or len(selected_halls) == 0:
        raise ValueError("❌ 错误：必须至少选择一个厅号进行处理")
    
    # 从 name_list 目录读取辅助名单
    all_known_names = gen_all_known_names(known_names_from_ui, selected_halls)
        
    # 将厅号列表拼接成正则表达式，例如：['醉春色', '百媚生'] -> '醉春色|百媚生'
    hall_filter = '|'.join(selected_halls)
    print(f"   - 使用选择的厅号筛选: {hall_filter}")
    df = df[df['厅号（必填）'].str.contains(hall_filter, na=False)]
    print(f"✅ 读取完成，筛选后共 {len(df)} 条记录")

 
    # 2. 批量解析
    print(f"\n🤖 开始 LLM 批量解析...")
    df_parsed = batch_parse_fields(
        df,
        date_str_from_file=date_str_from_file,
        strict_date_filter=strict_date_filter,
        assist_known_names=all_known_names
    )
    
    # 3. 生成统计
    generate_statistics(df_parsed)
    
    # 4. 仅保存带公式版本并生成着色版本（中间产物）
    print(f"\n📊 保存带公式版本并着色...")
    save_cleaned_data_with_formula(df_parsed, str(output_path))

    # 在最终路径上已完成写入与着色
    
    print("\n" + "="*60)
    print("✅ Pipeline 执行完成！")
    print("="*60)
    print(f"\n📋 输出文件：")
    print(f"1. {output_path}")
    print(f"   - 带公式且着色版本")
    print(f"   - 基于 CONFIG.COLS_CONFIG 配置着色")
    print(f"\n💡 说明：仅输出带颜色版本，便于直接人工校验")


from datetime import datetime 

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def get_date_str_from_text(text):
    PROMPT_DATE_EXTRACT = f'''
    当前时间是: {datetime.now().strftime('%Y年')}

    你是一个时间日期解析助手。请根据以下文本，返回这个文件合适的日期字符串。
    
    输入文本：{text}
    
    **重要规则：**
    1. 如果文件名中**没有明确的日期信息**，返回空字符串 ''
    2. 如果日期信息**有歧义**（如09223可能是0922或0923），返回空字符串 ''
    3. 如果日期信息**不完整**（如只有月份没有日期），返回空字符串 ''
    4. 只有**明确且无歧义**的日期才返回 YYYYMMDD 格式
    5. 支持多种日期格式的识别和转换
    
    返回格式为 YYYYMMDD 的日期字符串，不确定时返回空字符串 ''。
    
    示例：
    输入：映客10月03日打卡数据.xlsx， 且我们已知当年是2025年，则
    输出：20251003
    
    输入：主持打卡9.27.xlsx
    输出：20250927
    
    输入：打卡数据.xlsx
    输出：
    
    输入：主持打卡09223.xlsx
    这个有歧义
    输出：
    
    输入：10月打卡.xlsx
    输出：
    
    输入：主持打卡2024-10-03.xlsx
    输出：20241003
    
  

    **只返回日期字符串或空字符串，不要有任何其他文字！**
    '''
    try:
        response = model_client.get_completion(
            PROMPT_DATE_EXTRACT.format(text=text)
        )
        
        # 清理响应内容
        response = response.strip().replace("'", "")
        
        # 验证返回格式
        if response == '':
            return ''
        
        # 检查是否为8位数字格式
        if len(response) == 8 and response.isdigit():
            return response
        
        # 如果不符合格式，返回空字符串
        return ''
        
    except Exception as e:
        print(f"日期提取出错: {e}")
        return ''


model_client = _get_client(model_name=CURRENT_MODEL)


def gen_names():
    
    names = []
    for p in Path(r'F:\vscode_workspace\smart_table\daily_data').glob('*.xlsx'):
        if 'output' in p.stem:
            df = pd.read_excel(p, sheet_name=0)
            for idx, row in df.iterrows():
                host_list_text = row['主持人员列表_AI解析']
                paimai_list_text = row['排麦人员列表_AI解析']
                if host_list_text and paimai_list_text:
                    # 确保是字符串类型，处理 NaN 等异常值
                    host_list_text = str(host_list_text) if host_list_text is not None else ''
                    paimai_list_text = str(paimai_list_text) if paimai_list_text is not None else ''
                    
                    host_list = [name.strip() for name in host_list_text.split('|') if name.strip()]
                    paimai_list = [name.strip() for name in paimai_list_text.split('|') if name.strip()]
                    for host_name in host_list:
                        names.extend(host_name.split('-'))
                    for paimai_name in paimai_list:
                        names.extend(paimai_name.split('-'))
    names = list(set(names))
    # 写入temp目录而不是当前目录
    import tempfile
    temp_dir = tempfile.gettempdir()
    known_names_path = os.path.join(temp_dir, 'known_names_v2.txt')
    with open(known_names_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(names))
        

 

 
import time 

if __name__ == '__main__':
    # gen_names()
    # raise 
 
    paths = list(Path('daily_data').glob('*.xlsx'))
    for p in paths:
        p = Path(p)
 
        output_path = p.with_suffix(f'.{CURRENT_MODEL}.v3.output.xlsx')
        # if 'v3' in p.stem:
        #     continue
        if 'output' in p.stem:
            continue 

        date_str_from_file = get_date_str_from_text(p.stem)
        from global_config import STRICT_DATE_FILTER
        process_one_file(p, output_path=output_path, date_str_from_file=date_str_from_file, strict_date_filter=STRICT_DATE_FILTER)
        time.sleep(1)