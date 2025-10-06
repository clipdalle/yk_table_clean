"""
主入口文件
"""

import sys

def show_menu():
    """显示菜单"""
    print("="*60)
    print("语音厅考勤表 - 排麦人员字段清洗")
    print("="*60)
    print("\n请选择操作：")
    print("1. 查看数据概览")
    print("2. 测试清洗效果")
    print("3. 批量清洗全部数据")
    print("0. 退出")
    print("\n" + "="*60)

def main():
    """主函数"""
    while True:
        show_menu()
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == '1':
            print("\n正在加载数据...")
            import pandas as pd
            excel_path = 'data/主持打卡9.30.xlsx'
            df = pd.read_excel(excel_path, sheet_name=0)
            print(f"\n📊 数据概览:")
            print(f"  - 总记录数: {len(df)}")
            print(f"  - 字段数: {len(df.columns)}")
            print(f"  - 厅号数量: {df['厅号（必填）'].nunique()}")
            print(f"\n🏢 厅号分布:")
            print(df['厅号（必填）'].value_counts())
            input("\n按 Enter 返回菜单...")
        
        elif choice == '2':
            print("\n开始测试...")
            from pipeline.test_clean_sample import test_single_case, test_batch_sample
            test_single_case()
            input("\n按 Enter 继续批量测试...")
            test_batch_sample()
            input("\n按 Enter 返回菜单...")
        
        elif choice == '3':
            confirm = input("\n确认批量处理全部数据？(y/n): ").strip().lower()
            if confirm == 'y':
                print("\n开始批量处理...")
                from deprecated.clean_pipeline import main as clean_main
                clean_main()
                input("\n按 Enter 返回菜单...")
        
        elif choice == '0':
            print("\n再见！")
            break
        
        else:
            print("\n无效选项，请重新输入")
            input("\n按 Enter 继续...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已退出")
    except Exception as e:
        print(f"\n错误: {e}")
        input("\n按 Enter 退出...")
