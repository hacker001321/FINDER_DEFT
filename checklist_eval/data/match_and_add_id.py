#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据字段内容匹配来添加ID
将参考文件中的ID匹配到目标文件中
"""

import json
import sys
from pathlib import Path


def match_and_add_ids(reference_file, target_file, output_file, 
                      reference_match_field='prompt', 
                      target_match_field='question',
                      id_field='id'):
    """
    根据字段内容匹配来添加ID
    
    Args:
        reference_file: 参考文件（包含ID的文件）
        target_file: 目标文件（需要添加ID的文件）
        output_file: 输出文件
        reference_match_field: 参考文件中用于匹配的字段名
        target_match_field: 目标文件中用于匹配的字段名
        id_field: ID字段名
    """
    reference_path = Path(reference_file)
    target_path = Path(target_file)
    
    if not reference_path.exists():
        print(f"❌ 错误: 参考文件不存在: {reference_file}")
        return
    
    if not target_path.exists():
        print(f"❌ 错误: 目标文件不存在: {target_file}")
        return
    
    print(f"📖 读取参考文件: {reference_file}")
    print(f"📖 读取目标文件: {target_file}")
    
    # 第一步：读取参考文件，建立匹配字段到ID的映射
    match_to_id = {}
    reference_count = 0
    
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                reference_count += 1
                
                if reference_match_field in data and id_field in data:
                    match_key = data[reference_match_field].strip()
                    match_to_id[match_key] = data[id_field]
                else:
                    print(f"⚠️  参考文件行 {line_num} 缺少字段: {reference_match_field} 或 {id_field}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ 警告: 参考文件行 {line_num} 解析失败: {e}")
                continue
    
    print(f"✅ 参考文件读取完成: {reference_count} 条记录, {len(match_to_id)} 个唯一匹配键")
    
    # 第二步：读取目标文件，匹配并添加ID
    matched_count = 0
    unmatched_count = 0
    already_has_id = 0
    total_processed = 0
    
    with open(target_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line_num, line in enumerate(f_in, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    total_processed += 1
                    
                    # 检查是否已有ID
                    if id_field in data:
                        already_has_id += 1
                        # 保持原有ID
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        continue
                    
                    # 尝试匹配
                    if target_match_field in data:
                        match_key = data[target_match_field].strip()
                        
                        if match_key in match_to_id:
                            # 找到匹配，添加ID
                            matched_id = match_to_id[match_key]
                            data_with_id = {id_field: matched_id}
                            data_with_id.update(data)
                            matched_count += 1
                            print(f"✓ 行 {line_num}: 匹配成功，添加 ID={matched_id}")
                        else:
                            # 未找到匹配
                            data_with_id = data
                            unmatched_count += 1
                            print(f"✗ 行 {line_num}: 未找到匹配 (question前30字符: {match_key[:30]}...)")
                    else:
                        print(f"⚠️  目标文件行 {line_num} 缺少字段: {target_match_field}")
                        data_with_id = data
                        unmatched_count += 1
                    
                    # 写入输出
                    f_out.write(json.dumps(data_with_id, ensure_ascii=False) + '\n')
                    
                except json.JSONDecodeError as e:
                    print(f"❌ 警告: 目标文件行 {line_num} 解析失败: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print(f"✅ 处理完成!")
    print(f"{'='*60}")
    print(f"   📊 总处理行数: {total_processed}")
    print(f"   ✓ 匹配成功: {matched_count}")
    print(f"   ✗ 未匹配: {unmatched_count}")
    print(f"   ⚠️  已有ID: {already_has_id}")
    print(f"   💾 输出文件: {output_file}")
    print(f"{'='*60}")
    
    if unmatched_count > 0:
        print(f"\n⚠️  警告: 有 {unmatched_count} 条记录未找到匹配的ID")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='根据字段内容匹配来添加ID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本用法（默认匹配 prompt 和 question）
  python match_and_add_id.py reference.jsonl target.jsonl output.jsonl
  
  # 指定自定义匹配字段
  python match_and_add_id.py ref.jsonl target.jsonl output.jsonl \\
      --ref-field prompt --target-field question
  
  # 具体示例
  python match_and_add_id.py kimi-k2-0711-preview.jsonl \\
      tongyi_deep_research.jsonl \\
      tongyi_deep_research_with_id.jsonl
        """
    )
    
    parser.add_argument('reference_file', type=str, 
                       help='参考文件（包含ID的文件）')
    parser.add_argument('target_file', type=str, 
                       help='目标文件（需要添加ID的文件）')
    parser.add_argument('output_file', type=str, 
                       help='输出文件')
    parser.add_argument('--ref-field', type=str, default='prompt',
                       help='参考文件中用于匹配的字段名 (默认: prompt)')
    parser.add_argument('--target-field', type=str, default='question',
                       help='目标文件中用于匹配的字段名 (默认: question)')
    parser.add_argument('--id-field', type=str, default='id',
                       help='ID字段名 (默认: id)')
    
    args = parser.parse_args()
    
    match_and_add_ids(
        reference_file=args.reference_file,
        target_file=args.target_file,
        output_file=args.output_file,
        reference_match_field=args.ref_field,
        target_match_field=args.target_field,
        id_field=args.id_field
    )


if __name__ == "__main__":
    main()



