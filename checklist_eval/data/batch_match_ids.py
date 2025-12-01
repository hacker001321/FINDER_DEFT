#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量为多个文件匹配并添加ID
"""

import json
import sys
from pathlib import Path
from match_and_add_id import match_and_add_ids


def batch_process(reference_file, target_files, output_suffix='_with_id'):
    """
    批量处理多个目标文件
    
    Args:
        reference_file: 参考文件路径
        target_files: 目标文件列表
        output_suffix: 输出文件后缀
    """
    print("="*70)
    print("批量ID匹配处理")
    print("="*70)
    print(f"参考文件: {reference_file}")
    print(f"目标文件数量: {len(target_files)}")
    print("="*70)
    print()
    
    results = []
    
    for idx, target_file in enumerate(target_files, 1):
        target_path = Path(target_file)
        if not target_path.exists():
            print(f"⚠️  跳过不存在的文件: {target_file}")
            continue
        
        # 生成输出文件名
        output_file = target_path.stem + output_suffix + target_path.suffix
        output_path = target_path.parent / output_file
        
        print(f"\n{'='*70}")
        print(f"处理 [{idx}/{len(target_files)}]: {target_path.name}")
        print(f"{'='*70}")
        
        try:
            match_and_add_ids(
                reference_file=reference_file,
                target_file=target_file,
                output_file=str(output_path),
                reference_match_field='prompt',
                target_match_field='question',
                id_field='id'
            )
            results.append({
                'file': target_path.name,
                'output': output_path.name,
                'status': 'success'
            })
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            results.append({
                'file': target_path.name,
                'output': None,
                'status': 'failed',
                'error': str(e)
            })
    
    # 输出总结
    print("\n" + "="*70)
    print("处理总结")
    print("="*70)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    failed_count = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {failed_count}")
    print()
    
    for result in results:
        if result['status'] == 'success':
            print(f"  ✓ {result['file']} -> {result['output']}")
        else:
            print(f"  ✗ {result['file']} (错误: {result.get('error', 'Unknown')})")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='批量为多个文件匹配并添加ID',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单个文件
  python batch_match_ids.py kimi-k2-0711-preview.jsonl tongyi_deep_research.jsonl
  
  # 处理多个文件
  python batch_match_ids.py kimi-k2-0711-preview.jsonl \\
      tongyi_deep_research.jsonl \\
      o3-deep-research.jsonl \\
      o4-mini-deep-research.jsonl
  
  # 指定输出后缀
  python batch_match_ids.py kimi-k2-0711-preview.jsonl \\
      tongyi_deep_research.jsonl \\
      --suffix _matched
        """
    )
    
    parser.add_argument('reference_file', type=str, 
                       help='参考文件（包含ID和prompt的文件）')
    parser.add_argument('target_files', type=str, nargs='+',
                       help='目标文件列表（需要添加ID的文件）')
    parser.add_argument('--suffix', type=str, default='_with_id',
                       help='输出文件后缀 (默认: _with_id)')
    
    args = parser.parse_args()
    
    batch_process(
        reference_file=args.reference_file,
        target_files=args.target_files,
        output_suffix=args.suffix
    )


if __name__ == "__main__":
    main()



