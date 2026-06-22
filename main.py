#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
在线课程完课路径分析系统 - 主入口

功能：
1. 生成/加载用户、课程、章节、播放、测验、作业、讨论、退款、证书数据
2. 定义三级完课口径（严格/标准/宽松），八类学习行为分类
3. 计算：章节流失、测验卡点、作业拖延、互动贡献、证书转化、课程推荐
4. 学习行为清洗：标记播放时长异常长、倍速跳过、频繁刷新、后台挂播、多设备同时播放等异常记录，
   完课率报告说明哪些数据被剔除、保留或降权
5. 测验卡点深度定位：关联视频片段、作业分数、讨论区问题，输出内容优化线索和受影响学员群体
6. 输出：章节漏斗、用户分群、异常记录、数据清洗、卡点深度定位、课程对比、可复跑报告
7. 核心原则：不能只用播放总时长判断学习效果
"""

import sys
import os
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
else:
    import io
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

from src.report_generator import run_pipeline


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="在线课程完课路径分析系统 - 多维度行为分析与智能洞察",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py                              # 使用默认参数（生成模拟数据）
  python main.py --level strict               # 使用严格完课口径
  python main.py --use-existing               # 使用已有的data/目录下CSV数据
  python main.py --data-dir ./mydata --level lenient

数据格式（CSV列）：
  play_records.csv: user_id, course_id, chapter_id, play_duration_minutes, 
                    playback_speed, pause_count, is_background, progress_ratio, replay_count
        """
    )
    parser.add_argument("--data-dir", default="data", help="数据目录 (默认: data)")
    parser.add_argument("--output-dir", default="output", help="报告输出目录 (默认: output)")
    parser.add_argument("--level", default="standard",
                        choices=["strict", "standard", "lenient"],
                        help="完课口径等级: strict(严格)/standard(标准)/lenient(宽松) (默认: standard)")
    parser.add_argument("--use-existing", action="store_true",
                        help="使用 data-dir 中已有的CSV数据，不重新生成模拟数据")

    args = parser.parse_args()

    report_info = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        completion_level=args.level,
        use_existing_data=args.use_existing
    )

    print(f"\n📋 下一步操作建议：")
    print(f"  1. 在浏览器中打开报告：{report_info['report_path']}")
    print(f"  2. 查看导出的CSV数据：{report_info['csv_dir']}")
    print(f"  3. 如需更换完课口径，重新运行: python main.py --level strict")


if __name__ == "__main__":
    main()
