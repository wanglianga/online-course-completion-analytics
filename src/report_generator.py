import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generator import DataGenerator
from src.behavior_classifier import (
    LearningBehaviorClassifier, 
    CompletionDefiner, 
    COMPLETION_CRITERIA,
    BEHAVIOR_CLASSIFICATION_RULES
)
from src.metrics_calculator import MetricsCalculator
from src.visualization_engine import VisualizationEngine


HTML_REPORT_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>在线课程完课路径分析报告</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
            background: #f5f7fa; color: #333; line-height: 1.6;
        }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 40px; border-radius: 16px; margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        .header h1 { font-size: 32px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .header .meta { display: flex; gap: 30px; margin-top: 20px; flex-wrap: wrap; }
        .header .meta-item { background: rgba(255,255,255,0.15); padding: 10px 20px; border-radius: 8px; }
        .section { 
            background: white; border-radius: 12px; padding: 30px; margin-bottom: 24px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }
        .section-title { 
            font-size: 22px; color: #2c3e50; margin-bottom: 20px; 
            padding-bottom: 12px; border-bottom: 3px solid #667eea;
            display: flex; align-items: center; gap: 10px;
        }
        .section-title::before {
            content: ""; width: 4px; height: 24px; background: #667eea; border-radius: 2px;
        }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .kpi-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white; padding: 20px; border-radius: 12px;
        }
        .kpi-card.blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .kpi-card.green { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
        .kpi-card.orange { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
        .kpi-card.purple { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
        .kpi-card.teal { background: linear-gradient(135deg, #89f7fe 0%, #66a6ff 100%); }
        .kpi-value { font-size: 32px; font-weight: 700; margin: 8px 0; }
        .kpi-label { font-size: 13px; opacity: 0.9; }
        .kpi-sub { font-size: 12px; opacity: 0.8; margin-top: 4px; }
        .chart-container { 
            background: #fafbfc; border: 1px solid #eef0f4; 
            border-radius: 10px; padding: 16px; margin-bottom: 16px;
        }
        .chart-title { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 12px; }
        .data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .data-table th {
            background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600;
            border-bottom: 2px solid #e9ecef; color: #495057;
        }
        .data-table td { padding: 10px 12px; border-bottom: 1px solid #f1f3f5; }
        .data-table tr:hover { background: #f8f9ff; }
        .tag { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .tag-red { background: #ffe5e5; color: #d63031; }
        .tag-orange { background: #fff4e5; color: #e67e22; }
        .tag-green { background: #e5f9e8; color: #27ae60; }
        .tag-blue { background: #e5f2ff; color: #2980b9; }
        .tag-purple { background: #f3e5ff; color: #8e44ad; }
        .tag-gray { background: #f1f3f5; color: #6c757d; }
        .subsection { margin-top: 24px; }
        .subsection-title { font-size: 17px; color: #34495e; margin: 20px 0 12px; padding-left: 12px; border-left: 4px solid #667eea; }
        .description-box {
            background: #f0f7ff; border-left: 4px solid #3498db; padding: 16px 20px;
            border-radius: 8px; margin-bottom: 16px; font-size: 14px;
        }
        .warning-box {
            background: #fff8e1; border-left: 4px solid #ffc107; padding: 16px 20px;
            border-radius: 8px; margin-bottom: 16px; font-size: 14px;
        }
        .criteria-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
        .criteria-card {
            border: 2px solid #e9ecef; border-radius: 10px; padding: 20px;
            transition: all 0.2s;
        }
        .criteria-card:hover { border-color: #667eea; box-shadow: 0 4px 20px rgba(102,126,234,0.15); }
        .criteria-card h4 { color: #2c3e50; margin-bottom: 10px; }
        .criteria-card p { font-size: 13px; color: #6c757d; margin-bottom: 12px; }
        .criteria-list { list-style: none; font-size: 13px; }
        .criteria-list li { padding: 4px 0; color: #495057; }
        .criteria-list li::before { content: "✓ "; color: #27ae60; font-weight: bold; }
        .insight-list { list-style: none; }
        .insight-list li { 
            padding: 10px 16px; background: #f8f9ff; margin-bottom: 8px; 
            border-radius: 8px; font-size: 14px; border-left: 3px solid #667eea;
        }
        .chart-frame { width: 100%; border: none; border-radius: 8px; min-height: 500px; }
        .nav-bar { 
            position: sticky; top: 0; background: white; z-index: 100; padding: 12px 0;
            margin-bottom: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .nav-links { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 20px; }
        .nav-link {
            padding: 8px 16px; border-radius: 20px; font-size: 13px; cursor: pointer;
            text-decoration: none; color: #667eea; background: #f0f4ff;
            transition: all 0.2s;
        }
        .nav-link:hover { background: #667eea; color: white; }
    </style>
</head>
<body>
    <div class="container">
        {{header}}
        {{nav}}
        {{content}}
    </div>
    <script>
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', e => {
                e.preventDefault();
                const target = document.querySelector(link.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            });
        });
    </script>
</body>
</html>
"""


class ReportGenerator:
    """可复跑报告生成器"""

    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.run_timestamp = datetime.now()

    def _build_header(self, summary_stats):
        return f"""
        <div class="header" id="top">
            <h1>🎓 在线课程完课路径深度分析报告</h1>
            <p>基于多维度行为数据的完课口径定义、学习行为分类与智能洞察</p>
            <div class="meta">
                <div class="meta-item">📅 报告生成: {self.run_timestamp.strftime('%Y-%m-%d %H:%M')}</div>
                <div class="meta-item">👥 用户数: {summary_stats['total_users']:,}</div>
                <div class="meta-item">📚 课程数: {summary_stats['total_courses']}</div>
                <div class="meta-item">🎯 分析口径: {summary_stats['completion_level']}</div>
                <div class="meta-item">🏆 完课率: {summary_stats['overall_completion_rate']:.1%}</div>
            </div>
        </div>
        """

    def _build_nav(self):
        nav_items = [
            ("#top", "🏠 概览"),
            ("#criteria", "📏 口径说明"),
            ("#kpi", "📊 KPI总览"),
            ("#behavior", "👥 用户分群"),
            ("#funnel", "🔻 章节漏斗"),
            ("#quiz", "❓ 测验卡点"),
            ("#hw", "📝 作业拖延"),
            ("#abnormal", "⚠️ 异常记录"),
            ("#compare", "📈 课程对比"),
            ("#cert", "🏆 证书转化"),
            ("#recommend", "💡 推荐线索"),
            ("#insights", "🔍 核心洞察"),
        ]
        nav_links = "".join(
            f'<a href="{href}" class="nav-link">{label}</a>' for href, label in nav_items
        )
        return f'<div class="nav-bar"><div class="nav-links">{nav_links}</div></div>'

    def _build_criteria_section(self):
        criteria_cards = ""
        for level, criteria in COMPLETION_CRITERIA.items():
            level_name = {"strict": "严格", "standard": "标准", "lenient": "宽松"}.get(level, level)
            criteria_cards += f"""
            <div class="criteria-card">
                <h4>{level_name}完课口径</h4>
                <p>{criteria['description']}</p>
                <ul class="criteria-list">
                    <li>视频观看率 ≥ {criteria['watch_ratio_min']:.0%}</li>
                    <li>测验通过率 ≥ {criteria['quiz_pass_ratio_min']:.0%}</li>
                    <li>作业通过率 ≥ {criteria['homework_pass_ratio_min']:.0%}</li>
                    <li>有效时长占比 ≥ {criteria['min_effective_duration_ratio']:.0%}</li>
                    <li>章节覆盖率 ≥ 80%</li>
                </ul>
            </div>
            """

        behavior_cards = ""
        for key, rule in BEHAVIOR_CLASSIFICATION_RULES.items():
            behavior_cards += f"""
            <div class="criteria-card" style="border-top: 4px solid {rule['color']}">
                <h4><span style="color:{rule['color']}">●</span> {rule['name']}</h4>
                <p>{rule['description']}</p>
            </div>
            """

        return f"""
        <div class="section" id="criteria">
            <div class="section-title">📏 分析口径说明</div>
            
            <div class="description-box">
                <b>核心设计原则：</b>不能只用播放总时长判断学习效果。本分析采用<b>多维度综合口径</b>：
                有效学习时长（扣除后台挂时长）+ 进度完成率 + 测验/作业通过 + 互动参与，
                并对用户进行8类学习行为分类，识别真实学习与异常行为。
            </div>

            <div class="subsection">
                <div class="subsection-title">三级完课口径定义</div>
                <div class="criteria-grid">{criteria_cards}</div>
            </div>

            <div class="subsection">
                <div class="subsection-title">八类学习行为分类</div>
                <div class="criteria-grid">{behavior_cards}</div>
            </div>

            <div class="subsection">
                <div class="subsection-title">关键计算公式</div>
                <table class="data-table">
                    <thead>
                        <tr><th>指标</th><th>计算公式</th><th>说明</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>有效学习时长</td><td>总时长 × (1 - 后台占比 × 0.8)</td><td>扣除后台/异常时长</td></tr>
                        <tr><td>观看完成率</td><td>实际观看分钟 / 课程总分钟</td><td>上限100%</td></tr>
                        <tr><td>章节覆盖率</td><td>访问章节数 / 总章节数</td><td>是否按路径学习</td></tr>
                        <tr><td>跳看判定</td><td>平均进度<60% 或 高倍速(≥1.5x)占比>50%</td><td>疑似快速跳过</td></tr>
                        <tr><td>后台判定</td><td>后台播放记录>50% 或 暂停次数异常</td><td>疑似挂时长</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _build_kpi_section(self, summary_stats):
        s = summary_stats
        return f"""
        <div class="section" id="kpi">
            <div class="section-title">📊 核心KPI总览</div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">累计报名人次</div>
                    <div class="kpi-value">{s['total_enrollments']:,}</div>
                    <div class="kpi-sub">付费率 {s['paid_rate']:.1%}</div>
                </div>
                <div class="kpi-card blue">
                    <div class="kpi-label">标准完课人数</div>
                    <div class="kpi-value">{s['completers_standard']:,}</div>
                    <div class="kpi-sub">完课率 {s['completion_rate_standard']:.1%}</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-label">严格完课人数</div>
                    <div class="kpi-value">{s['completers_strict']:,}</div>
                    <div class="kpi-sub">严格完课率 {s['completion_rate_strict']:.1%}</div>
                </div>
                <div class="kpi-card orange">
                    <div class="kpi-label">证书领取数</div>
                    <div class="kpi-value">{s['total_certificates']:,}</div>
                    <div class="kpi-sub">证书转化 {s['cert_conversion_rate']:.1%}</div>
                </div>
                <div class="kpi-card purple">
                    <div class="kpi-label">真实学习用户</div>
                    <div class="kpi-value">{s['real_learners']:,}</div>
                    <div class="kpi-sub">占比 {s['real_learner_ratio']:.1%}</div>
                </div>
                <div class="kpi-card teal">
                    <div class="kpi-label">异常学习记录</div>
                    <div class="kpi-value">{s['abnormal_count']:,}</div>
                    <div class="kpi-sub">含后台/跳看/退款后观看</div>
                </div>
            </div>

            <div class="subsection">
                <div class="subsection-title">各课程完课情况对比</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>课程名称</th><th>难度</th><th>报名数</th><th>付费数</th>
                            <th>标准完课率</th><th>严格完课率</th><th>证书转化</th><th>退款率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {s['course_summary_rows']}
                    </tbody>
                </table>
            </div>
        </div>
        """

    def _build_chart_section(self, section_id, title, chart_names, description=""):
        chart_html = ""
        for chart_name in chart_names:
            chart_html += f"""
            <div class="chart-container">
                <div class="chart-title">{chart_name}</div>
                <iframe src="charts/{chart_name}.html" class="chart-frame" onload="this.style.height=this.contentWindow.document.body.scrollHeight + 'px'"></iframe>
            </div>
            """
        desc_html = f'<div class="description-box">{description}</div>' if description else ""
        return f"""
        <div class="section" id="{section_id}">
            <div class="section-title">{title}</div>
            {desc_html}
            {chart_html}
        </div>
        """

    def _build_abnormal_detail_section(self, abnormal_df):
        if len(abnormal_df) == 0:
            rows = '<tr><td colspan="6" style="text-align:center;padding:20px;color:#6c757d">暂无异常记录</td></tr>'
        else:
            top_df = abnormal_df.head(50)
            rows = ""
            for _, row in top_df.iterrows():
                sev_class = {
                    "高": "tag-red", "中": "tag-orange", "低": "tag-blue"
                }.get(row["abnormal_severity"], "tag-gray")
                rows += f"""
                <tr>
                    <td>{row['user_id']}</td>
                    <td>{row['course_id']}</td>
                    <td>{row['chapter_name'][:20] if pd.notna(row['chapter_name']) else '-'}</td>
                    <td><span class="tag {sev_class}">{row['abnormal_type']}</span></td>
                    <td>{row['indicator_1']}<br><small>{row['indicator_2']}</small></td>
                    <td>{row['suggestion']}</td>
                </tr>
                """

        type_summary = abnormal_df.groupby(["abnormal_type", "abnormal_severity"]).size().reset_index(name="count")
        type_rows = ""
        for _, r in type_summary.iterrows():
            sev_class = {"高": "tag-red", "中": "tag-orange", "低": "tag-blue"}.get(r["abnormal_severity"], "tag-gray")
            type_rows += f"""
            <tr>
                <td>{r['abnormal_type']}</td>
                <td><span class="tag {sev_class}">{r['abnormal_severity']}</span></td>
                <td>{r['count']}</td>
            </tr>
            """

        return f"""
        <div class="section" id="abnormal-detail">
            <div class="section-title">⚠️ 异常学习记录详情（Top 50）</div>
            <div class="warning-box">
                <b>说明：</b>异常记录仅作为疑似标记，需人工复核确认。
                后台挂时长=高暂停率+后台标记；跳看刷课=高倍速+低进度；
                快速完成=短时间内完成大量内容；退款后观看=退款后仍有学习记录。
            </div>
            <div class="subsection">
                <div class="subsection-title">异常类型汇总</div>
                <table class="data-table">
                    <thead><tr><th>异常类型</th><th>严重度</th><th>记录数</th></tr></thead>
                    <tbody>{type_rows}</tbody>
                </table>
            </div>
            <div class="subsection">
                <div class="subsection-title">异常详情列表</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>用户ID</th><th>课程ID</th><th>章节/范围</th>
                            <th>异常类型</th><th>关键指标</th><th>建议处理</th>
                        </tr>
                    </thead>
                    <tbody>{rows}</tbody>
                </table>
            </div>
        </div>
        """

    def _build_recommendations_section(self, recommendations_df):
        if len(recommendations_df) == 0:
            content = '<p style="color:#6c757d;padding:20px;text-align:center">暂无推荐线索</p>'
        else:
            by_type = recommendations_df.groupby("recommend_type").size().reset_index(name="count")
            type_rows = "".join(
                f'<tr><td>{r["recommend_type"]}</td><td>{r["count"]}</td></tr>'
                for _, r in by_type.iterrows()
            )

            priority_class = {"高": "tag-red", "中": "tag-orange", "低": "tag-blue"}
            top_recs = recommendations_df.head(30)
            rec_rows = ""
            for _, r in top_recs.iterrows():
                p_class = priority_class.get(r["priority"], "tag-gray")
                rec_rows += f"""
                <tr>
                    <td>{r['user_id']}</td>
                    <td>{r['user_type']}</td>
                    <td><span class="tag {p_class}">{r['priority']}优先</span></td>
                    <td>{r['target_course_name'][:15]}</td>
                    <td>{r['intervention_type']}</td>
                    <td style="font-size:12px">{r['reason']}</td>
                </tr>
                """

            content = f"""
            <div class="subsection">
                <div class="subsection-title">推荐线索类型分布</div>
                <table class="data-table">
                    <thead><tr><th>推荐类型</th><th>线索数量</th></tr></thead>
                    <tbody>{type_rows}</tbody>
                </table>
            </div>
            <div class="subsection">
                <div class="subsection-title">Top 30 推荐/干预线索</div>
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>用户</th><th>用户类型</th><th>优先级</th>
                            <th>目标课程</th><th>干预方式</th><th>推荐理由</th>
                        </tr>
                    </thead>
                    <tbody>{rec_rows}</tbody>
                </table>
            </div>
            """

        return f"""
        <div class="section" id="recommend">
            <div class="section-title">💡 课程推荐与运营干预线索</div>
            <div class="description-box">
                <b>生成逻辑：</b>
                ① 学习干预：基于当前进度和行为类型生成召回/激励策略；
                ② 进阶推荐：完课用户推荐更高难度课程；
                ③ 同难度推荐：基于历史偏好推荐相似课程；
                ④ 优先等级：流失召回和完课用户进阶为高优先级。
            </div>
            {content}
        </div>
        """

    def _build_insights_section(self, summary_stats):
        insights = summary_stats.get("key_insights", [])
        insight_items = "".join(f"<li>{ins}</li>" for ins in insights) if insights else (
            "<li>建议结合具体数据生成定制化洞察</li>"
        )

        return f"""
        <div class="section" id="insights">
            <div class="section-title">🔍 核心洞察与建议</div>
            <ul class="insight-list">{insight_items}</ul>

            <div class="subsection">
                <div class="subsection-title">可复跑性说明</div>
                <div class="description-box">
                    <b>本报告支持可复跑：</b>
                    <br>1. 将真实数据按模板格式放入 <code>data/</code> 目录（同名CSV文件）
                    <br>2. 重新运行 <code>python main.py</code> 或 <code>python -m src.main</code>
                    <br>3. 系统将自动加载数据、重新计算指标、重新生成所有图表和报告
                    <br>4. 支持通过环境变量调整完课口径：严格/标准/宽松
                    <br>5. 图表支持导出PNG/HTML，数据支持CSV下载
                </div>
            </div>

            <div class="subsection">
                <div class="subsection-title">数据文件模板</div>
                <table class="data-table">
                    <thead>
                        <tr><th>数据文件</th><th>核心字段</th><th>是否必须</th></tr>
                    </thead>
                    <tbody>
                        <tr><td>courses.csv</td><td>course_id, course_name, total_chapters, total_duration_minutes</td><td><span class="tag tag-red">必须</span></td></tr>
                        <tr><td>chapters.csv</td><td>chapter_id, course_id, chapter_index, duration_minutes</td><td><span class="tag tag-red">必须</span></td></tr>
                        <tr><td>users.csv</td><td>user_id, register_date</td><td><span class="tag tag-red">必须</span></td></tr>
                        <tr><td>enrollments.csv</td><td>enrollment_id, user_id, course_id, is_paid</td><td><span class="tag tag-red">必须</span></td></tr>
                        <tr><td>play_records.csv</td><td>play_id, user_id, course_id, chapter_id, play_duration, progress_ratio</td><td><span class="tag tag-red">必须</span></td></tr>
                        <tr><td>quizzes.csv / quiz_attempts.csv</td><td>测验及答题记录</td><td><span class="tag tag-orange">推荐</span></td></tr>
                        <tr><td>homeworks.csv / homework_submissions.csv</td><td>作业及提交记录</td><td><span class="tag tag-orange">推荐</span></td></tr>
                        <tr><td>discussions.csv</td><td>讨论/互动记录</td><td><span class="tag tag-blue">可选</span></td></tr>
                        <tr><td>refunds.csv</td><td>退款记录</td><td><span class="tag tag-blue">可选</span></td></tr>
                        <tr><td>certificates.csv</td><td>证书领取记录</td><td><span class="tag tag-blue">可选</span></td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        """

    def generate_report(self, data_dict, user_course_df, analysis_results, charts, completion_level="standard"):
        """生成完整HTML报告"""
        print("\n=== 生成分析报告 ===")

        summary_stats = self._calculate_summary_stats(data_dict, user_course_df, analysis_results, completion_level)

        header = self._build_header(summary_stats)
        nav = self._build_nav()

        content_parts = [
            self._build_criteria_section(),
            self._build_kpi_section(summary_stats),
            self._build_chart_section(
                "behavior", "👥 用户学习行为分群分析",
                ["user_behavior_segmentation", "behavior_radar_comparison", "effective_vs_total_duration"],
                "通过多维度行为特征将用户分为8类，证明<b>不能只用播放总时长判断学习效果</b>。有效时长vs总时长图展示了后台挂时长、跳看等行为导致的时长虚高。"
            ),
            self._build_chart_section(
                "funnel", "🔻 章节漏斗与流失分析",
                (["chapter_funnel_all"] + [f"chapter_funnel_{cid}" for cid in data_dict["courses"]["course_id"].tolist()] +
                 ["dropout_analysis_by_course"]),
                "每门课程各章节的5级漏斗：报名→访问→开始→50%进度→完成。标记流失率超过20%的高流失点，定位课程内容卡点。"
            ),
            self._build_chart_section(
                "quiz", "❓ 测验卡点深度分析",
                ["quiz_bottleneck_analysis"],
                "识别哪些测验造成用户卡顿：首次通过率低、尝试次数多、参与率低的测验标记为卡点，严重程度分为高/中/低。"
            ),
            self._build_chart_section(
                "hw", "📝 作业拖延模式分析",
                ["homework_delay_analysis"],
                "分析作业提交率、拖延率和分数差距：按时提交 vs 延迟提交的分数对比验证拖延对学习效果的负面影响。"
            ),
            self._build_abnormal_detail_section(analysis_results["abnormal_records"]),
            self._build_chart_section(
                "compare", "📈 课程综合对比看板",
                ["course_comparison_dashboard"],
                "6维度跨课程对比：完课率、转化漏斗、雷达画像、退款分析、流失分析、互动参与。"
            ),
            self._build_chart_section(
                "cert", "🏆 证书转化全景漏斗",
                ["certificate_conversion_funnel"],
                "7级完整转化路径：报名→付费→扣除退款→活跃→中期→高进度→拿证。计算全链路转化率定位瓶颈环节。"
            ),
            self._build_recommendations_section(analysis_results["recommendations"]),
            self._build_insights_section(summary_stats),
        ]

        content = "\n".join(content_parts)

        report_html = HTML_REPORT_TEMPLATE.replace("{{header}}", header).replace("{{nav}}", nav).replace("{{content}}", content)

        report_path = self.output_dir / "在线课程完课路径分析报告.html"
        report_path.write_text(report_html, encoding="utf-8")

        csv_dir = self.output_dir / "data_exports"
        csv_dir.mkdir(exist_ok=True)
        for name, df in analysis_results.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                export_path = csv_dir / f"{name}.csv"
                df.to_csv(export_path, index=False, encoding="utf-8-sig")

        user_course_df.to_csv(csv_dir / "user_course_behavior_classification.csv", index=False, encoding="utf-8-sig")

        summary_json = {
            "report_generated_at": self.run_timestamp.isoformat(),
            "completion_level": completion_level,
            "summary_stats": {k: (v if not isinstance(v, (pd.Series, pd.DataFrame)) else str(v)) for k, v in summary_stats.items() if k != "key_insights" and k != "course_summary_rows"},
            "charts_generated": len(charts),
            "total_data_records": {k: len(v) for k, v in data_dict.items() if isinstance(v, pd.DataFrame)},
            "analysis_outputs": {k: len(v) for k, v in analysis_results.items() if isinstance(v, pd.DataFrame)}
        }
        (self.output_dir / "run_summary.json").write_text(
            json.dumps(summary_json, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

        print(f"\n=== 报告生成完成 ===")
        print(f"主报告: {report_path}")
        print(f"数据导出: {csv_dir}/")
        print(f"图表数量: {len(charts)}")

        return {
            "report_path": str(report_path),
            "csv_dir": str(csv_dir),
            "charts_count": len(charts)
        }

    def _calculate_summary_stats(self, data_dict, user_course_df, analysis_results, completion_level):
        """计算摘要统计"""
        s = {}

        s["total_users"] = len(data_dict["users"])
        s["total_courses"] = len(data_dict["courses"])
        s["total_enrollments"] = len(data_dict["enrollments"])
        s["total_certificates"] = len(data_dict["certificates"])
        s["paid_rate"] = data_dict["enrollments"]["is_paid"].mean() if len(data_dict["enrollments"]) > 0 else 0
        s["completion_level"] = {"strict": "严格", "standard": "标准", "lenient": "宽松"}.get(completion_level, completion_level)

        completer_col = f"is_completed_{completion_level}"
        if completer_col in user_course_df.columns:
            s["overall_completion_rate"] = user_course_df[completer_col].mean()
            s["completers_standard"] = int(user_course_df["is_completed_standard"].sum()) if "is_completed_standard" in user_course_df.columns else 0
            s["completers_strict"] = int(user_course_df["is_completed_strict"].sum()) if "is_completed_strict" in user_course_df.columns else 0
        else:
            s["overall_completion_rate"] = user_course_df["is_completer"].mean()
            s["completers_standard"] = int(user_course_df["is_completer"].sum())
            s["completers_strict"] = int(user_course_df[user_course_df["completion_score_standard"] >= 100]["is_completer"].sum()) if "completion_score_standard" in user_course_df.columns else 0

        s["completion_rate_standard"] = s["completers_standard"] / max(len(user_course_df), 1)
        s["completion_rate_strict"] = s["completers_strict"] / max(len(user_course_df), 1)

        paid_users = user_course_df[user_course_df["is_paid"] == True]
        s["cert_conversion_rate"] = len(data_dict["certificates"]) / max(len(paid_users), 1)

        s["real_learners"] = int(user_course_df["is_real_learner"].sum())
        s["real_learner_ratio"] = s["real_learners"] / max(len(user_course_df), 1)
        s["abnormal_count"] = len(analysis_results["abnormal_records"])

        course_rows = []
        if "course_name" not in user_course_df.columns:
            course_summary = user_course_df.merge(
                data_dict["courses"][["course_id", "course_name", "difficulty", "price"]], on="course_id", how="left"
            )
        else:
            course_summary = user_course_df.copy()
        conv_df = analysis_results["certificate_conversion"]

        for cid in course_summary["course_id"].unique():
            cs = course_summary[course_summary["course_id"] == cid]
            conv = conv_df[conv_df["course_id"] == cid]
            if len(cs) == 0:
                continue

            cn = cs["course_name"].iloc[0]
            diff = cs["difficulty"].iloc[0]
            enrolls = len(cs)
            paid = cs["is_paid"].sum()
            std_rate = cs["is_completed_standard"].mean() if "is_completed_standard" in cs.columns else cs["is_completer"].mean()
            strict_rate = cs["is_completed_strict"].mean() if "is_completed_strict" in cs.columns else 0
            cert_conv = conv["conv_overall_paid_to_cert"].iloc[0] if len(conv) > 0 else 0
            refund_rate = conv["refund_rate"].iloc[0] if len(conv) > 0 else 0

            cert_tag = "tag-green" if cert_conv > 0.15 else ("tag-orange" if cert_conv > 0.05 else "tag-red")
            refund_tag = "tag-green" if refund_rate < 0.03 else ("tag-orange" if refund_rate < 0.08 else "tag-red")

            course_rows.append(f"""
            <tr>
                <td><b>{cn}</b></td>
                <td><span class="tag tag-purple">{diff}</span></td>
                <td>{enrolls}</td>
                <td>{paid}</td>
                <td><span class="tag tag-green">{std_rate:.1%}</span></td>
                <td><span class="tag tag-blue">{strict_rate:.1%}</span></td>
                <td><span class="tag {cert_tag}">{cert_conv:.1%}</span></td>
                <td><span class="tag {refund_tag}">{refund_rate:.1%}</span></td>
            </tr>
            """)
        s["course_summary_rows"] = "".join(course_rows)

        s["key_insights"] = self._generate_insights(data_dict, user_course_df, analysis_results)

        return s

    def _generate_insights(self, data_dict, user_course_df, analysis_results):
        """生成关键洞察"""
        insights = []

        behavior_counts = user_course_df["primary_behavior_name"].value_counts()
        top_behavior = behavior_counts.index[0] if len(behavior_counts) > 0 else "未知"
        insights.append(
            f"👥 <b>用户行为画像：</b>占比最高的行为类型是「{top_behavior}」({behavior_counts.iloc[0]/len(user_course_df):.1%})，"
            f"真实学习用户占比 {user_course_df['is_real_learner'].mean():.1%}，"
            f"疑似异常行为（后台/跳看）占比 {(user_course_df['is_background_idler']|user_course_df['is_skipper']).mean():.1%}。"
        )

        funnel_df = analysis_results["chapter_funnel"]
        high_drop = funnel_df[funnel_df["is_high_dropout_point"] == True]
        if len(high_drop) > 0:
            worst = high_drop.sort_values("dropout_rate_from_previous", ascending=False).iloc[0]
            insights.append(
                f"🔻 <b>章节流失预警：</b>共识别 {len(high_drop)} 个高流失章节，"
                f"最严重的是「{worst['course_name']}-{worst['chapter_name']}」，"
                f"上一章到本章流失率达 {worst['dropout_rate_from_previous']:.1%}，建议重点优化该章节内容难度或衔接。"
            )

        bottleneck_df = analysis_results["quiz_bottlenecks"]
        severe_bn = bottleneck_df[bottleneck_df["bottleneck_severity"] == "高"]
        if len(severe_bn) > 0:
            insights.append(
                f"❓ <b>测验卡点提示：</b>共 {len(bottleneck_df[bottleneck_df['is_bottleneck']==True])} 个测验存在卡点，"
                f"其中 {len(severe_bn)} 个严重卡点，建议降低题目难度或增加前置讲解。"
            )

        delay_df = analysis_results["homework_delays"]
        severe_delay = delay_df[delay_df["delay_level"] == "严重拖延"]
        if len(severe_delay) > 0:
            avg_gap = severe_delay["score_gap"].mean()
            insights.append(
                f"📝 <b>作业拖延影响：</b>严重拖延作业的准时提交vs延迟提交平均分差达 {avg_gap:.1f} 分，"
                f"验证了拖延与学习效果的强负相关，建议增加提交提醒和提前激励。"
            )

        abnormal_df = analysis_results["abnormal_records"]
        if len(abnormal_df) > 0:
            by_type = abnormal_df["abnormal_type"].value_counts()
            insights.append(
                f"⚠️ <b>异常行为提醒：</b>共识别 {len(abnormal_df)} 条异常记录，"
                f"其中「{by_type.index[0]}」占比最高({by_type.iloc[0]}条)，"
                f"建议按严重度分级处理，高风险记录优先人工复核。"
            )

        conv_df = analysis_results["certificate_conversion"]
        if len(conv_df) > 0:
            best = conv_df.sort_values("conv_overall_paid_to_cert", ascending=False).iloc[0]
            worst = conv_df.sort_values("conv_overall_paid_to_cert", ascending=True).iloc[0]
            insights.append(
                f"🏆 <b>证书转化差异：</b>"
                f"「{best['course_name']}」转化率最高({best['conv_overall_paid_to_cert']:.1%})，"
                f"「{worst['course_name']}」最低({worst['conv_overall_paid_to_cert']:.1%})，"
                f"可分析高转化课程的经验进行迁移。"
            )

        refund_df = analysis_results["abnormal_records"][
            analysis_results["abnormal_records"]["abnormal_type"] == "退款后继续观看"
        ]
        if len(refund_df) > 0:
            insights.append(
                f"💰 <b>退款后观看：</b>发现 {len(refund_df)} 个退款后仍继续观看的用户，"
                f"请核查课程权限控制是否生效，同时可作为「误退款」召回线索进行用户回访。"
            )

        recs_df = analysis_results["recommendations"]
        if len(recs_df) > 0:
            high_priority = len(recs_df[recs_df["priority"] == "高"])
            insights.append(
                f"💡 <b>运营线索：</b>共生成 {len(recs_df)} 条推荐/干预线索，"
                f"其中高优先级 {high_priority} 条（流失召回+进阶推荐），"
                f"建议本季度内优先完成高优先级用户的触达。"
            )

        return insights


def run_pipeline(data_dir="data", output_dir="output", completion_level="standard", use_existing_data=False):
    """运行完整分析管线"""
    print("=" * 60)
    print("🎓 在线课程完课路径分析系统")
    print("=" * 60)

    if not use_existing_data:
        print("\n[1/6] 生成模拟数据...")
        generator = DataGenerator()
        data_dict = generator.generate_all(data_dir)
    else:
        print("\n[1/6] 加载已有数据...")
        data_dict = {}
        data_path = Path(data_dir)
        required_files = {
            "courses": "courses.csv",
            "chapters": "chapters.csv",
            "users": "users.csv",
            "enrollments": "enrollments.csv",
            "play_records": "play_records.csv",
            "quizzes": "quizzes.csv",
            "quiz_attempts": "quiz_attempts.csv",
            "homeworks": "homeworks.csv",
            "homework_submissions": "homework_submissions.csv",
            "discussions": "discussions.csv",
            "refunds": "refunds.csv",
            "certificates": "certificates.csv"
        }
        for key, fname in required_files.items():
            fpath = data_path / fname
            if fpath.exists():
                data_dict[key] = pd.read_csv(fpath, encoding="utf-8-sig")
                print(f"  ✓ 加载 {fname}: {len(data_dict[key])} 行")
            else:
                print(f"  ⚠ 缺少 {fname}，将使用空数据")
                data_dict[key] = pd.DataFrame()

    print("\n[2/6] 计算用户-课程多维度指标...")
    classifier = LearningBehaviorClassifier(data_dict)
    user_course_df = classifier.run_classification(completion_level)

    user_course_df = user_course_df.merge(
        data_dict["courses"][["course_id", "course_name", "difficulty", "price"]],
        on="course_id", how="left"
    )

    print("\n[3/6] 应用三级完课口径评估...")
    user_course_df = CompletionDefiner.evaluate_all_levels(user_course_df)

    print("\n[4/6] 计算核心分析指标...")
    calculator = MetricsCalculator(data_dict, user_course_df)
    analysis_results = calculator.run_all_analyses()

    print("\n[5/6] 生成Plotly可视化图表...")
    viz = VisualizationEngine(output_dir)
    charts = viz.generate_all_charts(analysis_results, user_course_df, data_dict)

    print("\n[6/6] 生成可复跑HTML报告...")
    reporter = ReportGenerator(output_dir)
    report_info = reporter.generate_report(
        data_dict, user_course_df, analysis_results, charts, completion_level
    )

    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print(f"📄 报告文件: {report_info['report_path']}")
    print(f"📊 生成图表: {report_info['charts_count']} 个")
    print(f"📁 数据导出: {report_info['csv_dir']}")
    print("=" * 60)

    return report_info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="在线课程完课路径分析系统")
    parser.add_argument("--data-dir", default="data", help="数据目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--level", default="standard", choices=["strict", "standard", "lenient"],
                        help="完课口径等级: strict/standard/lenient")
    parser.add_argument("--use-existing", action="store_true", help="使用已有的数据文件而不重新生成")
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        completion_level=args.level,
        use_existing_data=args.use_existing
    )
