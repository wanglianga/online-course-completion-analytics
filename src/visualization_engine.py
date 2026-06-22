import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import json
import threading


BEHAVIOR_COLORS = {
    "试看用户": "#FFA07A",
    "跳看用户": "#FFD700",
    "重复播放用户": "#87CEEB",
    "后台挂时长用户": "#DDA0DD",
    "真实学习用户": "#90EE90",
    "退款后继续观看用户": "#F08080",
    "流失用户": "#C0C0C0",
    "完课用户": "#32CD32"
}


class VisualizationEngine:
    """可视化引擎 - 使用Plotly生成各类分析图表"""

    def __init__(self, output_dir="output", enable_png=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.charts_dir = self.output_dir / "charts"
        self.charts_dir.mkdir(exist_ok=True)
        self.generated_charts = []
        self._enable_png = enable_png
        self._png_failed = False

    def _save_png_with_timeout(self, fig, png_path, timeout=8):
        """带超时的PNG导出，避免kaleido卡死"""
        result = {"success": False, "error": None}

        def _export():
            try:
                fig.write_image(str(png_path), width=1200, height=800, scale=2)
                result["success"] = True
            except Exception as e:
                result["error"] = str(e)[:80]

        t = threading.Thread(target=_export, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            result["error"] = "导出超时"
        return result

    def _save_figure(self, fig, name):
        """保存图表为HTML和（可选）PNG"""
        html_path = self.charts_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)

        if self._enable_png and not self._png_failed:
            try:
                png_path = self.charts_dir / f"{name}.png"
                result = self._save_png_with_timeout(fig, png_path)
                if not result["success"]:
                    print(f"  提示: {name} PNG跳过({result['error']})")
                    self._png_failed = True
            except Exception as e:
                print(f"  提示: {name} PNG跳过({str(e)[:50]})")
                self._png_failed = True

        self.generated_charts.append({
            "name": name,
            "html_path": str(html_path),
            "relative_path": f"charts/{name}.html"
        })
        return str(html_path)

    def plot_chapter_funnel(self, funnel_df, course_id=None):
        """章节漏斗图"""
        if course_id:
            data = funnel_df[funnel_df["course_id"] == course_id].copy()
            course_name = data["course_name"].iloc[0] if len(data) > 0 else course_id
        else:
            data = funnel_df.groupby("chapter_index").agg({
                "total_enrolled": "sum",
                "users_accessed": "sum",
                "users_started": "sum",
                "users_progress_50": "sum",
                "users_completed": "sum",
                "chapter_name": "first"
            }).reset_index()
            course_name = "全部课程汇总"

        if len(data) == 0:
            return None

        stages = [
            ("报名人数", "total_enrolled"),
            ("访问章节", "users_accessed"),
            ("开始学习", "users_started"),
            ("进度50%", "users_progress_50"),
            ("完成学习", "users_completed")
        ]

        fig = go.Figure()

        colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

        for i, (stage_name, col) in enumerate(stages):
            values = data[col].tolist()
            labels = data["chapter_name"].tolist() if "chapter_name" in data.columns else [f"第{idx+1}章" for idx in range(len(data))]

            fig.add_trace(go.Bar(
                name=stage_name,
                x=labels,
                y=values,
                marker_color=colors[i],
                text=[f"{v:,}" for v in values],
                textposition="auto",
                hovertemplate=f"<b>{stage_name}</b><br>章节: %{{x}}<br>人数: %{{y:,}}<extra></extra>"
            ))

        fig.update_layout(
            title=f"章节学习漏斗 - {course_name}",
            barmode="group",
            xaxis_title="章节",
            yaxis_title="人数",
            legend_title="学习阶段",
            hovermode="x unified",
            height=600,
            showlegend=True
        )

        fig.update_xaxes(tickangle=-45)

        suffix = course_id if course_id else "all"
        return self._save_figure(fig, f"chapter_funnel_{suffix}")

    def plot_dropout_analysis(self, funnel_df):
        """章节流失分析图"""
        course_ids = funnel_df["course_id"].unique()

        fig = make_subplots(
            rows=len(course_ids), cols=1,
            subplot_titles=[funnel_df[funnel_df["course_id"] == cid]["course_name"].iloc[0] for cid in course_ids],
            vertical_spacing=0.08
        )

        for idx, cid in enumerate(course_ids):
            course_data = funnel_df[funnel_df["course_id"] == cid].copy()

            fig.add_trace(
                go.Bar(
                    x=course_data["chapter_name"],
                    y=course_data["dropout_rate_from_previous"] * 100,
                    name="章节间流失率",
                    marker_color="#EF553B",
                    text=[f"{v:.1f}%" for v in course_data["dropout_rate_from_previous"] * 100],
                    textposition="outside",
                    showlegend=(idx == 0)
                ),
                row=idx + 1, col=1
            )

            high_dropout = course_data[course_data["is_high_dropout_point"] == True]
            if len(high_dropout) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=high_dropout["chapter_name"],
                        y=high_dropout["dropout_rate_from_previous"] * 100,
                        mode="markers",
                        marker=dict(symbol="star", size=15, color="#FFD700"),
                        name="高流失点",
                        showlegend=(idx == 0)
                    ),
                    row=idx + 1, col=1
                )

            fig.update_xaxes(tickangle=-30, row=idx + 1, col=1)
            fig.update_yaxes(title_text="流失率(%)", range=[0, max(100, course_data["dropout_rate_from_previous"].max() * 120)], row=idx + 1, col=1)

        fig.update_layout(
            title="章节流失率分析（标记高流失点）",
            height=300 * len(course_ids),
            hovermode="x unified"
        )

        return self._save_figure(fig, "dropout_analysis_by_course")

    def plot_user_behavior_segmentation(self, user_course_df):
        """用户分群饼图和分布"""
        behavior_counts = user_course_df["primary_behavior_name"].value_counts().reset_index()
        behavior_counts.columns = ["behavior", "count"]

        colors = [BEHAVIOR_COLORS.get(b, "#999999") for b in behavior_counts["behavior"]]

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=["用户行为分布", "各行为平均关键指标"]
        )

        fig.add_trace(
            go.Pie(
                labels=behavior_counts["behavior"],
                values=behavior_counts["count"],
                marker=dict(colors=colors),
                textinfo="label+percent",
                hole=0.4,
                hovertemplate="<b>%{label}</b><br>用户数: %{value:,}<br>占比: %{percent}<extra></extra>"
            ),
            row=1, col=1
        )

        behavior_metrics = user_course_df.groupby("primary_behavior_name").agg({
            "watch_ratio": "mean",
            "effective_duration_ratio": "mean",
            "chapter_coverage": "mean",
            "quiz_pass_ratio": "mean",
            "hw_pass_ratio": "mean"
        }).reset_index()

        metrics = [
            ("观看完成率", "watch_ratio", "#636EFA"),
            ("有效时长率", "effective_duration_ratio", "#EF553B"),
            ("章节覆盖率", "chapter_coverage", "#00CC96"),
            ("测验通过率", "quiz_pass_ratio", "#AB63FA"),
            ("作业通过率", "hw_pass_ratio", "#FFA15A")
        ]

        for metric_name, metric_col, color in metrics:
            fig.add_trace(
                go.Bar(
                    name=metric_name,
                    x=behavior_metrics["primary_behavior_name"],
                    y=behavior_metrics[metric_col] * 100,
                    marker_color=color,
                    hovertemplate=f"<b>{metric_name}</b><br>%{{x}}<br>值: %{{y:.1f}}%<extra></extra>"
                ),
                row=1, col=2
            )

        fig.update_layout(
            title="用户学习行为分群分析",
            height=600,
            showlegend=True,
            barmode="group"
        )

        fig.update_yaxes(title_text="比例(%)", row=1, col=2, range=[0, 100])

        return self._save_figure(fig, "user_behavior_segmentation")

    def plot_behavior_radar(self, user_course_df):
        """用户行为雷达图对比"""
        behavior_groups = user_course_df.groupby("primary_behavior_name").agg({
            "watch_ratio": "mean",
            "chapter_coverage": "mean",
            "quiz_take_ratio": "mean",
            "hw_submit_ratio": "mean",
            "effective_duration_ratio": "mean",
            "discussion_count": lambda x: min(x.mean() / 5, 1)
        }).reset_index()

        categories = [
            "观看完成率", "章节覆盖率", "测验参与率",
            "作业提交率", "有效学习率", "互动活跃度"
        ]

        fig = go.Figure()

        for _, row in behavior_groups.iterrows():
            values = [
                row["watch_ratio"],
                row["chapter_coverage"],
                row["quiz_take_ratio"],
                row["hw_submit_ratio"],
                row["effective_duration_ratio"],
                min(row["discussion_count"], 1.0)
            ]
            values = [v * 100 for v in values]

            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=row["primary_behavior_name"],
                line=dict(color=BEHAVIOR_COLORS.get(row["primary_behavior_name"], "#999999")),
                opacity=0.7
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix="%"
                )),
            showlegend=True,
            title="用户行为画像雷达图对比",
            height=700
        )

        return self._save_figure(fig, "behavior_radar_comparison")

    def plot_quiz_bottlenecks(self, bottleneck_df):
        """测验卡点分析图"""
        if len(bottleneck_df) == 0:
            return None

        bottleneck_df = bottleneck_df.copy()
        bottleneck_df["label"] = bottleneck_df.apply(
            lambda x: f"{x['course_name'][:8]}-第{x['chapter_index']}章", axis=1
        )

        severity_colors = {"高": "#EF553B", "中": "#FFA15A", "低": "#FFD700", "无": "#90EE90"}

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "测验首次通过率分布",
                "测验平均尝试次数",
                "测验参与率 vs 通过率",
                "卡点严重程度分布"
            ],
            specs=[[{}, {}], [{}, {"type": "pie"}]]
        )

        colors_by_severity = [severity_colors.get(s, "#999") for s in bottleneck_df["bottleneck_severity"]]

        fig.add_trace(
            go.Bar(
                x=bottleneck_df["label"],
                y=bottleneck_df["first_attempt_pass_rate"] * 100,
                marker_color=colors_by_severity,
                text=[f"{v:.0f}%" for v in bottleneck_df["first_attempt_pass_rate"] * 100],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>首次通过率: %{y:.1f}%<extra></extra>"
            ),
            row=1, col=1
        )

        fig.add_shape(
            type="line", x0=-0.5, x1=len(bottleneck_df) - 0.5,
            y0=50, y1=50, line=dict(color="red", dash="dash"),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=bottleneck_df["label"],
                y=bottleneck_df["avg_attempts_per_user"],
                marker_color=colors_by_severity,
                text=[f"{v:.1f}" for v in bottleneck_df["avg_attempts_per_user"]],
                textposition="outside",
                showlegend=False,
                hovertemplate="%{x}<br>平均尝试: %{y:.1f}次<extra></extra>"
            ),
            row=1, col=2
        )

        fig.add_shape(
            type="line", x0=-0.5, x1=len(bottleneck_df) - 0.5,
            y0=2, y1=2, line=dict(color="red", dash="dash"),
            row=1, col=2
        )

        for severity in ["高", "中", "低", "无"]:
            subset = bottleneck_df[bottleneck_df["bottleneck_severity"] == severity]
            if len(subset) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=subset["quiz_take_rate"] * 100,
                        y=subset["overall_pass_rate"] * 100,
                        mode="markers",
                        marker=dict(
                            size=subset["users_attempted_quiz"] * 2 + 10,
                            color=severity_colors[severity],
                            opacity=0.8,
                            line=dict(width=1, color="white")
                        ),
                        text=subset["label"],
                        name=f"卡点-{severity}",
                        hovertemplate="%{text}<br>参与率: %{x:.0f}%<br>通过率: %{y:.0f}%<br>大小=参与人数<extra></extra>"
                    ),
                    row=2, col=1
                )

        severity_counts = bottleneck_df["bottleneck_severity"].value_counts()
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                marker=dict(colors=[severity_colors.get(s, "#999") for s in severity_counts.index]),
                textinfo="label+percent",
                showlegend=True,
                hovertemplate="<b>%{label}</b><br>测验数: %{value}<extra></extra>"
            ),
            row=2, col=2
        )

        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_xaxes(title_text="测验参与率(%)", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="首次通过率(%)", range=[0, 105], row=1, col=1)
        fig.update_yaxes(title_text="平均尝试次数", row=1, col=2)
        fig.update_yaxes(title_text="总体通过率(%)", range=[0, 100], row=2, col=1)

        fig.update_layout(
            title="测验卡点深度分析",
            height=900,
            showlegend=True
        )

        return self._save_figure(fig, "quiz_bottleneck_analysis")

    def plot_homework_delay_analysis(self, delay_df):
        """作业拖延分析图"""
        if len(delay_df) == 0:
            return None

        delay_df = delay_df.copy()
        delay_df["label"] = delay_df.apply(
            lambda x: f"{x['course_name'][:8]}-第{x['chapter_index']}章", axis=1
        )

        delay_colors = {
            "正常": "#90EE90",
            "轻度拖延": "#98FB98",
            "中度拖延": "#FFD700",
            "严重拖延": "#EF553B"
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "作业提交率 vs 拖延率",
                "平均拖延天数分布",
                "准时/延迟提交平均分数对比",
                "拖延等级分布"
            ],
            specs=[[{}, {}], [{}, {"type": "pie"}]]
        )

        colors = [delay_colors.get(l, "#999") for l in delay_df["delay_level"]]

        fig.add_trace(
            go.Bar(
                x=delay_df["label"],
                y=delay_df["submission_rate"] * 100,
                name="提交率",
                marker_color="#636EFA",
                offsetgroup=0
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=delay_df["label"],
                y=delay_df["late_rate"] * 100,
                name="拖延率",
                marker_color="#EF553B",
                offsetgroup=1
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(
                x=delay_df["label"],
                y=delay_df["avg_delay_days"],
                marker_color=colors,
                text=[f"{v:.1f}天" if v > 0 else "" for v in delay_df["avg_delay_days"]],
                textposition="outside",
                showlegend=False
            ),
            row=1, col=2
        )

        delay_levels = list(delay_colors.keys())
        avg_ontime = []
        avg_late = []
        for level in delay_levels:
            subset = delay_df[delay_df["delay_level"] == level]
            avg_ontime.append(subset["avg_score_ontime"].mean())
            avg_late.append(subset["avg_score_late"].mean())

        fig.add_trace(
            go.Bar(
                x=delay_levels,
                y=avg_ontime,
                name="准时提交均分",
                marker_color="#00CC96"
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Bar(
                x=delay_levels,
                y=avg_late,
                name="延迟提交均分",
                marker_color="#FFA15A"
            ),
            row=2, col=1
        )

        level_counts = delay_df["delay_level"].value_counts()
        pie_colors = [delay_colors.get(l, "#999") for l in level_counts.index]
        fig.add_trace(
            go.Pie(
                labels=level_counts.index,
                values=level_counts.values,
                marker=dict(colors=pie_colors),
                textinfo="label+percent",
                showlegend=True
            ),
            row=2, col=2
        )

        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        fig.update_yaxes(title_text="比例(%)", row=1, col=1)
        fig.update_yaxes(title_text="平均拖延天数", row=1, col=2)
        fig.update_yaxes(title_text="平均分数", range=[0, 100], row=2, col=1)

        fig.update_layout(
            title="作业拖延模式分析",
            barmode="group",
            height=900,
            showlegend=True
        )

        return self._save_figure(fig, "homework_delay_analysis")

    def plot_abnormal_records(self, abnormal_df):
        if len(abnormal_df) == 0:
            return None

        abnormal_df = abnormal_df.copy()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "table"}]],
            subplot_titles=["异常类型分布", "Top异常记录详情"]
        )

        type_counts = abnormal_df["abnormal_type"].value_counts()
        type_colors = {
            "后台挂播": "#DDA0DD",
            "倍速跳过": "#FFD700",
            "播放时长异常长": "#FF6B6B",
            "同一章节频繁刷新": "#4ECDC4",
            "多设备同时播放": "#FF8C42",
            "异常快速完成": "#EF553B",
            "退款后继续观看": "#F08080"
        }

        fig.add_trace(
            go.Pie(
                labels=type_counts.index,
                values=type_counts.values,
                marker=dict(colors=[type_colors.get(t, "#999") for t in type_counts.index]),
                textinfo="label+value+percent",
                hole=0.4,
                showlegend=True
            ),
            row=1, col=1
        )

        severity_order = {"高": 0, "中": 1, "低": 2}
        top_abnormal = abnormal_df.copy()
        top_abnormal["severity_order"] = top_abnormal["abnormal_severity"].map(severity_order)
        top_abnormal = top_abnormal.sort_values(["severity_order", "abnormal_type"]).head(10)

        fig.add_trace(
            go.Table(
                header=dict(
                    values=["用户", "课程/章节", "异常类型", "严重度", "处理方式"],
                    fill_color="#636EFA",
                    font=dict(color="white", size=11),
                    align="left"
                ),
                cells=dict(
                    values=[
                        top_abnormal["user_id"],
                        top_abnormal["chapter_name"].str[:15],
                        top_abnormal["abnormal_type"],
                        top_abnormal["abnormal_severity"],
                        top_abnormal.get("treatment", pd.Series(["—"]*len(top_abnormal)))
                    ],
                    fill_color=[
                        ["#FFE4E4" if s == "高" else "#FFF4E4" if s == "中" else "#FFFFFF"
                         for s in top_abnormal["abnormal_severity"]]
                    ],
                    align="left",
                    font=dict(size=10)
                )
            ),
            row=1, col=2
        )

        fig.update_layout(
            title="异常学习行为检测报告",
            height=600
        )

        return self._save_figure(fig, "abnormal_learning_records")

    def plot_course_comparison(self, conversion_df, user_course_df, funnel_df):
        """课程对比看板"""
        conversion_df = conversion_df.copy()

        course_avg_metrics = user_course_df.groupby("course_id").agg({
            "watch_ratio": "mean",
            "effective_duration_ratio": "mean",
            "chapter_coverage": "mean",
            "quiz_pass_ratio": "mean",
            "hw_pass_ratio": "mean",
            "is_completer": "mean"
        }).reset_index()

        course_avg_metrics = course_avg_metrics.merge(
            self._get_course_names(conversion_df), on="course_id"
        )

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "核心完课率对比",
                "证书转化漏斗对比",
                "各维度平均指标雷达",
                "退款率 vs 价格",
                "章节平均流失率",
                "互动参与率对比"
            ],
            specs=[
                [{}, {}],
                [{"type": "scatterpolar"}, {}],
                [{}, {}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.08
        )

        course_names = conversion_df["course_name"].tolist()
        course_colors = px.colors.qualitative.Set3[:len(course_names)]

        completer_rates = [
            course_avg_metrics[course_avg_metrics["course_id"] == cid]["is_completer"].iloc[0] * 100
            if len(course_avg_metrics[course_avg_metrics["course_id"] == cid]) > 0 else 0
            for cid in conversion_df["course_id"]
        ]
        cert_rates = conversion_df["conv_overall_paid_to_cert"] * 100
        refund_rates = conversion_df["refund_rate"] * 100

        x_pos = list(range(len(course_names)))

        fig.add_trace(
            go.Bar(
                x=course_names, y=completer_rates,
                name="行为完课率", marker_color="#636EFA",
                text=[f"{v:.1f}%" for v in completer_rates],
                textposition="outside"
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=course_names, y=cert_rates,
                name="证书转化率", marker_color="#00CC96",
                text=[f"{v:.1f}%" for v in cert_rates],
                textposition="outside"
            ),
            row=1, col=1
        )

        stages = [
            ("报名", conversion_df["stage_1_enrollments"]),
            ("付费", conversion_df["stage_2_paid"]),
            ("活跃", conversion_df["stage_4_active_learners"]),
            ("高进度", conversion_df["stage_6_high_progress"]),
            ("拿证", conversion_df["stage_7_certificates"])
        ]

        for idx, (stage_name, values) in enumerate(stages):
            normalized_values = []
            for i, v in enumerate(values):
                total = conversion_df["stage_1_enrollments"].iloc[i]
                normalized_values.append(v / max(total, 1) * 100)

            fig.add_trace(
                go.Bar(
                    x=course_names, y=normalized_values,
                    name=stage_name,
                    text=[f"{v:.0f}%" if v > 5 else "" for v in normalized_values],
                    textposition="inside"
                ),
                row=1, col=2
            )

        categories = [
            "观看率", "有效率", "章节覆盖",
            "测验通过", "作业通过", "完课率"
        ]

        for idx, (_, course) in enumerate(course_avg_metrics.iterrows()):
            values = [
                course["watch_ratio"],
                course["effective_duration_ratio"],
                course["chapter_coverage"],
                course["quiz_pass_ratio"],
                course["hw_pass_ratio"],
                course["is_completer"]
            ]
            values = [min(v * 100, 100) for v in values]
            values += [values[0]]
            cats = categories + [categories[0]]

            fig.add_trace(go.Scatterpolar(
                r=values, theta=cats,
                fill="toself",
                name=course["course_name"],
                line=dict(color=course_colors[idx % len(course_colors)]),
                opacity=0.6,
                showlegend=True
            ), row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=conversion_df["price"],
                y=refund_rates,
                mode="markers+text",
                marker=dict(
                    size=conversion_df["stage_2_paid"] / 2,
                    color=course_colors[:len(conversion_df)],
                    opacity=0.8
                ),
                text=course_names,
                textposition="top center",
                hovertemplate="%{text}<br>价格: ¥%{x}<br>退款率: %{y:.1f}%<br>大小=付费人数<extra></extra>",
                showlegend=False
            ),
            row=2, col=2
        )

        chapter_avg_dropout = funnel_df.groupby("course_name")["dropout_rate_from_previous"].mean() * 100
        chapter_avg_completion = funnel_df.groupby("course_name")["completion_rate"].mean() * 100

        fig.add_trace(
            go.Bar(
                x=chapter_avg_dropout.index, y=chapter_avg_dropout.values,
                name="平均流失率", marker_color="#EF553B",
                text=[f"{v:.1f}%" for v in chapter_avg_dropout.values],
                textposition="outside"
            ),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(
                x=chapter_avg_completion.index, y=chapter_avg_completion.values,
                name="平均完成率", marker_color="#00CC96",
                text=[f"{v:.1f}%" for v in chapter_avg_completion.values],
                textposition="outside"
            ),
            row=3, col=1
        )

        interaction_df = self._extract_interaction_data(user_course_df)
        fig.add_trace(
            go.Bar(
                x=interaction_df["course_name"],
                y=interaction_df["discussion_rate"] * 100,
                name="讨论参与率",
                marker_color="#AB63FA"
            ),
            row=3, col=2
        )
        fig.add_trace(
            go.Bar(
                x=interaction_df["course_name"],
                y=interaction_df["quiz_hw_rate"] * 100,
                name="测验/作业参与率",
                marker_color="#FFA15A"
            ),
            row=3, col=2
        )

        for r in range(1, 4):
            for c in range(1, 3):
                if (r, c) != (2, 1):
                    fig.update_xaxes(tickangle=-30, row=r, col=c)

        fig.update_yaxes(title_text="比例(%)", row=1, col=1)
        fig.update_yaxes(title_text="相对报名比例(%)", row=1, col=2)
        fig.update_xaxes(title_text="价格(¥)", row=2, col=2)
        fig.update_yaxes(title_text="退款率(%)", row=2, col=2)
        fig.update_yaxes(title_text="比例(%)", row=3, col=1)
        fig.update_yaxes(title_text="参与率(%)", row=3, col=2)

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")
            ),
            title="课程综合对比看板",
            barmode="group",
            height=1200,
            showlegend=True
        )

        return self._save_figure(fig, "course_comparison_dashboard")

    def _get_course_names(self, conversion_df):
        return conversion_df[["course_id", "course_name"]]

    def _extract_interaction_data(self, user_course_df):
        result = []
        for cid in user_course_df["course_id"].unique():
            subset = user_course_df[user_course_df["course_id"] == cid]
            cn = subset["course_name"].iloc[0] if "course_name" in subset.columns else cid
            if "course_name" not in subset.columns:
                continue

            result.append({
                "course_id": cid,
                "course_name": cn,
                "discussion_rate": (subset["discussion_count"] > 0).mean(),
                "quiz_hw_rate": ((subset["quiz_take_ratio"] > 0) | (subset["hw_submit_ratio"] > 0)).mean()
            })
        return pd.DataFrame(result)

    def plot_certificate_funnel(self, conversion_df):
        """证书转化全景漏斗"""
        if len(conversion_df) == 0:
            return None

        totals = conversion_df.select_dtypes(include=[np.number]).sum()

        funnel_stages = [
            ("课程报名", int(totals.get("stage_1_enrollments", 0)), "#636EFA"),
            ("付费转化", int(totals.get("stage_2_paid", 0)), "#00CC96"),
            ("扣除退款", int(totals.get("stage_3_paid_after_refund", 0)), "#EF553B"),
            ("活跃学习", int(totals.get("stage_4_active_learners", 0)), "#AB63FA"),
            ("中期进度", int(totals.get("stage_5_mid_progress", 0)), "#FFA15A"),
            ("高进度", int(totals.get("stage_6_high_progress", 0)), "#19D3F3"),
            ("获得证书", int(totals.get("stage_7_certificates", 0)), "#32CD32")
        ]

        values = [f[1] for f in funnel_stages]
        labels = [f[0] for f in funnel_stages]
        colors = [f[2] for f in funnel_stages]

        text_values = []
        for i, (stage, val, _) in enumerate(funnel_stages):
            if i == 0:
                text_values.append(f"{val:,}")
            else:
                prev = funnel_stages[i-1][1]
                conv = val / max(prev, 1) * 100
                text_values.append(f"{val:,}<br>(上一步转化:{conv:.1f}%)")

        fig = go.Figure(go.Funnel(
            y=labels,
            x=values,
            text=text_values,
            textposition="inside",
            marker=dict(color=colors),
            connector=dict(line=dict(color="#888", width=2)),
            hovertemplate="<b>%{y}</b><br>人数: %{x:,}<extra></extra>"
        ))

        overall_conv = funnel_stages[-1][1] / max(funnel_stages[0][1], 1) * 100
        paid_to_cert = funnel_stages[-1][1] / max(totals.get("stage_2_paid", 1), 1) * 100

        fig.update_layout(
            title=f"证书转化全景漏斗（报名→证书 整体转化:{overall_conv:.2f}%，付费→证书:{paid_to_cert:.2f}%）",
            height=600
        )

        return self._save_figure(fig, "certificate_conversion_funnel")

    def plot_effective_vs_total_watch(self, user_course_df):
        """总时长vs有效时长散点图"""
        user_course_df = user_course_df.copy()
        user_course_df["total_hours"] = user_course_df["total_watch_minutes"] / 60
        user_course_df["effective_hours"] = user_course_df["effective_watch_minutes"] / 60

        behavior_names = user_course_df["primary_behavior_name"].unique()

        fig = go.Figure()

        for behavior in behavior_names:
            subset = user_course_df[user_course_df["primary_behavior_name"] == behavior]
            fig.add_trace(go.Scatter(
                x=subset["total_hours"],
                y=subset["effective_hours"],
                mode="markers",
                name=behavior,
                marker=dict(
                    size=8,
                    color=BEHAVIOR_COLORS.get(behavior, "#999"),
                    opacity=0.7,
                    line=dict(width=1, color="white")
                ),
                text=subset.apply(
                    lambda x: f"{x['user_id']} - {x.get('course_name', x['course_id'])}<br>"
                              f"总时长: {x['total_hours']:.1f}h<br>"
                              f"有效: {x['effective_hours']:.1f}h<br>"
                              f"效率: {x['effective_duration_ratio']:.0%}",
                    axis=1
                ),
                hovertemplate="%{text}<extra></extra>"
            ))

        max_val = max(user_course_df["total_hours"].max(), user_course_df["effective_hours"].max())
        fig.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val],
            mode="lines", name="理想线（100%有效）",
            line=dict(color="red", dash="dash"),
            showlegend=True
        ))

        fig.update_layout(
            title="总播放时长 vs 有效学习时长（证明不能只用总时长判断学习效果）",
            xaxis_title="总播放时长 (小时)",
            yaxis_title="有效学习时长 (小时，扣除后台/异常)",
            height=700,
            showlegend=True,
            hovermode="closest"
        )

        return self._save_figure(fig, "effective_vs_total_duration")

    def plot_behavior_cleaning_summary(self, cleaning_data):
        if not cleaning_data or cleaning_data.get("total_records", 0) == 0:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "数据清洗总览",
                "各异常类型标记数量与处理方式",
                "处理方式分布",
                "完课率影响分布"
            ],
            specs=[
                [{"type": "pie"}, {}],
                [{"type": "pie"}, {}]
            ]
        )

        total = cleaning_data["total_records"]
        clean = cleaning_data["clean_records"]
        flagged = cleaning_data["flagged_records"]

        fig.add_trace(
            go.Pie(
                labels=["正常记录", "标记异常"],
                values=[clean, flagged],
                marker=dict(colors=["#90EE90", "#EF553B"]),
                textinfo="label+value+percent",
                hole=0.4,
                showlegend=True
            ),
            row=1, col=1
        )

        type_detail = cleaning_data.get("type_treatment_detail", [])
        if type_detail:
            types = [t["abnormal_type"] for t in type_detail]
            exclude_vals = [t["exclude_count"] for t in type_detail]
            downweight_vals = [t["downweight_count"] for t in type_detail]
            retain_vals = [t["retain_count"] for t in type_detail]

            fig.add_trace(
                go.Bar(name="剔除", x=types, y=exclude_vals, marker_color="#EF553B"),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name="降权", x=types, y=downweight_vals, marker_color="#FFD700"),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name="保留", x=types, y=retain_vals, marker_color="#90EE90"),
                row=1, col=2
            )

        fig.add_trace(
            go.Pie(
                labels=["剔除", "降权", "保留"],
                values=[
                    cleaning_data.get("exclude_count", 0),
                    cleaning_data.get("downweight_count", 0),
                    cleaning_data.get("retain_count", 0)
                ],
                marker=dict(colors=["#EF553B", "#FFD700", "#90EE90"]),
                textinfo="label+value+percent",
                hole=0.4
            ),
            row=2, col=1
        )

        completion_impact = cleaning_data.get("completion_rate_impact", {})
        if completion_impact:
            impact_pcts = [v["impact_pct"] for v in completion_impact.values()]
            bins = [0, 5, 10, 20, 50, 100]
            labels = ["0~5%", "5~10%", "10~20%", "20~50%", "50%+"]
            hist_vals = pd.cut(impact_pcts, bins=bins, labels=labels, right=False).value_counts().reindex(labels).fillna(0)

            fig.add_trace(
                go.Bar(
                    x=hist_vals.index.tolist(),
                    y=hist_vals.values.tolist(),
                    marker_color=["#90EE90", "#98FB98", "#FFD700", "#FFA15A", "#EF553B"],
                    text=[f"{int(v)}" for v in hist_vals.values],
                    textposition="outside",
                    showlegend=False
                ),
                row=2, col=2
            )

        fig.update_xaxes(tickangle=-30, row=1, col=2)
        fig.update_yaxes(title_text="记录数", row=1, col=2)
        fig.update_xaxes(title_text="完课率影响幅度", row=2, col=2)
        fig.update_yaxes(title_text="用户-课程数", row=2, col=2)
        fig.update_layout(
            title="学习行为数据清洗总览",
            barmode="stack",
            height=800,
            showlegend=True
        )

        return self._save_figure(fig, "behavior_cleaning_summary")

    def plot_quiz_bottleneck_deep(self, deep_df):
        if len(deep_df) == 0:
            return None

        deep_df = deep_df.copy()
        deep_df["label"] = deep_df.apply(
            lambda x: f"{x['course_name'][:8]}-第{x['chapter_index']}章", axis=1
        )

        bottleneck_only = deep_df[deep_df["is_bottleneck"] == True]

        if len(bottleneck_only) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="未发现测验卡点，所有测验通过率正常",
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=20), showarrow=False
            )
            fig.update_layout(title="测验卡点深度定位", height=400)
            return self._save_figure(fig, "quiz_bottleneck_deep_analysis")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "卡点测验：未通过学员 vs 通过学员视频进度对比",
                "卡点测验：关联作业均分对比",
                "讨论区活跃度与未回复率",
                "内容优化线索优先级分布"
            ],
            specs=[
                [{}, {}],
                [{}, {"type": "pie"}]
            ]
        )

        progress_data = []
        for _, row in bottleneck_only.iterrows():
            va = row.get("video_analysis", {})
            if isinstance(va, dict):
                progress_data.append({
                    "label": row["label"],
                    "failed_progress": va.get("failed_avg_progress", 0) * 100,
                    "passed_progress": va.get("passed_avg_progress", 0) * 100
                })
        if progress_data:
            pdf = pd.DataFrame(progress_data)
            fig.add_trace(
                go.Bar(name="未通过学员进度", x=pdf["label"], y=pdf["failed_progress"], marker_color="#EF553B"),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(name="通过学员进度", x=pdf["label"], y=pdf["passed_progress"], marker_color="#00CC96"),
                row=1, col=1
            )

        hw_data = []
        for _, row in bottleneck_only.iterrows():
            ha = row.get("homework_analysis", {})
            if isinstance(ha, dict) and ha.get("has_related_homework"):
                hw_data.append({
                    "label": row["label"],
                    "failed_score": ha.get("failed_avg_hw_score", 0) or 0,
                    "passed_score": ha.get("passed_avg_hw_score", 0) or 0
                })
        if hw_data:
            hdf = pd.DataFrame(hw_data)
            fig.add_trace(
                go.Bar(name="未通过学员作业均分", x=hdf["label"], y=hdf["failed_score"], marker_color="#EF553B", showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(name="通过学员作业均分", x=hdf["label"], y=hdf["passed_score"], marker_color="#00CC96", showlegend=False),
                row=1, col=2
            )

        disc_data = []
        for _, row in bottleneck_only.iterrows():
            da = row.get("discussion_analysis", {})
            if isinstance(da, dict):
                disc_data.append({
                    "label": row["label"],
                    "total": da.get("total_discussions", 0),
                    "unanswered_rate": da.get("unanswered_rate", 0) * 100
                })
        if disc_data:
            ddf = pd.DataFrame(disc_data)
            fig.add_trace(
                go.Bar(name="讨论数", x=ddf["label"], y=ddf["total"], marker_color="#AB63FA", showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=ddf["label"], y=ddf["unanswered_rate"],
                    mode="markers+lines", name="未回复率(%)",
                    marker=dict(size=10, color="#EF553B"),
                    yaxis="y2", showlegend=True
                ),
                row=2, col=1
            )

        all_clues = []
        for _, row in bottleneck_only.iterrows():
            clues = row.get("optimization_clues", [])
            if isinstance(clues, list):
                all_clues.extend(clues)
        if all_clues:
            priority_counts = {}
            for c in all_clues:
                p = c.get("priority", "低")
                priority_counts[p] = priority_counts.get(p, 0) + 1
            fig.add_trace(
                go.Pie(
                    labels=list(priority_counts.keys()),
                    values=list(priority_counts.values()),
                    marker=dict(colors=["#EF553B", "#FFD700", "#90EE90"]),
                    textinfo="label+value+percent",
                    hole=0.4
                ),
                row=2, col=2
            )

        fig.update_xaxes(tickangle=-30, row=1, col=1)
        fig.update_xaxes(tickangle=-30, row=1, col=2)
        fig.update_xaxes(tickangle=-30, row=2, col=1)
        fig.update_yaxes(title_text="进度(%)", row=1, col=1)
        fig.update_yaxes(title_text="分数", row=1, col=2)
        fig.update_yaxes(title_text="讨论数", row=2, col=1)
        fig.update_layout(
            title="测验卡点深度定位分析",
            barmode="group",
            height=900,
            showlegend=True
        )

        return self._save_figure(fig, "quiz_bottleneck_deep_analysis")

    def generate_all_charts(self, analysis_results, user_course_df, data_dict):
        """生成所有图表"""
        print("\n=== 开始生成可视化图表 ===")

        funnel_df = analysis_results["chapter_funnel"]
        bottleneck_df = analysis_results["quiz_bottlenecks"]
        delay_df = analysis_results["homework_delays"]
        interaction_df = analysis_results["interaction_contribution"]
        conversion_df = analysis_results["certificate_conversion"]
        recommendations_df = analysis_results["recommendations"]
        abnormal_df = analysis_results["abnormal_records"]

        print("1. 生成章节漏斗图...")
        for cid in funnel_df["course_id"].unique():
            self.plot_chapter_funnel(funnel_df, cid)
        self.plot_chapter_funnel(funnel_df)

        print("2. 生成章节流失分析...")
        self.plot_dropout_analysis(funnel_df)

        print("3. 生成用户行为分群...")
        self.plot_user_behavior_segmentation(user_course_df)
        self.plot_behavior_radar(user_course_df)

        print("4. 生成测验卡点分析...")
        self.plot_quiz_bottlenecks(bottleneck_df)

        print("5. 生成作业拖延分析...")
        self.plot_homework_delay_analysis(delay_df)

        print("6. 生成异常学习记录...")
        self.plot_abnormal_records(abnormal_df)

        print("7. 生成学习行为数据清洗图表...")
        cleaning_data = analysis_results.get("behavior_cleaning")
        if cleaning_data:
            self.plot_behavior_cleaning_summary(cleaning_data)

        print("8. 生成测验卡点深度定位图表...")
        deep_bottleneck_df = analysis_results.get("quiz_bottleneck_deep")
        if deep_bottleneck_df is not None and len(deep_bottleneck_df) > 0:
            self.plot_quiz_bottleneck_deep(deep_bottleneck_df)

        print("9. 生成课程综合对比...")
        self.plot_course_comparison(conversion_df, user_course_df, funnel_df)

        print("10. 生成证书转化漏斗...")
        self.plot_certificate_funnel(conversion_df)

        print("11. 生成有效时长vs总时长对比...")
        self.plot_effective_vs_total_watch(user_course_df)

        print(f"\n=== 图表生成完成！共生成 {len(self.generated_charts)} 个图表 ===")
        return self.generated_charts
