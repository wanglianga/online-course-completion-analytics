import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MetricsCalculator:
    """分析指标计算器 - 章节流失、测验卡点、作业拖延、互动贡献、证书转化、课程推荐线索"""

    def __init__(self, data_dict, user_course_metrics_df):
        self.data = data_dict
        self.courses_df = data_dict["courses"]
        self.chapters_df = data_dict["chapters"]
        self.users_df = data_dict["users"]
        self.enrollments_df = data_dict["enrollments"]
        self.play_records_df = data_dict["play_records"]
        self.quizzes_df = data_dict["quizzes"]
        self.quiz_attempts_df = data_dict["quiz_attempts"]
        self.homeworks_df = data_dict["homeworks"]
        self.homework_submissions_df = data_dict["homework_submissions"]
        self.discussions_df = data_dict["discussions"]
        self.refunds_df = data_dict["refunds"]
        self.certificates_df = data_dict["certificates"]
        self.user_course_df = user_course_metrics_df

    def calculate_chapter_funnel(self):
        """章节漏斗分析 - 计算每个章节的访问、学习、完成和流失情况"""
        funnel_data = []

        all_enrollments = self.enrollments_df.copy()

        for _, course in self.courses_df.iterrows():
            course_id = course["course_id"]
            course_name = course["course_name"]
            total_enrolled = len(all_enrollments[all_enrollments["course_id"] == course_id])

            course_chapters = self.chapters_df[
                self.chapters_df["course_id"] == course_id
            ].sort_values("chapter_index")

            prev_accessed = total_enrolled

            for _, chap in course_chapters.iterrows():
                chapter_id = chap["chapter_id"]
                chapter_name = chap["chapter_name"]
                chapter_index = chap["chapter_index"]
                chap_duration = chap["duration_minutes"]

                chapter_plays = self.play_records_df[
                    self.play_records_df["chapter_id"] == chapter_id
                ]

                users_accessed = chapter_plays["user_id"].nunique()

                users_started = chapter_plays[
                    chapter_plays["play_duration_minutes"] > 1
                ]["user_id"].nunique()

                users_progress_50 = chapter_plays[
                    chapter_plays["progress_ratio"] >= 0.5
                ]["user_id"].nunique()

                users_completed = chapter_plays[
                    chapter_plays["progress_ratio"] >= 0.9
                ]["user_id"].nunique()

                dropout_from_prev = prev_accessed - users_accessed if prev_accessed > 0 else 0
                dropout_rate_from_prev = dropout_from_prev / max(prev_accessed, 1)
                overall_dropout_rate = (total_enrolled - users_accessed) / max(total_enrolled, 1)
                completion_to_access = users_completed / max(users_accessed, 1)

                avg_progress = chapter_plays["progress_ratio"].mean() if len(chapter_plays) > 0 else 0
                avg_watch_minutes = chapter_plays["play_duration_minutes"].mean() if len(chapter_plays) > 0 else 0

                has_quiz = chap["has_quiz"]
                has_hw = chap["has_homework"]

                funnel_data.append({
                    "course_id": course_id,
                    "course_name": course_name,
                    "chapter_id": chapter_id,
                    "chapter_index": chapter_index,
                    "chapter_name": chapter_name,
                    "chapter_duration_minutes": chap_duration,
                    "total_enrolled": total_enrolled,
                    "users_accessed": users_accessed,
                    "users_started": users_started,
                    "users_progress_50": users_progress_50,
                    "users_completed": users_completed,
                    "access_rate": round(users_accessed / max(total_enrolled, 1), 4),
                    "start_rate": round(users_started / max(users_accessed, 1), 4),
                    "progress_50_rate": round(users_progress_50 / max(users_started, 1), 4),
                    "completion_rate": round(users_completed / max(users_accessed, 1), 4),
                    "dropout_from_previous": dropout_from_prev,
                    "dropout_rate_from_previous": round(dropout_rate_from_prev, 4),
                    "overall_dropout_rate": round(overall_dropout_rate, 4),
                    "completion_to_access_ratio": round(completion_to_access, 4),
                    "avg_progress": round(avg_progress, 4),
                    "avg_watch_minutes": round(avg_watch_minutes, 2),
                    "has_quiz": has_quiz,
                    "has_homework": has_hw,
                    "is_free_preview": chap["is_free_preview"]
                })

                prev_accessed = users_accessed

        funnel_df = pd.DataFrame(funnel_data)

        if len(funnel_df) > 0:
            high_dropout = funnel_df[funnel_df["dropout_rate_from_previous"] > 0.2]
            if len(high_dropout) > 0:
                funnel_df["is_high_dropout_point"] = funnel_df["dropout_rate_from_previous"] > 0.2
            else:
                funnel_df["is_high_dropout_point"] = False
        else:
            funnel_df["is_high_dropout_point"] = False

        return funnel_df

    def calculate_quiz_bottlenecks(self):
        """测验卡点分析 - 识别哪些测验是学习瓶颈"""
        bottlenecks = []

        for _, quiz in self.quizzes_df.iterrows():
            quiz_id = quiz["quiz_id"]
            chapter_id = quiz["chapter_id"]
            course_id = quiz["course_id"]

            attempts = self.quiz_attempts_df[self.quiz_attempts_df["quiz_id"] == quiz_id]
            users_attempted = attempts["user_id"].nunique()

            if users_attempted == 0:
                continue

            user_attempt_counts = attempts.groupby("user_id").size()
            users_multiple_attempts = len(user_attempt_counts[user_attempt_counts >= 2])

            first_attempts = attempts[attempts["attempt_number"] == 1]
            first_attempt_pass_rate = (
                first_attempts["passed"].sum() / len(first_attempts) if len(first_attempts) > 0 else 0
            )

            overall_pass_rate = (
                attempts.groupby("user_id")["passed"].max().sum() / users_attempted
            )

            avg_attempts_per_user = attempts.groupby("user_id").size().mean()
            avg_score_first = first_attempts["score"].mean() if len(first_attempts) > 0 else 0
            avg_time_spent = attempts["time_spent_minutes"].mean()

            users_with_chapter_access = self.play_records_df[
                self.play_records_df["chapter_id"] == chapter_id
            ]["user_id"].nunique()

            quiz_take_rate = users_attempted / max(users_with_chapter_access, 1)

            low_pass_threshold = 0.5
            high_attempts_threshold = 2.0
            low_take_threshold = 0.4

            is_bottleneck = (
                first_attempt_pass_rate < low_pass_threshold or
                avg_attempts_per_user > high_attempts_threshold or
                quiz_take_rate < low_take_threshold
            )

            bottleneck_severity = "低"
            bottleneck_factors = []
            if first_attempt_pass_rate < low_pass_threshold:
                bottleneck_factors.append(f"首次通过率低({first_attempt_pass_rate:.0%})")
                bottleneck_severity = "高"
            if avg_attempts_per_user > high_attempts_threshold:
                bottleneck_factors.append(f"平均尝试次数高({avg_attempts_per_user:.1f})")
                bottleneck_severity = "中" if bottleneck_severity == "低" else bottleneck_severity
            if quiz_take_rate < low_take_threshold:
                bottleneck_factors.append(f"参与率低({quiz_take_rate:.0%})")
                bottleneck_severity = "中" if bottleneck_severity == "低" else bottleneck_severity

            chap_info = self.chapters_df[self.chapters_df["chapter_id"] == chapter_id]
            course_info = self.courses_df[self.courses_df["course_id"] == course_id]

            bottlenecks.append({
                "quiz_id": quiz_id,
                "chapter_id": chapter_id,
                "course_id": course_id,
                "course_name": course_info["course_name"].iloc[0] if len(course_info) > 0 else "",
                "chapter_name": chap_info["chapter_name"].iloc[0] if len(chap_info) > 0 else "",
                "chapter_index": chap_info["chapter_index"].iloc[0] if len(chap_info) > 0 else 0,
                "quiz_difficulty": quiz["difficulty"],
                "total_questions": quiz["total_questions"],
                "users_accessed_chapter": users_with_chapter_access,
                "users_attempted_quiz": users_attempted,
                "quiz_take_rate": round(quiz_take_rate, 4),
                "total_attempts": len(attempts),
                "avg_attempts_per_user": round(avg_attempts_per_user, 2),
                "users_with_multiple_attempts": users_multiple_attempts,
                "first_attempt_pass_rate": round(first_attempt_pass_rate, 4),
                "overall_pass_rate": round(overall_pass_rate, 4),
                "avg_first_attempt_score": round(avg_score_first, 1),
                "avg_time_spent_minutes": round(avg_time_spent, 1),
                "is_bottleneck": is_bottleneck,
                "bottleneck_severity": bottleneck_severity if is_bottleneck else "无",
                "bottleneck_factors": "; ".join(bottleneck_factors) if bottleneck_factors else ""
            })

        return pd.DataFrame(bottlenecks)

    def calculate_homework_delays(self):
        """作业拖延分析 - 分析作业提交的延迟模式"""
        delays = []

        for _, hw in self.homeworks_df.iterrows():
            hw_id = hw["homework_id"]
            chapter_id = hw["chapter_id"]
            course_id = hw["course_id"]
            deadline_days = hw["deadline_days"]

            submissions = self.homework_submissions_df[
                self.homework_submissions_df["homework_id"] == hw_id
            ]

            if len(submissions) == 0:
                continue

            user_chapter_play = self.play_records_df[
                (self.play_records_df["chapter_id"] == chapter_id)
            ].groupby("user_id")["play_start_time"].min().reset_index()
            user_chapter_play.columns = ["user_id", "chapter_first_play_time"]

            submissions_with_play = submissions.merge(
                user_chapter_play, on="user_id", how="left"
            )

            submissions_with_play["days_after_play"] = (
                submissions_with_play["submit_time"] - 
                pd.to_datetime(submissions_with_play["chapter_first_play_time"])
            ).dt.total_seconds() / 86400

            late_submissions = submissions_with_play[submissions_with_play["is_late"] == True]
            on_time_submissions = submissions_with_play[submissions_with_play["is_late"] == False]

            delay_days_list = submissions_with_play["delay_days"].tolist()
            avg_delay = submissions_with_play["delay_days"].mean()
            median_delay = submissions_with_play["delay_days"].median()
            max_delay = submissions_with_play["delay_days"].max()

            late_rate = len(late_submissions) / len(submissions_with_play)
            avg_score_late = late_submissions["score"].mean() if len(late_submissions) > 0 else 0
            avg_score_ontime = on_time_submissions["score"].mean() if len(on_time_submissions) > 0 else 0

            users_eligible = self.play_records_df[
                self.play_records_df["chapter_id"] == chapter_id
            ]["user_id"].nunique()
            hw_submit_rate = len(submissions_with_play) / max(users_eligible, 1)

            resubmit_users = submissions_with_play[submissions_with_play["resubmission_count"] > 0]
            resubmit_rate = len(resubmit_users) / len(submissions_with_play)

            high_delay_threshold = 3
            severe_delay_threshold = 7

            delay_level = "正常"
            if avg_delay > severe_delay_threshold or late_rate > 0.4:
                delay_level = "严重拖延"
            elif avg_delay > high_delay_threshold or late_rate > 0.2:
                delay_level = "中度拖延"
            elif avg_delay > 1 or late_rate > 0.1:
                delay_level = "轻度拖延"

            chap_info = self.chapters_df[self.chapters_df["chapter_id"] == chapter_id]
            course_info = self.courses_df[self.courses_df["course_id"] == course_id]

            delays.append({
                "homework_id": hw_id,
                "chapter_id": chapter_id,
                "course_id": course_id,
                "course_name": course_info["course_name"].iloc[0] if len(course_info) > 0 else "",
                "chapter_name": chap_info["chapter_name"].iloc[0] if len(chap_info) > 0 else "",
                "chapter_index": chap_info["chapter_index"].iloc[0] if len(chap_info) > 0 else 0,
                "deadline_days": deadline_days,
                "users_eligible": users_eligible,
                "users_submitted": len(submissions_with_play),
                "submission_rate": round(hw_submit_rate, 4),
                "late_submissions": len(late_submissions),
                "late_rate": round(late_rate, 4),
                "avg_delay_days": round(avg_delay, 1),
                "median_delay_days": round(median_delay if pd.notna(median_delay) else 0, 1),
                "max_delay_days": round(max_delay, 1),
                "avg_score_late": round(avg_score_late, 1),
                "avg_score_ontime": round(avg_score_ontime, 1),
                "score_gap": round(avg_score_ontime - avg_score_late, 1),
                "resubmission_rate": round(resubmit_rate, 4),
                "delay_level": delay_level,
                "delay_distribution": {
                    "0天(准时)": len(submissions_with_play[submissions_with_play["delay_days"] == 0]),
                    "1-3天": len(submissions_with_play[
                        (submissions_with_play["delay_days"] > 0) & (submissions_with_play["delay_days"] <= 3)
                    ]),
                    "4-7天": len(submissions_with_play[
                        (submissions_with_play["delay_days"] > 3) & (submissions_with_play["delay_days"] <= 7)
                    ]),
                    "8天以上": len(submissions_with_play[submissions_with_play["delay_days"] > 7])
                }
            })

        return pd.DataFrame(delays)

    def calculate_interaction_contribution(self):
        """互动贡献分析 - 分析用户讨论、提问等互动行为的贡献度"""
        interactions = []

        for _, course in self.courses_df.iterrows():
            course_id = course["course_id"]
            course_name = course["course_name"]

            course_discussions = self.discussions_df[
                self.discussions_df["course_id"] == course_id
            ]

            total_enrolled = len(self.enrollments_df[self.enrollments_df["course_id"] == course_id])
            total_interactors = course_discussions["user_id"].nunique()

            interaction_rate = total_interactors / max(total_enrolled, 1)

            type_counts = course_discussions["type"].value_counts().to_dict()

            user_stats = course_discussions.groupby("user_id").agg({
                "discussion_id": "count",
                "reply_count": "sum",
                "like_count": "sum",
                "is_answered": lambda x: (x == True).sum() if len(x) > 0 else 0
            }).reset_index()
            user_stats.columns = ["user_id", "post_count", "total_replies", "total_likes", "answered_count"]

            total_posts = len(course_discussions)
            avg_posts_per_user = user_stats["post_count"].mean() if len(user_stats) > 0 else 0

            unanswered_questions = len(course_discussions[
                (course_discussions["type"] == "提问") & (course_discussions["is_answered"] == False)
            ])
            total_questions = len(course_discussions[course_discussions["type"] == "提问"])
            answer_rate = 1 - (unanswered_questions / max(total_questions, 1))

            user_stats["interaction_score"] = (
                user_stats["post_count"] * 10 +
                user_stats["total_replies"] * 2 +
                user_stats["total_likes"] * 1 +
                user_stats["answered_count"] * 5
            )

            high_contributors = len(user_stats[user_stats["interaction_score"] >= 50])
            medium_contributors = len(user_stats[
                (user_stats["interaction_score"] >= 10) & (user_stats["interaction_score"] < 50)
            ])
            low_contributors = len(user_stats[user_stats["interaction_score"] < 10])

            active_users_by_chapter = course_discussions.groupby("chapter_id").agg({
                "user_id": "nunique",
                "discussion_id": "count"
            }).reset_index()
            active_users_by_chapter.columns = ["chapter_id", "interacting_users", "total_posts"]
            active_users_by_chapter = active_users_by_chapter.merge(
                self.chapters_df[["chapter_id", "chapter_name", "chapter_index"]],
                on="chapter_id", how="left"
            ).sort_values("chapter_index")

            interactions.append({
                "course_id": course_id,
                "course_name": course_name,
                "total_enrolled": total_enrolled,
                "total_interactors": total_interactors,
                "interaction_rate": round(interaction_rate, 4),
                "total_posts": total_posts,
                "posts_breakdown": type_counts,
                "avg_posts_per_user": round(avg_posts_per_user, 2),
                "total_questions": total_questions,
                "unanswered_questions": unanswered_questions,
                "answer_rate": round(answer_rate, 4),
                "high_contributors": high_contributors,
                "medium_contributors": medium_contributors,
                "low_contributors": low_contributors,
                "top_contributors": user_stats.nlargest(5, "interaction_score")[
                    ["user_id", "post_count", "total_replies", "total_likes", "interaction_score"]
                ].to_dict("records"),
                "chapter_activity": active_users_by_chapter.to_dict("records")
            })

        return pd.DataFrame(interactions)

    def calculate_certificate_conversion(self):
        """证书转化分析 - 从报名到领证书的转化漏斗"""
        conversions = []

        for _, course in self.courses_df.iterrows():
            course_id = course["course_id"]
            course_name = course["course_name"]

            course_enrollments = self.enrollments_df[self.enrollments_df["course_id"] == course_id]
            total_enrollments = len(course_enrollments)
            paid_enrollments = len(course_enrollments[course_enrollments["is_paid"] == True])

            free_trial_users = len(course_enrollments[course_enrollments["is_paid"] == False])

            refunds = self.refunds_df[self.refunds_df["course_id"] == course_id]
            refund_count = len(refunds)
            refund_rate = refund_count / max(paid_enrollments, 1)

            paid_after_refund = paid_enrollments - refund_count

            active_learners = len(self.user_course_df[
                (self.user_course_df["course_id"] == course_id) &
                (self.user_course_df["watch_ratio"] >= 0.1)
            ])

            mid_progress = len(self.user_course_df[
                (self.user_course_df["course_id"] == course_id) &
                (self.user_course_df["watch_ratio"] >= 0.3)
            ])

            high_progress = len(self.user_course_df[
                (self.user_course_df["course_id"] == course_id) &
                (self.user_course_df["watch_ratio"] >= 0.6)
            ])

            course_certs = self.certificates_df[self.certificates_df["course_id"] == course_id]
            cert_count = len(course_certs)
            excellent_certs = len(course_certs[course_certs["certificate_type"] == "优秀证书"])

            enrollment_to_paid = paid_enrollments / max(total_enrollments, 1)
            paid_to_active = active_learners / max(paid_after_refund, 1)
            active_to_mid = mid_progress / max(active_learners, 1)
            mid_to_high = high_progress / max(mid_progress, 1)
            high_to_cert = cert_count / max(high_progress, 1)
            overall_conversion = cert_count / max(paid_enrollments, 1)

            conversions.append({
                "course_id": course_id,
                "course_name": course_name,
                "price": course["price"],
                "difficulty": course["difficulty"],
                "stage_1_enrollments": total_enrollments,
                "stage_2_paid": paid_enrollments,
                "stage_2_free_trial": free_trial_users,
                "stage_3_paid_after_refund": paid_after_refund,
                "refund_count": refund_count,
                "refund_rate": round(refund_rate, 4),
                "stage_4_active_learners": active_learners,
                "stage_5_mid_progress": mid_progress,
                "stage_6_high_progress": high_progress,
                "stage_7_certificates": cert_count,
                "excellent_certificates": excellent_certs,
                "conv_enroll_to_paid": round(enrollment_to_paid, 4),
                "conv_paid_to_active": round(paid_to_active, 4),
                "conv_active_to_mid": round(active_to_mid, 4),
                "conv_mid_to_high": round(mid_to_high, 4),
                "conv_high_to_cert": round(high_to_cert, 4),
                "conv_overall_paid_to_cert": round(overall_conversion, 4),
                "avg_completion_score": round(
                    course_certs["completion_score"].mean() if len(course_certs) > 0 else 0, 1
                )
            })

        return pd.DataFrame(conversions)

    def generate_course_recommendations(self):
        """课程推荐线索 - 基于用户行为生成个性化推荐和干预建议"""
        recommendations = []

        for _, user in self.users_df.iterrows():
            user_id = user["user_id"]

            user_enrollments = self.enrollments_df[self.enrollments_df["user_id"] == user_id]
            enrolled_course_ids = user_enrollments["course_id"].tolist()

            user_courses = self.user_course_df[self.user_course_df["user_id"] == user_id]

            if len(user_courses) == 0:
                continue

            user_behavior_types = user_courses["primary_behavior"].unique().tolist()
            avg_watch_ratio = user_courses["watch_ratio"].mean()
            avg_effective_ratio = user_courses["effective_duration_ratio"].mean()

            total_discussions = len(self.discussions_df[self.discussions_df["user_id"] == user_id])
            total_certs = len(self.certificates_df[self.certificates_df["user_id"] == user_id])
            total_refunds = len(self.refunds_df[self.refunds_df["user_id"] == user_id])

            unfinished_courses = user_courses[
                (user_courses["watch_ratio"] < 0.7) & (user_courses["watch_ratio"] > 0.1)
            ]

            for _, uc in unfinished_courses.iterrows():
                course_id = uc["course_id"]
                course_info = self.courses_df[self.courses_df["course_id"] == course_id].iloc[0]
                current_progress = uc["watch_ratio"]

                intervention_type = "常规跟进"
                if uc["primary_behavior"] == "dropout" and current_progress < 0.3:
                    intervention_type = "流失召回"
                elif uc["primary_behavior"] == "background_idler":
                    intervention_type = "学习质量提醒"
                elif uc["primary_behavior"] == "skipper":
                    intervention_type = "学习深度引导"
                elif uc["primary_behavior"] == "dropout" and current_progress >= 0.5:
                    intervention_type = "临门一脚激励"
                elif uc["is_refunded_learner"]:
                    intervention_type = "退款后关怀"

                next_chapter = None
                course_chapters = self.chapters_df[
                    self.chapters_df["course_id"] == course_id
                ].sort_values("chapter_index")
                accessed_count = int(uc["chapters_accessed"])
                if accessed_count < len(course_chapters):
                    next_chap = course_chapters.iloc[accessed_count] if accessed_count >= 0 else course_chapters.iloc[0]
                    next_chapter = next_chap["chapter_name"]

                recommendations.append({
                    "user_id": user_id,
                    "user_type": user["user_type"],
                    "recommend_type": "学习干预",
                    "target_course_id": course_id,
                    "target_course_name": course_info["course_name"],
                    "current_progress": f"{current_progress:.0%}",
                    "intervention_type": intervention_type,
                    "primary_behavior": uc["primary_behavior_name"],
                    "next_recommended_chapter": next_chapter,
                    "reason": f"用户在{course_info['course_name']}进度{current_progress:.0%}，{uc['primary_behavior_name']}，建议{intervention_type}",
                    "priority": "高" if intervention_type in ["流失召回", "临门一脚激励"] else "中"
                })

            completed_courses = user_courses[user_courses["is_completer"] == True]
            for _, cc in completed_courses.iterrows():
                course_id = cc["course_id"]
                course_info = self.courses_df[self.courses_df["course_id"] == course_id].iloc[0]

                next_level_courses = self.courses_df[
                    (self.courses_df["difficulty"] == 
                        ("高级" if course_info["difficulty"] in ["入门", "中级"] else "高级")) &
                    (~self.courses_df["course_id"].isin(enrolled_course_ids))
                ]

                for _, rec_course in next_level_courses.iterrows():
                    recommendations.append({
                        "user_id": user_id,
                        "user_type": user["user_type"],
                        "recommend_type": "进阶推荐",
                        "target_course_id": rec_course["course_id"],
                        "target_course_name": rec_course["course_name"],
                        "current_progress": f"已完成{course_info['course_name']}",
                        "intervention_type": "交叉销售",
                        "primary_behavior": cc["primary_behavior_name"],
                        "next_recommended_chapter": None,
                        "reason": f"用户已完成{course_info['course_name']}({course_info['difficulty']})，推荐进阶{rec_course['course_name']}({rec_course['difficulty']})",
                        "priority": "高" if total_certs > 0 else "中"
                    })

            if len(enrolled_course_ids) < 3:
                same_difficulty = self.courses_df[
                    ~self.courses_df["course_id"].isin(enrolled_course_ids)
                ]
                if len(user_courses) > 0:
                    diff_col = "difficulty" if "difficulty" in user_courses.columns else None
                    if diff_col:
                        diff_mode = user_courses[diff_col].mode()
                        most_common_diff = diff_mode.iloc[0] if len(diff_mode) > 0 else "入门"
                    else:
                        merged = user_courses[["course_id"]].merge(self.courses_df[["course_id", "difficulty"]], on="course_id")
                        diff_mode = merged["difficulty"].mode()
                        most_common_diff = diff_mode.iloc[0] if len(diff_mode) > 0 else "入门"

                    prefer_courses = same_difficulty[same_difficulty["difficulty"] == most_common_diff]
                    if len(prefer_courses) > 0:
                        rec = prefer_courses.iloc[0]
                        recommendations.append({
                            "user_id": user_id,
                            "user_type": user["user_type"],
                            "recommend_type": "同难度推荐",
                            "target_course_id": rec["course_id"],
                            "target_course_name": rec["course_name"],
                            "current_progress": f"已报名{len(enrolled_course_ids)}门课",
                            "intervention_type": "个性化推荐",
                            "primary_behavior": ",".join(user_behavior_types),
                            "next_recommended_chapter": None,
                            "reason": f"基于用户偏好{most_common_diff}难度课程，推荐{rec['course_name']}",
                            "priority": "低"
                        })

        return pd.DataFrame(recommendations)

    def calculate_abnormal_learning_records(self):
        """异常学习记录识别 - 识别异常的学习行为模式"""
        abnormal_records = []

        play_with_chapter = self.play_records_df.merge(
            self.chapters_df[["chapter_id", "duration_minutes", "chapter_name"]],
            on="chapter_id", how="left"
        )

        play_with_chapter["effective_speed_watch"] = (
            play_with_chapter["play_duration_minutes"] / 
            (play_with_chapter["duration_minutes"] * play_with_chapter["playback_speed"])
        )

        background_threshold = 0.7
        high_background_sessions = play_with_chapter[
            (play_with_chapter["is_background"] == True) &
            (play_with_chapter["pause_count"] > 10)
        ]

        for _, record in high_background_sessions.head(200).iterrows():
            abnormal_records.append({
                "record_id": record["play_id"],
                "user_id": record["user_id"],
                "course_id": record["course_id"],
                "chapter_id": record["chapter_id"],
                "chapter_name": record.get("chapter_name", ""),
                "abnormal_type": "疑似后台挂时长",
                "abnormal_severity": "高",
                "indicator_1": f"暂停次数: {record['pause_count']}",
                "indicator_2": f"标记后台: {record['is_background']}",
                "indicator_3": f"播放时长: {record['play_duration_minutes']:.1f}分钟",
                "suggestion": "核实是否为有效学习，必要时不计入有效时长"
            })

        speed_threshold = 0.3
        high_speed_skip = play_with_chapter[
            (play_with_chapter["playback_speed"] >= 1.5) &
            (play_with_chapter["progress_ratio"] < speed_threshold) &
            (play_with_chapter["play_duration_minutes"] > 5)
        ]

        for _, record in high_speed_skip.head(200).iterrows():
            abnormal_records.append({
                "record_id": record["play_id"],
                "user_id": record["user_id"],
                "course_id": record["course_id"],
                "chapter_id": record["chapter_id"],
                "chapter_name": record.get("chapter_name", ""),
                "abnormal_type": "疑似跳看/刷课",
                "abnormal_severity": "中",
                "indicator_1": f"倍速: {record['playback_speed']}x",
                "indicator_2": f"进度: {record['progress_ratio']:.0%}",
                "indicator_3": f"时长: {record['play_duration_minutes']:.1f}分钟",
                "suggestion": "高倍速低进度模式，建议人工复核学习有效性"
            })

        user_course_groups = self.play_records_df.groupby(
            ["user_id", "course_id"]
        ).agg({
            "play_duration_minutes": "sum",
            "play_start_time": ["min", "max"],
            "play_id": "count"
        }).reset_index()
        user_course_groups.columns = [
            "user_id", "course_id", "total_minutes", 
            "first_play", "last_play", "session_count"
        ]

        user_course_groups["days_span"] = (
            user_course_groups["last_play"] - user_course_groups["first_play"]
        ).dt.total_seconds() / 86400

        impossible_short = user_course_groups[
            (user_course_groups["total_minutes"] > 300) &
            (user_course_groups["days_span"] < 1)
        ]

        for _, record in impossible_short.head(100).iterrows():
            course = self.courses_df[self.courses_df["course_id"] == record["course_id"]]
            course_duration = course["total_duration_minutes"].iloc[0] if len(course) > 0 else 0
            completion_rate = record["total_minutes"] / max(course_duration, 1)

            abnormal_records.append({
                "record_id": f"SHORT_{record['user_id']}_{record['course_id']}",
                "user_id": record["user_id"],
                "course_id": record["course_id"],
                "chapter_id": "",
                "chapter_name": "整门课程",
                "abnormal_type": "异常快速完成",
                "abnormal_severity": "高" if completion_rate > 0.8 else "中",
                "indicator_1": f"总学习时长: {record['total_minutes']:.0f}分钟",
                "indicator_2": f"学习跨度: {record['days_span']:.1f}天",
                "indicator_3": f"学习会话: {record['session_count']}次",
                "suggestion": "短时间内完成大量内容，可能存在刷课行为，建议重点核查"
            })

        refund_ids = self.refunds_df["enrollment_id"].tolist()
        refund_enrolls = self.refunds_df[["enrollment_id", "user_id", "course_id", "refund_date"]]

        for _, refund in refund_enrolls.iterrows():
            refund_dt = pd.Timestamp(refund["refund_date"])
            after_refund_plays = self.play_records_df[
                (self.play_records_df["enrollment_id"] == refund["enrollment_id"]) &
                (pd.to_datetime(self.play_records_df["play_start_time"]) > refund_dt + timedelta(days=1))
            ]

            if len(after_refund_plays) > 0:
                total_min = after_refund_plays["play_duration_minutes"].sum()
                chapter_names = after_refund_plays.merge(
                    self.chapters_df[["chapter_id", "chapter_name"]], 
                    on="chapter_id", how="left"
                )["chapter_name"].unique().tolist()

                abnormal_records.append({
                    "record_id": f"REFUND_{refund['enrollment_id']}",
                    "user_id": refund["user_id"],
                    "course_id": refund["course_id"],
                    "chapter_id": "",
                    "chapter_name": f"{len(chapter_names)}个章节: {', '.join(chapter_names[:3])}",
                    "abnormal_type": "退款后继续观看",
                    "abnormal_severity": "中",
                    "indicator_1": f"退款日期: {refund['refund_date']}",
                    "indicator_2": f"退款后学习: {total_min:.1f}分钟",
                    "indicator_3": f"播放记录: {len(after_refund_plays)}条",
                    "suggestion": "退款后仍有学习行为，需确认权限控制是否正常，或是否为误退款"
                })

        return pd.DataFrame(abnormal_records)

    def run_all_analyses(self):
        """运行所有分析并返回结果字典"""
        print("计算章节漏斗...")
        chapter_funnel = self.calculate_chapter_funnel()

        print("识别测验卡点...")
        quiz_bottlenecks = self.calculate_quiz_bottlenecks()

        print("分析作业拖延...")
        homework_delays = self.calculate_homework_delays()

        print("计算互动贡献...")
        interaction_contribution = self.calculate_interaction_contribution()

        print("分析证书转化...")
        certificate_conversion = self.calculate_certificate_conversion()

        print("生成课程推荐线索...")
        recommendations = self.generate_course_recommendations()

        print("识别异常学习记录...")
        abnormal_records = self.calculate_abnormal_learning_records()

        return {
            "chapter_funnel": chapter_funnel,
            "quiz_bottlenecks": quiz_bottlenecks,
            "homework_delays": homework_delays,
            "interaction_contribution": interaction_contribution,
            "certificate_conversion": certificate_conversion,
            "recommendations": recommendations,
            "abnormal_records": abnormal_records
        }
