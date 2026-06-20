import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path


COMPLETION_CRITERIA = {
    "strict": {
        "watch_ratio_min": 0.85,
        "quiz_pass_ratio_min": 0.9,
        "homework_pass_ratio_min": 0.9,
        "min_effective_duration_ratio": 0.7,
        "description": "严格完课：观看率≥85%，测验通过率≥90%，作业通过率≥90%，有效时长≥70%"
    },
    "standard": {
        "watch_ratio_min": 0.7,
        "quiz_pass_ratio_min": 0.7,
        "homework_pass_ratio_min": 0.7,
        "min_effective_duration_ratio": 0.5,
        "description": "标准完课：观看率≥70%，测验通过率≥70%，作业通过率≥70%，有效时长≥50%"
    },
    "lenient": {
        "watch_ratio_min": 0.5,
        "quiz_pass_ratio_min": 0.5,
        "homework_pass_ratio_min": 0.5,
        "min_effective_duration_ratio": 0.3,
        "description": "宽松完课：观看率≥50%，测验通过率≥50%，作业通过率≥50%，有效时长≥30%"
    }
}


BEHAVIOR_CLASSIFICATION_RULES = {
    "free_trial_viewer": {
        "name": "试看用户",
        "description": "仅观看免费试看章节，未购买或未观看付费章节",
        "color": "#FFA07A"
    },
    "skipper": {
        "name": "跳看用户",
        "description": "章节平均进度<60%，倍速使用率>50%或多次跳过内容",
        "color": "#FFD700"
    },
    "repeater": {
        "name": "重复播放用户",
        "description": "同一章节重复观看次数≥2次，或回放率>30%",
        "color": "#87CEEB"
    },
    "background_idler": {
        "name": "后台挂时长用户",
        "description": "暂停次数异常多或标记为后台播放的时长占比>50%",
        "color": "#DDA0DD"
    },
    "real_learner": {
        "name": "真实学习用户",
        "description": "有效学习时长占比高，暂停/倍速正常，有完整的互动行为",
        "color": "#90EE90"
    },
    "refunded_learner": {
        "name": "退款后继续观看用户",
        "description": "已申请退款但仍有后续播放记录",
        "color": "#F08080"
    },
    "dropout": {
        "name": "流失用户",
        "description": "学习过程中中途停止，未完成课程",
        "color": "#C0C0C0"
    },
    "completer": {
        "name": "完课用户",
        "description": "满足完课口径的用户",
        "color": "#32CD32"
    }
}


class LearningBehaviorClassifier:
    """学习行为分类器 - 区分试看、跳看、重复播放、后台挂时长、真实学习、退款后继续观看"""

    def __init__(self, data_dict):
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

    def _calculate_user_course_metrics(self):
        """计算用户-课程维度的核心指标"""
        metrics = []

        user_course_play = self.play_records_df.groupby(
            ["user_id", "course_id", "enrollment_id"]
        ).agg({
            "play_duration_minutes": ["sum", "count"],
            "progress_ratio": ["mean", "min", "max"],
            "playback_speed": ["mean", "max"],
            "pause_count": "sum",
            "is_background": ["sum", "count"],
            "replay_count": "sum",
            "play_start_time": ["min", "max"],
            "chapter_id": "nunique"
        }).reset_index()

        user_course_play.columns = [
            "user_id", "course_id", "enrollment_id",
            "total_watch_minutes", "play_sessions",
            "avg_progress", "min_progress", "max_progress",
            "avg_speed", "max_speed",
            "total_pauses",
            "background_sessions", "total_sessions_for_bg",
            "total_replays",
            "first_play_time", "last_play_time",
            "chapters_accessed"
        ]

        for _, row in user_course_play.iterrows():
            course = self.courses_df[self.courses_df["course_id"] == row["course_id"]]
            if len(course) == 0:
                continue
            course = course.iloc[0]

            total_chapters = course["total_chapters"]
            total_course_duration = course["total_duration_minutes"]

            watch_ratio = min(row["total_watch_minutes"] / total_course_duration, 1.0)
            chapter_coverage = row["chapters_accessed"] / total_chapters
            background_ratio = row["background_sessions"] / max(row["total_sessions_for_bg"], 1)

            high_speed_sessions = len(self.play_records_df[
                (self.play_records_df["user_id"] == row["user_id"]) &
                (self.play_records_df["course_id"] == row["course_id"]) &
                (self.play_records_df["playback_speed"] >= 1.5)
            ])
            high_speed_ratio = high_speed_sessions / max(row["play_sessions"], 1)

            skip_indicators = (row["avg_progress"] < 0.6) or (high_speed_ratio > 0.5)
            replay_indicators = (row["total_replays"] > 0) or (row["max_progress"] > 1.0)
            background_indicators = (background_ratio > 0.5) or (row["total_pauses"] > row["play_sessions"] * 5)

            user_quizzes = self.quiz_attempts_df[
                (self.quiz_attempts_df["user_id"] == row["user_id"]) &
                (self.quiz_attempts_df["course_id"] == row["course_id"])
            ]
            total_quizzes_in_course = len(self.quizzes_df[self.quizzes_df["course_id"] == row["course_id"]])
            quizzes_taken = user_quizzes["quiz_id"].nunique()
            quizzes_passed = user_quizzes[user_quizzes["passed"] == True]["quiz_id"].nunique()

            user_hw = self.homework_submissions_df[
                (self.homework_submissions_df["user_id"] == row["user_id"]) &
                (self.homework_submissions_df["course_id"] == row["course_id"])
            ]
            total_hw_in_course = len(self.homeworks_df[self.homeworks_df["course_id"] == row["course_id"]])
            hw_submitted = len(user_hw)
            hw_passed = len(user_hw[user_hw["score"] >= 60])

            user_discussions = self.discussions_df[
                (self.discussions_df["user_id"] == row["user_id"]) &
                (self.discussions_df["course_id"] == row["course_id"])
            ]
            discussion_count = len(user_discussions)
            discussion_replies = user_discussions["reply_count"].sum()
            discussion_likes = user_discussions["like_count"].sum()

            refund = self.refunds_df[
                (self.refunds_df["enrollment_id"] == row["enrollment_id"])
            ]
            has_refund = len(refund) > 0
            refund_date = refund["refund_date"].iloc[0] if has_refund else None

            plays_after_refund = False
            if has_refund and refund_date is not None:
                refund_dt = pd.Timestamp(refund_date)
                last_play = pd.Timestamp(row["last_play_time"])
                plays_after_refund = last_play > refund_dt + timedelta(days=1)

            is_paid_enroll = self.enrollments_df[
                self.enrollments_df["enrollment_id"] == row["enrollment_id"]
            ]["is_paid"].iloc[0] if len(self.enrollments_df[
                self.enrollments_df["enrollment_id"] == row["enrollment_id"]
            ]) > 0 else False

            watched_paid_chapters = False
            if not is_paid_enroll:
                course_free_chapters = self.chapters_df[
                    (self.chapters_df["course_id"] == row["course_id"]) &
                    (self.chapters_df["is_free_preview"] == True)
                ]["chapter_id"].tolist()
                user_watched_chapters = self.play_records_df[
                    (self.play_records_df["user_id"] == row["user_id"]) &
                    (self.play_records_df["course_id"] == row["course_id"])
                ]["chapter_id"].unique().tolist()
                watched_paid_chapters = any(c not in course_free_chapters for c in user_watched_chapters)

            effective_minutes = row["total_watch_minutes"] * (1 - background_ratio * 0.8)
            effective_duration_ratio = effective_minutes / max(total_course_duration, 1)

            metrics.append({
                "user_id": row["user_id"],
                "course_id": row["course_id"],
                "enrollment_id": row["enrollment_id"],
                "is_paid": is_paid_enroll,
                "total_watch_minutes": round(row["total_watch_minutes"], 1),
                "effective_watch_minutes": round(effective_minutes, 1),
                "effective_duration_ratio": round(min(effective_duration_ratio, 1.0), 4),
                "watch_ratio": round(watch_ratio, 4),
                "chapter_coverage": round(chapter_coverage, 4),
                "total_chapters": total_chapters,
                "chapters_accessed": row["chapters_accessed"],
                "avg_progress": round(row["avg_progress"], 4),
                "avg_speed": round(row["avg_speed"], 2),
                "high_speed_ratio": round(high_speed_ratio, 4),
                "background_ratio": round(background_ratio, 4),
                "replay_count": row["total_replays"],
                "total_pauses": row["total_pauses"],
                "play_sessions": row["play_sessions"],
                "total_quizzes": total_quizzes_in_course,
                "quizzes_taken": quizzes_taken,
                "quizzes_passed": quizzes_passed,
                "quiz_take_ratio": round(quizzes_taken / max(total_quizzes_in_course, 1), 4),
                "quiz_pass_ratio": round(quizzes_passed / max(quizzes_taken, 1), 4),
                "total_homeworks": total_hw_in_course,
                "homework_submitted": hw_submitted,
                "homework_passed": hw_passed,
                "hw_submit_ratio": round(hw_submitted / max(total_hw_in_course, 1), 4),
                "hw_pass_ratio": round(hw_passed / max(hw_submitted, 1), 4),
                "discussion_count": discussion_count,
                "discussion_replies": discussion_replies,
                "discussion_likes": discussion_likes,
                "has_refund": has_refund,
                "plays_after_refund": plays_after_refund,
                "watched_paid_chapters": watched_paid_chapters,
                "skip_indicators": skip_indicators,
                "replay_indicators": replay_indicators,
                "background_indicators": background_indicators,
                "first_play_time": row["first_play_time"],
                "last_play_time": row["last_play_time"]
            })

        return pd.DataFrame(metrics)

    def classify_behavior(self, metrics_df, completion_level="standard"):
        """基于指标对用户-课程进行行为分类"""
        criteria = COMPLETION_CRITERIA[completion_level]
        classifications = []

        for _, row in metrics_df.iterrows():
            behaviors = []

            is_completer = (
                row["watch_ratio"] >= criteria["watch_ratio_min"] and
                row["quiz_pass_ratio"] >= criteria["quiz_pass_ratio_min"] and
                row["hw_pass_ratio"] >= criteria["homework_pass_ratio_min"] and
                row["effective_duration_ratio"] >= criteria["min_effective_duration_ratio"] and
                row["chapter_coverage"] >= 0.8
            )

            is_free_trial = (
                (not row["is_paid"]) and (not row["watched_paid_chapters"])
            ) or (
                row["chapter_coverage"] <= 0.2 and row["chapters_accessed"] <= 2
            )

            is_refunded_learner = row["has_refund"] and row["plays_after_refund"]

            is_background_idler = row["background_indicators"]

            is_repeater = row["replay_indicators"] and row["replay_count"] >= 2

            is_skipper = (
                row["skip_indicators"] and 
                not is_background_idler and 
                not is_repeater
            )

            is_dropout = (
                (not is_completer) and 
                (row["chapter_coverage"] < 0.5 or row["watch_ratio"] < 0.3) and
                not is_free_trial
            )

            is_real_learner = (
                (not is_skipper) and 
                (not is_background_idler) and 
                (not is_free_trial) and
                row["effective_duration_ratio"] >= 0.4 and
                (row["quiz_take_ratio"] >= 0.3 or row["hw_submit_ratio"] >= 0.3 or row["discussion_count"] > 0)
            )

            if is_completer:
                primary_behavior = "completer"
            elif is_refunded_learner:
                primary_behavior = "refunded_learner"
            elif is_free_trial:
                primary_behavior = "free_trial_viewer"
            elif is_background_idler:
                primary_behavior = "background_idler"
            elif is_repeater:
                primary_behavior = "repeater"
            elif is_skipper:
                primary_behavior = "skipper"
            elif is_dropout:
                primary_behavior = "dropout"
            elif is_real_learner:
                primary_behavior = "real_learner"
            else:
                primary_behavior = "dropout"

            all_behaviors = []
            if is_free_trial:
                all_behaviors.append("free_trial_viewer")
            if is_skipper:
                all_behaviors.append("skipper")
            if is_repeater:
                all_behaviors.append("repeater")
            if is_background_idler:
                all_behaviors.append("background_idler")
            if is_real_learner:
                all_behaviors.append("real_learner")
            if is_refunded_learner:
                all_behaviors.append("refunded_learner")
            if is_dropout:
                all_behaviors.append("dropout")
            if is_completer:
                all_behaviors.append("completer")

            classifications.append({
                "user_id": row["user_id"],
                "course_id": row["course_id"],
                "enrollment_id": row["enrollment_id"],
                "primary_behavior": primary_behavior,
                "primary_behavior_name": BEHAVIOR_CLASSIFICATION_RULES[primary_behavior]["name"],
                "all_behaviors": ",".join(all_behaviors),
                "is_completer": is_completer,
                "is_free_trial": is_free_trial,
                "is_refunded_learner": is_refunded_learner,
                "is_background_idler": is_background_idler,
                "is_repeater": is_repeater,
                "is_skipper": is_skipper,
                "is_dropout": is_dropout,
                "is_real_learner": is_real_learner,
                "completion_level_used": completion_level
            })

        return pd.DataFrame(classifications)

    def run_classification(self, completion_level="standard"):
        """运行完整的分类流程"""
        print("计算用户-课程指标...")
        metrics_df = self._calculate_user_course_metrics()
        print("进行学习行为分类...")
        classification_df = self.classify_behavior(metrics_df, completion_level)
        result_df = metrics_df.merge(classification_df, on=["user_id", "course_id", "enrollment_id"], how="left")
        print(f"分类完成，共 {len(result_df)} 条用户-课程记录")
        return result_df


class CompletionDefiner:
    """完课口径定义模块"""

    @staticmethod
    def get_criteria_description():
        """获取所有完课口径的说明"""
        descriptions = []
        for level, criteria in COMPLETION_CRITERIA.items():
            descriptions.append({
                "level": level,
                "description": criteria["description"],
                "watch_ratio_min": criteria["watch_ratio_min"],
                "quiz_pass_ratio_min": criteria["quiz_pass_ratio_min"],
                "homework_pass_ratio_min": criteria["homework_pass_ratio_min"],
                "effective_duration_ratio_min": criteria["min_effective_duration_ratio"],
                "additional_requirements": "章节覆盖率≥80%"
            })
        return pd.DataFrame(descriptions)

    @staticmethod
    def evaluate_completion(metrics_df, level="standard"):
        """根据指定口径评估完课情况"""
        criteria = COMPLETION_CRITERIA[level]
        result = metrics_df.copy()

        result["meets_watch_req"] = result["watch_ratio"] >= criteria["watch_ratio_min"]
        result["meets_quiz_req"] = result["quiz_pass_ratio"] >= criteria["quiz_pass_ratio_min"]
        result["meets_hw_req"] = result["hw_pass_ratio"] >= criteria["homework_pass_ratio_min"]
        result["meets_effective_req"] = result["effective_duration_ratio"] >= criteria["min_effective_duration_ratio"]
        result["meets_coverage_req"] = result["chapter_coverage"] >= 0.8

        result[f"is_completed_{level}"] = (
            result["meets_watch_req"] &
            result["meets_quiz_req"] &
            result["meets_hw_req"] &
            result["meets_effective_req"] &
            result["meets_coverage_req"]
        )

        passed_reqs = result[[
            "meets_watch_req", "meets_quiz_req", "meets_hw_req", 
            "meets_effective_req", "meets_coverage_req"
        ]].sum(axis=1)
        result[f"completion_score_{level}"] = round(passed_reqs / 5 * 100, 1)

        return result

    @staticmethod
    def evaluate_all_levels(metrics_df):
        """评估所有完课口径"""
        result = metrics_df.copy()
        for level in COMPLETION_CRITERIA.keys():
            result = CompletionDefiner.evaluate_completion(result, level)
        return result
