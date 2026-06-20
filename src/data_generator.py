import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
from pathlib import Path


class DataGenerator:
    """在线课程完课路径分析 - 模拟数据生成器"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.start_date = datetime(2026, 1, 1)
        self.end_date = datetime(2026, 3, 31)

    def _random_date(self, start=None, end=None):
        if start is None:
            start = self.start_date
        if end is None:
            end = self.end_date
        delta = end - start
        random_days = np.random.randint(0, delta.days + 1)
        random_seconds = np.random.randint(0, 86400)
        return start + timedelta(days=int(random_days), seconds=int(random_seconds))

    def generate_courses(self, n_courses=5):
        courses = []
        course_names = [
            "Python数据分析实战",
            "机器学习入门到精通",
            "Web前端开发基础",
            "数据结构与算法",
            "深度学习与神经网络"
        ]
        course_difficulties = ["入门", "中级", "高级", "入门", "高级"]
        course_prices = [299, 599, 199, 399, 799]

        for i in range(n_courses):
            rd = self._random_date()
            courses.append({
                "course_id": f"C{i+1:03d}",
                "course_name": course_names[i],
                "difficulty": course_difficulties[i],
                "price": course_prices[i],
                "total_chapters": np.random.randint(8, 15),
                "total_duration_minutes": np.random.randint(600, 1800),
                "publish_date": rd.date()
            })
        return pd.DataFrame(courses)

    def generate_chapters(self, courses_df):
        chapters = []
        for _, course in courses_df.iterrows():
            n_chapters = course["total_chapters"]
            course_total = course["total_duration_minutes"]
            chapter_names = [
                "课程介绍与环境搭建",
                "基础知识讲解",
                "核心概念深入",
                "实战案例一",
                "进阶技巧",
                "实战案例二",
                "性能优化",
                "项目综合实战",
                "常见问题解答",
                "扩展与延伸",
                "行业应用案例",
                "考前冲刺",
                "面试题精讲",
                "毕业项目指导"
            ]
            for j in range(n_chapters):
                chap_duration = max(15, int(course_total / n_chapters * np.random.uniform(0.7, 1.3)))
                chapters.append({
                    "chapter_id": f"{course['course_id']}_CH{j+1:02d}",
                    "course_id": course["course_id"],
                    "chapter_index": j + 1,
                    "chapter_name": chapter_names[j] if j < len(chapter_names) else f"第{j+1}章",
                    "duration_minutes": chap_duration,
                    "is_free_preview": j < 2,
                    "has_quiz": j >= 1,
                    "has_homework": j >= 2 and j % 2 == 0
                })
        return pd.DataFrame(chapters)

    def generate_users(self, n_users=200):
        users = []
        user_types = ["学生", "职场人士", "转行人员", "兴趣爱好者"]
        channels = ["官网", "推荐", "搜索引擎", "社交媒体", "线下活动"]

        for i in range(n_users):
            user_id = f"U{i+1:04d}"
            register_date = self._random_date()
            users.append({
                "user_id": user_id,
                "register_date": register_date.date(),
                "age": int(np.random.randint(18, 45)),
                "gender": np.random.choice(["男", "女"], p=[0.55, 0.45]),
                "user_type": np.random.choice(user_types, p=[0.3, 0.4, 0.2, 0.1]),
                "channel": np.random.choice(channels),
                "city": np.random.choice(["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京"])
            })
        return pd.DataFrame(users)

    def generate_enrollments(self, users_df, courses_df):
        enrollments = []
        for _, user in users_df.iterrows():
            n_enrolled = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            enrolled_courses = np.random.choice(
                courses_df["course_id"].tolist(),
                size=min(n_enrolled, len(courses_df)),
                replace=False
            )
            for cid in enrolled_courses:
                enroll_date = self._random_date(
                    start=datetime.combine(user["register_date"], datetime.min.time()))
                is_paid = np.random.random() > 0.25
                paid_amount = courses_df[courses_df["course_id"] == cid]["price"].values[0] if is_paid else 0
                enrollments.append({
                    "enrollment_id": str(uuid.uuid4())[:8],
                    "user_id": user["user_id"],
                    "course_id": cid,
                    "enroll_date": enroll_date.date(),
                    "is_paid": is_paid,
                    "paid_amount": paid_amount,
                    "payment_method": np.random.choice(["微信", "支付宝", "银行卡"]) if is_paid else None
                })
        return pd.DataFrame(enrollments)

    def generate_play_records(self, enrollments_df, chapters_df, courses_df):
        play_records = []
        for _, enroll in enrollments_df.iterrows():
            user_id = enroll["user_id"]
            course_id = enroll["course_id"]
            course_chapters = chapters_df[chapters_df["course_id"] == course_id].sort_values("chapter_index")

            behavior_type = np.random.choice(
                ["normal", "skipper", "background", "repeater", "dropout_early", "dropout_mid", "complete"],
                p=[0.25, 0.15, 0.1, 0.1, 0.15, 0.1, 0.15]
            )

            if behavior_type == "complete":
                chapters_to_play = course_chapters
            elif behavior_type == "dropout_early":
                chapters_to_play = course_chapters.head(max(2, np.random.randint(1, 4)))
            elif behavior_type == "dropout_mid":
                chapters_to_play = course_chapters.head(np.random.randint(4, len(course_chapters)))
            else:
                chapters_to_play = course_chapters.head(np.random.randint(3, len(course_chapters) + 1))

            for _, chap in chapters_to_play.iterrows():
                n_sessions = np.random.randint(1, 5)
                for s in range(n_sessions):
                    chap_duration = chap["duration_minutes"]
                    speed = np.random.choice([1.0, 1.25, 1.5, 2.0], p=[0.5, 0.2, 0.2, 0.1])

                    if behavior_type == "skipper":
                        progress_ratio = np.random.uniform(0.3, 0.7)
                    elif behavior_type == "background":
                        progress_ratio = np.random.uniform(0.9, 1.0)
                        speed = np.random.choice([1.0, 1.25])
                    elif behavior_type == "repeater" and np.random.random() < 0.4:
                        progress_ratio = np.random.uniform(1.0, 1.5)
                    elif behavior_type == "dropout_early" and chap["chapter_index"] > 2:
                        progress_ratio = np.random.uniform(0.1, 0.5)
                    else:
                        progress_ratio = np.random.uniform(0.7, 1.05)

                    actual_watch_minutes = chap_duration * min(progress_ratio, 1.0)

                    if chap["is_free_preview"] and not enroll["is_paid"]:
                        if chap["chapter_index"] > 2:
                            continue
                        actual_watch_minutes = min(actual_watch_minutes, 10)

                    pause_count = int(np.random.poisson(3 if behavior_type != "background" else 0))
                    if behavior_type == "background":
                        pause_count = int(np.random.poisson(15))

                    is_background = (behavior_type == "background") or (np.random.random() < 0.05)

                    play_start = self._random_date(
                        start=datetime.combine(enroll["enroll_date"], datetime.min.time()))

                    play_records.append({
                        "play_id": str(uuid.uuid4())[:8],
                        "user_id": user_id,
                        "course_id": course_id,
                        "chapter_id": chap["chapter_id"],
                        "enrollment_id": enroll["enrollment_id"],
                        "play_start_time": play_start,
                        "play_duration_minutes": actual_watch_minutes,
                        "playback_speed": speed,
                        "pause_count": pause_count,
                        "is_background": is_background,
                        "progress_ratio": min(progress_ratio, 1.0),
                        "replay_count": int(progress_ratio > 1.0)
                    })
        return pd.DataFrame(play_records)

    def generate_quizzes(self, chapters_df):
        quizzes = []
        for _, chap in chapters_df.iterrows():
            if chap["has_quiz"]:
                n_questions = np.random.randint(3, 8)
                quizzes.append({
                    "quiz_id": f"Q{chap['chapter_id']}",
                    "chapter_id": chap["chapter_id"],
                    "course_id": chap["course_id"],
                    "total_questions": n_questions,
                    "pass_score": 60,
                    "difficulty": np.random.choice(["简单", "中等", "困难"], p=[0.4, 0.4, 0.2])
                })
        return pd.DataFrame(quizzes)

    def generate_quiz_attempts(self, play_records_df, quizzes_df, enrollments_df):
        attempts = []
        played_chapters = play_records_df.groupby(
            ["user_id", "chapter_id", "course_id"]
        ).size().reset_index(name="play_count")

        for _, row in played_chapters.iterrows():
            quiz = quizzes_df[quizzes_df["chapter_id"] == row["chapter_id"]]
            if len(quiz) == 0:
                continue
            quiz = quiz.iloc[0]

            if np.random.random() < 0.6:
                n_attempts = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            else:
                n_attempts = 0

            for a in range(n_attempts):
                base_score = np.random.uniform(40, 100)
                if a > 0:
                    base_score += np.random.uniform(5, 20)
                score = min(100, base_score + np.random.normal(0, 10))
                passed = score >= quiz["pass_score"]
                attempts.append({
                    "attempt_id": str(uuid.uuid4())[:8],
                    "user_id": row["user_id"],
                    "quiz_id": quiz["quiz_id"],
                    "chapter_id": row["chapter_id"],
                    "course_id": row["course_id"],
                    "attempt_number": a + 1,
                    "score": round(score, 1),
                    "passed": passed,
                    "time_spent_minutes": max(2, np.random.exponential(10)),
                    "attempt_time": self._random_date()
                })
        return pd.DataFrame(attempts)

    def generate_homeworks(self, chapters_df):
        homeworks = []
        for _, chap in chapters_df.iterrows():
            if chap["has_homework"]:
                homeworks.append({
                    "homework_id": f"H{chap['chapter_id']}",
                    "chapter_id": chap["chapter_id"],
                    "course_id": chap["course_id"],
                    "title": f"{chap['chapter_name']}课后作业",
                    "deadline_days": 7,
                    "full_score": 100,
                    "pass_score": 60
                })
        return pd.DataFrame(homeworks)

    def generate_homework_submissions(self, play_records_df, homeworks_df, enrollments_df):
        submissions = []
        user_chapter_play = play_records_df.groupby(
            ["user_id", "chapter_id", "course_id", "enrollment_id"]
        ).agg({"play_start_time": "max"}).reset_index()

        for _, row in user_chapter_play.iterrows():
            hw = homeworks_df[homeworks_df["chapter_id"] == row["chapter_id"]]
            if len(hw) == 0:
                continue
            hw = hw.iloc[0]

            enroll = enrollments_df[enrollments_df["enrollment_id"] == row["enrollment_id"]]
            if len(enroll) == 0:
                continue
            enroll = enroll.iloc[0]

            submit_prob = 0.7 if enroll["is_paid"] else 0.2
            if np.random.random() < submit_prob:
                play_date = row["play_start_time"]
                deadline_date = play_date + timedelta(days=int(hw["deadline_days"]))

                delay_days = np.random.choice([-2, -1, 0, 1, 3, 5, 10, 15])
                submit_date = play_date + timedelta(days=int(delay_days))
                is_late = submit_date > deadline_date

                score = min(100, max(0, np.random.normal(75, 15)))
                submissions.append({
                    "submission_id": str(uuid.uuid4())[:8],
                    "user_id": row["user_id"],
                    "homework_id": hw["homework_id"],
                    "chapter_id": row["chapter_id"],
                    "course_id": row["course_id"],
                    "submit_time": submit_date,
                    "score": round(score, 1),
                    "is_late": is_late,
                    "delay_days": max(0, delay_days),
                    "resubmission_count": int(np.random.choice([0, 0, 0, 1, 2]))
                })
        return pd.DataFrame(submissions)

    def generate_discussions(self, users_df, courses_df, chapters_df, n_discussions=500):
        discussions = []
        discussion_types = ["提问", "分享", "讨论", "笔记"]
        for i in range(n_discussions):
            user = users_df.sample(1).iloc[0]
            course = courses_df.sample(1).iloc[0]
            chapter = chapters_df[chapters_df["course_id"] == course["course_id"]].sample(1).iloc[0]
            dtype = np.random.choice(discussion_types, p=[0.4, 0.2, 0.25, 0.15])
            discussions.append({
                "discussion_id": f"D{i+1:04d}",
                "user_id": user["user_id"],
                "course_id": course["course_id"],
                "chapter_id": chapter["chapter_id"],
                "type": dtype,
                "content_length": int(np.random.exponential(100)),
                "reply_count": int(np.random.poisson(2)),
                "like_count": int(np.random.poisson(3)),
                "is_answered": np.random.random() > 0.4 if dtype == "提问" else True,
                "post_time": self._random_date()
            })
        return pd.DataFrame(discussions)

    def generate_refunds(self, enrollments_df):
        refunds = []
        for _, enroll in enrollments_df.iterrows():
            if not enroll["is_paid"]:
                continue
            refund_prob = 0.08 if enroll["paid_amount"] > 300 else 0.05
            if np.random.random() < refund_prob:
                refund_date = self._random_date(
                    start=datetime.combine(enroll["enroll_date"], datetime.min.time()),
                    end=datetime.combine(enroll["enroll_date"], datetime.min.time()) + timedelta(days=15)
                )
                refunds.append({
                    "refund_id": str(uuid.uuid4())[:8],
                    "enrollment_id": enroll["enrollment_id"],
                    "user_id": enroll["user_id"],
                    "course_id": enroll["course_id"],
                    "refund_date": refund_date.date(),
                    "refund_amount": enroll["paid_amount"],
                    "refund_reason": np.random.choice(
                        ["内容不匹配", "时间不充裕", "难度过高", "教学质量", "其他"]),
                    "is_full_refund": np.random.random() > 0.2
                })
        return pd.DataFrame(refunds)

    def generate_certificates(self, enrollments_df, play_records_df, quiz_attempts_df, 
                                homework_submissions_df, chapters_df, courses_df, quizzes_df, homeworks_df):
        certificates = []

        user_course_stats = play_records_df.groupby(["user_id", "course_id", "enrollment_id"]).agg({
            "play_duration_minutes": "sum",
            "progress_ratio": "mean"
        }).reset_index()

        for _, stats in user_course_stats.iterrows():
            enroll = enrollments_df[enrollments_df["enrollment_id"] == stats["enrollment_id"]]
            if len(enroll) == 0 or not enroll.iloc[0]["is_paid"]:
                continue
            enroll = enroll.iloc[0]

            course_total = courses_df[courses_df["course_id"] == stats["course_id"]]
            if len(course_total) == 0:
                continue
            course_total = course_total.iloc[0]

            total_duration = course_total["total_duration_minutes"]
            watch_ratio = stats["play_duration_minutes"] / total_duration

            user_quizzes = quiz_attempts_df[
                (quiz_attempts_df["user_id"] == stats["user_id"]) & 
                (quiz_attempts_df["course_id"] == stats["course_id"])
            ]
            quizzes_passed = user_quizzes[user_quizzes["passed"] == True]["quiz_id"].nunique()
            total_quizzes = len(quizzes_df[quizzes_df["course_id"] == stats["course_id"]])

            user_hw = homework_submissions_df[
                (homework_submissions_df["user_id"] == stats["user_id"]) & 
                (homework_submissions_df["course_id"] == stats["course_id"])
            ]
            hw_passed = len(user_hw[user_hw["score"] >= 60])
            total_hw = len(homeworks_df[homeworks_df["course_id"] == stats["course_id"]])

            completion_score = (
                watch_ratio * 0.4 +
                (quizzes_passed / max(total_quizzes, 1)) * 0.3 +
                (hw_passed / max(total_hw, 1)) * 0.3
            )

            if completion_score >= 0.7 and watch_ratio >= 0.6:
                issue_date = self._random_date()
                if completion_score >= 0.9:
                    cert_type = np.random.choice(["结业证书", "优秀证书"], p=[0.15, 0.85])
                else:
                    cert_type = "结业证书"
                certificates.append({
                    "certificate_id": str(uuid.uuid4())[:8],
                    "user_id": stats["user_id"],
                    "course_id": stats["course_id"],
                    "enrollment_id": stats["enrollment_id"],
                    "issue_date": issue_date.date(),
                    "completion_score": round(completion_score * 100, 1),
                    "certificate_type": cert_type
                })
        return pd.DataFrame(certificates)

    def generate_all(self, output_dir="data"):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print("生成课程数据...")
        courses_df = self.generate_courses(5)
        courses_df.to_csv(output_path / "courses.csv", index=False, encoding="utf-8-sig")

        print("生成章节数据...")
        chapters_df = self.generate_chapters(courses_df)
        chapters_df.to_csv(output_path / "chapters.csv", index=False, encoding="utf-8-sig")

        print("生成用户数据...")
        users_df = self.generate_users(200)
        users_df.to_csv(output_path / "users.csv", index=False, encoding="utf-8-sig")

        print("生成报名数据...")
        enrollments_df = self.generate_enrollments(users_df, courses_df)
        enrollments_df.to_csv(output_path / "enrollments.csv", index=False, encoding="utf-8-sig")

        print("生成播放记录...")
        play_records_df = self.generate_play_records(enrollments_df, chapters_df, courses_df)
        play_records_df.to_csv(output_path / "play_records.csv", index=False, encoding="utf-8-sig")

        print("生成测验数据...")
        quizzes_df = self.generate_quizzes(chapters_df)
        quizzes_df.to_csv(output_path / "quizzes.csv", index=False, encoding="utf-8-sig")

        print("生成测验答题记录...")
        quiz_attempts_df = self.generate_quiz_attempts(play_records_df, quizzes_df, enrollments_df)
        quiz_attempts_df.to_csv(output_path / "quiz_attempts.csv", index=False, encoding="utf-8-sig")

        print("生成作业数据...")
        homeworks_df = self.generate_homeworks(chapters_df)
        homeworks_df.to_csv(output_path / "homeworks.csv", index=False, encoding="utf-8-sig")

        print("生成作业提交记录...")
        homework_submissions_df = self.generate_homework_submissions(
            play_records_df, homeworks_df, enrollments_df
        )
        homework_submissions_df.to_csv(output_path / "homework_submissions.csv", index=False, encoding="utf-8-sig")

        print("生成讨论数据...")
        discussions_df = self.generate_discussions(users_df, courses_df, chapters_df, 500)
        discussions_df.to_csv(output_path / "discussions.csv", index=False, encoding="utf-8-sig")

        print("生成退款数据...")
        refunds_df = self.generate_refunds(enrollments_df)
        refunds_df.to_csv(output_path / "refunds.csv", index=False, encoding="utf-8-sig")

        print("生成证书数据...")
        certificates_df = self.generate_certificates(
            enrollments_df, play_records_df, quiz_attempts_df, 
            homework_submissions_df, chapters_df, courses_df, quizzes_df, homeworks_df
        )
        certificates_df.to_csv(output_path / "certificates.csv", index=False, encoding="utf-8-sig")

        print("\n=== 数据生成完成！===")
        print(f"课程数: {len(courses_df)}")
        print(f"章节数: {len(chapters_df)}")
        print(f"用户数: {len(users_df)}")
        print(f"报名数: {len(enrollments_df)}")
        print(f"播放记录数: {len(play_records_df)}")
        print(f"测验数: {len(quizzes_df)}")
        print(f"答题记录数: {len(quiz_attempts_df)}")
        print(f"作业数: {len(homeworks_df)}")
        print(f"作业提交数: {len(homework_submissions_df)}")
        print(f"讨论数: {len(discussions_df)}")
        print(f"退款数: {len(refunds_df)}")
        print(f"证书数: {len(certificates_df)}")

        return {
            "courses": courses_df,
            "chapters": chapters_df,
            "users": users_df,
            "enrollments": enrollments_df,
            "play_records": play_records_df,
            "quizzes": quizzes_df,
            "quiz_attempts": quiz_attempts_df,
            "homeworks": homeworks_df,
            "homework_submissions": homework_submissions_df,
            "discussions": discussions_df,
            "refunds": refunds_df,
            "certificates": certificates_df
        }


if __name__ == "__main__":
    generator = DataGenerator()
    generator.generate_all()
