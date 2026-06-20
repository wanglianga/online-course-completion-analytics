# 在线课程完课路径深度分析系统

基于 **Python + Pandas + Plotly** 的多维度在线课程学习行为分析系统。

---

## 原始需求

> 请用 Python、Pandas 和 Plotly 完成在线课程完课路径分析，输入用户、课程、章节、播放时长、暂停、倍速、作业、测验、讨论、退款和证书领取数据。分析要定义完课口径，区分试看、跳看、重复播放、后台挂时长、真实学习和退款后继续观看，计算章节流失、测验卡点、作业拖延、互动贡献、证书转化和课程推荐线索。输出章节漏斗、用户分群、异常学习记录、课程对比、可复跑报告和口径说明，不能只用播放总时长判断学习效果。

---

## 项目简介

本系统突破传统"**播放总时长=学习效果**"的单一评价模式，采用**多维度综合分析框架**：

| 维度 | 说明 |
|------|------|
| **三级完课口径** | 严格 / 标准 / 宽松，每级都包含观看率、测验通过率、作业通过率、有效时长率、章节覆盖率 |
| **八类行为分类** | 试看、跳看、重复播放、后台挂时长、真实学习、退款后继续观看、流失、完课 |
| **七大核心分析** | 章节流失、测验卡点、作业拖延、互动贡献、证书转化、推荐线索、异常检测 |
| **九类可视化** | 章节漏斗、流失分析、用户分群、雷达画像、测验卡点、作业拖延、异常记录、课程对比、有效时长对比 |

---

## 启动方式

### 前置要求

- **Python 3.9+**（推荐 3.11）
- **pip** 或 **conda** 包管理工具
- **Docker 20.10+** 和 **Docker Compose v2+**（使用 Docker 方式时需要）
- 约 **500MB** 可用磁盘空间（含依赖、数据、图表）

---

### 方式一：Docker 一键启动（推荐，优先使用）

#### 1. 复制环境变量配置

```bash
# Windows PowerShell
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

#### 2. 一键构建并启动

```bash
docker compose up --build
```

如需后台运行：

```bash
docker compose up --build -d
```

查看运行日志：

```bash
docker compose logs -f course-analytics
```

停止并清理服务：

```bash
docker compose down
```

#### 3. 访问报告

服务启动后，在浏览器中访问：

```
http://localhost:8000/在线课程完课路径分析报告.html
```

> 💡 Docker 模式下，系统会：
> 1. 自动安装所有 Python 依赖
> 2. 自动生成模拟数据并完成全部分析
> 3. 自动启动 HTTP 服务，便于直接查看 HTML 报告
> 4. `data/` 和 `output/` 目录通过卷挂载，报告直接持久化到本地

如需调整完课口径，修改 `.env` 文件中的 `COMPLETION_LEVEL`：

```env
# 可选值：strict（严格）/ standard（标准）/ lenient（宽松）
COMPLETION_LEVEL=strict
```

如需使用自己的数据（不重新生成模拟数据），在 `.env` 中启用：

```env
USE_EXISTING=true
```

然后将 12 个 CSV 数据文件放入 `data/` 目录（命名见下文「数据格式要求」）。

---

### 方式二：本地 Python 环境启动

#### 1. 安装依赖

```bash
pip install -r requirements.txt
```

#### 2. 运行完整分析

**使用默认参数（生成模拟数据 + 标准完课口径）：**

```bash
python main.py
```

**使用严格完课口径：**

```bash
python main.py --level strict
```

**使用已有的真实数据（放入 data/ 目录）：**

```bash
python main.py --use-existing
```

**完整参数说明：**

```bash
python main.py --help
```

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `--data-dir` | 任意路径 | `data` | 数据输入目录 |
| `--output-dir` | 任意路径 | `output` | 报告输出目录 |
| `--level` | `strict` / `standard` / `lenient` | `standard` | 完课口径等级 |
| `--use-existing` | 开关（无需值） | 关闭 | 使用已有CSV，不重新生成模拟数据 |

#### 3. 查看报告

运行完成后，直接用浏览器打开生成的 HTML 报告文件：

```
output/在线课程完课路径分析报告.html
```

或使用 Python 内置 HTTP 服务：

```bash
cd output
python -m http.server 8000
# 然后访问 http://localhost:8000/在线课程完课路径分析报告.html
```

---

## 完课口径说明

### 三级完课口径

| 口径等级 | 观看完成率 | 测验通过率 | 作业通过率 | 有效时长占比 | 章节覆盖率 |
|----------|-----------|-----------|-----------|-------------|-----------|
| **严格** | ≥ 85% | ≥ 90% | ≥ 90% | ≥ 70% | ≥ 80% |
| **标准** | ≥ 70% | ≥ 70% | ≥ 70% | ≥ 50% | ≥ 80% |
| **宽松** | ≥ 50% | ≥ 50% | ≥ 50% | ≥ 30% | ≥ 80% |

### 八类学习行为分类

| 行为类型 | 判定规则 | 说明 |
|----------|----------|------|
| 🔶 **试看用户** | 仅观看免费章节 / 未购买 / 章节覆盖≤2章 | 只看了试看内容，未进入付费学习 |
| 🟡 **跳看用户** | 章节平均进度<60% **或** 倍速≥1.5x占比>50% | 疑似快速跳过内容，未深入学习 |
| 🔵 **重复播放用户** | 同一章节重复≥2次 **或** 回放率>30% | 反复学习，可能内容有难度或质量高 |
| 🟣 **后台挂时长用户** | 后台播放>50% **或** 暂停次数异常偏高 | 播放时长虚高，实际可能在挂机 |
| 🟢 **真实学习用户** | 有效时长率≥40% + 有测验/作业/讨论任一参与 | 学习行为真实且有交互参与 |
| 🔴 **退款后继续观看** | 已退款 + 退款1天后仍有播放记录 | 权限控制异常或潜在误退款召回线索 |
| ⚪ **流失用户** | 未达完课 + 章节覆盖<50% **或** 观看<30% | 学习中途放弃，需召回干预 |
| ✅ **完课用户** | 满足所选等级（严格/标准/宽松）全部5项指标 | 完整完成课程学习 |

### 核心计算公式

```
有效学习时长 = 总播放时长 × (1 - 后台占比 × 0.8)

观看完成率 = 实际观看分钟数 / 课程总时长分钟数  (上限100%)

章节覆盖率 = 访问过的章节数 / 课程总章节数

跳看判定 = 平均进度 < 60%  OR  高倍速(≥1.5x)会话占比 > 50%

后台判定 = 标记后台的会话占比 > 50%  OR  平均每会话暂停次数 > 5次

完课判定（标准级）= 观看≥70% AND 测验通过≥70% AND 作业通过≥70% 
                    AND 有效时长≥50% AND 章节覆盖≥80%
```

> ⚠️ **核心设计原则：不能只用播放总时长判断学习效果**
>
> 系统生成「**总时长 vs 有效时长散点图**」直观展示后台挂时长、跳看等行为导致的时长虚高问题。
> 两个用户播放总时长相同，有效时长差异可能高达 3~5 倍。

---

## 数据格式要求

将真实数据按以下文件名和字段放入 `data/` 目录，运行时加 `--use-existing`（或 Docker 模式下设置 `USE_EXISTING=true`）。

### 必填数据文件

| 文件名 | 核心字段 | 说明 |
|--------|----------|------|
| `courses.csv` | `course_id, course_name, total_chapters, total_duration_minutes, difficulty, price` | 课程主数据 |
| `chapters.csv` | `chapter_id, course_id, chapter_index, chapter_name, duration_minutes, is_free_preview, has_quiz, has_homework` | 章节主数据 |
| `users.csv` | `user_id, register_date, age, gender, user_type, channel, city` | 用户主数据 |
| `enrollments.csv` | `enrollment_id, user_id, course_id, enroll_date, is_paid, paid_amount, payment_method` | 报名/付费记录 |
| `play_records.csv` | `play_id, user_id, course_id, chapter_id, enrollment_id, play_start_time, play_duration_minutes, playback_speed, pause_count, is_background, progress_ratio, replay_count` | **最核心的播放行为日志** |

### 推荐数据文件（提升分析深度）

| 文件名 | 核心字段 | 说明 |
|--------|----------|------|
| `quizzes.csv` | `quiz_id, chapter_id, course_id, total_questions, pass_score, difficulty` | 测验元数据 |
| `quiz_attempts.csv` | `attempt_id, user_id, quiz_id, chapter_id, course_id, attempt_number, score, passed, time_spent_minutes, attempt_time` | 测验答题记录 |
| `homeworks.csv` | `homework_id, chapter_id, course_id, title, deadline_days, full_score, pass_score` | 作业元数据 |
| `homework_submissions.csv` | `submission_id, user_id, homework_id, chapter_id, course_id, submit_time, score, is_late, delay_days, resubmission_count` | 作业提交记录 |

### 可选数据文件

| 文件名 | 核心字段 | 说明 |
|--------|----------|------|
| `discussions.csv` | `discussion_id, user_id, course_id, chapter_id, type, content_length, reply_count, like_count, is_answered, post_time` | 讨论/提问/笔记互动 |
| `refunds.csv` | `refund_id, enrollment_id, user_id, course_id, refund_date, refund_amount, refund_reason, is_full_refund` | 退款记录 |
| `certificates.csv` | `certificate_id, user_id, course_id, enrollment_id, issue_date, completion_score, certificate_type` | 证书领取记录 |

> 💡 编码统一使用 **UTF-8 with BOM**（`utf-8-sig`），确保 Excel 打开中文不乱码。

---

## 输出目录结构

```
output/
├── 在线课程完课路径分析报告.html     # 📄 主报告（含所有图表嵌入）
├── run_summary.json                 # 📋 本次运行摘要与参数记录（可复跑）
├── charts/                          # 📊 所有图表（HTML + PNG双格式）
│   ├── chapter_funnel_all.html
│   ├── chapter_funnel_C001.html ...
│   ├── dropout_analysis_by_course.html
│   ├── user_behavior_segmentation.html
│   ├── behavior_radar_comparison.html
│   ├── effective_vs_total_duration.html
│   ├── quiz_bottleneck_analysis.html
│   ├── homework_delay_analysis.html
│   ├── abnormal_learning_records.html
│   ├── course_comparison_dashboard.html
│   └── certificate_conversion_funnel.html
└── data_exports/                    # 💾 分析结果CSV导出
    ├── user_course_behavior_classification.csv
    ├── chapter_funnel.csv
    ├── quiz_bottlenecks.csv
    ├── homework_delays.csv
    ├── interaction_contribution.csv
    ├── certificate_conversion.csv
    ├── recommendations.csv
    └── abnormal_records.csv
```

---

## 项目目录结构

```
.
├── main.py                          # 🚀 主入口
├── requirements.txt                 # 📦 Python依赖
├── Dockerfile                       # 🐳 Docker镜像定义
├── docker-compose.yml               # 🐳 Compose编排
├── .dockerignore                    # 🐳 Docker构建忽略
├── .env.example                     # 🔧 环境变量模板
├── README.md                        # 📖 本说明文档
├── src/                             # 核心源码
│   ├── __init__.py
│   ├── data_generator.py            # 模拟数据生成器（12类表）
│   ├── behavior_classifier.py       # 完课口径定义 + 行为分类器
│   ├── metrics_calculator.py        # 七大核心分析指标计算
│   ├── visualization_engine.py      # Plotly九类图表引擎
│   └── report_generator.py          # 可复跑HTML报告组装
├── data/                            # 📥 数据输入目录
│   └── (运行后自动生成12个CSV)
└── output/                          # 📤 报告输出目录
    └── (运行后自动生成报告+图表+CSV)
```

---

## 可复跑性说明

本系统完全支持可复跑：

1. **确定性模拟数据**：使用固定 `seed=42` 生成模拟数据，保证多次运行数据一致
2. **完整参数化**：完课口径、数据目录、输出目录全部参数化
3. **运行摘要 JSON**：每次运行在 `output/run_summary.json` 记录完整参数与指标
4. **零侵入真实数据**：真实数据按模板放入 `data/` 即可运行，无需改代码
5. **Docker 可复刻**：`docker compose up --build` 保证任何环境下运行结果一致

---

## 验证方式

### 验证 1：程序能否正常运行

```bash
python main.py --level standard
```

期望输出：
```
[1/6] 生成模拟数据...
...
[6/6] 生成可复跑HTML报告...
✅ 分析完成！
📄 报告文件: ...output/在线课程完课路径分析报告.html
```

### 验证 2：Docker 能否正常构建启动

```bash
docker compose config     # 检查compose配置语法
docker compose up --build  # 构建并启动
```

然后访问 `http://localhost:8000/在线课程完课路径分析报告.html`。

### 验证 3：数据与图表完整性检查

打开主报告后，依次确认：
- ✅ 顶部KPI卡片显示正常数值
- ✅ 口径说明章节显示三级口径卡片
- ✅ 每个分节下的iframe图表均可正常渲染
- ✅ 异常记录表格有数据（模拟数据约100~500条）
- ✅ 推荐线索表格有数据
- ✅ `output/charts/` 目录下存在约15+个HTML文件
- ✅ `output/data_exports/` 目录下存在8个CSV文件

---

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | **Python 3.11** | 核心开发语言 |
| 数据处理 | **Pandas 2.x + NumPy** | 数据清洗、聚合、指标计算 |
| 可视化 | **Plotly 5.x + Kaleido** | 交互式图表（HTML）+ 静态导出（PNG） |
| 报告模板 | **Jinja2 风格字符串模板** | 组装单文件HTML报告 |
| 容器化 | **Docker + Docker Compose** | 一键构建部署与环境隔离 |

---

## 注意事项

1. **图表PNG导出**：依赖 `kaleido` 浏览器内核，首次运行可能较慢；若失败会自动降级仅保存HTML（不影响报告查看）。
2. **大数据量**：单门课程用户超过10万时建议分课程并行运行，或使用Dask替代Pandas。
3. **完课口径调优**：建议先用 `standard` 跑一次，再根据业务需要在 `src/behavior_classifier.py` 中调整 `COMPLETION_CRITERIA` 阈值。
4. **异常记录人工复核**：系统标记的异常仅为**疑似**，退款后观看等需运营人工确认。
5. **数据校验**：使用真实数据前建议先用模拟数据跑通全流程，再逐步替换。

---

*本系统完课路径分析报告由程序自动生成，所有结论基于输入数据与上述公开口径。*
