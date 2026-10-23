# 云端筑盾 · XAI-NIDS 可解释对抗攻防平台

> 面向网络入侵检测（NIDS）的可解释 AI + 对抗攻防闭环系统。
> 用 SHAP 定位模型决策依据 → 在协议约束下引导攻击 → 用对抗重训加固模型。

## 项目亮点

- **协议一致性约束攻击**：只改攻击者现实中真能操控的流量特征（`duration / src_bytes / dst_bytes`），不碰协议标识，保证对抗样本在真实网络里能发出去。
- **三种攻击方法对比**：随机扰动（黑盒基线）→ SHAP 引导（定向高效）→ 白盒阈值穿越（利用 XGBoost 分裂点），攻击强度逐级递增。
- **三种防御手段**：对抗重训、输入噪声扰动、异常检测前置。
- **双数据集**：NSL-KDD（5 类）+ UNSW-NB15（10 类）一键切换。
- **DeepSeek AI 报告**：接入 DeepSeek 大模型，自动生成攻击/防御分析报告。

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 DeepSeek API Key（见下方说明）
#    编辑 .env 文件，填入你的 DeepSeek Key

# 3. 启动
python server.py
```

浏览器打开 http://127.0.0.1:8000 即可。

## 配置 DeepSeek API

编辑项目根目录 `.env`：

```env
XAI_NIDS_LLM_KEY=sk-你的DeepSeekKey
XAI_NIDS_LLM_URL=https://api.deepseek.com/v1/chat/completions
XAI_NIDS_LLM_MODEL=deepseek-chat
```

- DeepSeek 申请地址：https://platform.deepseek.com/
- 模型选 `deepseek-chat`（V3，便宜快，答辩够用）
- 不填 Key 也能用，只是 AI 报告区域会提示失败，其他功能正常。

## 五大功能模块

| 模块 | 功能 |
|---|---|
| 📡 流量采集 | 实时吞吐曲线、判定分布饼图、威胁事件流、攻击构成 |
| 🧠 流量训练 | XGBoost 准确率/F1、混淆矩阵、每类 P/R/F1 |
| 🔍 决策分析 | SHAP 全局特征重要性 Top15 + 可操控特征分布 + 当前样本高亮 |
| ⚔️ 对抗攻击 | 随机扰动 / SHAP 引导 / 白盒阈值穿越，实时对比扰动前后 |
| 🛡️ 鲁棒防御 | 对抗重训 / 噪声扰动 / 异常检测前置，对比加固前后拦截率 |

## 演示样本号（答辩用）

| 数据集 | 样本号 | 原始类别 | 三种攻击效果 |
|---|---|---|---|
| NSL-KDD | **56** | u2r | 三种攻击都能成功逃逸 |
| NSL-KDD | 48 | probe | SHAP 引导 5 步成功 |
| UNSW-NB15 | **87** | shellcode | 三种攻击都能成功逃逸（白盒需 step=0.5） |

> 注意：输 normal 类样本会显示"normal → normal 0 步假成功"，答辩时不要用。

## 项目结构

```
XAI-NIDS-Platform/
├── data/                  # NSL-KDD + UNSW-NB15 原始数据
├── models/                # 训练好的模型权重（启动自动生成）
├── src/
│   ├── preprocess.py      # 双数据集统一预处理
│   ├── attack.py          # 三种对抗攻击算法
│   └── defense.py         # 对抗重训
├── docs/                  # PPT / 报告 / 实验截图 / drawio 图
├── 演示视频/              # 答辩录屏
├── server.py              # FastAPI 后端
├── index.html             # 前端单页应用
├── echarts.min.js         # ECharts 本地库
├── start.bat              # 一键启动
├── requirements.txt
├── .env                   # DeepSeek API 配置
└── README.md
```

## 技术栈

- **后端**：Python 3.10+ / FastAPI / XGBoost / SHAP / scikit-learn
- **前端**：原生 HTML/CSS/JS / ECharts（本地引入，不依赖 CDN）
- **大模型**：DeepSeek-V3（OpenAI 兼容接口）

## 依赖

```
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
shap
```
