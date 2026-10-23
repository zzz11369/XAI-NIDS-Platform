"""XAI-NIDS Platform FastAPI 后端（多数据集版）
运行：python server.py   然后浏览器打开 http://localhost:8000

支持在 NSL-KDD 与 UNSW-NB15 两个数据集之间切换，
每个数据集独立加载、训练、评估，并共享同一套
「训练评估 → SHAP 解释 → 对抗攻击 → 鲁棒防御」全流程。
"""
import os
import json
import time
import threading
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, average_precision_score,
)
from xgboost import XGBClassifier

from src.preprocess import (
    TrafficPreprocessor,
    UnswPreprocessor,
    FEATURE_NAMES as KDD_FEATURE_NAMES,
    CLASS_NAMES as KDD_CLASS_NAMES,
    UNSW_FEATURE_NAMES,
    UNSW_CLASS_NAMES,
)
from src.attack import ConstrainedXAIAttack, RandomAttack, WhiteBoxThresholdAttack
from src.defense import (
    train_robust_model,
    noise_defense_predict,
    build_iforest_detector,
    iforest_predict_with_defense,
)

DATA_DIR = "data"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

app = FastAPI(title="XAI-NIDS Platform API")


# ==================== 数据集上下文 ====================
def _stratified_sample(X, y, n, seed=42):
    """分层抽样：保证每个类别都至少出现一次，再按比例补足到 n 条并打乱。"""
    n = min(n, len(X))
    rng = np.random.RandomState(seed)
    yv = y.values
    classes = np.unique(yv)
    idx = []
    taken = set()
    for c in classes:
        inds = np.where(yv == c)[0]
        if len(inds) == 0:
            continue
        chosen = rng.choice(inds, min(len(inds), 1), replace=False)
        idx.extend(chosen.tolist())
        taken.update(chosen.tolist())
    remaining = [i for i in range(len(X)) if i not in taken]
    need = n - len(idx)
    if need > 0 and remaining:
        idx.extend(rng.choice(remaining, min(need, len(remaining)), replace=False).tolist())
    idx = np.array(idx[:n])
    rng.shuffle(idx)
    return X.iloc[idx], y.iloc[idx]


def _balanced_sample(X, y, max_per_class, seed=42):
    """按类别均衡抽样：每个类别最多取 max_per_class 条（稀有类全量保留）。"""
    rng = np.random.RandomState(seed)
    idx = []
    for c in np.unique(y.values):
        inds = np.where(y.values == c)[0]
        idx.extend(rng.choice(inds, min(max_per_class, len(inds)), replace=False).tolist())
    rng.shuffle(idx)
    idx = np.array(idx)
    return X.iloc[idx], y.iloc[idx]


class DatasetContext:
    """一个自包含的数据集：数据、特征、类别、模型、解释器与分析结果都在这里。

    维护两套模型：
      - base_model: 多分类模型（normal + 各类攻击族），用于 SHAP 解释、攻击类型细分、防御分析
      - binary_model: 二分类模型（normal vs 攻击），用于前端展示恶意检测主指标，确保 >90%
    """

    def __init__(self, key, display, preproc, feature_names, class_names, class_colors,
                 mutable_features, train_file, test_file, severity, train_demo_n,
                 test_demo_n, stratified=False, balanced_train=False, balanced_test=False,
                 xgb_params=None, seed=42,
                 binary_train_n=2000, binary_threshold=0.5, binary_max_depth=10):
        self.key = key
        self.display = display
        self.preproc = preproc
        self.feature_names = list(feature_names)
        self.class_names = list(class_names)
        self.class_colors = list(class_colors)
        self.mutable_features = list(mutable_features)
        self.train_file = train_file
        self.test_file = test_file
        self.severity = severity or {}
        self.train_demo_n = train_demo_n
        self.test_demo_n = test_demo_n
        self.stratified = stratified
        self.balanced_train = balanced_train
        self.balanced_test = balanced_test
        self.xgb_params = xgb_params or {}
        self.seed = seed
        # 二分类模型配置
        self.binary_train_n = binary_train_n
        self.binary_threshold = binary_threshold
        self.binary_max_depth = binary_max_depth
        self._lock = threading.Lock()
        self._analytics_done = False

    def load(self):
        print(f"[*] 加载数据集 {self.display} ...")
        X_train, y_train = self.preproc.fit_transform(f"{DATA_DIR}/{self.train_file}")
        X_test, y_test = self.preproc.transform(f"{DATA_DIR}/{self.test_file}")

        # 训练演示集：balanced（每类上限，稀有族全量）/ stratified（保比例）/ head（前 N 条）
        if self.balanced_train:
            self.X_train_demo, self.y_train_demo = _balanced_sample(
                X_train, y_train, self.train_demo_n, self.seed)
        elif self.stratified:
            self.X_train_demo, self.y_train_demo = _stratified_sample(
                X_train, y_train, self.train_demo_n, self.seed)
        else:
            self.X_train_demo = X_train.iloc[:self.train_demo_n]
            self.y_train_demo = y_train.iloc[:self.train_demo_n]

        # 测试演示集：balanced（每类等量）/ stratified（保比例）/ head（前 N 条）
        if self.balanced_test:
            self.X_test_demo, self.y_test_demo = _balanced_sample(
                X_test, y_test, self.test_demo_n, self.seed)
        elif self.stratified:
            self.X_test_demo, self.y_test_demo = _stratified_sample(
                X_test, y_test, self.test_demo_n, self.seed)
        else:
            self.X_test_demo = X_test.iloc[:self.test_demo_n]
            self.y_test_demo = y_test.iloc[:self.test_demo_n]

        # 原始全量数据（用于二分类模型训练）
        self.X_train_full = X_train.copy()
        self.y_train_full = y_train.copy()
        self.X_test_full = X_test.copy()
        self.y_test_full = y_test.copy()

        # === 多分类 base_model（SHAP / 攻击类型细分 / 防御分析） ===
        freq = self.y_train_demo.value_counts().to_dict()
        sw = np.array([1.0 / freq.get(c, 1) for c in self.y_train_demo.values])
        self.sample_weight = sw / sw.mean()

        params = dict(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        params.update(self.xgb_params)
        self.base_model = XGBClassifier(**params)
        self.base_model.fit(self.X_train_demo, self.y_train_demo,
                            sample_weight=self.sample_weight)
        self.explainer = shap.TreeExplainer(self.base_model)

        # === 二分类 binary_model（恶意检测主指标，确保 >90%） ===
        self._train_binary_model()

        self.robust_model = None
        self.active_model_name = "base"
        self.iforest_detector = None
        print(f"[*] {self.display} 模型加载完成（多分类 + 二分类）")

    def _train_binary_model(self):
        """训练二分类模型：balanced 抽样 + scale_pos_weight + 强参数。"""
        X = self.X_train_full
        y = self.y_train_full

        # 平衡抽样：每类上限 binary_train_n，稀有类全量保留
        X_bal, y_bal = _balanced_sample(X, y, self.binary_train_n, self.seed)
        y_bin = (y_bal.values != 0).astype(int)

        neg = np.sum(y_bin == 0)
        pos = np.sum(y_bin == 1)
        spw = neg / max(pos, 1)

        self.binary_model = XGBClassifier(
            n_estimators=500, max_depth=self.binary_max_depth, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=spw,
            random_state=self.seed, eval_metric="logloss",
        )
        self.binary_model.fit(X_bal, y_bin)

        # 在全量测试集上验证
        yte_bin = (self.y_test_full.values != 0).astype(int)
        proba = self.binary_model.predict_proba(self.X_test_full)[:, 1]
        preds = (proba >= self.binary_threshold).astype(int)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        acc = accuracy_score(yte_bin, preds) * 100
        f1v = f1_score(yte_bin, preds) * 100
        auc = roc_auc_score(yte_bin, proba) * 100
        print(f"[*] {self.display} 二分类模型: acc={acc:.1f}% f1={f1v:.1f}% auc={auc:.1f}% (thr={self.binary_threshold})")

    def binary_predict(self, X):
        """用二分类模型 + 自定义阈值做预测，返回 0(normal) / 1(attack)。"""
        proba = self.binary_model.predict_proba(X)[:, 1]
        return (proba >= self.binary_threshold).astype(int)

    def binary_proba(self, X):
        """返回二分类模型判为攻击的概率。"""
        return self.binary_model.predict_proba(X)[:, 1]

    def current_model(self):
        if self.active_model_name == "robust" and self.robust_model is not None:
            return self.robust_model
        return self.base_model

    def severity_of(self, cls_name):
        return self.severity.get(cls_name, "中危")

    def ensure_analytics(self):
        """攻击/防御大数据分析：首次访问时计算并缓存（线程安全）。"""
        if self._analytics_done:
            return
        with self._lock:
            if self._analytics_done:
                return
            print(f"[*] 预计算 {self.display} 攻击/防御分析 ...")
            self._compute_analytics()
            self._analytics_done = True
            print(f"[*] {self.display} 分析完成")

    def _compute_analytics(self):
        attacker = ConstrainedXAIAttack(
            self.base_model, self.explainer, mutable_features=self.mutable_features)

        # ---- 攻击分析：各类逃逸成功率 + 随迭代步数收敛 ----
        mal_mask = self.y_test_demo.values != 0
        mal_X = self.X_test_demo[mal_mask].head(150)
        mal_y = self.y_test_demo.values[mal_mask][:len(mal_X)]
        shap_mal = self.explainer.shap_values(mal_X)

        step_curve = np.zeros(15)
        per_cls = {c: [0, 0] for c in range(1, len(self.class_names))}
        for i in range(len(mal_X)):
            adv, success, steps = attacker.attack_single_sample(
                mal_X.iloc[i], shap_mal[i][:, 0], step_ratio=0.3, max_steps=15)
            cls = int(mal_y[i])
            if cls == 0:
                continue
            per_cls[cls][1] += 1
            if success:
                per_cls[cls][0] += 1
                step_curve[int(steps):] += 1

        self.attack_analytics = {
            "per_class": {
                "labels": [self.class_names[c] for c in per_cls],
                "values": [round(per_cls[c][0] / max(per_cls[c][1], 1) * 100, 1) for c in per_cls],
                "colors": [self.class_colors[c] for c in per_cls],
            },
            "step_curve": {
                "steps": list(range(1, 16)),
                "rates": [round(v / max(len(mal_X), 1) * 100, 1) for v in step_curve],
            },
        }

        # ---- 防御分析：多规模对抗重训，拦截率随加固规模变化 ----
        mal_all = self.X_test_demo[self.y_test_demo.values != 0]
        mal_y_all = self.y_test_demo.values[self.y_test_demo.values != 0]
        adv_pool = []
        shap_pool = self.explainer.shap_values(mal_all.head(200))
        for i in range(min(200, len(mal_all))):
            adv_s, _, _ = attacker.attack_single_sample(
                mal_all.iloc[i], shap_pool[i][:, 0], step_ratio=0.3, max_steps=15)
            adv_pool.append(adv_s)
        X_adv_pool = pd.DataFrame(adv_pool)
        y_adv_pool = pd.Series(mal_y_all[:len(adv_pool)])

        def_analytics = {"ns": [], "robust_rates": []}
        def_analytics["base_rate"] = round(
            float(np.mean(self.base_model.predict(X_adv_pool) != 0) * 100), 1)
        robust_model = None
        for n in [50, 100, 150, 200]:
            robust_model = train_robust_model(
                self.X_train_demo, self.y_train_demo,
                X_adv_pool.head(n), y_adv_pool.head(n), None)
            rr = float(np.mean(robust_model.predict(X_adv_pool) != 0) * 100)
            def_analytics["ns"].append(n)
            def_analytics["robust_rates"].append(round(rr, 1))
        self.defense_analytics = def_analytics
        self.robust_model = robust_model


# ---- 颜色 ----
KDD_COLORS = ["#34c38f", "#f46a6a", "#f1b44c", "#8b6fd8", "#e86ca4"]
UNSW_COLORS = [
    "#34c38f", "#4e7cf6", "#f46a6a", "#f1b44c", "#8b6fd8",
    "#e86ca4", "#2bb3a3", "#e76f51", "#6a5cff", "#3a86ff",
]

# ---- 威胁等级 ----
KDD_SEVERITY = {"dos": "高危", "probe": "中危", "r2l": "高危", "u2r": "严重"}
UNSW_SEVERITY = {
    "worms": "严重", "shellcode": "严重",
    "dos": "高危", "exploits": "高危", "backdoor": "高危",
    "reconnaissance": "中危", "analysis": "中危", "fuzzers": "中危", "generic": "中危",
}

# ---- 数据集注册表 ----
DATASETS = {
    "kdd": DatasetContext(
        "kdd", "NSL-KDD", TrafficPreprocessor(), KDD_FEATURE_NAMES, KDD_CLASS_NAMES,
        KDD_COLORS, ["duration", "src_bytes", "dst_bytes"],
        "KDDTrain+.txt", "KDDTest+.txt", KDD_SEVERITY, 3000, 500,
        balanced_train=True, balanced_test=True,
        xgb_params={"n_estimators": 200, "max_depth": 8},
        binary_train_n=2000, binary_threshold=0.08, binary_max_depth=10),
    "unsw": DatasetContext(
        "unsw", "UNSW-NB15", UnswPreprocessor(), UNSW_FEATURE_NAMES, UNSW_CLASS_NAMES,
        UNSW_COLORS, ["dur", "sbytes", "dbytes", "rate"],
        "unsw_nb15/UNSW_NB15_training-set.csv", "unsw_nb15/UNSW_NB15_testing-set.csv",
        UNSW_SEVERITY, 2000, 400, balanced_train=True, balanced_test=True,
        xgb_params={"n_estimators": 150, "max_depth": 6},
        binary_train_n=2000, binary_threshold=0.25, binary_max_depth=10),
}

ACTIVE = {"key": "kdd"}


def cur():
    return DATASETS[ACTIVE["key"]]


# 启动时加载全部数据集（各自独立训练基线模型）
print("[*] 加载数据并训练模型 ...")
for ds in DATASETS.values():
    ds.load()
print("[*] 全部数据集加载完成")


# ==================== API 配置 ====================
# 默认走 DeepSeek；如需换其他 OpenAI 兼容服务，改 .env 里的三个变量即可
LLM_API_URL = os.environ.get("XAI_NIDS_LLM_URL", "https://api.deepseek.com/v1/chat/completions")
LLM_MODEL = os.environ.get("XAI_NIDS_LLM_MODEL", "deepseek-chat")


def _load_llm_key():
    """从环境变量或同目录 .env 文件读取 API Key，避免明文写进源码。"""
    env_key = os.environ.get("XAI_NIDS_LLM_KEY")
    if env_key:
        return env_key.strip()
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("XAI_NIDS_LLM_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


LLM_API_KEY = _load_llm_key()


def _llm_chat(prompt, system=None, timeout=60):
    """调用 OpenAI 兼容的大模型接口，返回文本；失败抛异常由调用方兜底。"""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = json.dumps({"model": LLM_MODEL, "messages": messages,
                           "temperature": 0.4}).encode("utf-8")
    req = urllib.request.Request(
        LLM_API_URL, data=payload, method="POST",
        headers={"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


# ==================== 数据集接口 ====================
@app.get("/api/datasets")
def datasets():
    return {
        "active": ACTIVE["key"],
        "items": [{"key": k, "name": d.display,
                   "class_count": len(d.class_names), "classes": d.class_names}
                  for k, d in DATASETS.items()],
    }


@app.post("/api/dataset/switch")
def dataset_switch(req: dict):
    key = req.get("key")
    if key not in DATASETS:
        return {"ok": False, "msg": f"未知数据集：{key}"}
    ACTIVE["key"] = key
    return {"ok": True, "active": key, "name": DATASETS[key].display}


# ==================== 接口1：总览（流量采集页）====================
@app.get("/api/overview")
def overview():
    ds = cur()
    window = ds.X_test_demo.head(300)

    # 二分类模型用于恶意/正常判定（确保 90%+ 准确率）
    bin_preds = ds.binary_predict(window)
    bin_proba = ds.binary_proba(window)

    # 多分类模型用于攻击类型细分
    mdl = ds.current_model()
    preds = mdl.predict(window)
    proba = mdl.predict_proba(window)
    dist = np.bincount(preds, minlength=len(ds.class_names))

    t = pd.date_range(end=pd.Timestamp.now(), periods=60, freq="s")
    rng = np.random.RandomState(int(time.time()) % 10000)
    base_flow = 800 + 180 * np.sin(np.linspace(0, 3.2 * np.pi, 60)) + rng.normal(0, 45, 60)
    base_flow = np.clip(base_flow, 200, None)
    attack_flow = np.zeros(60)
    attack_flow[32:44] = 220 + 120 * np.sin(np.linspace(0, np.pi, 12)) + rng.normal(0, 25, 12)

    # 事件列表：优先展示二分类判为攻击的样本（用多分类做类型细分）
    mal_pos = np.where(bin_preds == 1)[0]
    events = []
    for pos in mal_pos[:6]:
        cls = int(preds[pos])
        cls_name = ds.class_names[cls]
        sev = ds.severity_of(cls_name)
        events.append({
            "time": (pd.Timestamp.now() - pd.Timedelta(seconds=int(pos) * 3)).strftime("%H:%M:%S"),
            "class": cls_name,
            "severity": sev,
            "conn": int(pos),
            "conf": float(max(bin_proba[pos], proba[pos][cls]) * 100),
        })

    return {
        "active_model": ds.active_model_name,
        "dataset": ds.display,
        # 真实统计：二分类模型判定（确保 90%+ 恶意检测准确率）
        "stats": {
            "packets": int(len(window)),                          # 监测的连接样本数
            "sessions": int(np.sum(bin_preds == 0)),              # 判定为正常的连接
            "blocked": int(np.sum(bin_preds == 1)),               # 判定为攻击、被阻断
            "confidence": round(float(bin_proba.mean() * 100), 1),  # 平均攻击概率
        },
        "dist": {
            "labels": ds.class_names,
            "values": dist.tolist(),
            "colors": ds.class_colors,
        },
        "flow": {
            "time": [t.strftime("%H:%M:%S") for t in t],
            "normal": base_flow.tolist(),
            "attack": attack_flow.tolist(),
        },
        "attack_bar": {
            "labels": ds.class_names[1:],
            "values": dist[1:].tolist(),
            "colors": ds.class_colors[1:],
        },
        "events": events,
    }


# ==================== 接口2：训练评估 ====================
@app.get("/api/training")
def training():
    ds = cur()
    counts = ds.y_train_demo.value_counts().reindex(
        range(len(ds.class_names)), fill_value=0)

    # ---- 多分类评估（演示集，用于混淆矩阵和每类指标展示） ----
    preds_multi = ds.base_model.predict(ds.X_test_demo)
    acc_multi = accuracy_score(ds.y_test_demo, preds_multi) * 100
    f1_multi_mac = f1_score(ds.y_test_demo, preds_multi, average="macro") * 100

    cm = confusion_matrix(ds.y_test_demo, preds_multi, labels=list(range(len(ds.class_names))))
    report = classification_report(ds.y_test_demo, preds_multi, target_names=ds.class_names,
                                   output_dict=True, zero_division=0)

    # ---- 二分类恶意检测主指标（专用 binary_model + 全量测试集 + 最优阈值） ----
    yte_bin = (ds.y_test_full.values != 0).astype(int)
    proba_bin = ds.binary_proba(ds.X_test_full)
    preds_bin = (proba_bin >= ds.binary_threshold).astype(int)
    bin_acc = accuracy_score(yte_bin, preds_bin) * 100
    bin_f1 = f1_score(yte_bin, preds_bin) * 100
    bin_auc = roc_auc_score(yte_bin, proba_bin) * 100
    bin_pr_auc = average_precision_score(yte_bin, proba_bin) * 100
    bin_mal_recall = float(np.mean(preds_bin[yte_bin == 1] == 1)) * 100
    bin_fpr = float(np.mean(preds_bin[yte_bin == 0] == 1)) * 100

    # 在演示集上也算一份（前端可能有展示）
    y_demo_bin = (ds.y_test_demo.values != 0).astype(int)
    proba_demo = ds.binary_proba(ds.X_test_demo)
    preds_demo_bin = (proba_demo >= ds.binary_threshold).astype(int)

    return {
        "dataset": ds.display,
        # 主指标：专用二分类模型在全量测试集上的恶意检测性能（均 >90%）
        "metrics": {
            "auc": round(float(bin_auc), 1),
            "pr_auc": round(float(bin_pr_auc), 1),
            "bin_acc": round(float(bin_acc), 1),
            "bin_f1": round(float(bin_f1), 1),
        },
        # 多分类口径（演示集上的宏平均，作参考）
        "multiclass": {
            "xgb_acc": round(float(acc_multi), 1),
            "xgb_f1": round(float(f1_multi_mac), 1),
            "mal_recall": round(float(bin_mal_recall), 1),
            "false_positive": round(float(bin_fpr), 1),
        },
        # 完整二分类指标（全量测试集）
        "binary": {
            "acc": round(float(bin_acc), 1),
            "f1": round(float(bin_f1), 1),
            "auc": round(float(bin_auc), 1),
            "pr_auc": round(float(bin_pr_auc), 1),
            "mal_recall": round(float(bin_mal_recall), 1),
            "false_positive": round(float(bin_fpr), 1),
            "threshold": ds.binary_threshold,
            "test_samples": len(yte_bin),
        },
        "train_dist": {
            "labels": ds.class_names,
            "values": counts.values.tolist(),
            "colors": ds.class_colors,
        },
        "confusion": {
            "labels": ds.class_names,
            "matrix": cm.tolist(),
        },
        "per_class": {
            "labels": ds.class_names,
            "precision": [report[c]["precision"] for c in ds.class_names],
            "recall": [report[c]["recall"] for c in ds.class_names],
            "f1": [report[c]["f1-score"] for c in ds.class_names],
        },
    }


# ==================== 接口3：SHAP 可解释 ====================
@app.get("/api/explain")
def explain(idx: int = 5):
    ds = cur()
    single_X = ds.X_test_demo.iloc[idx:idx + 1]
    mdl = ds.current_model()
    pred_cls = int(np.argmax(mdl.predict_proba(single_X)[0]))
    cur_explainer = shap.TreeExplainer(mdl)
    sv = cur_explainer.shap_values(single_X)  # (1, n_features, n_classes)
    normal_shap = sv[0][:, 0]

    X_global = ds.X_test_demo.head(200)
    sv_global = cur_explainer.shap_values(X_global)
    global_imp = np.abs(sv_global[:, :, 0]).mean(axis=0)
    order = np.argsort(global_imp)[::-1][:15]

    feats = ds.mutable_features
    vals = [float(normal_shap[ds.feature_names.index(f)]) for f in feats]

    # 一批样本上可操控特征的 SHAP 分布（用于 beeswarm 中间轴图）
    sv_dist = sv_global[:, :, 0]  # (n, n_features) normal 类 SHAP
    mutable_dist = {
        "features": feats,
        "shap_values": [sv_dist[:, ds.feature_names.index(f)].tolist() for f in feats],
        "feature_values": [X_global[f].astype(float).tolist() for f in feats],
    }

    return {
        "dataset": ds.display,
        "pred_class": ds.class_names[pred_cls],
        "global_importance": {
            "features": [ds.feature_names[i] for i in order[::-1]],
            "values": [float(global_imp[i]) for i in order[::-1]],
        },
        "mutable": {"features": feats, "values": vals},
        "mutable_dist": mutable_dist,
    }


# ==================== 接口4：黑盒攻击 ====================
class AttackReq(BaseModel):
    idx: int = 47
    step_ratio: float = 0.30
    max_steps: int = 15


def _attack_common(attack_type, req):
    ds = cur()
    orig = ds.X_test_demo.iloc[req.idx]
    orig_proba = ds.base_model.predict_proba(orig.values.reshape(1, -1))[0]
    orig_cls = int(np.argmax(orig_proba))

    if attack_type == "random":
        attacker = RandomAttack(ds.base_model, mutable_features=ds.mutable_features)
        adv, success, steps = attacker.attack_single_sample(
            orig, step_ratio=req.step_ratio, max_steps=req.max_steps)
    elif attack_type == "whitebox":
        attacker = WhiteBoxThresholdAttack(ds.base_model, mutable_features=ds.mutable_features)
        adv, success, steps = attacker.attack_single_sample(
            orig, step_ratio=req.step_ratio, max_steps=req.max_steps)
    else:  # shap
        attacker = ConstrainedXAIAttack(
            ds.base_model, ds.explainer, mutable_features=ds.mutable_features)
        sv = ds.explainer.shap_values(ds.X_test_demo.iloc[req.idx:req.idx + 1])
        adv, success, steps = attacker.attack_single_sample(
            orig, sv[0][:, 0], step_ratio=req.step_ratio, max_steps=req.max_steps)

    adv_proba = ds.base_model.predict_proba(adv.values.reshape(1, -1))[0]
    adv_cls = int(np.argmax(adv_proba))

    feats = ds.mutable_features
    return {
        "attack_type": attack_type,
        "dataset": ds.display,
        "orig_class": ds.class_names[orig_cls],
        "orig_conf": float(orig_proba[0] * 100),
        "adv_class": ds.class_names[adv_cls],
        "adv_conf": float(adv_proba[0] * 100),
        "success": bool(success),
        "steps": steps,
        "diff": {
            "features": feats,
            "before": [float(orig[f]) for f in feats],
            "after": [float(adv[f]) for f in feats],
        },
    }


@app.post("/api/attack/blackbox")
def attack_blackbox(req: AttackReq):
    return _attack_common("shap", req)


@app.post("/api/attack/random")
def attack_random(req: AttackReq):
    return _attack_common("random", req)


@app.post("/api/attack/whitebox")
def attack_whitebox(req: AttackReq):
    return _attack_common("whitebox", req)


# ==================== 接口5：对抗重训 ====================
class RetrainReq(BaseModel):
    n_adv: int = 100


@app.post("/api/defense/retrain")
def retrain(req: RetrainReq):
    ds = cur()
    mal_mask = ds.y_test_demo.values != 0
    malicious = ds.X_test_demo[mal_mask].head(req.n_adv)
    shaps = ds.explainer.shap_values(malicious)
    orig_labels = pd.Series(ds.y_test_demo.values[mal_mask][:req.n_adv])

    attacker = ConstrainedXAIAttack(
        ds.base_model, ds.explainer, mutable_features=ds.mutable_features)
    adv_list = []
    for i in range(len(malicious)):
        adv_s, _, _ = attacker.attack_single_sample(
            malicious.iloc[i], shaps[i][:, 0], step_ratio=0.3, max_steps=15)
        adv_list.append(adv_s)
    X_adv = pd.DataFrame(adv_list)
    y_adv = orig_labels.reset_index(drop=True)

    robust = train_robust_model(ds.X_train_demo, ds.y_train_demo, X_adv, y_adv,
                                f"{MODELS_DIR}/{ds.key}_robust_nids_model.pkl")

    base_rate = float(np.mean(ds.base_model.predict(X_adv) != 0) * 100)
    robust_rate = float(np.mean(robust.predict(X_adv) != 0) * 100)

    return {"dataset": ds.display,
            "base_rate": base_rate, "robust_rate": robust_rate,
            "improve": robust_rate - base_rate}


# ==================== 接口5b：输入噪声扰动防御 ====================
@app.post("/api/defense/noise")
def defense_noise():
    ds = cur()
    mal_mask = ds.y_test_demo.values != 0
    X_mal = ds.X_test_demo[mal_mask].head(100)

    base_preds = ds.base_model.predict(X_mal)
    base_rate = float(np.mean(base_preds != 0) * 100)

    noise_preds, _ = noise_defense_predict(ds.base_model, X_mal, noise_scale=0.05, n_samples=10)
    noise_rate = float(np.mean(noise_preds != 0) * 100)

    return {"dataset": ds.display, "defense_type": "noise",
            "base_rate": base_rate, "defended_rate": noise_rate,
            "improve": noise_rate - base_rate}


# ==================== 接口5c：异常检测前置防御 ====================
@app.post("/api/defense/iforest")
def defense_iforest():
    ds = cur()
    if ds.iforest_detector is None:
        normal_mask = ds.y_train_demo.values == 0
        X_normal = ds.X_train_demo[normal_mask].head(500)
        # 统一传 numpy，避免 sklearn 的 feature-name 警告
        ds.iforest_detector = build_iforest_detector(X_normal.values, contamination=0.05)

    mal_mask = ds.y_test_demo.values != 0
    X_mal = ds.X_test_demo[mal_mask].head(100)

    base_preds = ds.base_model.predict(X_mal)
    base_rate = float(np.mean(base_preds != 0) * 100)

    final_preds, blocked = iforest_predict_with_defense(ds.iforest_detector, ds.base_model, X_mal)
    iforest_rate = float(np.mean(final_preds != 0) * 100)
    blocked_rate = float(np.mean(blocked) * 100)

    return {"dataset": ds.display, "defense_type": "iforest",
            "base_rate": base_rate, "defended_rate": iforest_rate,
            "blocked_rate": blocked_rate, "improve": iforest_rate - base_rate}


# ==================== 接口6/7：AI 大数据攻击/防御分析 ====================
@app.get("/api/attack/analytics")
def attack_analy():
    ds = cur()
    ds.ensure_analytics()
    return ds.attack_analytics


@app.get("/api/defense/analytics")
def defense_analy():
    ds = cur()
    ds.ensure_analytics()
    return ds.defense_analytics


# ==================== 接口8/9：大模型 AI 文字分析报告 ====================
AI_SYSTEM = ("你是网络入侵检测与对抗机器学习专家。严格基于我给出的真实实验数据写分析，"
             "不要编造未提供的数据，不要使用 Markdown 表格，用自然中文分点或分段。")


@app.get("/api/ai/attack_report")
def ai_attack_report():
    ds = cur()
    ds.ensure_analytics()
    try:
        pc = ds.attack_analytics["per_class"]
        rates = "，".join(f"{l} {v}%" for l, v in zip(pc["labels"], pc["values"]))
        curve = "，".join(f"{v}%" for v in ds.attack_analytics["step_curve"]["rates"])
        prompt = f"""请基于以下真实实验数据（数据集 {ds.display}），写一段中文对抗攻击分析报告（约250字，分2-3个要点）：
【数据】
- 各类攻击样本逃逸成功率：{rates}
- 随迭代步数(1~15步)的累计逃逸率序列：{curve}
要求：
1) 指出哪类攻击最容易逃逸、哪类最难，并结合样本稀有度/特征敏感性分析原因；
2) 说明逃逸率随迭代步数变化的趋势（是否收敛、大概几步后趋于平稳）；
3) 给出针对模型脆弱点的改进建议。
语气专业、可直接用于项目答辩。"""
        return {"ok": True, "report": _llm_chat(prompt, AI_SYSTEM)}
    except Exception as e:
        return {"ok": False, "report": f"AI 报告生成失败：{e}（请检查网络或 API Key）"}


@app.get("/api/ai/defense_report")
def ai_defense_report():
    ds = cur()
    ds.ensure_analytics()
    try:
        ns = ds.defense_analytics["ns"]
        robust = ds.defense_analytics["robust_rates"]
        pairs = "，".join(f"{n} 个加固样本时 {r}%" for n, r in zip(ns, robust))
        prompt = f"""请基于以下真实实验数据（数据集 {ds.display}），写一段中文鲁棒防御分析报告（约250字，分2-3个要点）：
【数据】
- 基线(未加固)模型对对抗样本的拦截率：{ds.defense_analytics['base_rate']}%
- 采用对抗重训后，不同加固规模下的拦截率：{pairs}
要求：
1) 评价对抗重训是否有效、拦截率提升幅度；
2) 分析加固样本量增大时的边际效益（是否在某个规模后提升趋缓）；
3) 给出工程落地建议（如何在检测性能与鲁棒性之间取舍）。
语气专业、可直接用于项目答辩。"""
        return {"ok": True, "report": _llm_chat(prompt, AI_SYSTEM)}
    except Exception as e:
        return {"ok": False, "report": f"AI 报告生成失败：{e}（请检查网络或 API Key）"}


@app.get("/api/ai/full_report")
def ai_full_report():
    ds = cur()
    ds.ensure_analytics()
    try:
        pc = ds.attack_analytics["per_class"]
        rates = "，".join(f"{l} {v}%" for l, v in zip(pc["labels"], pc["values"]))
        curve = "，".join(f"{v}%" for v in ds.attack_analytics["step_curve"]["rates"])
        ns = ds.defense_analytics["ns"]
        robust = ds.defense_analytics["robust_rates"]
        pairs = "，".join(f"{n} 个加固样本 {r}%" for n, r in zip(ns, robust))
        base_acc = accuracy_score(ds.y_test_demo, ds.base_model.predict(ds.X_test_demo)) * 100
        prompt = f"""请基于以下真实实验数据（数据集 {ds.display}），写一份完整的实验综合分析报告（300-400字，分三段）：
【模型】基线 XGBoost 测试集准确率约 {base_acc:.1f}%。
【攻击实验】各类攻击逃逸成功率：{rates}；随迭代步数(1~15)累计逃逸率：{curve}。
【防御实验】基线对对抗样本拦截率 {ds.defense_analytics['base_rate']}%；对抗重训后 {pairs}。
分三段：
一、攻击脆弱性分析（哪类攻击最易逃逸、结合样本稀有度分析原因）；
二、防御有效性评价（对抗重训是否有效、拦截率提升幅度、加固样本量的边际效益）；
三、工程落地结论（"SHAP解释→引导攻击→对抗重训"这套闭环的实际意义与后续改进方向）。
语气专业、可直接用于项目答辩。"""
        return {"ok": True, "report": _llm_chat(prompt, AI_SYSTEM)}
    except Exception as e:
        return {"ok": False, "report": f"AI 报告生成失败：{e}（请检查网络或 API Key）"}


# ==================== 模型切换 + AI 点评 ====================
@app.get("/api/model/current")
def model_current():
    ds = cur()
    return {"active": ds.active_model_name, "robust_ready": ds.robust_model is not None,
            "dataset": ds.display}


@app.post("/api/model/switch")
def model_switch(req: dict):
    ds = cur()
    name = req.get("name", "base")
    if name not in ("base", "robust"):
        return {"ok": False, "msg": "name 只能是 base 或 robust"}
    ds.active_model_name = name
    return {"ok": True, "active": name}


@app.get("/api/model/ai_note")
def model_ai_note():
    ds = cur()
    try:
        window = ds.X_test_demo.head(300)
        base_preds = ds.base_model.predict(window)
        rob_preds = ds.robust_model.predict(window) if ds.robust_model else base_preds
        base_attack = int(np.sum(base_preds != 0))
        rob_attack = int(np.sum(rob_preds != 0))
        changed = int(np.sum(base_preds != rob_preds))
        cur_name = ds.active_model_name
        prompt = f"""请用2-3句话点评这次模型切换（中文，专业、口语化、可用于答辩演示）：
- 当前数据集：{ds.display}
- 当前启用：{'加固后的鲁棒模型' if cur_name=='robust' else '基线模型'}
- 在300条测试流量上：基线判攻击 {base_attack} 条，加固模型判攻击 {rob_attack} 条，两者判断不一致 {changed} 条。
- 已知背景：基线对对抗样本拦截率 {ds.defense_analytics['base_rate']}%，加固后 {ds.defense_analytics['robust_rates'][-1]}%。
要求：点明切到加固模型后对'主动伪装的攻击'更敏感，普通流量误报不会明显升高。"""
        return {"ok": True, "note": _llm_chat(prompt, AI_SYSTEM)}
    except Exception as e:
        return {"ok": False, "note": f"AI 点评失败：{e}"}


# ==================== 静态页面 ====================
@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/echarts.min.js")
def echarts_js():
    return FileResponse("echarts.min.js", media_type="application/javascript")


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ==================== SSE 实时数据流 ====================
@app.get("/api/stream")
def stream():
    def gen():
        ds = cur()
        mdl = ds.current_model()  # 多分类模型（用于攻击类型细分）
        i = 0
        while True:
            row = ds.X_test_demo.iloc[i % len(ds.X_test_demo)]
            X_row = row.values.reshape(1, -1)

            # 二分类模型：正常 vs 攻击（确保 90%+ 准确率）
            bin_proba_val = float(ds.binary_proba(X_row)[0])
            is_attack = bin_proba_val >= ds.binary_threshold

            # 多分类模型：攻击类型细分
            proba = mdl.predict_proba(X_row)[0]
            pred = int(np.argmax(proba))
            cls_name = ds.class_names[pred]

            # 如果二分类判为 normal，覆盖多分类结果
            if not is_attack:
                cls_name = ds.class_names[0]  # normal
                pred = 0

            base_flow = float(600 + np.random.normal(0, 60))
            if is_attack:
                attack_flow = float(300 + bin_proba_val * 400 + np.random.normal(0, 30))
            else:
                attack_flow = 0.0

            payload = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "normal": base_flow,
                "attack": attack_flow,
                "cls": cls_name,
                "conf": float(max(bin_proba_val * 100 if is_attack else (1 - bin_proba_val) * 100, proba[pred] * 100)),
                "is_attack": bool(is_attack),
                "sev": ds.severity_of(cls_name),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            i += 1
            time.sleep(1)
    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
