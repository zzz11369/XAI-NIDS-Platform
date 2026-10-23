# -*- coding: utf-8 -*-
# 对抗防御训练：把攻击生成的对抗样本混入训练集重训
import os
import pickle

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


def train_robust_model(X_train, y_train, X_adv, y_adv, save_path=None, **kwargs):
    """对抗重训：原始数据 + 对抗样本合并训练，得到加固模型。

    X_train, y_train: 原始训练集
    X_adv, y_adv:     对抗样本和它们原本的标签
    save_path:        保存路径（可选）
    返回训练好的鲁棒模型。
    """
    X_combined = pd.concat([X_train, X_adv], ignore_index=True)
    y_combined = pd.concat([y_train, y_adv], ignore_index=True)

    params = dict(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    params.update(kwargs)

    robust_model = XGBClassifier(**params)
    robust_model.fit(X_combined, y_combined)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(robust_model, f)

    return robust_model


def train_base_model(X_train, y_train, save_path=None, **kwargs):
    """训练未加固的基线模型，用来当攻击的目标模型。"""
    params = dict(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
    )
    params.update(kwargs)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(model, f)

    return model


def noise_defense_predict(model, X, noise_scale=0.05, n_samples=10, seed=42):
    """输入噪声扰动防御：推理时对输入加微小随机噪声，多次预测取平均。

    原理：攻击样本通常在决策边界附近，微小噪声会扰乱攻击的精确扰动效果，
    使多次预测的平均结果更稳定。

    model: 检测模型
    X: 输入样本 (DataFrame 或 2D array)
    noise_scale: 噪声强度（相对值）
    n_samples: 多次采样次数
    返回预测概率和预测类别
    """
    rng = np.random.RandomState(seed)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_features = X.shape[1]
    avg_proba = np.zeros((X.shape[0], model.n_classes_))

    for _ in range(n_samples):
        # 对每个特征加高斯噪声（相对噪声）
        noise = rng.normal(0, noise_scale, X.shape) * (np.abs(X) + 1e-6)
        X_noisy = np.clip(X + noise, 0, None)
        proba = model.predict_proba(X_noisy)
        avg_proba += proba

    avg_proba /= n_samples
    preds = np.argmax(avg_proba, axis=1)
    return preds, avg_proba


def build_iforest_detector(X_train, contamination=0.05, random_state=42):
    """构建 Isolation Forest 异常检测器，用于前置拦截可疑输入。

    原理：正常流量在特征空间中形成致密簇，对抗样本通常偏离正常分布。
    在检测模型前加一层异常检测，可以直接拦截明显可疑的输入。

    X_train: 正常流量训练集（只用 normal 类样本）
    contamination: 预期异常比例
    返回训练好的 IsolationForest 模型
    """
    from sklearn.ensemble import IsolationForest
    detector = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )
    detector.fit(X_train)
    return detector


def iforest_predict_with_defense(detector, model, X, normal_class=0):
    """异常检测前置 + 检测模型联合预测。

    流程：先用 Isolation Forest 判断是否为异常输入，
    若判定为异常（-1）则直接标记为攻击，不再送入下游模型；
    若判定为正常（1）则送入检测模型做正常分类。

    detector: IsolationForest 模型
    model: 下游检测模型
    X: 输入样本
    normal_class: normal 类编号
    返回最终预测类别、是否被前置拦截、检测模型原始预测
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # 前置异常检测
    anomaly_flags = detector.predict(X)  # 1=正常, -1=异常
    blocked = (anomaly_flags == -1)

    # 对通过前置检测的样本做模型预测
    final_preds = np.full(X.shape[0], normal_class, dtype=int)
    if (~blocked).any():
        model_preds = model.predict(X[~blocked])
        final_preds[~blocked] = model_preds

    # 被前置拦截的样本标记为攻击（非 normal 类）
    final_preds[blocked] = 1  # 默认 dos 类作为"被拦截"标记

    return final_preds, blocked
