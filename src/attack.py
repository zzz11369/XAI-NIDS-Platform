# -*- coding: utf-8 -*-
# 协议约束下的 SHAP 引导对抗攻击
# 真实流量不能随便改特征（比如把 TCP 改成 UDP 连接就断了），
# 攻击者只能动现实中能控制的属性：
#   duration    引入延迟
#   src_bytes   填充垃圾字节
#   dst_bytes   诱导服务器多回包
# 按 normal 类 SHAP 值的符号决定放大还是压缩，把恶意样本推向 normal
import numpy as np
import pandas as pd


class ConstrainedXAIAttack:
    """协议约束的 SHAP 引导攻击器（五分类）。

    目标：让恶意样本被判成 normal（类别 0）。
    根据 normal 类 SHAP 值的符号决定扰动方向——正的放大，负的压缩。
    """

    # 现实中人类真正能改的流量特征
    MUTABLE_FEATURES = ["duration", "src_bytes", "dst_bytes"]
    # normal 在 CLASS_NAMES 里的下标（和 preprocess 保持一致）
    NORMAL_CLASS = 0

    def __init__(self, model, explainer, mutable_features=None, normal_class=None):
        self.model = model
        self.explainer = explainer
        if mutable_features is not None:
            self.mutable_features = list(mutable_features)
        else:
            self.mutable_features = self.MUTABLE_FEATURES
        self.normal_class = self.NORMAL_CLASS if normal_class is None else normal_class

    def _pred_class(self, sample):
        # 当前模型把这个样本判成哪一类
        return int(np.argmax(self.model.predict_proba(sample.values.reshape(1, -1))[0]))

    def attack_single_sample(self, sample, shap_normal, step_ratio=0.15, max_steps=10):
        """对单条恶意样本做协议约束下的定向扰动。

        sample: 原始特征样本（pd.Series，index 是特征名）
        shap_normal: 这条样本在 normal 类下的 SHAP 值（长度 = 特征数）
        返回 (扰动后的样本, 是否逃逸成功, 实际用了几步)
        """
        perturbed = sample.copy()
        feature_names = list(sample.index)

        for step in range(max_steps):
            # 已经被判成 normal 了，收手
            if self._pred_class(perturbed) == self.normal_class:
                return perturbed, True, step

            moved = False
            for feat in self.mutable_features:
                if feat not in feature_names:
                    continue
                feat_idx = feature_names.index(feat)
                shap_val = float(shap_normal[feat_idx])
                value = float(perturbed[feat])
                # normal 贡献为正 -> 放大，往正常那边推；为负 -> 压缩
                # value 为 0 时乘法扰动无效，加一个小的常数扰动保证能动起来；
                # 且零值特征若 SHAP 为负，"缩小"方向必然被 max(0,...) 截断，直接加增量
                if value <= 0:
                    # 零值特征：累加增量（不能赋值，否则下一步被乘法打回 0 反复横跳）
                    new_value = value + step_ratio
                elif shap_val > 0:
                    new_value = value * (1 + step_ratio) + step_ratio
                else:
                    new_value = value * (1 - step_ratio) - step_ratio
                perturbed[feat] = max(0.0, new_value)  # 特征值不能是负数
                moved = True

            if not moved:
                break

        success = self._pred_class(perturbed) == self.normal_class
        return perturbed, success, step + 1

    def batch_attack(self, X, shap_values, step_ratio=0.15, max_steps=10):
        """对一批恶意样本批量生成对抗样本。

        shap_values: 多分类 SHAP，形状 (n, n_features, n_classes)，内部取 normal 类引导。
        """
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_normal = shap_values[:, :, self.normal_class]  # (n, n_features)
        else:
            shap_normal = shap_values
        adv_list = []
        success_flags = []
        for i in range(len(X)):
            adv, success, _ = self.attack_single_sample(
                X.iloc[i], shap_normal[i], step_ratio=step_ratio, max_steps=max_steps
            )
            adv_list.append(adv)
            success_flags.append(success)
        return pd.DataFrame(adv_list), success_flags


class RandomAttack:
    """随机扰动攻击（黑盒基线）。

    不利用任何模型内部信息，随机选择特征方向进行扰动，
    作为攻击效果的对比基线。逃逸率通常远低于定向攻击。
    """

    MUTABLE_FEATURES = ["duration", "src_bytes", "dst_bytes"]
    NORMAL_CLASS = 0

    def __init__(self, model, mutable_features=None, normal_class=None, seed=42):
        self.model = model
        self.rng = np.random.RandomState(seed)
        self.mutable_features = mutable_features or self.MUTABLE_FEATURES
        self.normal_class = normal_class or self.NORMAL_CLASS

    def _pred_class(self, sample):
        return int(np.argmax(self.model.predict_proba(sample.values.reshape(1, -1))[0]))

    def attack_single_sample(self, sample, step_ratio=0.30, max_steps=15):
        perturbed = sample.copy()
        feature_names = list(sample.index)
        for step in range(max_steps):
            if self._pred_class(perturbed) == self.normal_class:
                return perturbed, True, step
            for feat in self.mutable_features:
                if feat not in feature_names:
                    continue
                value = float(perturbed[feat])
                # 零值特征：累加随机小增量（不能赋值，否则下一步乘法会打回 0 反复横跳）
                if value <= 0:
                    new_value = value + self.rng.rand() * step_ratio
                elif self.rng.rand() > 0.5:
                    new_value = value * (1 + step_ratio) + step_ratio
                else:
                    new_value = value * (1 - step_ratio) - step_ratio
                perturbed[feat] = max(0.0, new_value)
        success = self._pred_class(perturbed) == self.normal_class
        return perturbed, success, max_steps


class WhiteBoxThresholdAttack:
    """白盒阈值穿越攻击。

    利用 XGBoost 树模型的分裂点信息，直接将关键特征值推过分裂阈值，
    使样本沿树路径走到 normal 叶子节点。比黑盒攻击更高效。
    """

    MUTABLE_FEATURES = ["duration", "src_bytes", "dst_bytes"]
    NORMAL_CLASS = 0

    def __init__(self, model, mutable_features=None, normal_class=None):
        self.model = model
        self.mutable_features = mutable_features or self.MUTABLE_FEATURES
        self.normal_class = normal_class or self.NORMAL_CLASS
        self._extract_thresholds()

    def _extract_thresholds(self):
        """从 XGBoost booster 中提取各特征的分裂阈值集合。

        优化：每个特征只保留出现频率最高的 top-15 个阈值，避免遍历几百个
        分裂点导致每步要做几千次 predict_proba（原实现会超时）。
        """
        from collections import Counter
        self.feature_thresholds = {}
        try:
            booster = self.model.get_booster()
            trees = booster.get_dump()
            raw = {}
            for tree_str in trees:
                for line in tree_str.split("\n"):
                    line = line.strip()
                    if "yes=" in line and "no=" in line and "[" in line:
                        bracket = line.split("[")[1].split("]")[0]
                        if "<" in bracket:
                            feat_name, thresh = bracket.split("<")
                            feat_name = feat_name.strip()
                            thresh = float(thresh)
                            raw.setdefault(feat_name, []).append(thresh)
            # 每个特征只保留出现频率最高的 top-15 个阈值
            for feat, vals in raw.items():
                top = [t for t, _ in Counter(vals).most_common(15)]
                self.feature_thresholds[feat] = sorted(top)
        except Exception:
            pass

    def _pred_proba(self, sample):
        return self.model.predict_proba(sample.values.reshape(1, -1))[0]

    def attack_single_sample(self, sample, step_ratio=0.30, max_steps=15):
        """白盒攻击：利用阈值信息，选择最有利于逃逸的方向。"""
        perturbed = sample.copy()
        feature_names = list(sample.index)

        for step in range(max_steps):
            proba = self._pred_proba(perturbed)
            if int(np.argmax(proba)) == self.normal_class:
                return perturbed, True, step

            best_feat = None
            best_direction = 0
            best_normal_gain = 0

            for feat in self.mutable_features:
                if feat not in feature_names:
                    continue
                value = float(perturbed[feat])
                thresholds = self.feature_thresholds.get(feat, [])
                if not thresholds:
                    continue

                # 尝试把值推到每个阈值的左边和右边，看哪边 normal 概率提升最大
                for thresh in thresholds:
                    # 试推小到阈值以下
                    test_val = max(0.0, thresh - 0.01)
                    test_sample = perturbed.copy()
                    test_sample[feat] = test_val
                    test_proba = self._pred_proba(test_sample)
                    gain = test_proba[self.normal_class] - proba[self.normal_class]
                    if gain > best_normal_gain:
                        best_normal_gain = gain
                        best_feat = feat
                        best_direction = -1

                    # 试推大到阈值以上
                    test_val = thresh + 0.01
                    test_sample = perturbed.copy()
                    test_sample[feat] = test_val
                    test_proba = self._pred_proba(test_sample)
                    gain = test_proba[self.normal_class] - proba[self.normal_class]
                    if gain > best_normal_gain:
                        best_normal_gain = gain
                        best_feat = feat
                        best_direction = 1

            if best_feat is None or best_normal_gain <= 0:
                # 没有更好方向，退化为放大扰动
                for feat in self.mutable_features:
                    if feat in feature_names:
                        value = float(perturbed[feat])
                        perturbed[feat] = max(0.0, value * (1 + step_ratio) + step_ratio)
            else:
                value = float(perturbed[best_feat])
                if best_direction < 0:
                    perturbed[best_feat] = max(0.0, value * (1 - step_ratio) - step_ratio)
                else:
                    perturbed[best_feat] = value * (1 + step_ratio) + step_ratio

        success = int(np.argmax(self._pred_proba(perturbed))) == self.normal_class
        return perturbed, success, max_steps
