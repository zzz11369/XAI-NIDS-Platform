# -*- coding: utf-8 -*-
# 数据预处理：把 NSL-KDD 的原始流量转成模型能读的数值特征
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# NSL-KDD 官方 41 个特征名称（顺序必须与数据列严格一致，不可增删改）
FEATURE_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

# 类别型特征（Label Encoding）
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ==================== 五分类标签映射 ====================
# 将 NSL-KDD 的具体攻击类型归并为 4 大攻击家族 + normal（标准 NSL-KDD 五分类任务）
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos",
    "processtable": "dos", "mailbomb": "dos", "udpstorm": "dos", "worm": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "saint": "probe", "mscan": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "named": "r2l", "sendmail": "r2l", "xterm": "r2l",
    "sqlattack": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r",
}
CLASS_NAMES = ["normal", "dos", "probe", "r2l", "u2r"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# ==================== UNSW-NB15 数据集定义 ====================
# 官方 45 列：#, 42 个特征, attack_cat(攻击族), label(0/1)
UNSW_FEATURE_NAMES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean",
    "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
UNSW_CATEGORICAL_COLS = ["proto", "service", "state"]
UNSW_NUMERIC_COLS = [c for c in UNSW_FEATURE_NAMES if c not in UNSW_CATEGORICAL_COLS]

# UNSW-NB15 十分类：normal + 9 大攻击族（attack_cat 列，大小写不敏感）
UNSW_ATTACK_FAMILIES = [
    "analysis", "backdoor", "dos", "exploits", "fuzzers",
    "generic", "reconnaissance", "shellcode", "worms",
]
UNSW_CLASS_NAMES = ["normal"] + UNSW_ATTACK_FAMILIES
UNSW_CLASS_ID = {name: i for i, name in enumerate(UNSW_CLASS_NAMES)}


class UnswPreprocessor:
    """UNSW-NB15 流量数据预处理器（十分类：normal + 9 攻击族）。

    与 NSL-KDD 不同：
      - 42 个特征，其中 proto / service / state 为类别型；
      - 数据含缺失值占位 '-'（如 service 列），转成数值前统一替换；
      - 目标标签取 attack_cat 列（normal + 9 族）；attack_cat 缺失时按 label 兜底。
    """

    def __init__(self):
        self.encoders = {}
        self.categorical_cols = UNSW_CATEGORICAL_COLS
        self.feature_names = list(UNSW_FEATURE_NAMES)

    def _load_raw(self, filepath):
        df = pd.read_csv(filepath)
        # 去掉行号列 '#'（官方 45 列结构）
        if "#" in df.columns:
            df = df.drop(columns=["#"])
        # 缺失占位 '-' -> NaN，数值列再转成 float
        df = df.replace("-", np.nan)
        for col in UNSW_NUMERIC_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        # 类别列缺失值兜底
        for col in self.categorical_cols:
            df[col] = df[col].fillna("missing").astype(str)
        return df

    def _make_target(self, df):
        # attack_cat 大小写不敏感映射；已知类别 -> id
        raw = df["attack_cat"].astype(str).str.strip().str.lower()
        known = raw.map(UNSW_CLASS_ID)
        missing = known.isna()
        target = known.fillna(-1).astype(int)
        # label 列（0/1）用于兜底：label==0 -> normal，label==1 -> generic(最大攻击族)
        lab = pd.to_numeric(df.get("label", pd.Series(np.zeros(len(df)))),
                            errors="coerce").fillna(0.0)
        target[missing & (lab == 0)] = UNSW_CLASS_ID["normal"]
        target[missing & (lab != 0)] = UNSW_CLASS_ID["generic"]
        return target.astype(int)

    def _encode(self, df, fit):
        X = {}
        for col in self.feature_names:
            if col in self.categorical_cols:
                if fit:
                    le = LabelEncoder()
                    le.fit(df[col])
                    self.encoders[col] = le
                le = self.encoders[col]
                # 测试集未见过的类别 -> 训练集字母序最小类别
                vals = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                X[col] = le.transform(vals)
            else:
                X[col] = df[col].astype(float).values
        return pd.DataFrame(X, columns=self.feature_names)

    def fit_transform(self, filepath):
        df = self._load_raw(filepath)
        y = self._make_target(df)
        X = self._encode(df, fit=True)
        return X, y

    def transform(self, filepath):
        df = self._load_raw(filepath)
        y = self._make_target(df)
        X = self._encode(df, fit=False)
        return X, y


class TrafficPreprocessor:
    """NSL-KDD 流量数据预处理器（五分类：normal/dos/probe/r2l/u2r）。"""

    def __init__(self):
        self.encoders = {}
        self.categorical_cols = CATEGORICAL_COLS

    @staticmethod
    def _load_raw(filepath):
        df = pd.read_csv(filepath, header=None)
        assert df.shape[1] == 43, \
            f"NSL-KDD 文件应包含 43 列（41 特征 + label + difficulty），实际为 {df.shape[1]} 列"
        df.columns = FEATURE_NAMES + ["label", "difficulty"]
        return df

    @staticmethod
    def _make_target(df):
        # 将具体攻击标签映射到五分类整数编码（normal=0, dos=1, probe=2, r2l=3, u2r=4）
        cat = df["label"].astype(str).str.strip().map(ATTACK_CATEGORY)
        # 未识别的标签兜底为 dos（NSL-KDD 完整覆盖，正常情况下不会触发）
        df["target"] = cat.map(CLASS_ID).fillna(CLASS_ID["dos"]).astype(int)
        return df

    def fit_transform(self, filepath):
        """在训练集上拟合并转换：学习类别编码，同时返回 X, y。"""
        df = self._load_raw(filepath)
        df = self._make_target(df)

        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        X = df[FEATURE_NAMES].copy()
        y = df["target"].copy()
        return X, y

    def transform(self, filepath):
        """在测试集上转换：复用训练集的编码器；未见过的类别映射到训练集最常见类别。"""
        df = self._load_raw(filepath)
        df = self._make_target(df)

        X = df[FEATURE_NAMES].copy()
        for col in self.categorical_cols:
            le = self.encoders[col]
            # 未见过的类别 -> 编码后的第一个类别（LabelEncoder.classes_ 已排序，[0] 是字母序最小）
            X[col] = X[col].astype(str).map(
                lambda s: s if s in le.classes_ else le.classes_[0]
            )
            X[col] = le.transform(X[col])

        y = df["target"].copy()
        return X, y
