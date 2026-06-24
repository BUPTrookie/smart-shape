"""
在线推理模块
============

加载训练时持久化的 model/scaler/feature_names，对单条「压头参数 + Pre 曲线」
预测 20 个点位的整形变化量 Δ。

设计要点：特征构造必须与训练时的 ``FeatureEngineer`` 完全一致（同样的位置档位、
Pre 全局特征、交互项），否则预测错位。本模块以持久化的 ``feature_names`` 为准，
反推出位置特征档位与交互组合，从而保证训练/推理特征严格对齐。
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

import rs_impact_config as config


class ImpactPredictor:
    """启动时加载模型常驻内存，提供单样本推理。"""

    def __init__(
        self,
        model_path: str | None = None,
        scaler_path: str | None = None,
        feature_names_path: str | None = None,
    ) -> None:
        self.model = joblib.load(model_path or config.MODEL_PATH)
        self.scaler = joblib.load(scaler_path or config.SCALER_PATH)
        with open(feature_names_path or config.FEATURE_NAMES_PATH, encoding="utf-8") as f:
            self.feature_names: list[str] = json.load(f)

        # 从特征名反推位置档位与交互组合，保证推理与训练一致
        self.position_features: list[tuple[str, float]] = []
        self.interaction_features: list[tuple[str, str]] = []
        self._parse_feature_structure()

    def _parse_feature_structure(self) -> None:
        """解析 feature_names，得到 (压头,档位) 列表与 (压头A,压头B) 交互列表。"""
        for name in self.feature_names:
            if name.startswith("inter_"):
                # 形如 inter_RS1_RS2
                parts = name.split("_")[1:]
                if len(parts) >= 2:
                    self.interaction_features.append((parts[0], parts[1]))
            elif "_pos_" in name:
                # 形如 RS1_pos_-32.2500
                rs_name, pos_str = name.split("_pos_")
                self.position_features.append((rs_name, float(pos_str)))

    @staticmethod
    def _pre_features(pre_curve: dict) -> dict:
        """从 Pre 的 P1-P20 计算全局特征，与 FeatureEngineer.create_pre_features 一致。"""
        arr = np.array([float(pre_curve[f"P{i}"]) for i in range(1, 21)], dtype=float)
        return {
            "pre_mean": float(arr.mean()),
            "pre_std": float(arr.std()),
            "pre_slope": float(arr[-1] - arr[0]),
            "pre_seg1_mean": float(arr[0:4].mean()),  # P1-P4
            "pre_seg2_mean": float(arr[4:8].mean()),  # P5-P8
            "pre_seg3_mean": float(arr[8:16].mean()),  # P9-P16
            "pre_seg4_mean": float(arr[16:20].mean()),  # P17-P20
        }

    def _build_features(self, rs_params: dict, pre_curve: dict) -> dict:
        """构造单样本特征字典（键=feature_names，值与训练一致）。"""
        feats: dict[str, float] = {}
        # 位置特征：RSZ * 1[RSX == pos]
        for rs_name, pos in self.position_features:
            rsx = float(rs_params.get(f"{rs_name}X", 0.0))
            rsz = float(rs_params.get(f"{rs_name}Z", 0.0))
            feats[f"{rs_name}_pos_{pos:.4f}"] = rsz * (1.0 if rsx == pos else 0.0)
        # Pre 曲线特征
        feats.update(self._pre_features(pre_curve))
        # 交互特征：RS_aZ * RS_bZ
        for rsa, rsb in self.interaction_features:
            feats[f"inter_{rsa}_{rsb}"] = float(
                rs_params.get(f"{rsa}Z", 0.0)
            ) * float(rs_params.get(f"{rsb}Z", 0.0))
        return feats

    def predict(self, rs_params: dict, pre_curve: dict) -> dict:
        """
        预测 20 个点位的整形变化量 Δ。

        参数:
            rs_params: {RS1X, RS1Z, RS2X, RS2Z, RS3X, RS3Z, RS4X, RS4Z}
            pre_curve: {P1..P20}

        返回:
            ``{delta_P1: ..., delta_P20: ...}``
        """
        feats = self._build_features(rs_params, pre_curve)
        # 按 feature_names 顺序构造特征向量，缺失补 0
        row = [feats.get(name, 0.0) for name in self.feature_names]
        x = pd.DataFrame([row], columns=self.feature_names)
        # scaler.transform 返回 ndarray，转回 DataFrame 保留 feature names，
        # 与训练时（DataFrame fit）保持一致，避免 sklearn feature name 警告
        x_scaled = pd.DataFrame(self.scaler.transform(x), columns=self.feature_names)
        delta = self.model.predict(x_scaled)[0]
        return {f"delta_P{i + 1}": float(delta[i]) for i in range(20)}
