import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from bituslabs_ds.utils import keep_numeric_columns, save_list

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Safe for import; silent if no config


def smart_feature_selection(
    data: pd.DataFrame, threshold: float = 0.9, prefer_keywords: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    remove features (one of two) that are highly correlated with each other
    data: DataFrame，完整数据集
    threshold: float，相关性阈值，比如 0.9
    prefer_keywords: list，优先保留的关键词，比如 'mean', 'median'

    return:
        to_drop: list，需要删除的特征
        kept_features: list，保留的特征
    """

    if prefer_keywords is None:
        prefer_keywords = ["mean", "median"]

    data = keep_numeric_columns(data)
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = set()
    kept = set()
    for column in upper.columns:
        # 找到当前列与其他列高度相关的
        high_corr = upper[column][upper[column] > threshold].index.tolist()
        if high_corr:
            # 包括自己和高度相关的列
            high_corr = [column] + high_corr
            # 保留这一组的第一个
            keep_feat = high_corr[0]
            # 看这组里面有没有带 prefer_keywords 的特征
            for feat in high_corr:
                if any(key in feat.lower() for key in prefer_keywords):
                    # 如果有偏好的，保留偏好的第一个，删除其他
                    keep_feat = feat
                    break
            kept.add(keep_feat)
            high_corr.remove(keep_feat)  # 删掉自己
            to_drop.update(high_corr)
        else:
            # 如果没有高度相关的，可以直接保留
            kept.add(column)

    logger.info(f"kept: {len(kept)} features: \n({kept}")
    logger.info(f"remove: {len(to_drop)} features: \n({to_drop}")
    return list(to_drop), list(kept)


def feature_selection_by_variance(data: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
    """
    remove features with variance < threshold.
    :param data:
    :param threshold:
    :return:
    """
    selector = VarianceThreshold(threshold=threshold)
    data = keep_numeric_columns(data)
    data_filtered = selector.fit_transform(data)
    # 获取保留的列名
    selected_features = data.columns[selector.get_support()].tolist()
    logging.info(f"selected features by variance：{list(selected_features)}")
    return data_filtered, selected_features


def feature_selection_by_pca(data: pd.DataFrame, output_path: str = ".") -> pd.DataFrame:
    """
    remove features with variance < threshold.
    :param data:
    :param output_path:
    :return:
    """

    data = keep_numeric_columns(data)
    pca = PCA(n_components=data.shape[1])  # 保留所有主成分
    pca.fit(data)

    # 计算每个特征在所有主成分上的贡献度（取绝对值求和）
    importance = np.abs(pca.components_).sum(axis=0).tolist()

    features = data.columns.tolist()
    feature_importance = pd.DataFrame({"Feature": features, "Importance": importance})
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

    features = [features[i] for i in np.argsort(importance)[::-1]]
    # save features ordered by importance to json support model deployment
    save_list(features, f"{output_path}/features/important_features.json")

    return feature_importance
