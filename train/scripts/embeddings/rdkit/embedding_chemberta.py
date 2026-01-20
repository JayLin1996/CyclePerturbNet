import sys
import numpy as np
from CyclePerturbNet.train.scripts.models.paths import DATA_DIR, EMBEDDING_DIR, ROOT
from tqdm.auto import tqdm
import pandas as pd
import logging
from pathlib import Path
from CyclePerturbNet.train.scripts.models.helper import canonicalize_smiles
import anndata
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# ===================== 核心配置 - 无需修改 =====================
# 你的ChemBERTa模型路径
CHEMBERTa_MODEL_PATH = str(ROOT / "ChemBERTa-zinc-base-v1")
# 自动选择设备：有GPU用GPU，无GPU用CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 批量推理大小，越大越快（GPU显存不足则调小，比如32/16）
BATCH_SIZE = 64
# ===================== 核心配置结束 =====================

# Set up logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration (和RDKit代码完全一致)"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "chemberta_embedding.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# 全局加载模型和分词器 - 只加载一次，避免重复加载浪费内存
logger.info(f"Loading ChemBERTa model from: {CHEMBERTa_MODEL_PATH}")
logger.info(f"Using device: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(CHEMBERTa_MODEL_PATH)
model = AutoModel.from_pretrained(CHEMBERTa_MODEL_PATH, weights_only=False).to(DEVICE)
# 设置为推理模式，关闭dropout，加速+稳定
model.eval()
logger.info("✅ ChemBERTa model and tokenizer loaded successfully!")

def embed_smile_chemberta(smile: str) -> Optional[np.ndarray]:
    """
    单条SMILES生成ChemBERTa嵌入，和原embed_smile函数对应
    :param smile: 单个SMILES字符串
    :return: 768维numpy数组，失败返回None
    """
    try:
        # 分词 + 编码
        encoding = tokenizer(
            smile,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        # 无梯度推理，核心提速点
        with torch.no_grad():
            outputs = model(**encoding)
        
        # 取 <s> token的向量作为分子嵌入 (标准做法)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_embedding
    
    except Exception as e:
        logger.error(f"Error processing SMILES '{smile}': {str(e)}")
        return None

def batch_embed_smiles_chemberta(smiles_list: List[str]) -> np.ndarray:
    """
    批量生成ChemBERTa嵌入（核心提速，比单条快100倍+），替代原embed_smiles_list
    保持和原函数一致的输入输出格式
    """
    logger.info(f"Starting ChemBERTa embedding generation for {len(smiles_list)} SMILES strings")
    
    # 步骤1: 去重 + 过滤空值
    unique_smiles_list = [s for s in list(set(smiles_list)) if s and s.strip()]
    logger.info(f"Found {len(unique_smiles_list)} unique valid SMILES strings")
    
    all_embeddings = []
    failed_smiles = []
    
    # 步骤2: 批量推理
    for i in tqdm(range(0, len(unique_smiles_list), BATCH_SIZE), desc="Generating ChemBERTa embeddings", position=1, leave=False):
        batch_smiles = unique_smiles_list[i:i+BATCH_SIZE]
        
        # 批量编码
        batch_encoding = tokenizer(
            batch_smiles,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)
        
        # 无梯度推理
        with torch.no_grad():
            batch_outputs = model(**batch_encoding)
        
        # 提取batch的cls向量
        batch_embeddings = batch_outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend(batch_embeddings)
    
    # 步骤3: 映射回原始SMILES列表，缺失的填充0向量
    smiles_to_embedding = dict(zip(unique_smiles_list, all_embeddings))
    embedding_dim = 768 # ChemBERTa固定768维
    full_embedding = []
    
    for smile in smiles_list:
        if smile and smile.strip() in smiles_to_embedding:
            full_embedding.append(smiles_to_embedding[smile])
        else:
            if smile: logger.warning(f"SMILES '{smile}' missing from embeddings, filling with zeros")
            full_embedding.append(np.zeros(embedding_dim))
    
    full_embedding = np.array(full_embedding)
    
    # 步骤4: 处理NaN/Inf 完全复用你的RDKit修复逻辑
    drug_idx, feature_idx = np.where(np.isnan(full_embedding))
    drug_idx_infs, feature_idx_infs = np.where(np.isinf(full_embedding))
    drug_idx = np.concatenate((drug_idx, drug_idx_infs))
    feature_idx = np.concatenate((feature_idx, feature_idx_infs))
    
    if len(drug_idx) > 0:
        logger.warning(f"Found {len(drug_idx)} NaN/Inf values in embeddings, replacing with 0")
        full_embedding[drug_idx, feature_idx] = 0
    
    logger.info(f"✅ Successfully generated ChemBERTa embeddings with shape {full_embedding.shape}")
    return full_embedding

def embed_and_save_embeddings(smiles_list, threshold=0.01, embedding_path=None, skip_variance_filter=False, fixed_embedding_dim = None):
    """
    完全复用你修复后的RDKit核心处理函数，逻辑一模一样，只改了嵌入生成的调用
    输出格式/标准化/异常处理 完全和RDKit一致，无缝兼容
    """
    logger.info("Starting ChemBERTa embedding processing")
    logger.info(f"Number of SMILES strings loaded: {len(smiles_list)}")
    
    # Step 1: SMILES规范化清洗 (和RDKit完全一致)
    canon_smiles_list = []
    for smile in smiles_list:
        canon_smile = canonicalize_smiles(smile)
        if canon_smile is not None:
            canon_smiles_list.append(canon_smile)
        else:
            logger.warning(f"Failed to canonicalize SMILES: {smile}")
    
    logger.info(f"Number of valid canonicalized SMILES: {len(canon_smiles_list)}")
    
    # Step 2: 生成ChemBERTa嵌入 (替换RDKit的embed_smiles_list)
    full_embedding = batch_embed_smiles_chemberta(canon_smiles_list)
    
    # Step3: 转成DataFrame，列名格式和RDKit一致 latent_0 ~ latent_767
    df = pd.DataFrame(
        data=full_embedding,
        index=canon_smiles_list,
        columns=[f"latent_{i}" for i in range(full_embedding.shape[1])],
    )
    
    # Step4: 处理重复索引 (和RDKit一致)
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate SMILES indices")
        df = df.loc[~df.index.duplicated(keep='first')]
    
    # ✅ 关键：ChemBERTa没有无用的latent_0列，无需删除！！！和RDKit唯一的区别点
    logger.info(f"ChemBERTa embedding dimension after processing: {df.shape[1]} (fixed 768)")

    # Step5: 零方差/低方差处理 - 完全复用你修复后的逻辑，无除以0错误
    if skip_variance_filter:
        logger.info("Skipping low variance column filtering (ChemBERTa)")
        std_values = df.std()
        zero_variance_cols = std_values[std_values == 0].index.tolist()
        low_variance_cols = std_values[std_values <= 1e-10].index.tolist()
        
        if zero_variance_cols:
            logger.warning(f"Found {len(zero_variance_cols)} columns with zero variance (will be handled in normalization)")
        
        if low_variance_cols:
            logger.warning(f"Found {len(low_variance_cols)} columns with extremely low variance")
    else:
        low_std_cols = [f"latent_{idx}" for idx in np.where(df.std() <= threshold)[0]]
        logger.info(f"Deleting columns with std<={threshold}: {low_std_cols}")
        df.drop(columns=low_std_cols, inplace=True)
    
    # Step6: 安全标准化 - 完全复用你的修复逻辑，工业级健壮性
    logger.info("Applying safe normalization (same as RDKit)...")
    mean_vals = df.mean()
    std_vals = df.std()

    zero_std_mask = std_vals == 0
    very_small_std_mask = std_vals < 1e-10
    if zero_std_mask.any():
        logger.warning(f"Found {zero_std_mask.sum()} features with zero standard deviation")
        std_vals[zero_std_mask] = 1.0  # 避免除以0
    if very_small_std_mask.any():
        logger.warning(f"Found {very_small_std_mask.sum()} features with very small standard deviation (<1e-10)")
        std_vals[very_small_std_mask] = np.maximum(std_vals[very_small_std_mask], 1e-10)

    # 预处理NaN/Inf
    if np.any(np.isinf(df.values)):
        logger.warning("Found infinite values in embeddings, replacing with large finite values")
        df = df.replace([np.inf, -np.inf], [1e6, -1e6])
    if np.any(np.isnan(df.values)):
        logger.warning("Found NaN values in embeddings, replacing with 0")
        df = df.fillna(0)

    # 标准化核心逻辑
    normalized_df = pd.DataFrame(
        (df - mean_vals) / std_vals,
        index=df.index,
        columns=df.columns
    )

    # 后处理NaN/Inf和极端值
    normalized_df = normalized_df.fillna(0)
    normalized_df = normalized_df.clip(-10, 10) # 和RDKit一致的裁剪范围

    # 维度固定裁剪（如果传入fixed_embedding_dim）
    if fixed_embedding_dim is not None and fixed_embedding_dim < normalized_df.shape[1]:
        logger.info(f"Selecting top {fixed_embedding_dim} features by variance (ChemBERTa)")
        variances = normalized_df.var().sort_values(ascending=False)
        selected_features = variances.head(fixed_embedding_dim).index.tolist()
        normalized_df = normalized_df[selected_features]

    # Step7: 保存文件 (格式和RDKit完全一致：parquet)
    if embedding_path is None:
        directory = EMBEDDING_DIR / "chemberta" / "data" / "embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "chemberta_embedding.parquet"
    else:
        output_path = Path(embedding_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving ChemBERTa embeddings for {len(normalized_df)} SMILES to {output_path}")
    logger.info(f"Final ChemBERTa embedding shape: {normalized_df.shape}")
    logger.info(f"Final embedding stats - Min: {normalized_df.values.min():.6f}, Max: {normalized_df.values.max():.6f}, Mean: {normalized_df.values.mean():.6f}, Std: {normalized_df.values.std():.6f}")
    
    normalized_df.to_parquet(output_path)
    return output_path

# 完全复用你的校验函数，一字不改
def validate(embedding_df, adata, smiles_key='SMILES'):
    logger.info("Starting validation of ChemBERTa embeddings against dataset SMILES")
    dataset_canonical_smiles = set()
    for raw_smile in adata.obs[smiles_key]:
        splitted = [raw_smile]
        if ".." in raw_smile:
            splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
        for s in splitted:
            c = canonicalize_smiles(s)
            if c is not None:
                dataset_canonical_smiles.add(c)

    embedding_smiles = set(embedding_df.index)
    missing_smiles = dataset_canonical_smiles - embedding_smiles
    if missing_smiles:
        logger.warning(f"Found {len(missing_smiles)} SMILES in dataset that are missing from embeddings.")
        for smile in list(missing_smiles)[:10]:
            logger.warning(f"  {smile}")
        logger.warning("Continuing without raising an error.")
    else:
        logger.info("✅ Validation successful! All combined SMILES are accounted for.")

    extra_smiles = embedding_smiles - dataset_canonical_smiles
    if extra_smiles:
        logger.info(f"Note: Embeddings contain {len(extra_smiles)} additional SMILES not in dataset.")

# ===================== 对外暴露的主函数 - 和RDKit完全一致的入参 =====================
def compute_chemberta_embeddings(h5ad_path, output_path=None, smiles_key='SMILES', skip_variance_filter=False, fixed_embedding_dim = None):
    """
    ChemBERTa分子嵌入生成主函数，和原compute_rdkit_embeddings 完全一致的调用方式！！！
    你只需要把代码里的 compute_rdkit_embeddings 改成这个函数名，就能无缝替换
    """
    main_steps = ['Loading SMILES', 'Computing embeddings', 'Saving results', 'Validating']
    with tqdm(total=len(main_steps), desc="Overall ChemBERTa progress", position=0) as pbar:
        # Step 1: 读取数据集
        logger.info(f"Loading dataset from: {h5ad_path}")
        adata = anndata.read_h5ad(h5ad_path)
        
        logger.info("Available keys in adata.obs:")
        logger.info(f"{list(adata.obs.columns)}")
        
        if smiles_key not in adata.obs.columns:
            logger.error(f"SMILES key '{smiles_key}' not found in available columns!")
            logger.info(f"Please use one of the available keys: {list(adata.obs.columns)}")
            raise KeyError(f"SMILES key '{smiles_key}' not found in dataset")
            
        raw_smiles_data = adata.obs[smiles_key].tolist()
        
        if not raw_smiles_data:
            logger.error("Failed to load SMILES data")
            return
        
        # 处理组合药物 SMILES1..SMILES2
        expanded_smiles_data = []
        for raw_smile in raw_smiles_data:
            if ".." in raw_smile:
                splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
                expanded_smiles_data.extend(splitted)
            else:
                expanded_smiles_data.append(raw_smile)
        
        smiles_data = list(set(expanded_smiles_data))
        logger.info(f"Total unique SMILES (after splitting '..'): {len(smiles_data)}")
        pbar.update(1)
        
        # Step 2: 生成嵌入
        pbar.set_description("Computing ChemBERTa embeddings")
        output_file = embed_and_save_embeddings(
            smiles_data,
            embedding_path=output_path,
            skip_variance_filter=skip_variance_filter,
            fixed_embedding_dim = fixed_embedding_dim
        )
        pbar.update(1)
        
        # Step3: 校验保存结果
        pbar.set_description("Saving results")
        df = pd.read_parquet(output_file)
        logger.info(f"✅ Successfully generated and saved ChemBERTa embeddings with shape: {df.shape}")
        logger.info(f"Embeddings saved to: {output_file}")
        pbar.update(1)
        
        # Step4: 校验一致性
        pbar.set_description("Validating")
        validate(df, adata, smiles_key)
        pbar.update(1)

# ===================== 命令行入口 - 和RDKit完全一致 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate ChemBERTa embeddings from SMILES data (same as RDKit)')
    parser.add_argument('h5ad_path', type=str, help='Path to the h5ad file containing SMILES data')
    parser.add_argument('--output_path', type=str, help='Path to save the embeddings', default=None)
    parser.add_argument('--smiles_key', type=str, default='SMILES', help='Key for SMILES data in the h5ad file')
    parser.add_argument('--skip_variance_filter', action='store_true', help='Skip dropping low-variance columns')
    parser.add_argument('--fixed_embedding_dim', type=int, default=None, help='Fixed embedding dimension to keep')

    args = parser.parse_args()
    
    compute_chemberta_embeddings(
        h5ad_path=args.h5ad_path,
        output_path=args.output_path,
        smiles_key=args.smiles_key,
        skip_variance_filter=args.skip_variance_filter,
        fixed_embedding_dim = args.fixed_embedding_dim
    )