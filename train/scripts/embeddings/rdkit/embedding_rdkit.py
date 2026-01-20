import sys
import numpy as np
from CyclePerturbNet.train.scripts.models.paths import DATA_DIR, EMBEDDING_DIR
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd
import multiprocessing
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
import logging
from pathlib import Path
from CyclePerturbNet.train.scripts.models.helper import canonicalize_smiles
import h5py
import argparse
import anndata

# Set up logging
def setup_logging(log_dir="logs"):
    """Set up logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "rdkit_embedding.log"
    
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

def embed_smile(smile):
    """Function to process a single SMILES string."""
    try:
        local_generator = MakeGenerator(("RDKit2D",))
        result = local_generator.process(smile)
        if result is None:
            logger.warning(f"Failed to process SMILES: {smile}")
        return result
    except Exception as e:
        logger.error(f"Error processing SMILES '{smile}': {str(e)}")
        return None

def embed_smiles_list(smiles_list, n_processes=16):
    """Create RDKit embeddings for a list of SMILES strings."""
    logger.info(f"Starting embedding generation for {len(smiles_list)} SMILES strings")
    
    # Filter down to unique SMILES
    unique_smiles_list = list(set(smiles_list))
    logger.info(f"Found {len(unique_smiles_list)} unique SMILES strings")
    
    # Generate embeddings in parallel
    with multiprocessing.Pool(processes=n_processes) as pool:
        data = list(tqdm(
            pool.imap(embed_smile, unique_smiles_list),
            total=len(unique_smiles_list),
            desc="Generating RDKit embeddings",
            position=1,
            leave=False
        ))
    
    # Track failed SMILES
    failed_smiles = [s for s, d in zip(unique_smiles_list, data) if d is None]
    if failed_smiles:
        logger.warning(f"\nFailed to process {len(failed_smiles)} SMILES:")
        for s in failed_smiles[:10]:  # Show first 10
            logger.warning(f"  {s}")
        if len(failed_smiles) > 10:
            logger.warning("  ...")
    
    # Filter out None values
    valid_data = [(s, d) for s, d in zip(unique_smiles_list, data) if d is not None]
    unique_smiles_list = [s for s, _ in valid_data]
    data = [d for _, d in valid_data]
    
    embedding = np.array(data)
    
    # Handle NaNs and Infs
    drug_idx, feature_idx = np.where(np.isnan(embedding))
    drug_idx_infs, feature_idx_infs = np.where(np.isinf(embedding))
    drug_idx = np.concatenate((drug_idx, drug_idx_infs))
    feature_idx = np.concatenate((feature_idx, feature_idx_infs))
    
    if len(drug_idx) > 0:
        logger.warning(f"Found {len(drug_idx)} NaN/Inf values in embeddings")
                
    embedding[drug_idx, feature_idx] = 0
    
    # Map back to original SMILES list, filling with zeros if missing
    smiles_to_embedding = dict(zip(unique_smiles_list, embedding))
    embedding_dim = embedding.shape[1]
    full_embedding = []
    for smile in smiles_list:
        if smile in smiles_to_embedding:
            full_embedding.append(smiles_to_embedding[smile])
        else:
            logger.warning(f"SMILES '{smile}' missing from embeddings, filling with zeros")
            full_embedding.append(np.zeros(embedding_dim))
    
    full_embedding = np.array(full_embedding)
    
    logger.info(f"Successfully generated embeddings with shape {full_embedding.shape}")
    return full_embedding

# def embed_and_save_embeddings(smiles_list, threshold=0.01, embedding_path=None, skip_variance_filter=False, fixed_embedding_dim = None):
#     """Process embeddings and save to parquet file."""
#     logger.info("Starting embedding processing")
#     logger.info(f"Number of SMILES strings loaded: {len(smiles_list)}")
    
#     # Canonicalize SMILES
#     canon_smiles_list = []
#     for smile in smiles_list:
#         canon_smile = canonicalize_smiles(smile)
#         if canon_smile is not None:
#             canon_smiles_list.append(canon_smile)
#         else:
#             logger.warning(f"Failed to canonicalize SMILES: {smile}")
    
#     logger.info(f"Number of valid canonicalized SMILES: {len(canon_smiles_list)}")
    
#     # Create embeddings using canonicalized SMILES
#     full_embedding = embed_smiles_list(canon_smiles_list)
    
#     # Create DataFrame with canonicalized SMILES as index
#     df = pd.DataFrame(
#         data=full_embedding,
#         index=canon_smiles_list,
#         columns=[f"latent_{i}" for i in range(full_embedding.shape[1])],
#     )
    
#     # Handle duplicate indices before processing
#     if df.index.duplicated().any():
#         logger.warning(f"Found {df.index.duplicated().sum()} duplicate SMILES indices")
#         df = df.loc[~df.index.duplicated(keep='first')]
    
#     # Drop the first descriptor column (latent_0)
#     df.drop(columns=["latent_0"], inplace=True)
    
#     # Optionally drop low-variance columns
#     if not skip_variance_filter:
#         low_std_cols = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
#         logger.info(f"Deleting columns with std<={threshold}: {low_std_cols}")
#         df.drop(columns=low_std_cols, inplace=True)
#     else:
#         logger.info("Skipping low variance column filtering")
    
#     # if fixed_embedding_dim is not None:
#     #     # 在归一化之前，进行维度选择
#     #     current_dim = df.shape[1]
#     #     if current_dim > fixed_embedding_dim:
#     #         # 选择方差最大的target_dim个特征
#     #         logger.info(f"Selecting top {fixed_embedding_dim} features by variance")
            
#     #         # 计算每个特征的方差
#     #         variances = df.var().sort_values(ascending=False)
            
#     #         # 选择前target_dim个特征
#     #         selected_features = variances.head(fixed_embedding_dim).index.tolist()
#     #         df = df[selected_features]

#     # Normalize
#     normalized_df = pd.DataFrame(
#         (df - df.mean()) / df.std(),
#         index=df.index,
#         columns=df.columns
#     )
    
#     # Set output path
#     if embedding_path is None:
#         directory = EMBEDDING_DIR / "rdkit" / "data" / "embeddings"
#         directory.mkdir(parents=True, exist_ok=True)
#         output_path = directory / "rdkit2D_embedding.parquet"
#     else:
#         output_path = Path(embedding_path)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
    
#     logger.info(f"Saving embeddings for {len(normalized_df)} SMILES to {output_path}")
#     normalized_df.to_parquet(output_path)
#     return output_path

def embed_and_save_embeddings(smiles_list, threshold=0.01, embedding_path=None, skip_variance_filter=False, fixed_embedding_dim = None):
    """Process embeddings and save to parquet file."""
    logger.info("Starting embedding processing")
    logger.info(f"Number of SMILES strings loaded: {len(smiles_list)}")
    
    # Canonicalize SMILES
    canon_smiles_list = []
    for smile in smiles_list:
        canon_smile = canonicalize_smiles(smile)
        if canon_smile is not None:
            canon_smiles_list.append(canon_smile)
        else:
            logger.warning(f"Failed to canonicalize SMILES: {smile}")
    
    logger.info(f"Number of valid canonicalized SMILES: {len(canon_smiles_list)}")
    
    # Create embeddings using canonicalized SMILES
    full_embedding = embed_smiles_list(canon_smiles_list)
    
    # Create DataFrame with canonicalized SMILES as index
    df = pd.DataFrame(
        data=full_embedding,
        index=canon_smiles_list,
        columns=[f"latent_{i}" for i in range(full_embedding.shape[1])],
    )
    
    # Handle duplicate indices before processing
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate SMILES indices")
        df = df.loc[~df.index.duplicated(keep='first')]
    
    # Drop the first descriptor column (latent_0)
    df.drop(columns=["latent_0"], inplace=True)
    
    # 关键修复：即使跳过方差过滤，也需要处理方差为0的特征
    # 因为我们知道RDKit2D会有一些方差为0的特征
    if skip_variance_filter:
        logger.info("Skipping low variance column filtering")
        # 找出方差为0或极低的列
        std_values = df.std()
        zero_variance_cols = std_values[std_values == 0].index.tolist()
        low_variance_cols = std_values[std_values <= 1e-10].index.tolist()
        
        if zero_variance_cols:
            logger.warning(f"Found {len(zero_variance_cols)} columns with zero variance (will be handled in normalization): {zero_variance_cols}")
        
        if low_variance_cols:
            logger.warning(f"Found {len(low_variance_cols)} columns with extremely low variance: {low_variance_cols}")
    else:
        # 原来的低方差过滤逻辑
        low_std_cols = [f"latent_{idx+1}" for idx in np.where(df.std() <= threshold)[0]]
        logger.info(f"Deleting columns with std<={threshold}: {low_std_cols}")
        df.drop(columns=low_std_cols, inplace=True)
    
    # 关键修复：安全的标准化，避免除以0或无穷大
    logger.info("Applying safe normalization...")
    
    # 计算均值和标准差
    mean_vals = df.mean()
    std_vals = df.std()
    
    # 找出标准差为0或接近0的特征
    zero_std_mask = std_vals == 0
    very_small_std_mask = std_vals < 1e-10
    small_std_mask = std_vals < 1e-6
    
    if zero_std_mask.any():
        logger.warning(f"Found {zero_std_mask.sum()} features with zero standard deviation")
        # 对于标准差为0的特征，标准化后应该为0（因为所有值相同）
        # 我们直接将这些特征设置为0
        std_vals[zero_std_mask] = 1.0  # 避免除以0
    
    if very_small_std_mask.any():
        logger.warning(f"Found {very_small_std_mask.sum()} features with very small standard deviation (<1e-10)")
        # 增加一个小的epsilon避免数值问题
        std_vals[very_small_std_mask] = np.maximum(std_vals[very_small_std_mask], 1e-10)
    
    if small_std_mask.any():
        logger.info(f"Found {small_std_mask.sum()} features with small standard deviation (<1e-6)")
    
    # 检查无穷大或NaN值
    if np.any(np.isinf(df.values)):
        logger.warning("Found infinite values in embeddings, replacing with large finite values")
        df = df.replace([np.inf, -np.inf], [1e6, -1e6])
    
    if np.any(np.isnan(df.values)):
        logger.warning("Found NaN values in embeddings, replacing with 0")
        df = df.fillna(0)
    
    # 应用标准化
    normalized_df = pd.DataFrame(
        (df - mean_vals) / std_vals,
        index=df.index,
        columns=df.columns
    )
    
    # 再次检查NaN和无穷大
    if np.any(np.isnan(normalized_df.values)):
        logger.warning("NaN values appeared after normalization, replacing with 0")
        normalized_df = normalized_df.fillna(0)
    
    if np.any(np.isinf(normalized_df.values)):
        logger.warning("Infinite values appeared after normalization, clipping to [-10, 10]")
        normalized_df = normalized_df.clip(-10, 10)
    
    # 验证标准化结果
    normalized_mean = normalized_df.mean()
    normalized_std = normalized_df.std()
    
    logger.info(f"After normalization: mean range = [{normalized_mean.min():.6f}, {normalized_mean.max():.6f}]")
    logger.info(f"After normalization: std range = [{normalized_std.min():.6f}, {normalized_std.max():.6f}]")
    
    # 检查是否有异常值
    extreme_values = (normalized_df.abs() > 10).sum().sum()
    if extreme_values > 0:
        logger.warning(f"Found {extreme_values} extreme values (|value| > 10) after normalization")
        # 可以可选地裁剪极端值
        normalized_df = normalized_df.clip(-10, 10)
    
    # Set output path
    if embedding_path is None:
        directory = EMBEDDING_DIR / "rdkit" / "data" / "embeddings"
        directory.mkdir(parents=True, exist_ok=True)
        output_path = directory / "rdkit2D_embedding.parquet"
    else:
        output_path = Path(embedding_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving embeddings for {len(normalized_df)} SMILES to {output_path}")
    logger.info(f"Final embedding shape: {normalized_df.shape}")
    logger.info(f"Final embedding - Min: {normalized_df.values.min():.6f}, Max: {normalized_df.values.max():.6f}, Mean: {normalized_df.values.mean():.6f}, Std: {normalized_df.values.std():.6f}")
    
    normalized_df.to_parquet(output_path)
    return output_path


def validate(embedding_df, adata, smiles_key='SMILES'):
    """
    Validate by comparing canonical SMILES from the dataset vs. the
    canonical SMILES in the embedding DataFrame index.
    
    Splits on '..' if present, but NOT on single '.'.
    Then each piece is canonicalized the same way we do in embed_and_save_embeddings.
    If the canonical form is found in embedding_df.index, it won't be listed as missing.
    """
    logger.info("Starting validation of embeddings against dataset SMILES")

    dataset_canonical_smiles = set()
    for raw_smile in adata.obs[smiles_key]:
        # If ".." in raw_smile, split it into multiple
        splitted = [raw_smile]
        if ".." in raw_smile:
            splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
        
        # Canonicalize each splitted piece
        for s in splitted:
            c = canonicalize_smiles(s)
            if c is not None:
                dataset_canonical_smiles.add(c)

    # Compare canonical forms
    embedding_smiles = set(embedding_df.index)
    missing_smiles = dataset_canonical_smiles - embedding_smiles
    if missing_smiles:
        logger.warning(
            f"Found {len(missing_smiles)} SMILES in dataset that are missing from embeddings."
        )
        for smile in list(missing_smiles)[:10]:
            logger.warning(f"  {smile}")
        logger.warning("Continuing without raising an error.")
    else:
        logger.info("Validation successful! All combined SMILES are accounted for.")

    # Optional: note any extra SMILES in embeddings but not in the dataset
    extra_smiles = embedding_smiles - dataset_canonical_smiles
    if extra_smiles:
        logger.info(
            f"Note: Embeddings contain {len(extra_smiles)} additional SMILES not in dataset."
        )


def compute_rdkit_embeddings(h5ad_path, output_path=None, smiles_key='SMILES', skip_variance_filter=False, fixed_embedding_dim = None):
    """
    Generate RDKit embeddings for SMILES strings from an h5ad file.
    
    Args:
        h5ad_path (str): Path to the h5ad file containing SMILES data
        output_path (str, optional): Path to save the embeddings. If None, saves to default location
        smiles_key (str): Key for SMILES data in the h5ad file
        skip_variance_filter (bool): If True, keeps all features without filtering low variance ones
    """
    # Create progress bar for main steps
    main_steps = ['Loading SMILES', 'Computing embeddings', 'Saving results', 'Validating']
    
    with tqdm(total=len(main_steps), desc="Overall progress", position=0) as pbar:
        # Step 1: Load SMILES and dataset
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
        
        # Expand any ".." into multiple SMILES, but leave single-dot lines as-is
        expanded_smiles_data = []
        for raw_smile in raw_smiles_data:
            if ".." in raw_smile:
                splitted = [x.strip() for x in raw_smile.split("..") if x.strip()]
                expanded_smiles_data.extend(splitted)
            else:
                expanded_smiles_data.append(raw_smile)
        
        # De-duplicate
        smiles_data = list(set(expanded_smiles_data))
        logger.info(f"Total unique SMILES (after splitting '..'): {len(smiles_data)}")
        pbar.update(1)
        
        # Step 2: Process and compute embeddings
        pbar.set_description("Computing embeddings")
        output_file = embed_and_save_embeddings(
            smiles_data,
            embedding_path=output_path,
            skip_variance_filter=skip_variance_filter,
            fixed_embedding_dim = fixed_embedding_dim
        )
        pbar.update(1)
        
        # Step 3: Save and load verification
        pbar.set_description("Saving results")
        df = pd.read_parquet(output_file)
        logger.info(f"Successfully generated and saved embeddings with shape: {df.shape}")
        logger.info(f"Embeddings saved to: {output_file}")
        pbar.update(1)
        
        # Step 4: Validate (no error if missing)
        pbar.set_description("Validating")
        validate(df, adata, smiles_key)
        pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RDKit embeddings from SMILES data')
    parser.add_argument('h5ad_path', type=str, help='Path to the h5ad file containing SMILES data')
    parser.add_argument('--output_path', type=str, help='Path to save the embeddings', default=None)
    parser.add_argument('--smiles_key', type=str, default='SMILES', help='Key for SMILES data in the h5ad file')
    parser.add_argument('--skip_variance_filter', action='store_true', help='Skip dropping low-variance columns')

    args = parser.parse_args()
    
    compute_rdkit_embeddings(
        h5ad_path=args.h5ad_path,
        output_path=args.output_path,
        smiles_key=args.smiles_key,
        skip_variance_filter=args.skip_variance_filter
    )


