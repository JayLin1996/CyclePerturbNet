import hydra
import torch
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
import json
from datetime import datetime  # 修正导入方式

# 导入必要的模块
from data.data import load_dataset_splits
from models.train import evaluate_r2
from models.lightning_module import ChemCPA  

@hydra.main(version_base=None, config_path="./config/", config_name="lincs")
def test_model(args):
    """加载训练好的模型并在所有数据集上评估"""
    
    # 1. 加载配置和数据
    data_params = args["dataset"]
    datasets, dataset = load_dataset_splits(**data_params, return_dataset=True)
    
    # 数据集配置
    dataset_config = {
        "num_genes": datasets["training"].num_genes,
        "num_drugs": datasets["training"].num_drugs,
        "num_covariates": datasets["training"].num_covariates,
        "use_drugs_idx": dataset.use_drugs_idx,
        "canon_smiles_unique_sorted": dataset.canon_smiles_unique_sorted,
    }
    
    print(f"数据集信息:")
    print(f"  基因数量: {dataset_config['num_genes']}")
    print(f"  药物数量: {dataset_config['num_drugs']}")
    print(f"  协变量数量: {dataset_config['num_covariates']}")
    print()
    
    # 2. 初始化模型（不训练）
    model = ChemCPA(args, dataset_config)
    
    # 3. 加载训练好的checkpoint
    checkpoint_path = input("请输入checkpoint文件路径（或按回车使用默认路径）: ").strip()
    if not checkpoint_path:
        # 如果没有提供路径，尝试使用默认配置路径
        run_id = args.get("wandb", {}).get("run_id") or input("请输入wandb run_id: ")
        checkpoint_path = Path(args["training"]["save_dir"]) / run_id / "last.ckpt"
    
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        return
    
    print(f"正在加载checkpoint: {checkpoint_path}")
    
    # 使用PyTorch Lightning的load_from_checkpoint方法
    trained_model = ChemCPA.load_from_checkpoint(
        checkpoint_path,
        config=args,
        dataset_config=dataset_config
    )
    
    # 将模型设置为评估模式
    trained_model.eval()
    trained_model.freeze()
    
    print("模型加载成功！")
    
    # 4. 获取数据集的基因数据用于评估
    # 注意：根据您的数据结构调整以下代码
    if hasattr(datasets['training'], 'genes'):
        train_control_genes = datasets['training'].genes
        test_control_genes = datasets['test'].genes
        ood_control_genes = datasets['ood'].genes if 'ood' in datasets else None
    else:
        # 如果数据集中没有直接包含genes属性，可能需要从其他方式获取
        print("警告: 无法直接从数据集中获取control genes")
        print("请确保数据加载正确")
        train_control_genes = None
        test_control_genes = None
        ood_control_genes = None
    
    # 5. 在三个数据集上进行评估
    print("\n" + "="*60)
    print("开始模型评估...")
    print("="*60)
    
    results = {}
    
    # 训练集评估
    print("\n评估训练集...")
    try:
        train_eval = evaluate_r2(
            trained_model.model,  # 使用ComPert模型
            datasets['training'],
            train_control_genes
        )
        results["train"] = {
            "R2_mean": float(train_eval[0]),
            "R2_mean_de": float(train_eval[1]),
            "R2_var": float(train_eval[2]),
            "R2_var_de": float(train_eval[3])
        }
        print(f"  训练集R2_mean: {train_eval[0]:.4f}")
        print(f"  训练集R2_mean_de: {train_eval[1]:.4f}")
    except Exception as e:
        print(f"  训练集评估失败: {e}")
        results["train"] = {"error": str(e)}
    
    # 测试集评估
    print("\n评估测试集...")
    try:
        test_eval = evaluate_r2(
            trained_model.model,
            datasets['test'],
            test_control_genes
        )
        results["test"] = {
            "R2_mean": float(test_eval[0]),
            "R2_mean_de": float(test_eval[1]),
            "R2_var": float(test_eval[2]),
            "R2_var_de": float(test_eval[3])
        }
        print(f"  测试集R2_mean: {test_eval[0]:.4f}")
        print(f"  测试集R2_mean_de: {test_eval[1]:.4f}")
    except Exception as e:
        print(f"  测试集评估失败: {e}")
        results["test"] = {"error": str(e)}
    
    # OOD集评估（如果存在）
    if 'ood' in datasets and ood_control_genes is not None:
        print("\n评估OOD集...")
        try:
            ood_eval = evaluate_r2(
                trained_model.model,
                datasets['ood'],
                ood_control_genes
            )
            results["ood"] = {
                "R2_mean": float(ood_eval[0]),
                "R2_mean_de": float(ood_eval[1]),
                "R2_var": float(ood_eval[2]),
                "R2_var_de": float(ood_eval[3])
            }
            print(f"  OOD集R2_mean: {ood_eval[0]:.4f}")
            print(f"  OOD集R2_mean_de: {ood_eval[1]:.4f}")
        except Exception as e:
            print(f"  OOD集评估失败: {e}")
            results["ood"] = {"error": str(e)}
    else:
        print("\n警告: OOD数据集不存在或未提供control genes")
        results["ood"] = {"status": "not_available"}
    
    # 6. 打印详细结果
    print("\n" + "="*60)
    print("评估结果汇总:")
    print("="*60)
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name.upper()} SET:")
        if "error" in metrics:
            print(f"  错误: {metrics['error']}")
        elif "status" in metrics:
            print(f"  状态: {metrics['status']}")
        else:
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
    
    # 7. 保存结果到文件
    # 获取当前工作目录
    current_dir = Path.cwd()
    print(f"\n当前工作目录: {current_dir}")
    
    # 创建输出目录
    output_dir = current_dir / "evaluation_results"
    output_dir.mkdir(exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"
    
    # 添加元数据
    results_with_metadata = {
        "checkpoint": str(checkpoint_path),
        "evaluation_date": timestamp,
        "config": OmegaConf.to_container(args, resolve=True),
        "results": results
    }
    
    with open(results_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    print(f"\n结果已保存到: {results_file}")
    print(f"完整路径: {results_file.absolute()}")
    
    return results

if __name__ == "__main__":
    test_model()