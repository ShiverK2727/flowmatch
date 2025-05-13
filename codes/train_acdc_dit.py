import os
import sys
import yaml
import argparse
import datetime
import shutil
import logging
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.dit import MODEL_CONFIGS
from models.ema import EMACallback
from dataset.acdc_dataset import DataSet4ACDC, RandomGeneratorImage, StandGeneratorImage
from utils.logger import setup_logger, log_info, log_error

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_experiment_dir(base_path, config_path):
    """创建唯一的实验目录并复制配置文件"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_path, f"exp_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # 复制配置文件到实验目录
    config_filename = os.path.basename(config_path)
    shutil.copy2(config_path, os.path.join(exp_dir, config_filename))

    # 保存运行时配置
    with open(os.path.join(exp_dir, 'runtime_config.yaml'), 'w') as f:
        runtime_config = {
            'config_file': config_path,
            'timestamp': timestamp,
            'python_path': os.sys.executable,
            'working_dir': os.getcwd(),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'None'),
        }
        yaml.dump(runtime_config, f, default_flow_style=False)

    return exp_dir


def setup_dataloaders(config):
    """根据配置设置数据加载器"""
    data_config = config['data']

    train_dataset = DataSet4ACDC(
        data_config['base_dir'],
        split='train',
        transform=transforms.Compose([
            RandomGeneratorImage(
                [data_config['image_size'], data_config['image_size']],
                scale_to_neg1_pos1=data_config.get('scale_to_neg1_pos1', True)
            )
        ])
    )

    val_dataset = DataSet4ACDC(
        data_config['base_dir'],
        split='val',
        transform=transforms.Compose([
            StandGeneratorImage(
                [data_config['image_size'], data_config['image_size']],
                scale_to_neg1_pos1=data_config.get('scale_to_neg1_pos1', True)
            )
        ])
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config.get('num_workers', 4),
        drop_last=True,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config.get('num_workers', 4),
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader, val_dataloader


def setup_model(config, latent_shape, output_dir):
    """根据配置设置模型"""
    model_config = config['model']
    model_type = model_config['type']

    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    img_save_path = os.path.join(output_dir, model_config.get('image_save_name', 'log_images'))

    # 构建完整的模型参数
    model_kwargs = {
        # 架构参数
        'input_size': model_config['input_size'],
        'in_channels': model_config['in_channels'],
        'class_dropout_prob': model_config['class_dropout_prob'],
        'num_classes': model_config['num_classes'],
        'learn_sigma': model_config['learn_sigma'],

        # 训练参数
        'learning_rate': model_config['learning_rate'],
        'training_type': model_config['training_type'],
        'vae_model_path': model_config['vae_model_path'],

        # 流匹配参数
        'denoise_timesteps': model_config['denoise_timesteps'],
        'denoise_timesteps_target': model_config['denoise_timesteps_target'],
        'bootstrap_every': model_config['bootstrap_every'],
        'bootstrap_dt_bias': model_config['bootstrap_dt_bias'],
        'bootstrap_cfg': model_config['bootstrap_cfg'],
        'cfg_scale': model_config['cfg_scale'],
        'bootstrap_ema': model_config['bootstrap_ema'],
        'eval_size': model_config['eval_size'],
        'force_t': model_config['force_t'],
        'force_dt': model_config['force_dt'],

        # 运行时参数
        'lightning_mode': True,
        'latent_shape': latent_shape,
        'image_save_path': img_save_path,
    }

    model = MODEL_CONFIGS[model_type](**model_kwargs)
    print(f"Model type: {model_type}")
    print(f"Parameter count: {count_parameters(model)}")

    return model


def setup_trainer(config, wandb_logger, exp_dir, gpu_ids, train_dataset_size):
    """设置训练器"""
    trainer_config = config['trainer']
    data_config = config['data']
    
    # 计算总epoch数
    total_steps = trainer_config['max_steps']
    batch_size = data_config['batch_size']
    steps_per_epoch = train_dataset_size // batch_size
    total_epochs = total_steps // steps_per_epoch + (1 if total_steps % steps_per_epoch else 0)
    
    log_info(f"Training configuration:", print_message=True)
    log_info(f"  - Total steps: {total_steps}", print_message=True)
    log_info(f"  - Steps per epoch: {steps_per_epoch}", print_message=True)
    log_info(f"  - Total epochs: {total_epochs}", print_message=True)
    log_info(f"  - Batch size: {batch_size}", print_message=True)
    log_info(f"  - Dataset size: {train_dataset_size}", print_message=True)

    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = []

    # 设置检查点回调
    if 'every_n_train_steps' in trainer_config:
        # 基于步数保存
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(exp_dir, 'checkpoints'),
            filename="model-{step:06d}",
            every_n_train_steps=trainer_config['every_n_train_steps'],
            save_last=True,
            verbose=True,
            save_on_train_epoch_end=False,
            enable_version_counter=True,
        )
        log_info(f"Checkpoint configuration (step-based):")
        log_info(f"  - Save every {trainer_config['every_n_train_steps']} steps")
    else:
        # 基于epoch保存
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(exp_dir, 'checkpoints'),
            filename="model-{epoch:03d}",
            every_n_epochs=trainer_config.get('every_n_epochs', 100),
            save_last=True,
            verbose=True,
            save_on_train_epoch_end=True,
            enable_version_counter=True,
        )
        log_info(f"Checkpoint configuration (epoch-based):")
        log_info(f"  - Save every {trainer_config['every_n_epochs']} epochs")
        
    callbacks.append(checkpoint_callback)

    # 如果需要保存最优模型，添加监控配置
    if trainer_config.get('save_best_model', False):
        checkpoint_callback.monitor = "val_loss"
        checkpoint_callback.mode = "min"
        checkpoint_callback.save_top_k = trainer_config.get('save_top_k', 3)
        log_info(f"  - Save best model based on val_loss")
        log_info(f"  - Save top {trainer_config.get('save_top_k', 3)} models")

    # 设置EMA回调
    if trainer_config.get('use_ema', False):
        ema_callback = EMACallback(decay=trainer_config.get('ema_decay', 0.999))
        callbacks.append(ema_callback)
        log_info(f"  - Using EMA with decay {trainer_config.get('ema_decay', 0.999)}")

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=total_epochs,  # 使用计算得到的epoch数
        max_steps=trainer_config.get('max_steps', -1),
        check_val_every_n_epoch=trainer_config.get('check_val_every_n_epoch', 200),
        log_every_n_steps=trainer_config.get('log_every_n_steps', 50),
        gradient_clip_val=trainer_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=trainer_config.get('accumulate_grad_batches', 1),
        num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', 1),
        limit_val_batches=trainer_config.get('limit_val_batches', 1.0),
        callbacks=callbacks,
        logger=wandb_logger,
        default_root_dir=exp_dir,
        accelerator="gpu",
        devices=len(gpu_ids),
        strategy="ddp_find_unused_parameters_false" if len(gpu_ids) > 1 else "auto"
    )

    return trainer

# single gpu: [00:55<10:24,  0.64it/s, v_num=5]
# 3 gpus: [03:35<00:00,  0.68it/s, v_num=7]
# by github
def main(args):
    # 加载配置
    config = load_config(args.configs)

    # 创建实验目录
    exp_dir = create_experiment_dir(args.exp, args.configs)
    # print(f"Experiment directory: {exp_dir}")
    log_info(f"Experiment directory: {exp_dir}", print_message=True)

    setup_logger(exp_dir, log_level=logging.INFO)

    # 设置数据加载器
    train_dataloader, val_dataloader = setup_dataloaders(config)

    log_info(f"len(train_dataset): {len(train_dataloader.dataset)}", print_message=True)
    log_info(f"len(val_dataset): {len(val_dataloader.dataset)}", print_message=True)

    # 获取示例batch以确定latent shape
    images_i, _ = next(iter(train_dataloader))
    batch_size = config['data']['batch_size']
    in_channels = config['model'].get('in_channels', 16)
    latent_size = images_i.shape[2] // 8
    latent_shape = (batch_size, in_channels, latent_size, latent_size)

    # 设置模型
    model = setup_model(config, latent_shape, exp_dir)

    log_info(f"Model type: {config['model']['type']}")
    log_info(f"Parameter count: {count_parameters(model)}")

    # 设置logger
    logger_name = f"{config['model']['type']}_{Path(exp_dir).name}"
    wandb_logger = WandbLogger(
        name=logger_name,
        project=config['logging'].get('project_name', 'shortcut_model'),
        save_dir=exp_dir,
    )

    # 保存模型配置到wandb
    if wandb_logger:
        wandb_logger.experiment.config.update(config)
        wandb_logger.experiment.config.update({"args": vars(args)})

    # 处理设备配置
    gpu_ids = [int(x) for x in args.gpus.split(',')]
    log_info(f"Using GPUs: {gpu_ids}")

    # 设置trainer
    trainer = setup_trainer(config, wandb_logger, exp_dir, gpu_ids, len(train_dataloader.dataset))

    log_info("Starting training...")
    # 训练模型
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path=args.resume if args.resume else None
    )
    log_info("Training completed!")


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser(description='DiT Training Script')
    parser.add_argument('--configs', type=str, required=True, help='Path to configs yaml file')
    parser.add_argument('--gpus', type=str, default='0', help='GPU ids to use, e.g., "0,1,2"')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--exp', type=str, default='./pl/', help='Path to checkpoint to resume training')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main(args)
