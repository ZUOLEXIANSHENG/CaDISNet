"""
CaDISNet 2-Class Training Script
Includes: HSIC, VIB, Domain Adversarial, EMA, TensorBoard
Matches 3-class training pipeline structure
"""
import os
import sys
import re
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io
import glob
import uuid
import copy
from CaDISNet import CaDISNet
import random
from scipy.signal import butter, lfilter, resample
from scipy.stats import zscore
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------------------
# 1. 工具函数：HSIC Loss 计算
# ----------------------------------------------------------------------
def compute_hsic_loss(z_s, z_u):
    """HSIC with adaptive RBF kernel widths"""
    N = z_s.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=z_s.device, requires_grad=True)

    z_s_norm = (z_s ** 2).sum(1).view(-1, 1)
    dist_s = z_s_norm + z_s_norm.t() - 2.0 * torch.mm(z_s, z_s.t())

    z_u_norm = (z_u ** 2).sum(1).view(-1, 1)
    dist_u = z_u_norm + z_u_norm.t() - 2.0 * torch.mm(z_u, z_u.t())

    # median heuristic for sigma
    sigma_s = torch.median(dist_s.detach())
    sigma_u = torch.median(dist_u.detach())

    if sigma_s <= 1e-5:
        sigma_s = torch.tensor(1.0, device=z_s.device)
    if sigma_u <= 1e-5:
        sigma_u = torch.tensor(1.0, device=z_s.device)

    K = torch.exp(-dist_s / (2 * sigma_s))
    L = torch.exp(-dist_u / (2 * sigma_u))

    H = torch.eye(N, device=z_s.device) - (1.0 / N) * torch.ones((N, N), device=z_s.device)

    KH = torch.mm(K, H)
    LH = torch.mm(L, H)
    hsic = torch.trace(torch.mm(KH, LH))
    
    return hsic / ((N - 1) ** 2)


# ----------------------------------------------------------------------
# VIB: KL 散度到标准正态 (低配版，未使用重参数化时等价于 L2)
# ----------------------------------------------------------------------
def compute_kl_loss(mu, logvar=None):
    """
    KL( N(mu, sigma^2) || N(0, I) )
    若未提供 logvar（确定性编码器），退化为 0.5 * ||mu||^2
    """
    if logvar is None:
        return 0.5 * torch.mean(mu.pow(2))
    return 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)


# ----------------------------------------------------------------------
# Zu 对比学习: Supervised Contrastive Loss
# 目标: 让同被试的 Zu 聚集，不同被试的 Zu 远离
# 这样 Zu 能更好地"吸收"被试特异性噪声，净化 Zs
# ----------------------------------------------------------------------
def compute_contrastive_loss_zu(zu, domain_ids, temperature=0.1):
    """
    Supervised Contrastive Loss for Zu (variation features).
    
    Args:
        zu: (N, D) 变异特征向量
        domain_ids: (N,) 被试/域标签
        temperature: 温度参数，控制分布锐度
    
    Returns:
        对比损失值 (标量)
    """
    N = zu.size(0)
    if N <= 1:
        return torch.tensor(0.0, device=zu.device, requires_grad=True)
    
    # L2 归一化
    zu_norm = F.normalize(zu, p=2, dim=1)  # (N, D)
    
    # 计算相似度矩阵: (N, N)
    sim_matrix = torch.mm(zu_norm, zu_norm.t()) / temperature
    
    # 创建正样本掩码: 同一被试为正样本
    domain_ids = domain_ids.view(-1, 1)  # (N, 1)
    pos_mask = (domain_ids == domain_ids.t()).float()  # (N, N)
    pos_mask.fill_diagonal_(0)  # 排除自身
    
    # 负样本掩码: 不同被试
    neg_mask = 1.0 - pos_mask
    neg_mask.fill_diagonal_(0)
    
    # 如果某个样本没有正样本，跳过
    pos_count = pos_mask.sum(dim=1)
    valid_mask = pos_count > 0
    
    if not valid_mask.any():
        return torch.tensor(0.0, device=zu.device, requires_grad=True)
    
    # 数值稳定性
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    
    exp_logits = torch.exp(logits)
    
    all_mask = torch.ones_like(pos_mask)
    all_mask.fill_diagonal_(0)
    denominator = (exp_logits * all_mask).sum(dim=1, keepdim=True) + 1e-8
    
    log_prob = logits - torch.log(denominator)
    
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)
    mean_log_prob_pos = mean_log_prob_pos / (pos_count + 1e-8)
    
    loss = -mean_log_prob_pos[valid_mask].mean()
    
    return loss

# ----------------------------------------------------------------------
# 2. 数据处理与加载
# ----------------------------------------------------------------------
class EEGDataset(Dataset):
    def __init__(self, data, labels, domains=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        if domains is not None:
            self.domains = torch.tensor(domains, dtype=torch.long)
        else:
            self.domains = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.domains is not None:
            return self.data[idx], self.labels[idx], self.domains[idx]
        return self.data[idx], self.labels[idx]

DEFAULT_PREPROCESS_CFG = {
    'max_trial_len_sec': 3.5,
    'window_len_sec': 2.0,
    'window_step_samples': 128,
    'target_rate': 100,
    'bandpass': (4, 40),
    'filter_padding': 100,
    'enable_memmap': False,  # ✅ 默认关闭 memmap，避免超算配额问题
    'memmap_threshold_mb': 4096,
    'memmap_cache_dir': '/tmp/eeg_memmap_cache',  # 使用 /tmp 避免占用用户配额
}


def segment_trials(data, labels, segment_size, step_size, desired_len=None, domains=None):
    num_trials, num_channels, num_samples = data.shape
    segments = []
    label_list = []
    domain_list = [] if domains is not None else None

    for trial_idx in range(num_trials):
        trial = data[trial_idx]
        if np.isnan(trial).all():
            continue
        max_start = trial.shape[1] - segment_size + 1
        if max_start <= 0:
            continue
        for start in range(0, max_start, max(1, step_size)):
            end = start + segment_size
            if end > trial.shape[1]:
                break
            segment = trial[:, start:end].astype(np.float32, copy=False)
            if desired_len is not None and segment.shape[1] != desired_len:
                segment = resample(segment, desired_len, axis=1).astype(np.float32, copy=False)
            segments.append(segment)
            label_list.append(labels[trial_idx])
            if domain_list is not None:
                domain_list.append(domains[trial_idx])

    if not segments:
        return None, None, None

    segmented_data = np.stack(segments, axis=0)
    mask = ~np.isnan(segmented_data).any(axis=(1, 2))
    segmented_data = segmented_data[mask]
    label_array = np.asarray(label_list, dtype=np.int64)[mask]
    domain_array = None
    if domain_list is not None:
        domain_array = np.asarray(domain_list, dtype=np.int64)[mask]
    return segmented_data, label_array, domain_array


def bandpass_filter_in_batches(data, b, a, padding, batch_size=512):
    if padding < 0:
        padding = 0
    n_segments = data.shape[0]
    filtered = np.empty_like(data, dtype=np.float32)
    for start in range(0, n_segments, batch_size):
        end = min(start + batch_size, n_segments)
        chunk = data[start:end]
        if padding > 0:
            chunk_padded = np.pad(chunk, ((0, 0), (0, 0), (padding, padding)), mode='constant')
            chunk_filtered = lfilter(b, a, chunk_padded, axis=2).astype(np.float32, copy=False)
            filtered[start:end] = chunk_filtered[:, :, padding:-padding]
        else:
            filtered[start:end] = lfilter(b, a, chunk, axis=2).astype(np.float32, copy=False)
    return filtered


# ==========================================
# 🔥 新增：定义安全的 Z-score 函数
# ==========================================
def safe_zscore(data, axis=2):
    """
    安全的 Z-score，防止除以 0 产生 NaN
    输入形状通常是 (N, C, T)，axis=2 表示对时间轴归一化
    """
    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (data - mean) / std


def preprocess_eeg_trials(data, labels, cfg, domains=None):
    if data is None or labels is None or len(data) == 0:
        if domains is not None:
            return None, None, None
        return None, None

    data = np.asarray(data, dtype=np.float32)
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        car = np.nanmean(data, axis=1, keepdims=True)
    car = np.nan_to_num(car, nan=0.0)
    data = data - car

    target_rate = cfg.get('target_rate', 100)
    source_rate = cfg.get('source_rate', target_rate)
    window_sec = cfg.get('window_len_sec', 1.0)
    segment_size = max(1, int(window_sec * source_rate))
    step_size = max(1, cfg.get('window_step_samples', 128))
    desired_len = int(window_sec * target_rate)
    resample_len = desired_len if segment_size != desired_len else None

    segments, seg_labels, seg_domains = segment_trials(
        data, labels, segment_size, step_size, resample_len, domains
    )

    if segments is None:
        if domains is not None:
            return None, None, None
        return None, None

    padding = min(cfg.get('filter_padding', 100), max(0, segments.shape[2] // 2))
    b, a = butter(4, cfg.get('bandpass', (4, 40)), btype='bandpass', fs=target_rate)
    b = b.astype(np.float32)
    a = a.astype(np.float32)
    segments = bandpass_filter_in_batches(segments, b, a, padding)

    if np.any(np.isnan(segments)) or np.any(np.isinf(segments)):
        segments = np.nan_to_num(segments, nan=0.0, posinf=0.0, neginf=0.0)

    segments = safe_zscore(segments, axis=2)
    segments = np.nan_to_num(segments, nan=0.0, posinf=0.0, neginf=0.0)
    segments = segments[:, np.newaxis, :, :].astype(np.float32, copy=False)

    if seg_domains is not None:
        return segments, seg_labels, seg_domains
    return segments, seg_labels

def load_mat_file(file_path, cfg):
    """加载 main_model_training.py 生成的 .mat 数据，返回未分段 trial。"""
    try:
        mat = scipy.io.loadmat(file_path, variable_names=['eeg', 'event'])
    except Exception as e:
        print(f"[WARN] 无法读取 {os.path.basename(file_path)}: {e}")
        return None, None, None

    if 'eeg' not in mat or 'event' not in mat:
        return None, None, None

    try:
        eeg_struct = mat['eeg'][0, 0]
        raw_data = eeg_struct['data'] 
        fs = eeg_struct['fsample'][0, 0]
        events = mat['event'][0] 
    except (IndexError, KeyError):
        return None, None, None

    if raw_data.ndim == 3 and raw_data.shape[0] == 1:
        raw_data = raw_data[0]
    raw_data = np.asarray(raw_data, dtype=np.float32)

    if isinstance(fs, np.ndarray):
        fs = fs.item()
    fs = float(fs)

    # 2-Class Label Map
    label_map = {1: 0, 4: 1}
    window_sec = cfg.get('max_trial_len_sec', 3.5)
    samples_per_trial = max(1, int(window_sec * fs))

    def _to_scalar(value):
        if isinstance(value, np.ndarray):
            value = value.squeeze()
            if value.size == 0:
                return None
            return value.item()
        return value

    def _to_str(value):
        if isinstance(value, np.ndarray):
            value = value.flatten()
            if value.size == 0:
                return ""
            value = value[0]
        if isinstance(value, bytes):
            return value.decode(errors='ignore')
        return str(value)

    trials, labels = [], []
    skipped_labels = set()
    
    for ev in events:
        try:
            etype = _to_str(ev['type']).lower()
            esample = _to_scalar(ev['sample'])
            evalue = _to_scalar(ev['value'])
        except Exception:
            continue

        if 'target' not in etype:
            continue

        if evalue not in label_map or esample is None:
            skipped_labels.add(evalue)
            continue

        start = int(esample)
        end = min(raw_data.shape[1], start + samples_per_trial)
        if end <= start:
            continue
                        
        trial = raw_data[:, start:end]
        if trial.shape[1] < samples_per_trial:
            pad_len = samples_per_trial - trial.shape[1]
            trial = np.pad(trial, ((0, 0), (0, pad_len)), mode='constant', constant_values=np.nan)
        else:
            trial = trial[:, :samples_per_trial]

        trials.append(trial.astype(np.float32, copy=False))
        labels.append(label_map[evalue])

    if not trials:
        return None, None, None
        
    X = np.stack(trials, axis=0)
    y = np.asarray(labels, dtype=np.int64)
    return X, y, fs

def _apply_gaussian_noise(batch, idx_tensor, snr_db):
    if idx_tensor.numel() == 0:
        return
    subset = batch[idx_tensor]
    signal_power = subset.pow(2).mean(dim=(1, 2, 3), keepdim=True)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_std = torch.sqrt(signal_power / snr_linear + 1e-8)
    noise = torch.randn_like(subset) * noise_std
    batch[idx_tensor] = subset + noise

def _apply_channel_dropout(batch, idx_tensor, drop_prob):
    if idx_tensor.numel() == 0 or drop_prob <= 0.0:
        return
    subset = batch[idx_tensor]
    keep_mask = (torch.rand(subset.size(0), subset.size(2), device=subset.device) > drop_prob).float()
    keep_mask = keep_mask.view(subset.size(0), 1, subset.size(2), 1)
    batch[idx_tensor] = subset * keep_mask

def _apply_impulse_noise(batch, idx_tensor, prob, strength):
    if idx_tensor.numel() == 0 or prob <= 0.0:
        return
    subset = batch[idx_tensor]
    mask = (torch.rand_like(subset) < prob).float()
    sign = (torch.randint(0, 2, subset.shape, device=subset.device) * 2 - 1).to(subset.dtype)
    impulses = sign * strength
    batch[idx_tensor] = subset + mask * impulses

def _generate_fgsm_samples(model, inputs, labels, eps, device):
    prev_mode = model.training
    model.zero_grad(set_to_none=True)
    if prev_mode:
        model.eval()
    adv_inputs = inputs.clone().detach().to(device)
    adv_inputs.requires_grad = True
    outputs = model(adv_inputs)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = nn.CrossEntropyLoss()(outputs, labels)
    loss.backward()
    adv = adv_inputs + eps * adv_inputs.grad.sign()
    adv = adv.detach()
    model.zero_grad(set_to_none=True)
    if prev_mode:
        model.train()
    return adv

def apply_training_augmentations(model, inputs, labels, device, cfg):
    """
    综合多种扰动 (高覆盖率 + 多噪声类型) 以激活 Z_u
    """
    coverage = cfg.get('coverage', 0.5)
    if coverage <= 0.0:
        return inputs
    
    gaussian_snrs = cfg.get('gaussian_snrs', [])
    option_pool = [('gaussian', snr) for snr in gaussian_snrs]
    
    if not option_pool:
        return inputs
    
    batch = inputs.clone()
    batch_size = batch.size(0)
    mask = torch.rand(batch_size, device=device) < coverage
    if not mask.any():
        return batch
    
    selected_idx = mask.nonzero(as_tuple=False).squeeze(1)
    choice_indices = torch.from_numpy(
        np.random.choice(len(option_pool), size=selected_idx.numel())
    ).to(selected_idx.device)
    
    for option_id in range(len(option_pool)):
        idx_tensor = selected_idx[choice_indices == option_id]
        if idx_tensor.numel() == 0:
            continue
        aug_type, aug_param = option_pool[option_id]
        if aug_type == 'gaussian':
            _apply_gaussian_noise(batch, idx_tensor, aug_param)
    
    return batch

def load_file_auto(file_path, preprocess_cfg=None):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.mat':
        cfg = DEFAULT_PREPROCESS_CFG.copy()
        if preprocess_cfg:
            cfg.update(preprocess_cfg)
        return load_mat_file(file_path, cfg)
    else:
        return None, None, None

def extract_subject_id(file_path):
    match = re.search(r'(S\d{2})', file_path, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _create_memmap(shape, dtype, cache_dir, prefix):
    """
    为大规模分段结果创建磁盘映射，避免一次性占用全部内存。
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.getcwd(), "memmap_cache")
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex}.dat"
    path = os.path.join(cache_dir, filename)
    memmap_arr = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    return memmap_arr, path


def unpack_causal_outputs(outputs):
    """
    兼容不同版本 CaDISNet 的输出，确保训练流程始终可运行。
    """
    if not isinstance(outputs, tuple):
        raise ValueError(f"模型返回类型异常: 期望 tuple, 实际 {type(outputs)}")
    if len(outputs) >= 11:
        return outputs[:11]
    if len(outputs) == 7:
        preds, reconstructed, raw_features, z_s, z_u, domain_logits, preds_zu_adv = outputs
        return (preds, reconstructed, raw_features, z_s, z_u,
                domain_logits, preds_zu_adv, None, None, None, None)
    if len(outputs) == 6:
        preds, reconstructed, raw_features, z_s, z_u, domain_logits = outputs
        return (preds, reconstructed, raw_features, z_s, z_u,
                domain_logits, None, None, None, None, None)
    raise ValueError(f"模型返回 {len(outputs)} 个元素 (期望 6/7/11)")

def load_folder_filtered(folder_path, collect_domains=False, subject_to_idx=None, preprocess_cfg=None):
    print(f"[SCAN] 扫描并筛选文件夹: {folder_path}")
    
    cfg = DEFAULT_PREPROCESS_CFG.copy()
    if preprocess_cfg:
        cfg.update(preprocess_cfg)
    
    if not os.path.exists(folder_path):
        print(f"[WARN] 路径不存在: {folder_path}")
        if collect_domains:
            return None, None, None, subject_to_idx
        return None, None

    mat_files = []
    subject_dirs = [
        d for d in sorted(os.listdir(folder_path))
        if os.path.isdir(os.path.join(folder_path, d))
    ]

    for subj in subject_dirs:
        subj_path = os.path.join(folder_path, subj)
        print(f"   [SCAN] 受试者 {subj} ...")
        for root, dirs, _ in os.walk(subj_path):
            name_lower = os.path.basename(root).lower()
            has_offline = 'offlineimagery' in name_lower
            has_2class = '2class' in name_lower
            if has_2class or has_offline:
                mat_files.extend(glob.glob(os.path.join(root, "*.mat")))

    if not mat_files:
        print("   [WARN] 未找到 OfflineImagery + 2class 数据，尝试直接扫描根目录 .mat ...")
        mat_files = glob.glob(os.path.join(folder_path, "*.mat"))
        if not mat_files:
            if collect_domains:
                return None, None, None, subject_to_idx
            return None, None
    
    print(f"   [INFO] 共找到 {len(mat_files)} 个 .mat 文件")

    if collect_domains and subject_to_idx is None:
        subject_to_idx = {}

    # -------------------------------
    # Pass 1: 只统计段数，避免占内存
    # -------------------------------
    total_segments = 0
    sample_shape = None
    for f in mat_files:
        xt, yt, fs = load_mat_file(f, cfg)
        if xt is None:
            continue
        local_cfg = cfg.copy()
        local_cfg['source_rate'] = fs
        domain_vector = None
        if collect_domains:
            subj = extract_subject_id(f) or "UNKNOWN"
            if subj not in subject_to_idx:
                subject_to_idx[subj] = len(subject_to_idx)
            dom_id = subject_to_idx[subj]
            domain_vector = np.full(len(yt), dom_id, dtype=np.int64)
        result = preprocess_eeg_trials(xt, yt, local_cfg, domains=domain_vector)
        del xt, yt, domain_vector
        if result is None or result[0] is None:
            gc.collect()
            continue
        if collect_domains:
            seg_data, seg_labels, seg_domains = result
        else:
            seg_data, seg_labels = result
        if seg_data.size == 0:
            del seg_data, seg_labels
            gc.collect()
            continue
        if sample_shape is None:
            sample_shape = seg_data.shape[1:]
        total_segments += seg_data.shape[0]
        del seg_data, seg_labels
        if collect_domains:
            del seg_domains
        gc.collect()

    if total_segments == 0 or sample_shape is None:
        if collect_domains:
            return None, None, None, subject_to_idx
        return None, None

    bytes_per_sample = np.prod(sample_shape, dtype=np.int64) * np.dtype(np.float32).itemsize
    estimated_mb = total_segments * bytes_per_sample / (1024 ** 2)
    memmap_enabled = cfg.get('enable_memmap', True)
    memmap_threshold_mb = cfg.get('memmap_threshold_mb', 4096)
    use_memmap = memmap_enabled and estimated_mb >= memmap_threshold_mb
    seg_len_sec = cfg.get('window_len_sec', 1.0)
    if use_memmap:
        print(f"   [INFO] 预计生成 {total_segments} 个 {seg_len_sec:g}s 段 (~{estimated_mb/1024:.2f} GB)，启用磁盘映射缓解内存压力")
    else:
        print(f"   [INFO] 预计生成 {total_segments} 个 {seg_len_sec:g}s 段 (~{estimated_mb/1024:.2f} GB)，直接载入内存")

    cache_dir = cfg.get('memmap_cache_dir')
    data_shape = (total_segments,) + tuple(sample_shape)

    if use_memmap:
        data_mem, data_path = _create_memmap(data_shape, np.float32, cache_dir, "segments")
        label_mem, label_path = _create_memmap((total_segments,), np.int64, cache_dir, "labels")
        domain_mem = domain_path = None
        if collect_domains:
            domain_mem, domain_path = _create_memmap((total_segments,), np.int64, cache_dir, "domains")
    else:
        segments_list, labels_list = [], []
        domain_segment_list = []

    # -------------------------------
    # Pass 2: 真正写入 / 拼接数据
    # -------------------------------
    cursor = 0
    for f in mat_files:
        xt, yt, fs = load_mat_file(f, cfg)
        if xt is None:
            continue
        local_cfg = cfg.copy()
        local_cfg['source_rate'] = fs
        domain_vector = None
        if collect_domains:
            subj = extract_subject_id(f) or "UNKNOWN"
            dom_id = subject_to_idx.setdefault(subj, len(subject_to_idx))
            domain_vector = np.full(len(yt), dom_id, dtype=np.int64)
        result = preprocess_eeg_trials(xt, yt, local_cfg, domains=domain_vector)
        del xt, yt, domain_vector
        if result is None or result[0] is None:
            gc.collect()
            continue
        if collect_domains:
            seg_data, seg_labels, seg_domains = result
        else:
            seg_data, seg_labels = result
            seg_domains = None
        if seg_data.size == 0:
            del seg_data, seg_labels, seg_domains
            gc.collect()
            continue

        n_seg = seg_data.shape[0]
        if use_memmap:
            data_mem[cursor:cursor + n_seg] = seg_data
            label_mem[cursor:cursor + n_seg] = seg_labels
            if collect_domains and seg_domains is not None:
                domain_mem[cursor:cursor + n_seg] = seg_domains
            cursor += n_seg
            del seg_data, seg_labels, seg_domains
        else:
            segments_list.append(seg_data)
            labels_list.append(seg_labels)
            if collect_domains and seg_domains is not None:
                domain_segment_list.append(seg_domains)
        gc.collect()

    if use_memmap:
        data_mem.flush()
        label_mem.flush()
        X = np.memmap(data_path, dtype=np.float32, mode='r+', shape=data_shape)
        y = np.memmap(label_path, dtype=np.int64, mode='r+', shape=(total_segments,))
        if collect_domains:
            domain_mem.flush()
            domains = np.memmap(domain_path, dtype=np.int64, mode='r+', shape=(total_segments,))
            return X, y, domains, subject_to_idx
        return X, y

    if not segments_list:
        if collect_domains:
            return None, None, None, subject_to_idx
        return None, None
        
    X = np.concatenate(segments_list, axis=0)
    y = np.concatenate(labels_list, axis=0)
    
    if collect_domains:
        domains = np.concatenate(domain_segment_list, axis=0)
        return X, y, domains, subject_to_idx
    return X, y

# ----------------------------------------------------------------------
# 3. 训练主循环
# ----------------------------------------------------------------------
def train_causal_model(X_train_full, Y_train_full, domains_train_full,
                       X_test, Y_test, save_name, params, num_domains,
                       domains_test=None):
    print("=" * 60)
    print(f'[INFO] 启动因果驱动训练 (Causal-Driven Training) - 2 Class')
    print(f'[INFO] 训练集: {X_train_full.shape}')
    if X_test is not None:
        print(f'[INFO] 测试集: {X_test.shape}')
    else:
        print(f'[INFO] 测试集: None')
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # TensorBoard setup
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'Causal_2Class_Random_{run_id}')
    writer = SummaryWriter(log_dir=log_dir)
    print(f'[INFO] TensorBoard Log Dir: {log_dir}')

    X_train_tensor = torch.from_numpy(X_train_full).float()
    Y_train_tensor = torch.from_numpy(Y_train_full).long()
    domains_tensor = torch.from_numpy(domains_train_full).long()

    full_dataset = TensorDataset(X_train_tensor, Y_train_tensor, domains_tensor)
    if len(full_dataset) < 2:
        raise ValueError("训练样本数量不足，无法划分训练/验证集。")

    # ----------------------------------------------------------
    # ✅ 随机划分验证集 (90/10 Split)
    # ----------------------------------------------------------
    val_size = max(1, int(0.1 * len(full_dataset)))
    if val_size >= len(full_dataset):
        val_size = len(full_dataset) - 1
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size], generator=generator)
    print(f"[SPLIT] Random Split | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=params['batch_size'],
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params['batch_size'],
        shuffle=False
    )

    test_loader = None
    if X_test is not None and Y_test is not None:
        if domains_test is not None:
            test_dataset = TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(Y_test).long(),
                torch.from_numpy(domains_test).long()
            )
        else:
            test_dataset = TensorDataset(
                torch.from_numpy(X_test).float(),
                torch.from_numpy(Y_test).long()
            )
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    n_chans = X_train_full.shape[2]
    n_samples = X_train_full.shape[3]
    
    model = CaDISNet(
        nb_classes=params['nclass'],
        Chans=n_chans,
        Samples=n_samples,
        latent_dim=32,
        dropoutRate=params['dropout_ratio'],
        num_domains=num_domains
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0.0))
    criterion_rec = nn.MSELoss()
    criterion_domain = nn.CrossEntropyLoss() if num_domains > 0 else None
    
    base_alpha_hsic = params.get('alpha_hsic', 0.1)
    lambda_rec = params.get('lambda_rec', 1.0)
    gamma_vib = params.get('gamma_vib', 0.01)
    warmup_epochs = params.get('warmup_epochs', 20)
    base_lambda_domain = params.get('lambda_domain', 0.1)
    lambda_zu_adv = params.get('lambda_zu_adv', 0.1)
    lambda_contrastive = params.get('lambda_contrastive', 0.1)  # 🔥 Zu对比学习权重
    augment_cfg = params.get('augment', {})

    best_acc = 0.0
    epochs = params['epochs']
    early_stop_patience = params.get('early_stop_patience', 30)
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        
        if epoch < warmup_epochs:
            current_alpha = 0.0
            phase = "Warmup"
        else:
            current_alpha = base_alpha_hsic
            phase = "Causal"
        
        # ----------------------------------------------------
        # 🔥 动态调整对抗权重: Sigmoid Growth
        # ----------------------------------------------------
        p = float(epoch) / epochs
        lambda_p = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
        
        target_domain_weight = params.get('lambda_domain', 0.1)
        current_lambda_domain = lambda_p * target_domain_weight

        running_losses = {'total': 0., 'cls': 0., 'rec': 0., 'hsic': 0., 'domain': 0., 'zu_adv': 0., 'vib': 0., 'contrastive': 0.}
        
        for i, batch in enumerate(train_loader):
            if len(batch) == 3:
                inputs, labels, domains = batch
                domains = domains.to(device)
            else:
                inputs, labels = batch
                domains = None
            inputs, labels = inputs.to(device), labels.to(device)
            aug_inputs = apply_training_augmentations(model, inputs, labels, device, augment_cfg)
            optimizer.zero_grad()
            
            outputs = model(aug_inputs)
            preds, reconstructed, raw_features, z_s, z_u, domain_logits, preds_zu_adv, \
                z_s_mu, z_s_logvar, z_u_mu, z_u_logvar = unpack_causal_outputs(outputs)
            
            loss_cls = criterion_cls(preds, labels)
            loss_rec = criterion_rec(reconstructed, raw_features)
            if preds_zu_adv is not None:
                loss_zu_adv = criterion_cls(preds_zu_adv, labels)
            else:
                loss_zu_adv = torch.tensor(0.0, device=device)
            
            if current_alpha > 0:
                loss_hsic = compute_hsic_loss(z_s, z_u)
            else:
                loss_hsic = torch.tensor(0.0, device=device)
            
            # VIB
            mu_s = z_s_mu if z_s_mu is not None else z_s
            mu_u = z_u_mu if z_u_mu is not None else z_u
            loss_vib = compute_kl_loss(mu_s, z_s_logvar) + compute_kl_loss(mu_u, z_u_logvar)

            if criterion_domain is not None and domain_logits is not None and domains is not None:
                loss_domain = criterion_domain(domain_logits, domains)
            else:
                loss_domain = torch.tensor(0.0, device=device)
            
            # 🔥 Zu 对比学习损失
            if domains is not None and current_alpha > 0:
                loss_contrastive = compute_contrastive_loss_zu(z_u, domains, temperature=0.1)
            else:
                loss_contrastive = torch.tensor(0.0, device=device)
            
            loss_total = (loss_cls + lambda_rec * loss_rec +
                          current_alpha * loss_hsic +
                          gamma_vib * loss_vib +
                          current_lambda_domain * loss_domain +
                          lambda_zu_adv * loss_zu_adv +
                          lambda_contrastive * loss_contrastive)
            
            loss_total.backward()
            optimizer.step()
            
            running_losses['total'] += loss_total.item()
            running_losses['cls'] += loss_cls.item()
            running_losses['rec'] += loss_rec.item()
            running_losses['hsic'] += loss_hsic.item()
            running_losses['domain'] += loss_domain.item() if loss_domain is not None else 0.0
            running_losses['zu_adv'] += loss_zu_adv.item()
            running_losses['vib'] += loss_vib.item()
            running_losses['contrastive'] += loss_contrastive.item()

        n_batches = len(train_loader)
        
        # TensorBoard Logging
        writer.add_scalar('Loss/Total', running_losses['total']/n_batches, epoch)
        writer.add_scalar('Loss/Cls', running_losses['cls']/n_batches, epoch)
        writer.add_scalar('Loss/Rec', running_losses['rec']/n_batches, epoch)
        writer.add_scalar('Loss/HSIC', running_losses['hsic']/n_batches, epoch)
        writer.add_scalar('Loss/Domain', running_losses['domain']/n_batches, epoch)
        writer.add_scalar('Loss/Zu_Adv', running_losses['zu_adv']/n_batches, epoch)
        writer.add_scalar('Loss/Contrastive', running_losses['contrastive']/n_batches, epoch)
        
        log_str = (f"Epoch {epoch+1:03d} [{phase}] | "
                   f"T_Loss:{running_losses['total']/n_batches:.3f} "
                   f"(C:{running_losses['cls']/n_batches:.2f}, "
                   f"R:{running_losses['rec']/n_batches:.6f}, "
                   f"D:{running_losses['domain']/n_batches:.2f}, "
                   f"H:{running_losses['hsic']/n_batches:.6f}, "
                   f"Con:{running_losses['contrastive']/n_batches:.4f})")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    preds = outputs[0]
                else:
                    preds = outputs
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0
        writer.add_scalar('Metrics/Val_Acc', val_acc, epoch)
        log_str += f" | Val Acc: {val_acc:.4f}"
        print(log_str)

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"[EARLY STOP] 连续 {early_stop_patience} 轮验证集无提升，提前结束训练。")
                break

    print(f"[DONE] 训练完成! Best Val Acc: {best_acc:.4f}")

    test_acc = None
    if test_loader is not None:
        print("\n[TEST] 加载最佳模型进行最终测试并提取 t-SNE 特征...")
        model.load_state_dict(torch.load(save_name, map_location=device))
        model.eval()
        correct = 0
        total = 0
        
        # 收集 t-SNE 特征
        all_zs = []
        all_zu = []
        all_labels = []
        all_domains = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 处理有无 domain 的情况
                if len(batch) == 3:
                    inputs, labels, domains = batch
                else:
                    inputs, labels = batch
                    domains = torch.zeros_like(labels) # 如果没有 domain，用 0 占位
                    
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                preds, reconstructed, raw_features, z_s, z_u, domain_logits, preds_zu_adv, \
                    z_s_mu, z_s_logvar, z_u_mu, z_u_logvar = unpack_causal_outputs(outputs)
                
                _, predicted = torch.max(preds.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集特征
                all_zs.append(z_s.cpu().numpy())
                all_zu.append(z_u.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_domains.append(domains.numpy())
                
        test_acc = correct / total if total > 0 else 0.0
        print(f"[FINAL RESULT] CaDISNet Test Accuracy: {test_acc:.4f}")
        
        # 保存 t-SNE 特征到 npz 文件
        tsne_save_path = save_name.replace('.pth', '_tsne_features.npz')
        np.savez(tsne_save_path, 
                 z_s=np.concatenate(all_zs, axis=0), 
                 z_u=np.concatenate(all_zu, axis=0), 
                 labels=np.concatenate(all_labels, axis=0), 
                 domains=np.concatenate(all_domains, axis=0))
        print(f"[INFO] t-SNE 特征已保存至: {tsne_save_path}")
        
    else:
        print("[WARN] 测试集为空，跳过最终评估。")

    # ------------------------------------------------------------------
    # 🔥 方案B: 从训练集提取 t-SNE 特征 (包含所有训练受试者)
    # 优势: 训练集有多个受试者 (LOSO中8个)，一次运行即可画出完整4面板图
    # ------------------------------------------------------------------
    print("\n[t-SNE] 从训练集提取多受试者 Zs/Zu 特征 (用于 t-SNE 可视化)...")
    actual_model_tsne = model
    actual_model_tsne.load_state_dict(torch.load(save_name, map_location=device))
    actual_model_tsne.eval()
    
    full_train_loader = DataLoader(
        full_dataset, batch_size=params['batch_size'], shuffle=False
    )
    
    train_zs_list = []
    train_zu_list = []
    train_labels_list = []
    train_domains_list = []
    
    with torch.no_grad():
        for batch in full_train_loader:
            inputs, labels_b, domains_b = batch
            inputs = inputs.to(device)
            outputs = actual_model_tsne(inputs)
            preds, reconstructed, raw_features, z_s, z_u, domain_logits, preds_zu_adv, \
                z_s_mu, z_s_logvar, z_u_mu, z_u_logvar = unpack_causal_outputs(outputs)
            train_zs_list.append(z_s.cpu().numpy())
            train_zu_list.append(z_u.cpu().numpy())
            train_labels_list.append(labels_b.numpy())
            train_domains_list.append(domains_b.numpy())
    
    tsne_train_path = save_name.replace('.pth', '_train_tsne_features.npz')
    np.savez(tsne_train_path,
             z_s=np.concatenate(train_zs_list, axis=0),
             z_u=np.concatenate(train_zu_list, axis=0),
             labels=np.concatenate(train_labels_list, axis=0),
             domains=np.concatenate(train_domains_list, axis=0))
    print(f"[INFO] 训练集 t-SNE 特征已保存至: {tsne_train_path}")
    print(f"[INFO] 包含 {len(np.unique(np.concatenate(train_domains_list)))} 个受试者的数据")
    print(f"[TIP]  使用以下命令生成 t-SNE 4面板图:")
    print(f"       python plot_tsne_causal.py --npz {tsne_train_path}")

    return test_acc if test_acc is not None else best_acc

# ----------------------------------------------------------------------
# 4. 入口
# ----------------------------------------------------------------------
if __name__ == "__main__":
    params = {
        'nclass': 2,
        'batch_size': 64,
        'epochs': 100,
        'early_stop_patience': 30,
        'lr': 0.005,
        'dropout_ratio': 0.5,
        'alpha_hsic': 0.8,
        'lambda_rec': 0.5,
        'lambda_domain': 0.05,
        'domain_warmup': 20,
        'lambda_zu_adv': 0.1,
        'augment': {
            'coverage': 0.0,
            'gaussian_snrs': []
        },
        'preprocess': {
            'max_trial_len_sec': 3.5,
            'window_len_sec': 2.0,
            'window_step_samples': 128,
            'target_rate': 100,
            'bandpass': (4, 40),
            'filter_padding': 100,
            'enable_memmap': False,  # ✅ 关闭 memmap，直接载入内存
        }
    }
    set_seed(42)
    save_folder = '/data/home/sczc681/run/zqs'
    os.makedirs(save_folder, exist_ok=True)

    train_dir = '/data/home/sczc681/run/zqs/train'
    test_dir = '/data/home/sczc681/run/zqs/test'
    preprocess_cfg = params.get('preprocess')
    
    print(f"[TRAIN] 加载训练集 (仅 OfflineImagery & 2class)...")
    X_train, Y_train, domains_train, subject_map = load_folder_filtered(
        train_dir, collect_domains=True, preprocess_cfg=preprocess_cfg
    )
    
    if X_train is None or domains_train is None:
        print("[ERROR] 训练集加载失败或缺少受试者标签，退出")
        sys.exit(1)
        
    print(f"[EVAL] 加载测试集 (仅 OfflineImagery & 2class)...")
    X_test, Y_test, domains_test, _ = load_folder_filtered(
        test_dir, collect_domains=True, subject_to_idx=subject_map, preprocess_cfg=preprocess_cfg
    )
    
    num_domains = len(subject_map) if subject_map else 0
    if num_domains == 0:
        print("[ERROR] 未解析到任何受试者 ID，无法训练域分类器")
        sys.exit(1)
    
    save_path = os.path.join(save_folder, 'CaDISNet_2Class.pth')
    train_causal_model(X_train, Y_train, domains_train, X_test, Y_test, save_path, params, num_domains,
                       domains_test=domains_test)
