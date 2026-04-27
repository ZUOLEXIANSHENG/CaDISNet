import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# ==============================================================================
# 1. 梯度反转层 (GRL)
# ==============================================================================
class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReverseLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReverseLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReverseFunction.apply(x, self.alpha)

# ==============================================================================
# 2. 时序 Transformer 模块 (Temporal Transformer)
# ==============================================================================
class TemporalTransformer(nn.Module):
    """
    使用多头自注意力捕捉全局时序逻辑。
    """
    def __init__(self, input_dim, time_steps, num_heads=4, num_layers=1, dropout=0.5):
        super(TemporalTransformer, self).__init__()
        # input_dim 是 F2 (通常为 16)
        # MHSA 需要 embed_dim 是 num_heads 的倍数
        self.embed_dim = input_dim 
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=num_heads, 
            dim_feedforward=128, 
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 可学习的位置编码 [Batch=1, Time_steps, F2]
        self.pos_embedding = nn.Parameter(torch.zeros(1, time_steps, input_dim))

    def forward(self, x):
        # x shape: (Batch, F2, Time) -> (Batch, Time, F2)
        x = x.permute(0, 2, 1)
        # 加入位置信息
        x = x + self.pos_embedding
        # Transformer 处理
        x = self.transformer(x)
        # Flatten: (Batch, Time * F2)
        x = x.reshape(x.size(0), -1)
        return x

# ==============================================================================
# 3. CausalEEGNet 主模型 (含分支对抗)
# ==============================================================================
class CausalEEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, 
                 dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, 
                 latent_dim=32, num_domains=0): 
        super(CausalEEGNet, self).__init__()

        self.F2 = F2

        # --- A. 特征提取 (EEGNet Backbone) ---
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        self.separable_conv_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.separable_conv_point = nn.Conv2d(F1 * D, F2, (1, 1), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # --- 🔥 [核心修复] 自动检测时序维度 (Auto Dimension Detection) ---
        # 抛弃硬编码的 Samples // 32，改用 Dummy Forward 精确获取输出大小
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, Chans, Samples)
            dummy_x = self.conv1(dummy_x)
            dummy_x = self.bn1(dummy_x)
            dummy_x = self.depthwise_conv(dummy_x)
            dummy_x = self.bn2(dummy_x)
            dummy_x = self.elu1(dummy_x)
            dummy_x = self.avg_pool1(dummy_x)
            dummy_x = self.dropout1(dummy_x)
            dummy_x = self.separable_conv_depth(dummy_x)
            dummy_x = self.separable_conv_point(dummy_x)
            dummy_x = self.bn3(dummy_x)
            dummy_x = self.elu2(dummy_x)
            dummy_x = self.avg_pool2(dummy_x)
            # dummy_x shape now: (1, F2, 1, Time_Steps)
            self.final_time_steps = dummy_x.shape[3]
            # print(f"[DEBUG] CausalEEGNet: Input Samples={Samples}, Detected Time_Steps={self.final_time_steps}")
        
        self.flatten_dim = F2 * self.final_time_steps 
        
        # --- B. 时序 Transformer 与特征整合 ---
        self.temporal_transformer = TemporalTransformer(
            input_dim=F2, 
            time_steps=self.final_time_steps, # 使用检测到的精确时间步
            num_heads=2, 
            dropout=dropoutRate
        )
        
        # --- C. 双流编码器 (Dual Encoders) ---
        # 引入均值 / 方差头以支持重参数化 (VIB)
        self.semantic_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.sem_mu = nn.Linear(128, latent_dim)
        self.sem_logvar = nn.Linear(128, latent_dim)

        self.variation_head = nn.Sequential(
            nn.LazyLinear(128),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.Dropout(0.5)
        )
        self.var_mu = nn.Linear(128, latent_dim)
        self.var_logvar = nn.Linear(128, latent_dim)

        # --- D. 任务头 ---
        self.classifier = nn.Linear(latent_dim, nb_classes)

        # --- E. 域判别器 (对抗训练 Zs) ---
        self.domain_classifier = None
        if num_domains > 0:
            self.domain_grl = GradientReverseLayer(alpha=1.0)
            self.domain_classifier = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ELU(),
                nn.Linear(64, num_domains) # 预测 Subject ID
            )

        # --- F. 重构器 (Decoder) ---
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 128),
            nn.ELU(),
            nn.LazyLinear(self.flatten_dim)
        )

        # ============================================================
        # 🔥 G. 新增：Zu 的分支对抗模块 (Branch Adversarial)
        # ============================================================
        # 目标：让 Zu 无法预测类别 (Class Label)，从而剥离语义信息
        self.zu_adv_grl = GradientReverseLayer(alpha=1.0)
        self.zu_adv_classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ELU(),
            nn.Linear(32, nb_classes) # 预测 Class Label (如左手/右手)
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        # 1. Conv Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)

        x = self.separable_conv_depth(x)
        x = self.separable_conv_point(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        x = x.squeeze(2)
        
        # 2. Transformer & Flatten
        features_flat = self.temporal_transformer(x)

        # 3. Encoders with reparameterization
        sem_feat = self.semantic_head(features_flat)
        z_s_mu = self.sem_mu(sem_feat)
        z_s_logvar = self.sem_logvar(sem_feat)
        z_s = self.reparameterize(z_s_mu, z_s_logvar)

        var_feat = self.variation_head(features_flat)
        z_u_mu = self.var_mu(var_feat)
        z_u_logvar = self.var_logvar(var_feat)
        z_u = self.reparameterize(z_u_mu, z_u_logvar)

        # 4. Main Outputs (Zs -> Class)
        preds = self.classifier(z_s)
        
        # 5. Domain Adversarial (Zs -> Subject ID)
        dom_logits = None
        if self.domain_classifier is not None:
            z_s_rev = self.domain_grl(z_s)
            dom_logits = self.domain_classifier(z_s_rev)
            
        # 6. Reconstruction (Zs + Zu -> Raw Features)
        # 主路：Zu 直接参与重构，梯度为正，保留信息
        combined_z = torch.cat([z_s, z_u], dim=1)
        rec_features = self.decoder(combined_z)

        # ============================================================
        # 🔥 7. Zu Branch Adversarial (Zu -> Class)
        # ============================================================
        # 支路：Zu 经过 GRL 预测类别，梯度反转，去除类别信息
        z_u_rev = self.zu_adv_grl(z_u)
        preds_zu_adv = self.zu_adv_classifier(z_u_rev)

        return (preds, rec_features, features_flat,
                z_s, z_u, dom_logits, preds_zu_adv,
                z_s_mu, z_s_logvar, z_u_mu, z_u_logvar)