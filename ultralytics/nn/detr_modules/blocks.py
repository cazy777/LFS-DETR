import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
# ------------ Basic ops ------------
def conv_1x1(c1, c2, act=True):
    m = [nn.Conv2d(c1, c2, 1, 1, 0, bias=False), nn.BatchNorm2d(c2)]
    if act:
        m.append(nn.SiLU(inplace=True))
    return nn.Sequential(*m)

def dwconv_3x3(c, s=1, d=1, act=True):
    m = [nn.Conv2d(c, c, 3, s, d, dilation=d, groups=c, bias=False), nn.BatchNorm2d(c)]
    if act:
        m.append(nn.SiLU(inplace=True))
    return nn.Sequential(*m)

class LowRankConv1x1(nn.Module):
    def __init__(self, c: int, r: float = 0.25, act: bool = True):
        super().__init__()
        hidden = max(1, int(c * r))
        self.conv1 = nn.Conv2d(c, hidden, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(hidden, c, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.act(x)
import torch
import torch.nn as nn


def _to_int_channels(ch):
    """
    将 Ultralytics parse_model 传入的通道数转换为 int。
    兼容 int、[int]、(int,)、[[int]] 等单输入情况。

    注意：
    SAFM 是单输入模块，如果 ch 是 [256, 512] 这种多输入通道列表，
    说明 YAML 的 from 可能写成了多个来源，需要检查 YAML。
    """
    while isinstance(ch, (list, tuple)) and len(ch) == 1:
        ch = ch[0]

    if isinstance(ch, (list, tuple)):
        raise TypeError(
            f"SAFM expects a single input feature, but got multiple channels: {ch}. "
            f"Please check the YAML 'from' field. For single input, use -1 instead of [-1, -2]."
        )

    return int(ch)


class PointwiseConv1x1(nn.Module):
    """
    普通 1×1 Conv，用于 w/o LRConv 消融。
    替代 LowRankConv1x1。
    """
    def __init__(self, channels, act=True):
        super().__init__()

        channels = _to_int_channels(channels)

        layers = [
            nn.Conv2d(channels, channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(channels)
        ]

        if act:
            layers.append(nn.SiLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SAFM_Base(nn.Module):
    """
    SAFM 基础类。
    通过开关控制不同消融版本。

    use_ampgate:
        是否使用 AMPgate，即幅值统计 + Z-score 标准化 + 残差式频谱增益。

    use_msdr:
        是否使用 MSDR，即多尺度深度卷积细节重构。

    use_simam:
        是否使用 SimAM 无参数注意力。

    use_lrconv:
        是否使用 LowRankConv1x1。若为 False，则使用普通 1×1 Conv 替代。

    use_freq_w:
        是否使用频域通道权重 freq_w。
    """

    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4,
                 use_ampgate: bool = True,
                 use_msdr: bool = True,
                 use_simam: bool = True,
                 use_lrconv: bool = True,
                 use_freq_w: bool = True):
        super().__init__()

        # ======================================================
        # 关键修复：先把 in_channels 转成 int
        # ======================================================
        in_channels = _to_int_channels(in_channels)

        C = in_channels
        c1 = max(1, int(C * split_ratio))
        c2 = C - c1

        # 防止极端情况下 c2=0，导致 torch.split 出问题
        if c2 <= 0:
            c1 = C - 1
            c2 = 1

        self.c1 = c1
        self.c2 = c2
        self.e_lambda = e_lambda

        self.use_ampgate = use_ampgate
        self.use_msdr = use_msdr
        self.use_simam = use_simam
        self.use_lrconv = use_lrconv
        self.use_freq_w = use_freq_w

        # ======================================================
        # 1. LRConv 轻量通道重构
        # ======================================================
        if use_lrconv:
            self.proj = LowRankConv1x1(C, r=r_proj, act=True)
            self.fuse = LowRankConv1x1(C, r=r_proj, act=True)
        else:
            # w/o LRConv：用普通 1×1 Conv 替代
            self.proj = PointwiseConv1x1(C, act=True)
            self.fuse = PointwiseConv1x1(C, act=True)

        # ======================================================
        # 2. 频域通道权重 freq_w
        # ======================================================
        if use_freq_w:
            hidden_fw = max(1, c1 // r_freq)
            self.freq_w = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, hidden_fw, 1, 1, 0, bias=False),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_fw, c1, 1, 1, 0, bias=True),
                nn.Sigmoid()
            )
        else:
            self.freq_w = None

        # ======================================================
        # 3. MSDR 多尺度细节重构
        # ======================================================
        if use_msdr:
            d1, d2 = dilations
            self.ms_dw = nn.Sequential(
                nn.Conv2d(c1, c1, 3, 1, d1, dilation=d1, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),

                nn.Conv2d(c1, c1, 3, 1, d2, dilation=d2, groups=c1, bias=False),
                nn.BatchNorm2d(c1),
                nn.SiLU(inplace=True),
            )
        else:
            # w/o MSDR：频域调制后直接进入融合
            self.ms_dw = nn.Identity()

    def _simam(self, x):
        """
        SimAM 无参数注意力。
        """
        B, C, H, W = x.shape
        n = H * W

        mean = x.mean(dim=(2, 3), keepdim=True)
        var = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / (n + 1e-6)
        e = ((x - mean) ** 2) / (4 * (var + self.e_lambda)) + 0.5

        return torch.sigmoid(e)

    @staticmethod
    def channel_shuffle(x, groups: int = 2):
        """
        通道 shuffle。
        如果通道数不能被 groups 整除，则直接返回原特征。
        """
        B, C, H, W = x.size()

        if C % groups != 0:
            return x

        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, H, W)

        return x

    def forward(self, x):
        identity = x

        # ======================================================
        # 1. LRConv + 通道划分
        # ======================================================
        y = self.proj(x)
        xs, xm = torch.split(y, [self.c1, self.c2], dim=1)

        # ======================================================
        # 2. 频域增强路径
        # ======================================================
        Xf = torch.fft.fft2(xs, dim=(-2, -1))

        # ------------------------------
        # AMPgate：幅值统计 + Z-score + 残差式频谱增益
        # w/o AMPgate 时跳过该部分
        # ------------------------------
        if self.use_ampgate:
            amp = Xf.abs()
            amp_mean = amp.mean(dim=1, keepdim=True)

            amp_norm = amp_mean - amp_mean.mean(dim=(-2, -1), keepdim=True)
            amp_norm = amp_norm / (amp_norm.std(dim=(-2, -1), keepdim=True) + 1e-6)

            amp_att = torch.sigmoid(amp_norm)
            Xf = Xf * (1.0 + amp_att)

        # ------------------------------
        # 频域通道权重 freq_w
        # ------------------------------
        if self.freq_w is not None:
            Wc = self.freq_w(xs)
            Wf = torch.fft.fft2(Wc, s=xs.shape[-2:], dim=(-2, -1))
            Xf = Xf * Wf

        # IFFT 回到空间域
        xs = torch.fft.ifft2(Xf, dim=(-2, -1)).real

        # ======================================================
        # 3. MSDR 多尺度细节重构
        # w/o MSDR 时这里是 Identity
        # ======================================================
        xs = self.ms_dw(xs)

        # ======================================================
        # 4. 拼接空间旁路
        # ======================================================
        y = torch.cat([xs, xm], dim=1)

        # ======================================================
        # 5. SimAM 响应重标定
        # w/o SimAM 时跳过
        # ======================================================
        if self.use_simam:
            att = self._simam(y)
            y = y * att

        # ======================================================
        # 6. LRConv 融合 + Channel Shuffle + 残差
        # ======================================================
        y = self.fuse(y)
        y = self.channel_shuffle(y, groups=2)

        if identity.shape == y.shape:
            y = y + identity

        return y


# ==========================================================
# 完整 SAFM
# ==========================================================
class SAFM(SAFM_Base):
    """
    完整 SAFM：
    LRConv + AMPgate + freq_w + MSDR + SimAM + LRConv
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=True,
            use_msdr=True,
            use_simam=True,
            use_lrconv=True,
            use_freq_w=True
        )


# ==========================================================
# 1. w/o AMPgate
# ==========================================================
class SAFM_wo_AMPgate(SAFM_Base):
    """
    去掉 AMPgate：
    删除幅值统计、Z-score 标准化和残差式频谱增益。
    保留 freq_w、MSDR、SimAM 和 LRConv。
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=False,
            use_msdr=True,
            use_simam=True,
            use_lrconv=True,
            use_freq_w=True
        )


# ==========================================================
# 2. w/o MSDR
# ==========================================================
class SAFM_wo_MSDR(SAFM_Base):
    """
    去掉 MSDR：
    频域调制后 IFFT 直接进入拼接融合，不再经过多尺度深度卷积细化。
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=True,
            use_msdr=False,
            use_simam=True,
            use_lrconv=True,
            use_freq_w=True
        )


# ==========================================================
# 3. w/o SimAM
# ==========================================================
class SAFM_wo_SimAM(SAFM_Base):
    """
    去掉 SimAM：
    频域增强特征和空间旁路特征拼接后，直接进入 LRConv 融合。
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=True,
            use_msdr=True,
            use_simam=False,
            use_lrconv=True,
            use_freq_w=True
        )


# ==========================================================
# 4. w/o LRConv
# ==========================================================
class SAFM_wo_LRConv(SAFM_Base):
    """
    去掉 LRConv：
    将前后两个 LowRankConv1x1 替换为普通 1×1 Conv。
    这个实验可选，不是最低限度必须做。
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=True,
            use_msdr=True,
            use_simam=True,
            use_lrconv=False,
            use_freq_w=True
        )


# ==========================================================
# 5. w/o freq_w
# ==========================================================
class SAFM_wo_FreqW(SAFM_Base):
    """
    去掉频域通道权重 freq_w：
    只保留 AMPgate 的幅值统计调制。
    如果论文中没有单独描述 freq_w，可以不做这个消融。
    """
    def __init__(self,
                 in_channels: int,
                 split_ratio: float = 0.75,
                 dilations=(1, 2),
                 e_lambda=1e-4,
                 r_proj: float = 0.25,
                 r_freq: int = 4):
        super().__init__(
            in_channels=in_channels,
            split_ratio=split_ratio,
            dilations=dilations,
            e_lambda=e_lambda,
            r_proj=r_proj,
            r_freq=r_freq,
            use_ampgate=True,
            use_msdr=True,
            use_simam=True,
            use_lrconv=True,
            use_freq_w=False
        )



class BlurPool2d(nn.Module):
    """
    Anti-aliased downsampling with fixed binomial kernel (no learnable params).
    stride=2 to replace MaxPool2d(3,2,1).
    """
    def __init__(self, stride=2, kernel_size=3):
        super().__init__()
        assert kernel_size in (3, 5)
        if kernel_size == 3:
            k = torch.tensor([1., 2., 1.])
        else:
            k = torch.tensor([1., 4., 6., 4., 1.])
        kernel = (k[:, None] * k[None, :])
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel[None, None, :, :])  # (1,1,kh,kw)
        self.stride = stride
        self.pad = kernel_size // 2

    def forward(self, x):
        c = x.shape[1]
        w = self.kernel.expand(c, 1, *self.kernel.shape[-2:])   # depthwise shared
        return F.conv2d(x, w, stride=self.stride, padding=self.pad, groups=c)
def _shift2d_zero(x: torch.Tensor, dx: int, dy: int):
    if dx == 0 and dy == 0:
        return x
    y = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
    if dy > 0:
        y[:, :, :dy, :] = 0
    elif dy < 0:
        y[:, :, dy:, :] = 0
    if dx > 0:
        y[:, :, :, :dx] = 0
    elif dx < 0:
        y[:, :, :, dx:] = 0
    return y

class CSAF(nn.Module):
    """
    Correlation-Shift Alignment Fusion (robust)
    输入: [xh, xl]
    输出: cat([xh_align, xl], dim=1)
    自动将 xh resize 到 xl 的空间尺寸
    """
    def __init__(self, c: int, r: int = 4, k: int = 3, tau: float = 1.0, act: bool = True,
                 resize_mode: str = "nearest"):
        super().__init__()
        assert k % 2 == 1, "k must be odd, e.g., 3 or 5"
        cr = max(8, c // r)

        self.q = nn.Sequential(
            nn.Conv2d(c, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )
        self.k = nn.Sequential(
            nn.Conv2d(c, cr, 1, 1, 0, bias=False),
            nn.BatchNorm2d(cr),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )

        self.k_size = k
        self.tau = tau
        self.resize_mode = resize_mode

        t = k // 2
        self.offsets = [(dx, dy) for dy in range(-t, t + 1) for dx in range(-t, t + 1)]

    def forward(self, x):
        xh, xl = x

        # 关键：自动对齐空间尺度（默认把 xh 对齐到 xl）
        if xh.shape[-2:] != xl.shape[-2:]:
            xh = F.interpolate(xh, size=xl.shape[-2:], mode=self.resize_mode)


        assert xh.shape[1] == xl.shape[1], f"CSAF expects same channels, got {xh.shape[1]} vs {xl.shape[1]}"

        ql = self.q(xl)
        kh = self.k(xh)

        ql = F.normalize(ql, dim=1)
        kh = F.normalize(kh, dim=1)

        corr = []
        for dx, dy in self.offsets:
            kh_s = _shift2d_zero(kh, dx, dy)
            corr.append((ql * kh_s).sum(dim=1, keepdim=True))
        corr = torch.cat(corr, dim=1)

        w = F.softmax(corr / max(self.tau, 1e-6), dim=1)

        xh_align = 0.0
        for i, (dx, dy) in enumerate(self.offsets):
            xh_s = _shift2d_zero(xh, dx, dy)
            xh_align = xh_align + xh_s * w[:, i:i+1, :, :]

        return torch.cat([xh_align, xl], dim=1)