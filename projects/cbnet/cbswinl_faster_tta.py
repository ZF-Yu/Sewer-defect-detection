_base_ = [
    '../swin/swinl_faster.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'

model = dict(
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained
    ),
    neck=dict(
        type='CBFPN',
        in_channels=[192, 384, 768, 1536])
)
