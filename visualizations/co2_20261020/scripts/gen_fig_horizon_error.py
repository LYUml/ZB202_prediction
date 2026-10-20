"""5至60分钟误差变化；数据源：冻结确认期predictions.npz；尺寸按PAPER_PLAN。"""
from _figbase import draw, setup_style, PALETTE, save_fig
setup_style()
draw('horizon_error')
