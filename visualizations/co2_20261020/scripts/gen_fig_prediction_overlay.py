"""30分钟预测与实测，固定选择中间完整日及首个点位；数据源：冻结确认期predictions.npz；尺寸按PAPER_PLAN。"""
from _figbase import draw, setup_style, PALETTE, save_fig
setup_style()
draw('prediction_overlay')
