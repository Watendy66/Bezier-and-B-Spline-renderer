import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)
RES = 800
NUM_SEGMENTS = 1000
MAX_CONTROL_POINTS = 100
control_points = []

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))
curve_points_field = ti.Vector.field(2, dtype=ti.f32, shape=NUM_SEGMENTS + 1)
gui_points = ti.Vector.field(2, dtype=ti.f32, shape=MAX_CONTROL_POINTS)

# --- 算法部分 ---

def de_casteljau(points, t):
    if len(points) == 1: return points[0]
    next_points = [(1 - t) * points[i] + t * points[i+1] for i in range(len(points) - 1)]
    return de_casteljau(next_points, t)

def get_b_spline_point(points, t):
    """均匀三次 B 样条的矩阵实现"""
    n = len(points)
    if n < 4: return points[0] # B样条至少需要4个点
    
    # 确定当前 t 落在第几个段 (共有 n-3 段)
    num_intervals = n - 3
    t_scaled = t * num_intervals
    i = min(int(t_scaled), num_intervals - 1)
    local_t = t_scaled - i
    
    # 选取相邻的4个控制点
    p0, p1, p2, p3 = points[i], points[i+1], points[i+2], points[i+3]
    
    # 三次 B 样条基函数 (Matrix Form)
    t2 = local_t * local_t
    t3 = t2 * local_t
    
    f0 = (-t3 + 3*t2 - 3*local_t + 1) / 6.0
    f1 = (3*t3 - 6*t2 + 4) / 6.0
    f2 = (-3*t3 + 3*t2 + 3*local_t + 1) / 6.0
    f3 = t3 / 6.0
    
    return f0 * p0 + f1 * p1 + f2 * p2 + f3 * p3

window = ti.ui.Window(name="Bezier vs B-Spline (B:BSpline, V:Bezier, C:Clear)", res=(800, 800))
canvas = window.get_canvas()
control_points = []
mode = "Bezier" # 初始模式

# ... 之前的定义保持不变 ...

@ti.kernel
def render_all(curr_num_points: ti.i32, mode_val: ti.i32):
    # 1. 每一帧开始时，彻底清空画布（解决清除不干净的问题）
    for i, j in pixels:
        pixels[i, j] = ti.Vector([0.0, 0.0, 0.0])

    # 2. 绘制辅助折线 (只有在有控制点时才绘制)
    if curr_num_points >= 2:
        for i in range(curr_num_points - 1):
            # 获取相邻两个控制点
            p_start = gui_points[i] * float(RES)
            p_end = gui_points[i+1] * float(RES)
            
            # 这里的辅助线也可以简单反走样，颜色设为灰色 [0.3, 0.3, 0.3]
            for k in range(200): # 采样辅助线
                t = k / 200.0
                pos = p_start * (1.0 - t) + p_end * t
                px, py = int(pos[0]), int(pos[1])
                if 0 <= px < RES and 0 <= py < RES:
                    pixels[px, py] = ti.Vector([0.3, 0.3, 0.3])

    # 3. 绘制贝塞尔或 B 样条曲线 (带反走样)
    # 只有当点数足够时才渲染曲线
    can_render_curve = (mode_val == 0 and curr_num_points >= 2) or (mode_val == 1 and curr_num_points >= 4)
    
    if can_render_curve:
        for k in range(NUM_SEGMENTS + 1):
            pos = curve_points_field[k] * float(RES)
            base_i, base_j = int(pos[0]), int(pos[1])
            for offset_i in range(-1, 2):
                for offset_j in range(-1, 2):
                    ni, nj = base_i + offset_i, base_j + offset_j
                    if 0 <= ni < RES and 0 <= nj < RES:
                        dist = (ti.Vector([ni + 0.5, nj + 0.5]) - pos).norm()
                        weight = ti.max(0.0, 1.0 - dist / 1.5)
                        # 曲线颜色：粉红色
                        pixels[ni, nj] += ti.Vector([1.0, 0.4, 0.7]) * weight * 0.5

# --- 主循环逻辑修改 ---
mode_val = 0 # 0 for Bezier, 1 for BSpline

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.LMB:
            if len(control_points) < MAX_CONTROL_POINTS:
                pos = window.get_cursor_pos()
                control_points.append(np.array([pos[0], pos[1]], dtype=np.float32))
        elif window.event.key == 'c':
            control_points.clear()
            # 清空 field 确保 GUI 也不显示点
            gui_points.fill(-10.0) 
        elif window.event.key == 'b':
            mode_val = 1
        elif window.event.key == 'v':
            mode_val = 0

    # 更新控制点 Field (供渲染 Kernel 使用)
    if len(control_points) > 0:
        np_pts = np.full((MAX_CONTROL_POINTS, 2), -10.0, dtype=np.float32)
        np_pts[:len(control_points)] = np.array(control_points)
        gui_points.from_numpy(np_pts)

    # 计算曲线几何点
    if mode_val == 0 and len(control_points) >= 2:
        curve_list = [de_casteljau(control_points, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS + 1)]
        curve_points_field.from_numpy(np.array(curve_list, dtype=np.float32))
    elif mode_val == 1 and len(control_points) >= 4:
        curve_list = [get_b_spline_point(control_points, i/NUM_SEGMENTS) for i in range(NUM_SEGMENTS + 1)]
        curve_points_field.from_numpy(np.array(curve_list, dtype=np.float32))

    # 调用统一的渲染 Kernel
    render_all(len(control_points), mode_val)

    canvas.set_image(pixels)
    # 绘制控制点
    if len(control_points) > 0:
        canvas.circles(gui_points, radius=0.005, color=(0.2, 0.8, 1.0))
    window.show()