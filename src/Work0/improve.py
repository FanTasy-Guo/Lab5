import taichi as ti
import taichi.math as tm

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

# 交互参数
light_pos_x  = ti.field(ti.f32, shape=())
light_pos_y  = ti.field(ti.f32, shape=())
light_pos_z  = ti.field(ti.f32, shape=())
max_bounces  = ti.field(ti.i32, shape=())
msaa_samples = ti.field(ti.i32, shape=())

# 材质 ID
MAT_DIFFUSE = 0
MAT_MIRROR  = 1
MAT_GLASS   = 2

IOR_GLASS = 1.5   # 玻璃折射率

# ──────────────────────────────────────────────────────────
# 基础工具函数
# ──────────────────────────────────────────────────────────

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect_vec(I, N):
    """I 为入射方向（朝向表面），N 为法线，返回反射方向"""
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract_vec(I, N, eta):
    """
    斯涅尔定律：计算折射方向。
      I   : 归一化入射方向（射向表面）
      N   : 归一化法线，必须与 I 反向（即 dot(I,N) < 0）
      eta : n_入射 / n_折射
    返回 (can_refract: int, T: 折射方向向量)
    全内反射时 can_refract=0
    """
    cos_i  = -I.dot(N)                      # 入射角余弦，>0
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    can    = 1
    T      = ti.Vector([0.0, 0.0, 0.0])
    if sin2_t >= 1.0:                        # 全内反射
        can = 0
    else:
        cos_t = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
        T     = eta * I + (eta * cos_i - cos_t) * N  # 标准斯涅尔折射公式
    return can, T

@ti.func
def schlick(cos_theta, ior):
    """Schlick 近似菲涅耳反射率"""
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5

# ──────────────────────────────────────────────────────────
# 场景几何求交
# ──────────────────────────────────────────────────────────

@ti.func
def intersect_sphere(ro, rd, center, radius):
    """
    球体求交。
    始终返回【几何外法线】（从球心指向交点方向），
    让调用方自行判断是从内部还是外部击中。
    返回: (t, outward_normal)，t<0 表示未击中。
    """
    t      = -1.0
    normal = ti.Vector([0.0, 0.0, 0.0])
    oc     = ro - center
    b      = 2.0 * oc.dot(rd)
    c      = oc.dot(oc) - radius * radius
    disc   = b * b - 4.0 * c
    if disc > 0.0:
        sqrt_d = ti.sqrt(disc)
        t1     = (-b - sqrt_d) * 0.5
        t2     = (-b + sqrt_d) * 0.5
        # 取最近的正根（用 1e-4 过滤掉数值噪声）
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
        if t > 0.0:
            # 外法线：永远从球心指向交点
            normal = normalize(ro + rd * t - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    """水平无限平面求交，法线朝上"""
    t      = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-4:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    """
    遍历场景，寻找最近交点。
    返回 (t, 几何外法线 N, 物体颜色 color, 材质 mat_id)
    """
    min_t   = 1e10
    hit_n   = ti.Vector([0.0, 0.0, 0.0])
    hit_c   = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    # 玻璃球（左侧）
    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([1.0, 1.0, 1.0])   # 无色玻璃，透射不染色
        hit_mat = MAT_GLASS

    # 银色镜面球（右侧）
    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    # 棋盘格地板
    t, n = intersect_plane(ro, rd, -1.0)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_mat = MAT_DIFFUSE
        p       = ro + rd * t
        ix      = ti.floor(p.x * 2.0)
        iz      = ti.floor(p.z * 2.0)
        if (ix + iz) % 2 == 0:
            hit_c = ti.Vector([0.3, 0.3, 0.3])
        else:
            hit_c = ti.Vector([0.8, 0.8, 0.8])

    return min_t, hit_n, hit_c, hit_mat

# ──────────────────────────────────────────────────────────
# 核心：迭代式光线追踪
# ──────────────────────────────────────────────────────────

@ti.func
def trace_ray(ro_init, rd_init, light_pos, nb):
    bg_color   = ti.Vector([0.05, 0.15, 0.2])
    final_c    = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    ro = ro_init
    rd = rd_init

    for _ in range(nb):
        t, N_geo, obj_color, mat_id = scene_intersect(ro, rd)

        # 未命中任何物体 → 返回背景色
        if t > 1e9:
            final_c += throughput * bg_color
            break

        p = ro + rd * t   # 交点世界坐标

        # ══════════════════════════════════════
        #  镜面反射材质
        # ══════════════════════════════════════
        if mat_id == MAT_MIRROR:
            # N_geo 已是外法线，反射时直接用
            ro         = p + N_geo * 1e-4
            rd         = normalize(reflect_vec(rd, N_geo))
            throughput *= 0.85 * obj_color
            # 不 break，继续追踪反射射线

        # ══════════════════════════════════════
        #  玻璃折射材质（斯涅尔定律 + 菲涅耳）
        # ══════════════════════════════════════
        elif mat_id == MAT_GLASS:
            # ── 步骤1：判断射线从哪侧入射 ──
            #   rd·N_geo < 0 → 从外部射入（空气→玻璃）
            #   rd·N_geo > 0 → 从内部射出（玻璃→空气）
            from_outside = rd.dot(N_geo) < 0.0

            # 根据入射方向设置法线和折射率比值
            N_for_refract = N_geo if from_outside else -N_geo
            eta           = (1.0 / IOR_GLASS) if from_outside else (IOR_GLASS / 1.0)

            # ── 步骤2：计算菲涅耳反射率（Schlick） ──
            cos_i = ti.abs(rd.dot(N_geo))   # 入射角余弦
            fres  = schlick(cos_i, IOR_GLASS)

            # ── 步骤3：尝试折射 ──
            can_refract, T = refract_vec(rd, N_for_refract, eta)

            if can_refract == 0:
                # 全内反射（只发生在从内部射出时，掠射角大时）
                ro         = p + N_for_refract * 1e-4
                rd         = normalize(reflect_vec(rd, N_for_refract))
                throughput *= obj_color
            else:
                # 折射光路：偏移方向与 N_for_refract 相反（进入另一侧）
                ro         = p - N_for_refract * 1e-4
                rd         = normalize(T)
                # 菲涅耳权重：折射携带 (1-fres) 的能量
                throughput *= (1.0 - fres) * obj_color
            # 不 break，继续追踪穿透射线

        # ══════════════════════════════════════
        #  漫反射材质（Phong + 硬阴影）
        # ══════════════════════════════════════
        else:  # MAT_DIFFUSE
            L = normalize(light_pos - p)

            # 硬阴影射线，从交点沿法线偏移防止自交
            shadow_orig = p + N_geo * 1e-4
            shadow_t, _, _, _ = scene_intersect(shadow_orig, L)
            dist_to_light     = (light_pos - p).norm()

            ambient      = 0.15 * obj_color
            direct_light = ambient

            # shadow_t < 0 → 未击中任何东西（无阴影）
            # shadow_t > dist_to_light → 击中物体但比光源远（无阴影）
            if shadow_t < 0.0 or shadow_t > dist_to_light:
                diff         = ti.max(0.0, N_geo.dot(L))
                direct_light += 0.8 * diff * obj_color

            final_c += throughput * direct_light
            break   # 漫反射终止

    return final_c

# ──────────────────────────────────────────────────────────
# 渲染 Kernel（含 MSAA 多重采样抗锯齿）
# ──────────────────────────────────────────────────────────

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    nb        = max_bounces[None]
    ns        = msaa_samples[None]

    for i, j in pixels:
        accum = ti.Vector([0.0, 0.0, 0.0])

        for s in range(ns):
            # 均匀网格抖动：4×4 子像素采样（ns 最大 16 时覆盖完整 4×4 格）
            col = s % 4
            row = s // 4
            dx  = (col + 0.5) / ti.max(1, ti.min(ns, 4)) - 0.5
            dy  = (row + 0.5) / ti.max(1, ti.min(ns // 4 + 1, 4)) - 0.5

            u  = (i + 0.5 + dx - res_x * 0.5) / res_y * 2.0
            v  = (j + 0.5 + dy - res_y * 0.5) / res_y * 2.0

            ro = ti.Vector([0.0, 1.0, 5.0])
            rd = normalize(ti.Vector([u, v - 0.2, -1.0]))

            accum += trace_ray(ro, rd, light_pos, nb)

        pixels[i, j] = tm.clamp(accum / float(ns), 0.0, 1.0)

# ──────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────

def main():
    window = ti.ui.Window("Ray Tracing: Refraction + MSAA", (res_x, res_y))
    canvas = window.get_canvas()
    gui    = window.get_gui()

    light_pos_x[None]  = 2.0
    light_pos_y[None]  = 4.0
    light_pos_z[None]  = 3.0
    max_bounces[None]  = 6    # 玻璃球需要至少 2 次弹射（入射+出射）
    msaa_samples[None] = 1    # 默认无 MSAA，可拉到 4/8/16

    while window.running:
        render()
        canvas.set_image(pixels)

        with gui.sub_window("Controls", 0.72, 0.04, 0.27, 0.30):
            light_pos_x[None]  = gui.slider_float("Light X",      light_pos_x[None],  -5.0, 5.0)
            light_pos_y[None]  = gui.slider_float("Light Y",      light_pos_y[None],   1.0, 8.0)
            light_pos_z[None]  = gui.slider_float("Light Z",      light_pos_z[None],  -5.0, 5.0)
            max_bounces[None]  = gui.slider_int(  "Max Bounces",  max_bounces[None],   2,   8  )
            msaa_samples[None] = gui.slider_int(  "MSAA Samples", msaa_samples[None],  1,  16  )

        window.show()

if __name__ == "__main__":
    main()
