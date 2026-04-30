import taichi as ti
import taichi.math as tm
import math

ti.init(arch=ti.gpu)

res_x, res_y = 800, 600
pixels = ti.Vector.field(3, dtype=ti.f32, shape=(res_x, res_y))

light_pos_x  = ti.field(ti.f32, shape=())
light_pos_y  = ti.field(ti.f32, shape=())
light_pos_z  = ti.field(ti.f32, shape=())
max_bounces  = ti.field(ti.i32, shape=())
msaa_samples = ti.field(ti.i32, shape=())

MAT_DIFFUSE = 0
MAT_MIRROR  = 1
MAT_GLASS   = 2

IOR_GLASS = 1.5

@ti.func
def normalize(v):
    return v / v.norm(1e-5)

@ti.func
def reflect_vec(I, N):
    return I - 2.0 * I.dot(N) * N

@ti.func
def refract_vec(I, N, eta):
    cos_i  = -I.dot(N)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    can    = 1
    T      = ti.Vector([0.0, 0.0, 0.0])
    if sin2_t >= 1.0:
        can = 0
    else:
        cos_t = ti.sqrt(ti.max(0.0, 1.0 - sin2_t))
        T     = eta * I + (eta * cos_i - cos_t) * N
    return can, T

@ti.func
def schlick(cos_theta, ior):
    r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
    return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5

@ti.func
def intersect_sphere(ro, rd, center, radius):
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
        if t1 > 1e-4:
            t = t1
        elif t2 > 1e-4:
            t = t2
        if t > 0.0:
            normal = normalize(ro + rd * t - center)
    return t, normal

@ti.func
def intersect_plane(ro, rd, plane_y):
    t      = -1.0
    normal = ti.Vector([0.0, 1.0, 0.0])
    if ti.abs(rd.y) > 1e-5:
        t1 = (plane_y - ro.y) / rd.y
        if t1 > 1e-4:
            t = t1
    return t, normal

@ti.func
def scene_intersect(ro, rd):
    min_t   = 1e10
    hit_n   = ti.Vector([0.0, 0.0, 0.0])
    hit_c   = ti.Vector([0.0, 0.0, 0.0])
    hit_mat = MAT_DIFFUSE

    t, n = intersect_sphere(ro, rd, ti.Vector([-1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([1.0, 1.0, 1.0])
        hit_mat = MAT_GLASS

    t, n = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([0.9, 0.9, 0.9])
        hit_mat = MAT_MIRROR

    t, n = intersect_sphere(ro, rd, ti.Vector([2.0, 4.0, 3.0]), 0.3)
    if 0.0 < t < min_t:
        min_t   = t
        hit_n   = n
        hit_c   = ti.Vector([1.0, 1.0, 0.8])
        hit_mat = MAT_DIFFUSE

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

@ti.func
def shadow_intersect(ro, rd):
    min_t = 1e10

    t, _ = intersect_sphere(ro, rd, ti.Vector([1.2, 0.0, 0.0]), 1.0)
    if 0.0 < t < min_t:
        min_t = t

    t, _ = intersect_sphere(ro, rd, ti.Vector([2.0, 4.0, 3.0]), 0.3)
    if 0.0 < t < min_t:
        min_t = t

    t, _ = intersect_plane(ro, rd, -1.0)
    if 0.0 < t < min_t:
        min_t = t

    result = -1.0
    if min_t < 1e9:
        result = min_t
    return result

@ti.func
def soft_shadow_factor(hit_p, N, light_center, light_radius, num_samples):
    to_light = light_center - hit_p
    dist = to_light.norm()
    L = to_light / dist

    shadow_factor = 1.0
    valid_L = L.norm() > 1e-4

    if valid_L and dist > 1e-4:
        unblocked = 0.0
        count = 0

        for i in range(12):
            r = ti.sqrt(ti.random())
            theta = 2.0 * math.pi * ti.random()
            offset_x = r * ti.cos(theta) * light_radius
            offset_z = r * ti.sin(theta) * light_radius

            perp = ti.Vector([-L.z, 0.0, L.x])
            perp_len = perp.norm()
            if perp_len > 1e-4:
                perp = perp / perp_len

            parallel = ti.Vector([L.x, 0.0, L.z])
            parallel_len = parallel.norm()
            if parallel_len > 1e-4:
                parallel = parallel / parallel_len

            sample_pos = light_center + perp * offset_x + parallel * offset_z

            to_sample = sample_pos - hit_p
            sample_dist = to_sample.norm()
            sample_dir = to_sample / sample_dist

            shadow_orig = hit_p + N * 1e-4
            shadow_t = shadow_intersect(shadow_orig, sample_dir)

            if shadow_t < 0.0 or shadow_t > sample_dist:
                unblocked += 1.0

            count += 1

        shadow_factor = unblocked / float(count) if count > 0 else 1.0

    return shadow_factor

@ti.func
def trace_ray(ro_init, rd_init, light_pos, nb):
    bg_color   = ti.Vector([0.05, 0.15, 0.2])
    final_c    = ti.Vector([0.0, 0.0, 0.0])
    throughput = ti.Vector([1.0, 1.0, 1.0])

    ro = ro_init
    rd = rd_init

    for _ in range(nb):
        t, N_geo, obj_color, mat_id = scene_intersect(ro, rd)

        if t > 1e9:
            final_c += throughput * bg_color
            break

        p = ro + rd * t

        if mat_id == MAT_MIRROR:
            ro         = p + N_geo * 1e-4
            rd         = normalize(reflect_vec(rd, N_geo))
            throughput *= 0.85 * obj_color

        elif mat_id == MAT_GLASS:
            from_outside = rd.dot(N_geo) < 0.0

            N_for_refract = N_geo if from_outside else -N_geo
            eta           = (1.0 / IOR_GLASS) if from_outside else (IOR_GLASS / 1.0)

            cos_i = ti.abs(rd.dot(N_geo))
            fres  = schlick(cos_i, IOR_GLASS)

            can_refract, T = refract_vec(rd, N_for_refract, eta)

            if can_refract == 0:
                ro         = p + N_for_refract * 1e-4
                rd         = normalize(reflect_vec(rd, N_for_refract))
                throughput *= obj_color
            else:
                ro         = p - N_for_refract * 1e-4
                rd         = normalize(T)
                throughput *= (1.0 - fres) * obj_color

        else:
            L = normalize(light_pos - p)

            shadow_orig = p + N_geo * 1e-4
            shadow_t = shadow_intersect(shadow_orig, L)
            dist_to_light = (light_pos - p).norm()

            ambient      = 0.15 * obj_color
            direct_light = ambient

            in_penumbra = (shadow_t > 0.0 and shadow_t <= dist_to_light)

            if in_penumbra:
                light_radius = 0.5
                sf = soft_shadow_factor(shadow_orig, N_geo, light_pos, light_radius, 8)
                diff = ti.max(0.0, N_geo.dot(L))
                direct_light += 0.8 * diff * obj_color * sf
            elif shadow_t < 0.0 or shadow_t > dist_to_light:
                diff = ti.max(0.0, N_geo.dot(L))
                direct_light += 0.8 * diff * obj_color

            final_c += throughput * direct_light
            break

    return final_c

@ti.kernel
def render():
    light_pos = ti.Vector([light_pos_x[None], light_pos_y[None], light_pos_z[None]])
    nb        = max_bounces[None]
    ns        = msaa_samples[None]

    for i, j in pixels:
        accum = ti.Vector([0.0, 0.0, 0.0])

        for s in range(ns):
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

def main():
    window = ti.ui.Window("Ray Tracing: Refraction + Soft Shadow", (res_x, res_y))
    canvas = window.get_canvas()
    gui    = window.get_gui()

    light_pos_x[None]  = 2.0
    light_pos_y[None]  = 4.0
    light_pos_z[None]  = 3.0
    max_bounces[None]  = 6
    msaa_samples[None] = 1

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