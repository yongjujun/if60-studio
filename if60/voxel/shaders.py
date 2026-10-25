"""복셀 렌더러 GLSL 셰이더 (OpenGL 3.3 core — macOS/Windows/Linux 공통).

재질(material) 번호
  0 단색            1 잔노이즈(돌·콘크리트·흙)   2 건물 외벽(창문 격자, 밤엔 불 켜짐)
  3 물(움직이는 픽셀 물결)   4 아스팔트           5 발광(조명·간판·창밖 하늘)
  6 강한 노이즈(잔디)        7 유리(차·지하철 창, 밤엔 불 켜짐)

텍스처 이미지를 쓰지 않고 월드 좌표를 12.5cm 격자로 잘라 해시 노이즈를 입힌다.
그래서 큰 상자 하나로도 마인크래프트식 '픽셀 질감'이 나오고, 꼭짓점 수가 적어 빠르다.
"""

COMMON_FS = """
#version 330
in vec3 v_world;
in vec3 v_norm;
in vec3 v_col;
flat in float v_mat;
in vec4 v_light;

uniform vec3 u_cam;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform vec3 u_amb_sky;
uniform vec3 u_amb_ground;
uniform vec3 u_fog_col;
uniform float u_fog_density;
uniform float u_time;
uniform float u_night;
uniform float u_texel;          // 픽셀 질감 밀도 (1m 당 칸 수)
uniform sampler2DShadow u_shadow;
uniform float u_shadow_on;

out vec4 f_color;

float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float shadow_at(vec4 lp, vec3 n) {
    if (u_shadow_on < 0.5) return 1.0;
    vec3 p = lp.xyz / lp.w * 0.5 + 0.5;
    if (p.x < 0.0 || p.x > 1.0 || p.y < 0.0 || p.y > 1.0 || p.z > 1.0) return 1.0;
    float bias = 0.0015 + 0.003 * (1.0 - abs(dot(n, u_sun_dir)));
    vec2 ts = 1.0 / vec2(textureSize(u_shadow, 0));
    float s = 0.0;
    for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
            s += texture(u_shadow, vec3(p.xy + vec2(x, y) * ts, p.z - bias));
    return s / 9.0;
}

void main() {
    vec3 n = normalize(v_norm);
    vec3 cell = floor((v_world - n * 0.01) * u_texel);
    vec3 albedo = v_col;
    float emissive = 0.0;
    int m = int(v_mat + 0.5);

    if (m == 1) {
        albedo *= 0.88 + 0.24 * hash(cell);
    } else if (m == 6) {
        float h = hash(cell);
        albedo *= 0.72 + 0.5 * h;
    } else if (m == 4) {
        float h = hash(cell);
        albedo *= 0.86 + 0.22 * h + (h > 0.97 ? 0.3 : 0.0);
    } else if (m == 3) {
        vec3 wc = floor(vec3(v_world.x * 3.0 + u_time * 0.6, 0.0, v_world.z * 3.0 + u_time * 0.35));
        float h = hash(wc);
        albedo *= 0.86 + 0.18 * hash(wc + 7.0);
        if (h > 0.86) albedo = mix(albedo, vec3(0.85, 0.93, 1.0), 0.45);
    } else if (m == 2) {
        albedo *= 0.9 + 0.16 * hash(cell);
        if (abs(n.y) < 0.5) {
            float u = abs(n.x) > 0.5 ? v_world.z : v_world.x;
            vec2 g = vec2(u / 2.2, v_world.y / 3.2);
            vec2 fr = fract(g);
            if (fr.x > 0.18 && fr.x < 0.82 && fr.y > 0.28 && fr.y < 0.82 && v_world.y > 2.5) {
                float r = hash(vec3(floor(g), floor(dot(v_world, n)) * 0.37));
                vec3 glass = mix(vec3(0.22, 0.28, 0.38), vec3(0.45, 0.55, 0.66), r * 0.6);
                if (u_night > 0.05 && r > 0.52) {
                    glass = mix(glass, vec3(1.0, 0.86, 0.55), u_night);
                    emissive = u_night;
                }
                albedo = glass;
            }
        }
    } else if (m == 7) {
        float r = hash(floor(v_world * 0.5));
        albedo = vec3(0.2, 0.26, 0.34) + 0.1 * r;
        if (u_night > 0.05) { albedo = mix(albedo, vec3(1.0, 0.92, 0.7), u_night * 0.8); emissive = u_night * 0.8; }
    } else if (m == 5) {
        emissive = 1.0;
    }

    float ndl = max(dot(n, u_sun_dir), 0.0);
    float sh = ndl > 0.0 ? shadow_at(v_light, n) : 1.0;
    vec3 amb = mix(u_amb_ground, u_amb_sky, n.y * 0.5 + 0.5);
    vec3 col = albedo * (amb + u_sun_col * ndl * sh);
    col = mix(col, albedo, emissive);

    float d = length(v_world - u_cam);
    float fog = 1.0 - exp(-pow(d * u_fog_density, 1.4));
    col = mix(col, u_fog_col, clamp(fog, 0.0, 1.0));
    f_color = vec4(col, 1.0);
}
"""

WORLD_VS = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_light_vp;
in vec3 in_pos;
in vec3 in_norm;
in vec3 in_col;
in float in_mat;
out vec3 v_world;
out vec3 v_norm;
out vec3 v_col;
flat out float v_mat;
out vec4 v_light;
void main() {
    v_world = in_pos;
    v_norm = in_norm;
    v_col = in_col;
    v_mat = in_mat;
    v_light = u_light_vp * vec4(in_pos, 1.0);
    gl_Position = u_vp * vec4(in_pos, 1.0);
}
"""

# 군중: 같은 모델을 수천 번 그린다(인스턴싱). 이동·순환(wrap)·들썩임을 GPU에서 계산.
INSTANCE_VS = """
#version 330
uniform mat4 u_vp;
uniform mat4 u_light_vp;
uniform float u_time;
uniform vec3 u_wrap_min;
uniform vec3 u_wrap_size;
uniform float u_bob;
uniform float u_bob_freq;
uniform float u_spin;
uniform float u_gscale;         // 자라나기(grow) 배율 — 0이면 안 보임
in vec3 in_pos;
in vec3 in_norm;
in vec3 in_col;
in float in_slot;
in float in_mat;
in vec3 i_off;
in float i_yaw;
in vec3 i_c1;
in vec3 i_c2;
in float i_phase;
in vec3 i_vel;
in float i_scale;
out vec3 v_world;
out vec3 v_norm;
out vec3 v_col;
flat out float v_mat;
out vec4 v_light;
void main() {
    float yaw = i_yaw + u_spin * u_time * (i_phase > 3.14159 ? 1.0 : -1.0);
    float c = cos(yaw), s = sin(yaw);
    vec3 p = in_pos * i_scale * u_gscale;
    p = vec3(c * p.x + s * p.z, p.y, -s * p.x + c * p.z);
    vec3 n = vec3(c * in_norm.x + s * in_norm.z, in_norm.y, -s * in_norm.x + c * in_norm.z);
    vec3 base = i_off + i_vel * u_time;
    if (u_wrap_size.x > 0.0) base.x = u_wrap_min.x + mod(base.x - u_wrap_min.x, u_wrap_size.x);
    if (u_wrap_size.z > 0.0) base.z = u_wrap_min.z + mod(base.z - u_wrap_min.z, u_wrap_size.z);
    if (u_wrap_size.y > 0.0) base.y = u_wrap_min.y + mod(base.y - u_wrap_min.y, u_wrap_size.y);
    base.y += u_bob * abs(sin(u_time * u_bob_freq + i_phase));
    vec3 w = base + p;
    v_world = w;
    v_norm = n;
    v_col = in_slot < 0.5 ? in_col : (in_slot < 1.5 ? i_c1 : i_c2);
    v_mat = in_mat;
    v_light = u_light_vp * vec4(w, 1.0);
    gl_Position = u_vp * vec4(w, 1.0);
}
"""

DEPTH_FS = """
#version 330
void main() {}
"""

# 하늘: 화면 전체 삼각형. 시간대별 그라디언트 + 픽셀 해 + 밤하늘 별 + (선택) 하늘의 행성
SKY_VS = """
#version 330
in vec2 in_pos;
out vec2 v_ndc;
void main() { v_ndc = in_pos; gl_Position = vec4(in_pos, 0.9999, 1.0); }
"""

SKY_FS = """
#version 330
in vec2 v_ndc;
uniform mat4 u_inv_vp;
uniform vec3 u_cam;
uniform vec3 u_zenith;
uniform vec3 u_horizon;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_col;
uniform float u_sun_size;
uniform float u_night;
uniform vec3 u_planet_dir;
uniform vec3 u_planet_col;
uniform float u_planet_size;
uniform float u_skyq;           // 하늘 픽셀 격자 (복셀 260, 매끈한 룩은 크게)
out vec4 f_color;
float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
void main() {
    vec4 w = u_inv_vp * vec4(v_ndc, 1.0, 1.0);
    vec3 d = normalize(w.xyz / w.w - u_cam);
    // 픽셀 느낌을 위해 방향을 격자로 양자화
    vec3 q = floor(d * u_skyq) / u_skyq;
    float h = max(q.y, 0.0);
    vec3 col = mix(u_horizon, u_zenith, pow(h, 0.55));
    if (q.y < 0.0) col = u_horizon * 0.92;
    float sd = dot(normalize(q), u_sun_dir);
    if (u_sun_size > 0.0 && sd > 1.0 - u_sun_size) col = u_sun_col;
    else if (u_sun_size > 0.0 && sd > 1.0 - u_sun_size * 6.0) col = mix(col, u_sun_col, 0.25);
    if (u_planet_size > 0.0) {
        float pd = dot(normalize(q), u_planet_dir);
        if (pd > 1.0 - u_planet_size) {
            float shade = hash(floor(q * 900.0));
            col = u_planet_col * (0.85 + 0.25 * shade);
        }
    }
    if (u_night > 0.05 && q.y > 0.02) {
        float st = hash(floor(d * 420.0));
        if (st > 0.9965) col = mix(col, vec3(1.0), u_night * (0.5 + 0.5 * fract(st * 91.0)));
    }
    f_color = vec4(col, 1.0);
}
"""

# VFX: 반투명 충격파 돔·구, 커지는 불덩이·버섯구름. 매끈한 단위 구 메시를 크기·위치만 바꿔 그린다.
VFX_VS = """
#version 330
uniform mat4 u_vp;
uniform vec3 u_center;
uniform vec3 u_scale;
in vec3 in_pos;
out vec3 v_world;
out vec3 v_n;
void main() {
    vec3 w = u_center + in_pos * u_scale;
    v_world = w;
    v_n = normalize(in_pos / max(u_scale, vec3(1e-4)));
    gl_Position = u_vp * vec4(w, 1.0);
}
"""

VFX_FS = """
#version 330
in vec3 v_world;
in vec3 v_n;
uniform vec3 u_cam;
uniform vec3 u_col;      // 바깥 색
uniform vec3 u_col2;     // 안쪽(뜨거운 중심) 색
uniform float u_alpha;
uniform float u_mode;    // 0 = 껍질(돔·구, 가산 합성)  1 = 불덩이(불투명 발광)
uniform float u_time;
uniform float u_floor;   // 이 높이 아래는 그리지 않음 (반구 돔)
uniform float u_cell;    // 픽셀 노이즈 밀도
out vec4 f_color;
float hash(vec3 p) {
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
void main() {
    if (v_world.y < u_floor) discard;
    vec3 v = normalize(u_cam - v_world);
    vec3 n = normalize(v_n);
    float rim = pow(1.0 - abs(dot(n, v)), 2.0);
    float h = hash(floor(v_world * u_cell + vec3(0.0, -u_time * 3.0, 0.0)));
    if (u_mode < 0.5) {
        vec3 c = mix(u_col2, u_col, rim) * (0.8 + 0.35 * h);
        f_color = vec4(c, u_alpha * (0.10 + 0.9 * rim));
    } else {
        float k = smoothstep(0.15, 0.95, rim + 0.35 * (h - 0.5));
        vec3 c = mix(u_col2, u_col, k) * (0.88 + 0.28 * h);
        f_color = vec4(c, u_alpha);
    }
}
"""
