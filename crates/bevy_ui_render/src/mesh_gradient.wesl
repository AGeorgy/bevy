import bevy_render::view::{View, position_world_to_clip};
import package::ui::{
    compute_uinode_background_coverage,
    compute_uinode_border_coverage,
};
import package::gradient_color::convert_to_linear_rgba;
import bevy_render::color_operations::oklab_to_linear_rgb;
import bevy_render::maths::PI_2;

const MAX_POINTS: u32 = 256u;
const MAX_CLIPS: u32 = 128u;

struct MeshGradientPoints {
    positions: array<vec4<f32>, 256>,
    colors: array<vec4<f32>, 256>,
};

struct MeshGradientStyle {
    transform_x: vec4<f32>,
    transform_y: vec4<f32>,
    radius_x: vec4<f32>,
    radius_y: vec4<f32>,
    border: vec4<f32>,
    size: vec4<f32>,
    metadata: vec4<u32>,
};

@if(MESH_CLIPPED)
struct MeshGradientClip {
    metadata: vec4<u32>,
    clip_rects: array<vec4<f32>, 128>,
    clip_transform_x: array<vec4<f32>, 128>,
    clip_transform_y: array<vec4<f32>, 128>,
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> points: MeshGradientPoints;
@group(1) @binding(1) var<uniform> mesh: MeshGradientStyle;
@if(MESH_CLIPPED)
@group(1) @binding(2) var<uniform> clips: MeshGradientClip;

struct MeshVertexOutput {
    @if(MESH_PARAMETER_UV)
    @location(0) uv: vec2<f32>,
    @if(VERTEX_COLOR)
    @location(1) interpolation_color: vec4<f32>,
    @if(BICUBIC_COLOR || HUE_COLOR)
    @location(1) @interpolate(flat) patch_index: vec2<u32>,
    @location(2) local_point: vec2<f32>,
    @if(MESH_CLIPPED)
    @location(3) world_point: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

fn compute_point_index(x: u32, y: u32) -> u32 {
    return min(y * mesh.metadata.x + x, MAX_POINTS - 1u);
}

fn unpack_bytes(value: u32) -> vec4<u32> {
    return vec4(
        value & 0xffu,
        (value >> 8u) & 0xffu,
        (value >> 16u) & 0xffu,
        value >> 24u,
    );
}

fn sample_position_in_row(x: i32, y: u32) -> vec2<f32> {
    let width = i32(mesh.metadata.x);
    if x < 0 {
        return 2.0 * points.positions[compute_point_index(0u, y)].xy
            - points.positions[compute_point_index(1u, y)].xy;
    }
    if x >= width {
        return 2.0 * points.positions[compute_point_index(u32(width - 1), y)].xy
            - points.positions[compute_point_index(u32(width - 2), y)].xy;
    }
    return points.positions[compute_point_index(u32(x), y)].xy;
}

fn sample_control_position(x: i32, y: i32) -> vec2<f32> {
    let height = i32(mesh.metadata.y);
    if y < 0 {
        return 2.0 * sample_position_in_row(x, 0u) - sample_position_in_row(x, 1u);
    }
    if y >= height {
        return 2.0 * sample_position_in_row(x, u32(height - 1))
            - sample_position_in_row(x, u32(height - 2));
    }
    return sample_position_in_row(x, u32(y));
}

// Control hues are lifted onto a consistent, unwrapped field before upload.
// Interpolate that field normally and wrap once at the final RGB conversion.
@if(HUE_COLOR)
fn wrap_mesh_hue(color: vec4<f32>) -> vec4<f32> {
@if(IN_OKLCH || IN_OKLCH_LONG)
    return vec4(color.xy, fract(color.z), color.a);
@else
    return vec4(fract(color.x), color.yzw);
}

@if(IN_OKHSL || IN_OKHSL_LONG)
fn evaluate_fallback_lightness_chroma(
    patch_index: vec2<u32>,
    uv: vec2<f32>
) -> vec2<f32> {
    let x = patch_index.x;
    let y = patch_index.y;
@if(BILINEAR_COLOR) {
    let top = mix(
        points.positions[compute_point_index(x, y)].zw,
        points.positions[compute_point_index(x + 1u, y)].zw,
        uv.x
    );
    let bottom = mix(
        points.positions[compute_point_index(x, y + 1u)].zw,
        points.positions[compute_point_index(x + 1u, y + 1u)].zw,
        uv.x
    );
    return mix(top, bottom, uv.y);
} @else {
    let u_weights = compute_catmull_rom_weights(uv.x);
    let v_weights = compute_catmull_rom_weights(uv.y);
    let ix = i32(x);
    let iy = i32(y);
    let row0 = evaluate_cubic_vec2(
        sample_control_lightness_chroma(ix - 1, iy - 1),
        sample_control_lightness_chroma(ix, iy - 1),
        sample_control_lightness_chroma(ix + 1, iy - 1),
        sample_control_lightness_chroma(ix + 2, iy - 1),
        u_weights
    );
    let row1 = evaluate_cubic_vec2(
        sample_control_lightness_chroma(ix - 1, iy),
        sample_control_lightness_chroma(ix, iy),
        sample_control_lightness_chroma(ix + 1, iy),
        sample_control_lightness_chroma(ix + 2, iy),
        u_weights
    );
    let row2 = evaluate_cubic_vec2(
        sample_control_lightness_chroma(ix - 1, iy + 1),
        sample_control_lightness_chroma(ix, iy + 1),
        sample_control_lightness_chroma(ix + 1, iy + 1),
        sample_control_lightness_chroma(ix + 2, iy + 1),
        u_weights
    );
    let row3 = evaluate_cubic_vec2(
        sample_control_lightness_chroma(ix - 1, iy + 2),
        sample_control_lightness_chroma(ix, iy + 2),
        sample_control_lightness_chroma(ix + 1, iy + 2),
        sample_control_lightness_chroma(ix + 2, iy + 2),
        u_weights
    );
    return evaluate_cubic_vec2(row0, row1, row2, row3, v_weights);
}
}

@if(IN_OKHSL || IN_OKHSL_LONG)
fn blend_okhsl_blue_cusp(
    patch_index: vec2<u32>,
    uv: vec2<f32>,
    color: vec4<f32>,
    rgb: vec3<f32>
) -> vec3<f32> {
    // OKHSL's gamut intersection switches roots at saturated blue. The local
    // Oklab surface keeps the colored points exact and bridges that narrow cusp.
    let distance = abs(fract(color.x - 264.052 / 360.0 + 0.5) - 0.5);
    if distance >= 0.01 {
        return rgb;
    }
    let lc = evaluate_fallback_lightness_chroma(patch_index, uv);
    let angle = color.x * PI_2;
    let chroma = max(lc.y, 0.0);
    let smooth_rgb = oklab_to_linear_rgb(vec3(lc.x, chroma * cos(angle), chroma * sin(angle)));
    if distance <= 0.002 {
        return smooth_rgb;
    }
    let weight = 1.0 - smoothstep(0.002, 0.01, distance);
    return mix(rgb, smooth_rgb, weight);
}

@if(BICUBIC_COLOR && (IN_OKHSL || IN_OKHSL_LONG))
fn sample_fallback_lightness_chroma_row(x: i32, y: u32) -> vec2<f32> {
    let width = i32(mesh.metadata.x);
    if x < 0 {
        return 2.0 * points.positions[compute_point_index(0u, y)].zw
            - points.positions[compute_point_index(1u, y)].zw;
    }
    if x >= width {
        return 2.0 * points.positions[compute_point_index(u32(width - 1), y)].zw
            - points.positions[compute_point_index(u32(width - 2), y)].zw;
    }
    return points.positions[compute_point_index(u32(x), y)].zw;
}

@if(BICUBIC_COLOR && (IN_OKHSL || IN_OKHSL_LONG))
fn sample_control_lightness_chroma(x: i32, y: i32) -> vec2<f32> {
    let height = i32(mesh.metadata.y);
    if y < 0 {
        return 2.0 * sample_fallback_lightness_chroma_row(x, 0u)
            - sample_fallback_lightness_chroma_row(x, 1u);
    }
    if y >= height {
        return 2.0 * sample_fallback_lightness_chroma_row(x, u32(height - 1))
            - sample_fallback_lightness_chroma_row(x, u32(height - 2));
    }
    return sample_fallback_lightness_chroma_row(x, u32(y));
}

@if(BICUBIC_COLOR)
fn sample_color_in_row(x: i32, y: u32) -> vec4<f32> {
    let width = i32(mesh.metadata.x);
    if x < 0 {
        return 2.0 * points.colors[compute_point_index(0u, y)]
            - points.colors[compute_point_index(1u, y)];
    }
    if x >= width {
        return 2.0 * points.colors[compute_point_index(u32(width - 1), y)]
            - points.colors[compute_point_index(u32(width - 2), y)];
    }
    return points.colors[compute_point_index(u32(x), y)];
}

@if(BICUBIC_COLOR)
fn sample_control_color(x: i32, y: i32) -> vec4<f32> {
    let height = i32(mesh.metadata.y);
    if y < 0 {
        return 2.0 * sample_color_in_row(x, 0u) - sample_color_in_row(x, 1u);
    }
    if y >= height {
        return 2.0 * sample_color_in_row(x, u32(height - 1)) - sample_color_in_row(x, u32(height - 2));
    }
    return sample_color_in_row(x, u32(y));
}

fn compute_catmull_rom_weights(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4(0.0, 1.0, 0.0, 0.0);
    }
    if t == 1.0 {
        return vec4(0.0, 0.0, 1.0, 0.0);
    }
    let t2 = t * t;
    let t3 = t2 * t;
    return 0.5 * vec4(
        -t + 2.0 * t2 - t3,
        2.0 - 5.0 * t2 + 3.0 * t3,
        t + 4.0 * t2 - 3.0 * t3,
        -t2 + t3,
    );
}

fn evaluate_cubic_vec2(
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    weights: vec4<f32>,
) -> vec2<f32> {
    return p0 * weights.x + p1 * weights.y + p2 * weights.z + p3 * weights.w;
}

@if(BICUBIC_COLOR)
fn evaluate_cubic_vec4(
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    weights: vec4<f32>,
) -> vec4<f32> {
    return p0 * weights.x + p1 * weights.y + p2 * weights.z + p3 * weights.w;
}

fn evaluate_surface_position(patch_index: vec2<u32>, uv: vec2<f32>) -> vec2<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let u_weights = compute_catmull_rom_weights(uv.x);
    let v_weights = compute_catmull_rom_weights(uv.y);
    let row0 = evaluate_cubic_vec2(
        sample_control_position(x - 1, y - 1),
        sample_control_position(x, y - 1),
        sample_control_position(x + 1, y - 1),
        sample_control_position(x + 2, y - 1),
        u_weights,
    );
    let row1 = evaluate_cubic_vec2(
        sample_control_position(x - 1, y),
        sample_control_position(x, y),
        sample_control_position(x + 1, y),
        sample_control_position(x + 2, y),
        u_weights,
    );
    let row2 = evaluate_cubic_vec2(
        sample_control_position(x - 1, y + 1),
        sample_control_position(x, y + 1),
        sample_control_position(x + 1, y + 1),
        sample_control_position(x + 2, y + 1),
        u_weights,
    );
    let row3 = evaluate_cubic_vec2(
        sample_control_position(x - 1, y + 2),
        sample_control_position(x, y + 2),
        sample_control_position(x + 1, y + 2),
        sample_control_position(x + 2, y + 2),
        u_weights,
    );
    return evaluate_cubic_vec2(row0, row1, row2, row3, v_weights);
}

// A finer edge is evaluated on the coarser neighbor's piecewise-linear chord.
// Its unchanged UV still drives fragment color, so both sides interpolate the
// same parameters and colors along the shared geometric edge.
fn evaluate_snapped_surface_position(
    patch_index: vec2<u32>,
    uv: vec2<f32>,
    subdivisions: vec2<u32>,
    position_subdivisions: vec2<u32>,
) -> vec2<f32> {
    if all(position_subdivisions == subdivisions) {
        return evaluate_surface_position(patch_index, uv);
    }
    var start = uv;
    var end = uv;
    var weight = 0.0;
    if position_subdivisions.x < subdivisions.x {
        let grid = uv.x * f32(position_subdivisions.x);
        let segment = floor(grid);
        start.x = segment / f32(position_subdivisions.x);
        end.x = (segment + 1.0) / f32(position_subdivisions.x);
        weight = fract(grid);
    } else {
        let grid = uv.y * f32(position_subdivisions.y);
        let segment = floor(grid);
        start.y = segment / f32(position_subdivisions.y);
        end.y = (segment + 1.0) / f32(position_subdivisions.y);
        weight = fract(grid);
    }
    return mix(
        evaluate_surface_position(patch_index, start),
        evaluate_surface_position(patch_index, end),
        weight,
    );
}

@if(VERTEX_COLOR || BILINEAR_COLOR)
fn evaluate_bilinear_surface_color(patch_index: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let x = patch_index.x;
    let y = patch_index.y;
    let top = mix(
        points.colors[compute_point_index(x, y)],
        points.colors[compute_point_index(x + 1u, y)],
        uv.x,
    );
    let bottom = mix(
        points.colors[compute_point_index(x, y + 1u)],
        points.colors[compute_point_index(x + 1u, y + 1u)],
        uv.x,
    );
    return mix(top, bottom, uv.y);
}

@if(BICUBIC_COLOR)
fn evaluate_bicubic_surface_color(patch_index: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let u_weights = compute_catmull_rom_weights(uv.x);
    let v_weights = compute_catmull_rom_weights(uv.y);
    let row0 = evaluate_cubic_vec4(
        sample_control_color(x - 1, y - 1),
        sample_control_color(x, y - 1),
        sample_control_color(x + 1, y - 1),
        sample_control_color(x + 2, y - 1),
        u_weights,
    );
    let row1 = evaluate_cubic_vec4(
        sample_control_color(x - 1, y),
        sample_control_color(x, y),
        sample_control_color(x + 1, y),
        sample_control_color(x + 2, y),
        u_weights,
    );
    let row2 = evaluate_cubic_vec4(
        sample_control_color(x - 1, y + 1),
        sample_control_color(x, y + 1),
        sample_control_color(x + 1, y + 1),
        sample_control_color(x + 2, y + 1),
        u_weights,
    );
    let row3 = evaluate_cubic_vec4(
        sample_control_color(x - 1, y + 2),
        sample_control_color(x, y + 2),
        sample_control_color(x + 1, y + 2),
        sample_control_color(x + 2, y + 2),
        u_weights,
    );
    return evaluate_cubic_vec4(row0, row1, row2, row3, v_weights);
}

@vertex
fn vertex(
    @location(0) packed: vec2<u32>,
) -> MeshVertexOutput {
    let parameter = unpack_bytes(packed.x);
    let factors = unpack_bytes(packed.y);
    let patch_index = parameter.xy;
    let subdivisions = factors.xy;
    let position_subdivisions = factors.zw;
    let uv = vec2<f32>(parameter.zw) / vec2<f32>(subdivisions);
    let normalized = evaluate_snapped_surface_position(
        patch_index,
        uv,
        subdivisions,
        position_subdivisions,
    );
    let local_point = (normalized - vec2(0.5)) * mesh.size.xy;
    let homogeneous = vec3(local_point, 1.0);
    let world_point = vec2(
        dot(mesh.transform_x.xyz, homogeneous),
        dot(mesh.transform_y.xyz, homogeneous),
    );
    var out: MeshVertexOutput;
    out.position = position_world_to_clip(vec3(world_point, 0.0), view.clip_from_world);
@if(MESH_PARAMETER_UV)
    out.uv = uv;
@if(VERTEX_COLOR)
    out.interpolation_color = evaluate_bilinear_surface_color(patch_index, uv);
@if(BICUBIC_COLOR || HUE_COLOR)
    out.patch_index = patch_index;
    out.local_point = local_point;
@if(MESH_CLIPPED)
    out.world_point = world_point;
    return out;
}

@if(MESH_CLIPPED)
fn is_outside_clip(world_point: vec2<f32>) -> bool {
    for (var index = 0u; index < min(clips.metadata.x, MAX_CLIPS); index += 1u) {
        let homogeneous = vec3(world_point, 1.0);
        let local = vec2(
            dot(clips.clip_transform_x[index].xyz, homogeneous),
            dot(clips.clip_transform_y[index].xyz, homogeneous),
        );
        let rect = clips.clip_rects[index];
        if local.x < rect.x || local.y < rect.y || local.x > rect.z || local.y > rect.w {
            return true;
        }
    }
    return false;
}

@fragment
fn fragment(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    let flags = mesh.metadata.w;
@if(MESH_BORDER)
    let coverage = compute_uinode_border_coverage(
        in.local_point,
        mesh.size.xy,
        mesh.radius_x,
        mesh.radius_y,
        mesh.border,
        flags,
    );
@else
    let coverage = compute_uinode_background_coverage(
        in.local_point,
        mesh.size.xy,
        mesh.radius_x,
        mesh.radius_y,
        mesh.border,
        flags,
    );
    if coverage <= 0.0 {
        discard;
    }
@if(MESH_CLIPPED)
    if is_outside_clip(in.world_point) {
        discard;
    }
@if(VERTEX_COLOR)
    let field_color = in.interpolation_color;
@if(BILINEAR_COLOR)
    let field_color = evaluate_bilinear_surface_color(in.patch_index, in.uv);
@if(BICUBIC_COLOR)
    let field_color = evaluate_bicubic_surface_color(in.patch_index, in.uv);
@if(HUE_COLOR)
    let interpolation_color = wrap_mesh_hue(field_color);
@else
    let interpolation_color = field_color;
@if(VERTEX_COLOR && IN_LINEAR_RGB)
    let color = interpolation_color;
@else
    let color = convert_to_linear_rgba(interpolation_color);
@if(VERTEX_COLOR && IN_LINEAR_RGB)
    let alpha = color.a;
@else
    let alpha = clamp(color.a, 0.0, 1.0);
@if(IN_OKHSL || IN_OKHSL_LONG)
    let rgb = blend_okhsl_blue_cusp(in.patch_index, in.uv, interpolation_color, color.rgb);
@else
    let rgb = color.rgb;
    return vec4(rgb, saturate(alpha * coverage));
}
