import bevy_render::view::View;
import package::ui::{
    uinode_background_coverage,
    uinode_border_coverage,
};
import bevy_render::color_operations::{
    oklab_to_linear_rgb,
    srgb_to_linear,
};

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
    tessellation: vec4<f32>,
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
    @if(BICUBIC_COLOR)
    @location(1) @interpolate(flat) patch_index: vec2<u32>,
    @location(2) local_point: vec2<f32>,
    @location(3) world_point: vec2<f32>,
    @if(MESH_WIREFRAME)
    @location(4) @interpolate(flat) subdivisions: vec2<u32>,
    @builtin(position) position: vec4<f32>,
};

fn point_index(x: u32, y: u32) -> u32 {
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

fn position_in_row(x: i32, y: u32) -> vec2<f32> {
    let width = i32(mesh.metadata.x);
    if x < 0 {
        return 2.0 * points.positions[point_index(0u, y)].xy
            - points.positions[point_index(1u, y)].xy;
    }
    if x >= width {
        return 2.0 * points.positions[point_index(u32(width - 1), y)].xy
            - points.positions[point_index(u32(width - 2), y)].xy;
    }
    return points.positions[point_index(u32(x), y)].xy;
}

fn control_position(x: i32, y: i32) -> vec2<f32> {
    let height = i32(mesh.metadata.y);
    if y < 0 {
        return 2.0 * position_in_row(x, 0u) - position_in_row(x, 1u);
    }
    if y >= height {
        return 2.0 * position_in_row(x, u32(height - 1))
            - position_in_row(x, u32(height - 2));
    }
    return position_in_row(x, u32(y));
}

@if(BICUBIC_COLOR)
fn color_in_row(x: i32, y: u32) -> vec4<f32> {
    let width = i32(mesh.metadata.x);
    if x < 0 {
        return 2.0 * points.colors[point_index(0u, y)]
            - points.colors[point_index(1u, y)];
    }
    if x >= width {
        return 2.0 * points.colors[point_index(u32(width - 1), y)]
            - points.colors[point_index(u32(width - 2), y)];
    }
    return points.colors[point_index(u32(x), y)];
}

@if(BICUBIC_COLOR)
fn control_color(x: i32, y: i32) -> vec4<f32> {
    let height = i32(mesh.metadata.y);
    if y < 0 {
        return 2.0 * color_in_row(x, 0u) - color_in_row(x, 1u);
    }
    if y >= height {
        return 2.0 * color_in_row(x, u32(height - 1))
            - color_in_row(x, u32(height - 2));
    }
    return color_in_row(x, u32(y));
}

fn cubic_weights(t: f32) -> vec4<f32> {
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

fn cubic2(
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    weights: vec4<f32>,
) -> vec2<f32> {
    return p0 * weights.x + p1 * weights.y + p2 * weights.z + p3 * weights.w;
}

@if(BICUBIC_COLOR)
fn cubic4(
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    weights: vec4<f32>,
) -> vec4<f32> {
    return p0 * weights.x + p1 * weights.y + p2 * weights.z + p3 * weights.w;
}

fn surface_position(patch_index: vec2<u32>, uv: vec2<f32>) -> vec2<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let u_weights = cubic_weights(uv.x);
    let v_weights = cubic_weights(uv.y);
    let row0 = cubic2(
        control_position(x - 1, y - 1),
        control_position(x, y - 1),
        control_position(x + 1, y - 1),
        control_position(x + 2, y - 1),
        u_weights,
    );
    let row1 = cubic2(
        control_position(x - 1, y),
        control_position(x, y),
        control_position(x + 1, y),
        control_position(x + 2, y),
        u_weights,
    );
    let row2 = cubic2(
        control_position(x - 1, y + 1),
        control_position(x, y + 1),
        control_position(x + 1, y + 1),
        control_position(x + 2, y + 1),
        u_weights,
    );
    let row3 = cubic2(
        control_position(x - 1, y + 2),
        control_position(x, y + 2),
        control_position(x + 1, y + 2),
        control_position(x + 2, y + 2),
        u_weights,
    );
    return cubic2(row0, row1, row2, row3, v_weights);
}

// A finer edge is evaluated on the coarser neighbor's piecewise-linear chord.
// Its unchanged UV still drives fragment color, so both sides interpolate the
// same parameters and colors along the shared geometric edge.
fn snapped_surface_position(
    patch_index: vec2<u32>,
    uv: vec2<f32>,
    subdivisions: vec2<u32>,
    position_subdivisions: vec2<u32>,
) -> vec2<f32> {
    if all(position_subdivisions == subdivisions) {
        return surface_position(patch_index, uv);
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
        surface_position(patch_index, start),
        surface_position(patch_index, end),
        weight,
    );
}

@if(VERTEX_COLOR)
fn bilinear_surface_color(patch_index: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let x = patch_index.x;
    let y = patch_index.y;
    let top = mix(
        points.colors[point_index(x, y)],
        points.colors[point_index(x + 1u, y)],
        uv.x,
    );
    let bottom = mix(
        points.colors[point_index(x, y + 1u)],
        points.colors[point_index(x + 1u, y + 1u)],
        uv.x,
    );
    return mix(top, bottom, uv.y);
}

@if(BICUBIC_COLOR)
fn bicubic_surface_color(patch_index: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let u_weights = cubic_weights(uv.x);
    let v_weights = cubic_weights(uv.y);
    let row0 = cubic4(
        control_color(x - 1, y - 1),
        control_color(x, y - 1),
        control_color(x + 1, y - 1),
        control_color(x + 2, y - 1),
        u_weights,
    );
    let row1 = cubic4(
        control_color(x - 1, y),
        control_color(x, y),
        control_color(x + 1, y),
        control_color(x + 2, y),
        u_weights,
    );
    let row2 = cubic4(
        control_color(x - 1, y + 1),
        control_color(x, y + 1),
        control_color(x + 1, y + 1),
        control_color(x + 2, y + 1),
        u_weights,
    );
    let row3 = cubic4(
        control_color(x - 1, y + 2),
        control_color(x, y + 2),
        control_color(x + 1, y + 2),
        control_color(x + 2, y + 2),
        u_weights,
    );
    return cubic4(row0, row1, row2, row3, v_weights);
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
    let normalized = snapped_surface_position(
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
    out.position = view.clip_from_world * vec4(world_point, 0.0, 1.0);
@if(MESH_PARAMETER_UV)
    out.uv = uv;
@if(VERTEX_COLOR)
    out.interpolation_color = bilinear_surface_color(patch_index, uv);
@if(BICUBIC_COLOR)
    out.patch_index = patch_index;
    out.local_point = local_point;
    out.world_point = world_point;
@if(MESH_WIREFRAME)
    out.subdivisions = subdivisions;
    return out;
}

@if(MESH_CLIPPED)
fn outside_clip(world_point: vec2<f32>) -> bool {
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

fn convert_to_linear_rgba(color: vec4<f32>) -> vec4<f32> {
@if(IN_OKLAB)
    let rgb = oklab_to_linear_rgb(color.xyz);
@elif(IN_SRGB)
    let rgb = srgb_to_linear(color.xyz);
@else
    let rgb = color.xyz;
    return vec4(rgb, clamp(color.a, 0.0, 1.0));
}

@if(MESH_WIREFRAME)
fn tessellation_wireframe(uv: vec2<f32>, subdivisions: vec2<u32>) -> f32 {
    let grid = uv * vec2<f32>(subdivisions);
    let cell = fract(grid);
    let x_distance = min(cell.x, 1.0 - cell.x);
    let y_distance = min(cell.y, 1.0 - cell.y);
    let diagonal_distance = abs(cell.x - cell.y);
    let width = mesh.tessellation.y;
    let x_line = 1.0 - smoothstep(0.0, max(fwidth(grid.x) * width, 1e-6), x_distance);
    let y_line = 1.0 - smoothstep(0.0, max(fwidth(grid.y) * width, 1e-6), y_distance);
    let diagonal_line = 1.0 - smoothstep(
        0.0,
        max(fwidth(grid.x - grid.y) * width, 1e-6),
        diagonal_distance,
    );
    return max(max(x_line, y_line), diagonal_line);
}

@fragment
fn fragment(in: MeshVertexOutput) -> @location(0) vec4<f32> {
@if(MESH_CLIPPED)
    if outside_clip(in.world_point) {
        discard;
    }
    let flags = mesh.metadata.w;
@if(MESH_BORDER)
    let coverage = uinode_border_coverage(
        in.local_point,
        mesh.size.xy,
        mesh.radius_x,
        mesh.radius_y,
        mesh.border,
        flags,
    );
@else
    let coverage = uinode_background_coverage(
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
@if(VERTEX_COLOR)
    let interpolation_color = in.interpolation_color;
@if(BICUBIC_COLOR)
    let interpolation_color = bicubic_surface_color(in.patch_index, in.uv);
@if(VERTEX_COLOR && IN_LINEAR_RGB)
    var color = interpolation_color;
@else
    var color = convert_to_linear_rgba(interpolation_color);
@if(MESH_WIREFRAME)
    let wireframe = tessellation_wireframe(in.uv, in.subdivisions) * mesh.tessellation.z;
@if(MESH_WIREFRAME)
    color = vec4(mix(color.rgb, vec3(1.0), wireframe), color.a);
    return vec4(color.rgb, saturate(color.a * coverage));
}
