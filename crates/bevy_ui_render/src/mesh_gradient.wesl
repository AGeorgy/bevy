import bevy_render::view::View;
import package::ui::{
    draw_uinode_background,
    draw_uinode_border,
};
import bevy_render::color_operations::{
    oklab_to_linear_rgb,
    srgb_to_linear,
};

const MAX_POINTS: u32 = 256u;
const MAX_CLIPS: u32 = 128u;
const BORDER_LEFT: u32 = 256u;
const BORDER_TOP: u32 = 512u;
const BORDER_RIGHT: u32 = 1024u;
const BORDER_BOTTOM: u32 = 2048u;
const BORDER_ANY: u32 = BORDER_LEFT + BORDER_TOP + BORDER_RIGHT + BORDER_BOTTOM;

fn enabled(flags: u32, mask: u32) -> bool {
    return (flags & mask) != 0u;
}

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
    clip_rects: array<vec4<f32>, 128>,
    clip_transform_x: array<vec4<f32>, 128>,
    clip_transform_y: array<vec4<f32>, 128>,
};

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<uniform> points: MeshGradientPoints;
@group(1) @binding(1) var<uniform> mesh: MeshGradientStyle;

struct MeshVertexOutput {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) patch_index: vec2<u32>,
    @location(2) local_point: vec2<f32>,
    @location(3) world_point: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

fn point_index(x: u32, y: u32) -> u32 {
    return min(y * mesh.metadata.x + x, MAX_POINTS - 1u);
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

fn cubic2(
    p0: vec2<f32>,
    p1: vec2<f32>,
    p2: vec2<f32>,
    p3: vec2<f32>,
    t: f32,
) -> vec2<f32> {
    if t == 0.0 {
        return p1;
    }
    if t == 1.0 {
        return p2;
    }
    return 0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
}

fn cubic4(
    p0: vec4<f32>,
    p1: vec4<f32>,
    p2: vec4<f32>,
    p3: vec4<f32>,
    t: f32,
) -> vec4<f32> {
    if t == 0.0 {
        return p1;
    }
    if t == 1.0 {
        return p2;
    }
    return 0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t * t
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t * t * t);
}

fn surface_position(patch_index: vec2<u32>, uv: vec2<f32>) -> vec2<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let row0 = cubic2(
        control_position(x - 1, y - 1),
        control_position(x, y - 1),
        control_position(x + 1, y - 1),
        control_position(x + 2, y - 1),
        uv.x,
    );
    let row1 = cubic2(
        control_position(x - 1, y),
        control_position(x, y),
        control_position(x + 1, y),
        control_position(x + 2, y),
        uv.x,
    );
    let row2 = cubic2(
        control_position(x - 1, y + 1),
        control_position(x, y + 1),
        control_position(x + 1, y + 1),
        control_position(x + 2, y + 1),
        uv.x,
    );
    let row3 = cubic2(
        control_position(x - 1, y + 2),
        control_position(x, y + 2),
        control_position(x + 1, y + 2),
        control_position(x + 2, y + 2),
        uv.x,
    );
    return cubic2(row0, row1, row2, row3, uv.y);
}

fn surface_color(patch_index: vec2<u32>, uv: vec2<f32>) -> vec4<f32> {
    let x = i32(patch_index.x);
    let y = i32(patch_index.y);
    let row0 = cubic4(
        control_color(x - 1, y - 1),
        control_color(x, y - 1),
        control_color(x + 1, y - 1),
        control_color(x + 2, y - 1),
        uv.x,
    );
    let row1 = cubic4(
        control_color(x - 1, y),
        control_color(x, y),
        control_color(x + 1, y),
        control_color(x + 2, y),
        uv.x,
    );
    let row2 = cubic4(
        control_color(x - 1, y + 1),
        control_color(x, y + 1),
        control_color(x + 1, y + 1),
        control_color(x + 2, y + 1),
        uv.x,
    );
    let row3 = cubic4(
        control_color(x - 1, y + 2),
        control_color(x, y + 2),
        control_color(x + 1, y + 2),
        control_color(x + 2, y + 2),
        uv.x,
    );
    return cubic4(row0, row1, row2, row3, uv.y);
}

@vertex
fn vertex(
    @location(0) patch_index: vec2<u32>,
    @location(1) uv: vec2<f32>,
) -> MeshVertexOutput {
    let normalized = surface_position(patch_index, uv);
    let local_point = (normalized - vec2(0.5)) * mesh.size.xy;
    let homogeneous = vec3(local_point, 1.0);
    let world_point = vec2(
        dot(mesh.transform_x.xyz, homogeneous),
        dot(mesh.transform_y.xyz, homogeneous),
    );
    var out: MeshVertexOutput;
    out.position = view.clip_from_world * vec4(world_point, 0.0, 1.0);
    out.uv = uv;
    out.patch_index = patch_index;
    out.local_point = local_point;
    out.world_point = world_point;
    return out;
}

fn outside_clip(world_point: vec2<f32>) -> bool {
    for (var index = 0u; index < min(mesh.metadata.z, MAX_CLIPS); index += 1u) {
        let homogeneous = vec3(world_point, 1.0);
        let local = vec2(
            dot(mesh.clip_transform_x[index].xyz, homogeneous),
            dot(mesh.clip_transform_y[index].xyz, homogeneous),
        );
        let rect = mesh.clip_rects[index];
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

@fragment
fn fragment(in: MeshVertexOutput) -> @location(0) vec4<f32> {
    if outside_clip(in.world_point) {
        discard;
    }
    let color = convert_to_linear_rgba(surface_color(in.patch_index, in.uv));
    let flags = mesh.metadata.w;
    if enabled(flags, BORDER_ANY) {
        return draw_uinode_border(
            color,
            in.local_point,
            mesh.size.xy,
            mesh.radius_x,
            mesh.radius_y,
            mesh.border,
            flags,
        );
    }
    return draw_uinode_background(
        color,
        in.local_point,
        mesh.size.xy,
        mesh.radius_x,
        mesh.radius_y,
        mesh.border,
        flags,
    );
}
