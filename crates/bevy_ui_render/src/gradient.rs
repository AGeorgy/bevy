use core::{
    f32::consts::{FRAC_PI_2, TAU},
    hash::Hash,
    ops::Range,
};

use super::shader_flags::BORDER_ALL;
use crate::clipping::clip_polygon;
use crate::mesh_gradient::{
    interpolation_color, physical_axes, ParameterVertex, QualityState, SurfaceBounds,
    TopologyCache, TopologyKey,
};
use crate::*;
use bevy_asset::*;
use bevy_color::{ColorToComponents, Hsla, Hsva, LinearRgba, Okhsla, Oklaba, Oklcha, Srgba};
use bevy_ecs::{
    prelude::Component,
    system::{
        lifetimeless::{Read, SRes},
        *,
    },
};
use bevy_math::{
    ops::{cos, sin},
    FloatOrd, Rect, Vec2,
};
use bevy_math::{Affine2, UVec4, Vec2Swizzles, Vec4};
use bevy_mesh::VertexBufferLayout;
use bevy_platform::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use bevy_render::sync_world::MainEntity;
use bevy_render::{
    render_phase::*,
    render_resource::{binding_types::uniform_buffer, *},
    renderer::{RenderDevice, RenderQueue},
    view::*,
    Extract, ExtractSchedule, Render, RenderSystems,
};
use bevy_render::{GpuResourceAppExt, RenderStartup};
use bevy_shader::Shader;
use bevy_sprite::BorderRect;
use bevy_text::{EmSize, RemSize};
use bevy_ui::{
    BackgroundGradient, BorderGradient, ColorStop, ComputedStackIndex, ComputedUiRenderTargetInfo,
    ConicGradient, Gradient, InterpolationColorSpace, LinearGradient, MeshGradient,
    MeshGradientColorInterpolation, MeshGradientGeometry, MeshGradientWireframe, RadialGradient,
    ResolvedBorderRadius, Val, MAX_MESH_GRADIENT_DIMENSION,
};
use bevy_utils::default;
use bytemuck::{cast_slice, Pod, Zeroable};
use tracing::warn;

pub struct GradientPlugin;

impl Plugin for GradientPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "gradient.wesl");
        embedded_asset!(app, "mesh_gradient.wesl");

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .add_render_command::<TransparentUi, DrawGradientFns>()
                .add_render_command::<TransparentUi, DrawMeshGradientFns>()
                .init_resource::<ExtractedGradients>()
                .init_resource::<MeshGradientQualityCache>()
                .init_resource::<MeshGradientSurfaceCache>()
                .init_resource::<MeshGradientBindingCache>()
                .init_resource::<GpuMeshTopologyCache>()
                .init_resource::<MeshGradientDiagnostics>()
                .init_gpu_resource::<GradientMeta>()
                .init_gpu_resource::<SpecializedRenderPipelines<GradientPipeline>>()
                .add_systems(RenderStartup, init_gradient_pipeline)
                .add_systems(
                    ExtractSchedule,
                    extract_gradients
                        .in_set(RenderUiSystems::ExtractGradient)
                        .after(extract_uinode_background_colors),
                )
                .add_systems(
                    Render,
                    (
                        queue_gradient.in_set(RenderSystems::Queue),
                        (prepare_gradient, prepare_mesh_gradients)
                            .in_set(RenderSystems::PrepareBindGroups),
                    ),
                );
        }
    }
}

#[derive(Component)]
pub struct GradientBatch {
    pub range: Range<u32>,
}

#[derive(Resource)]
pub struct GradientMeta {
    vertices: RawBufferVec<UiGradientVertex>,
    indices: RawBufferVec<u32>,
    view_bind_group: Option<BindGroup>,
}

impl Default for GradientMeta {
    fn default() -> Self {
        Self {
            vertices: RawBufferVec::new(BufferUsages::VERTEX),
            indices: RawBufferVec::new(BufferUsages::INDEX),
            view_bind_group: None,
        }
    }
}

#[derive(Resource)]
pub struct GradientPipeline {
    pub view_layout: BindGroupLayoutDescriptor,
    pub mesh_layout: BindGroupLayoutDescriptor,
    pub mesh_clipped_layout: BindGroupLayoutDescriptor,
    pub shader: Handle<Shader>,
    pub mesh_shader: Handle<Shader>,
}

pub fn init_gradient_pipeline(mut commands: Commands, asset_server: Res<AssetServer>) {
    let view_layout = BindGroupLayoutDescriptor::new(
        "ui_gradient_view_layout",
        &BindGroupLayoutEntries::single(
            ShaderStages::VERTEX_FRAGMENT,
            uniform_buffer::<ViewUniform>(true),
        ),
    );
    let mesh_layout = BindGroupLayoutDescriptor::new(
        "ui_mesh_gradient_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<MeshGradientPointsUniform>(false),
                uniform_buffer::<MeshGradientStyleUniform>(false),
            ),
        ),
    );
    let mesh_clipped_layout = BindGroupLayoutDescriptor::new(
        "ui_mesh_gradient_clipped_layout",
        &BindGroupLayoutEntries::sequential(
            ShaderStages::VERTEX_FRAGMENT,
            (
                uniform_buffer::<MeshGradientPointsUniform>(false),
                uniform_buffer::<MeshGradientStyleUniform>(false),
                uniform_buffer::<MeshGradientClipUniform>(false),
            ),
        ),
    );

    commands.insert_resource(GradientPipeline {
        view_layout,
        mesh_layout,
        mesh_clipped_layout,
        shader: load_embedded_asset!(asset_server.as_ref(), "gradient.wesl"),
        mesh_shader: load_embedded_asset!(asset_server.as_ref(), "mesh_gradient.wesl"),
    });
}

pub fn compute_gradient_line_length(angle: f32, size: Vec2) -> f32 {
    let center = 0.5 * size;
    let v = Vec2::new(sin(angle), -cos(angle));

    let (pos_corner, neg_corner) = if v.x >= 0.0 && v.y <= 0.0 {
        (size.with_y(0.), size.with_x(0.))
    } else if v.x >= 0.0 && v.y > 0.0 {
        (size, Vec2::ZERO)
    } else if v.x < 0.0 && v.y <= 0.0 {
        (Vec2::ZERO, size)
    } else {
        (size.with_x(0.), size.with_y(0.))
    };

    let t_pos = (pos_corner - center).dot(v);
    let t_neg = (neg_corner - center).dot(v);

    (t_pos - t_neg).abs()
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct UiGradientPipelineKey {
    anti_alias: bool,
    color_space: InterpolationColorSpace,
    mesh: bool,
    mesh_color_interpolation: MeshGradientColorInterpolation,
    mesh_wireframe: bool,
    mesh_border: bool,
    mesh_clipped: bool,
    mesh_cull_folds: bool,
    mesh_flipped: bool,
    pub target_format: TextureFormat,
}

fn mesh_gradient_primitive_state(cull_folds: bool, flipped: bool) -> PrimitiveState {
    PrimitiveState {
        // Parameter triangles are clockwise after the UI coordinate system's
        // downward Y axis is projected to the render target. A mirrored node
        // reverses that winding without changing which surface side is valid.
        front_face: if flipped {
            FrontFace::Ccw
        } else {
            FrontFace::Cw
        },
        cull_mode: cull_folds.then_some(Face::Back),
        ..default()
    }
}

impl SpecializedRenderPipeline for GradientPipeline {
    type Key = UiGradientPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let vertex_layout = if key.mesh {
            VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Vertex,
                vec![VertexFormat::Uint32x2],
            )
        } else {
            VertexBufferLayout::from_vertex_formats(
                VertexStepMode::Vertex,
                vec![
                    // position
                    VertexFormat::Float32x3,
                    // uv
                    VertexFormat::Float32x2,
                    // flags
                    VertexFormat::Uint32,
                    // border radius x values (top left, top right, bottom right, bottom left)
                    VertexFormat::Float32x4,
                    // border radius y values (top left, top right, bottom right, bottom left)
                    VertexFormat::Float32x4,
                    // border
                    VertexFormat::Float32x4,
                    // size
                    VertexFormat::Float32x2,
                    // point
                    VertexFormat::Float32x2,
                    // start_point
                    VertexFormat::Float32x2,
                    // dir
                    VertexFormat::Float32x2,
                    // start_color
                    VertexFormat::Float32x4,
                    // start_len
                    VertexFormat::Float32,
                    // end_len
                    VertexFormat::Float32,
                    // end color
                    VertexFormat::Float32x4,
                    // hint
                    VertexFormat::Float32,
                ],
            )
        };
        let color_space = match key.color_space {
            InterpolationColorSpace::Oklaba => "IN_OKLAB",
            InterpolationColorSpace::Oklcha => "IN_OKLCH",
            InterpolationColorSpace::OklchaLong => "IN_OKLCH_LONG",
            InterpolationColorSpace::Okhsla => "IN_OKHSL",
            InterpolationColorSpace::OkhslaLong => "IN_OKHSL_LONG",
            InterpolationColorSpace::Srgba => "IN_SRGB",
            InterpolationColorSpace::LinearRgba => "IN_LINEAR_RGB",
            InterpolationColorSpace::Hsla => "IN_HSL",
            InterpolationColorSpace::HslaLong => "IN_HSL_LONG",
            InterpolationColorSpace::Hsva => "IN_HSV",
            InterpolationColorSpace::HsvaLong => "IN_HSV_LONG",
        };

        let mut shader_defs = if key.anti_alias {
            vec![color_space.into(), "ANTI_ALIAS".into()]
        } else {
            vec![color_space.into()]
        };
        match key.mesh_color_interpolation {
            MeshGradientColorInterpolation::Vertex => shader_defs.push("VERTEX_COLOR".into()),
            MeshGradientColorInterpolation::Bicubic => {
                shader_defs.push("BICUBIC_COLOR".into());
            }
        }
        if key.mesh_wireframe
            || key.mesh_color_interpolation == MeshGradientColorInterpolation::Bicubic
        {
            shader_defs.push("MESH_PARAMETER_UV".into());
        }
        if key.mesh_wireframe {
            shader_defs.push("MESH_WIREFRAME".into());
        }
        if key.mesh_border {
            shader_defs.push("MESH_BORDER".into());
        }
        if key.mesh_clipped {
            shader_defs.push("MESH_CLIPPED".into());
        }

        let shader = if key.mesh {
            self.mesh_shader.clone()
        } else {
            self.shader.clone()
        };
        RenderPipelineDescriptor {
            vertex: VertexState {
                shader: shader.clone(),
                shader_defs: shader_defs.clone(),
                buffers: vec![vertex_layout],
                ..default()
            },
            fragment: Some(FragmentState {
                shader,
                shader_defs,
                targets: vec![Some(ColorTargetState {
                    format: key.target_format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            layout: if key.mesh {
                vec![
                    self.view_layout.clone(),
                    if key.mesh_clipped {
                        self.mesh_clipped_layout.clone()
                    } else {
                        self.mesh_layout.clone()
                    },
                ]
            } else {
                vec![self.view_layout.clone()]
            },
            label: Some(if key.mesh {
                "ui_mesh_gradient_pipeline".into()
            } else {
                "ui_gradient_pipeline".into()
            }),
            primitive: mesh_gradient_primitive_state(key.mesh_cull_folds, key.mesh_flipped),
            ..default()
        }
    }
}

pub enum ResolvedGradient {
    Linear { angle: f32 },
    Conic { center: Vec2, start: f32 },
    Radial { center: Vec2, size: Vec2 },
    Mesh(ResolvedMeshGradient),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshGradientId {
    main_entity: MainEntity,
    layer: u32,
    border: bool,
}

pub struct ResolvedMeshGradient {
    mesh: MeshGradient,
    bounds: Arc<SurfaceBounds>,
    id: MeshGradientId,
    display_scale: f32,
    wireframe: bool,
}

#[derive(Resource, Default)]
pub(crate) struct MeshGradientSurfaceCache {
    entries: HashMap<MeshGradientId, (MeshGradient, Arc<SurfaceBounds>)>,
}

impl MeshGradientSurfaceCache {
    fn get(&mut self, id: MeshGradientId, mesh: &MeshGradient) -> Arc<SurfaceBounds> {
        if let Some((cached_mesh, bounds)) = self.entries.get(&id)
            && surface_bounds_inputs_match(cached_mesh, mesh)
        {
            return Arc::clone(bounds);
        }

        let bounds = Arc::new(SurfaceBounds::new(mesh));
        self.entries.insert(id, (mesh.clone(), Arc::clone(&bounds)));
        bounds
    }

    fn retain(&mut self, active: &HashSet<MeshGradientId>) {
        self.entries.retain(|id, _| active.contains(id));
    }
}

fn surface_bounds_inputs_match(left: &MeshGradient, right: &MeshGradient) -> bool {
    left.dimensions() == right.dimensions()
        && left.color_interpolation() == right.color_interpolation()
        && left
            .points()
            .iter()
            .zip(right.points())
            .all(|(left, right)| left.position == right.position)
        && (left.color_interpolation() != MeshGradientColorInterpolation::Vertex
            || left.color_space() == right.color_space()
                && left
                    .points()
                    .iter()
                    .zip(right.points())
                    .all(|(left, right)| left.color == right.color))
}

impl ResolvedGradient {
    fn is_mesh(&self) -> bool {
        matches!(self, Self::Mesh(_))
    }
}

pub struct ExtractedGradient {
    pub stack_index: u32,
    pub transform: Affine2,
    pub rect: Rect,
    pub clip: Option<CalculatedClip>,
    pub stops: Vec<(LinearRgba, f32, f32)>,
    pub node_type: NodeType,
    /// Border radius of the UI node.
    /// Ordering: top left, top right, bottom right, bottom left.
    pub border_radius: ResolvedBorderRadius,
    /// Border thickness of the UI node.
    /// Ordering: left, top, right, bottom.
    pub border: BorderRect,
    pub resolved_gradient: ResolvedGradient,
    pub color_space: InterpolationColorSpace,
}

/// A render-world resource that stores all gradients in the scene.
#[derive(Resource, Default)]
pub struct ExtractedGradients {
    /// The list of gradients grouped by their main-world entity, along with each group's target camera entity.
    ///
    /// This is a two-level data structure so that we can quickly remove all
    /// gradients associated with a main-world entity when it changes.
    pub items: MainEntityHashMap<(Entity, EntityIndexMap<ExtractedGradient>)>,
}

// Interpolate implicit stops (where position is `f32::NAN`)
// If the first and last stops are implicit set them to the `min` and `max` values
// so that we always have explicit start and end points to interpolate between.
fn interpolate_color_stops(stops: &mut [(LinearRgba, f32, f32)], min: f32, max: f32) {
    if stops[0].1.is_nan() {
        stops[0].1 = min;
    }
    if stops.last().unwrap().1.is_nan() {
        stops.last_mut().unwrap().1 = max;
    }

    let mut i = 1;

    while i < stops.len() - 1 {
        let point = stops[i].1;
        if point.is_nan() {
            let start = i;
            let mut end = i + 1;
            while end < stops.len() - 1 && stops[end].1.is_nan() {
                end += 1;
            }
            let start_point = stops[start - 1].1;
            let end_point = stops[end].1;
            let steps = end - start;
            let step = (end_point - start_point) / (steps + 1) as f32;
            for j in 0..steps {
                stops[i + j].1 = start_point + step * (j + 1) as f32;
            }
            i = end;
        }
        i += 1;
    }
}

fn compute_color_stops(
    stops: &[ColorStop],
    scale_factor: f32,
    length: f32,
    target_size: Vec2,
    scratch: &mut Vec<(LinearRgba, f32, f32)>,
    em_size: EmSize,
    rem_size: RemSize,
) -> Vec<(LinearRgba, f32, f32)> {
    let mut extracted_color_stops = vec![];

    // resolve the physical distances of explicit stops and sort them
    scratch.extend(stops.iter().filter_map(|stop| {
        stop.point
            .resolve(scale_factor, length, target_size, em_size, rem_size)
            .ok()
            .map(|physical_point| (stop.color.to_linear(), physical_point, stop.hint))
    }));
    scratch.sort_by_key(|(_, point, _)| FloatOrd(*point));

    let min = scratch
        .first()
        .map(|(_, min, _)| *min)
        .unwrap_or(0.)
        .min(0.);

    // get the position of the last explicit stop and use the full length of the gradient if no explicit stops
    let max = scratch
        .last()
        .map(|(_, max, _)| *max)
        .unwrap_or(length)
        .max(length);

    let mut sorted_stops_drain = scratch.drain(..);

    // Fill the extracted color stops buffer
    extracted_color_stops.extend(stops.iter().map(|stop| {
        if stop.point == Val::Auto {
            (stop.color.to_linear(), f32::NAN, stop.hint)
        } else {
            sorted_stops_drain.next().unwrap()
        }
    }));

    interpolate_color_stops(&mut extracted_color_stops, min, max);

    extracted_color_stops
}

pub fn extract_gradients(
    mut commands: Commands,
    mut extracted_gradients: ResMut<ExtractedGradients>,
    mut surface_cache: ResMut<MeshGradientSurfaceCache>,
    gradients_query: Extract<
        Query<
            (
                Entity,
                &ComputedNode,
                &ComputedStackIndex,
                &ComputedUiTargetCamera,
                &ComputedUiRenderTargetInfo,
                &UiGlobalTransform,
                &InheritedVisibility,
                Option<&CalculatedClip>,
                Option<&MeshGradientWireframe>,
                AnyOf<(&BackgroundGradient, &BorderGradient)>,
            ),
            Or<(
                Changed<ComputedNode>,
                Changed<ComputedStackIndex>,
                Changed<ComputedUiTargetCamera>,
                Changed<ComputedUiRenderTargetInfo>,
                Changed<UiGlobalTransform>,
                Changed<InheritedVisibility>,
                Changed<CalculatedClip>,
                Changed<MeshGradientWireframe>,
                Changed<BackgroundGradient>,
                Changed<BorderGradient>,
            )>,
        >,
    >,
    unfilitered_gradients_query: Extract<
        Query<(
            Entity,
            &ComputedNode,
            &ComputedStackIndex,
            &ComputedUiTargetCamera,
            &ComputedUiRenderTargetInfo,
            &UiGlobalTransform,
            &InheritedVisibility,
            Option<&CalculatedClip>,
            Option<&MeshGradientWireframe>,
            AnyOf<(&BackgroundGradient, &BorderGradient)>,
        )>,
    >,
    (
        mut removed_computed_node_query,
        mut removed_computed_stack_index_query,
        mut removed_computed_ui_target_camera_query,
        mut removed_computed_ui_render_target_info_query,
        mut removed_ui_global_transform_query,
        mut removed_inherited_visibility_query,
        mut removed_calculated_clip_query,
        mut removed_mesh_gradient_wireframe_query,
        mut removed_background_gradient_query,
        mut removed_border_gradient_query,
    ): (
        Extract<RemovedComponents<ComputedNode>>,
        Extract<RemovedComponents<ComputedStackIndex>>,
        Extract<RemovedComponents<ComputedUiTargetCamera>>,
        Extract<RemovedComponents<ComputedUiRenderTargetInfo>>,
        Extract<RemovedComponents<UiGlobalTransform>>,
        Extract<RemovedComponents<InheritedVisibility>>,
        Extract<RemovedComponents<CalculatedClip>>,
        Extract<RemovedComponents<MeshGradientWireframe>>,
        Extract<RemovedComponents<BackgroundGradient>>,
        Extract<RemovedComponents<BorderGradient>>,
    ),
    camera_map: Extract<UiCameraMap>,
    mut nodes_processed_this_frame: Local<MainEntityHashSet>,
) {
    nodes_processed_this_frame.clear();
    let mut camera_mapper = camera_map.get_mapper();
    let mut sorted_stops = vec![];

    for (
        entity,
        uinode,
        stack_index,
        camera,
        target,
        transform,
        inherited_visibility,
        clip,
        wireframe,
        (gradient, gradient_border),
    ) in gradients_query.iter().chain(
        removed_calculated_clip_query
            .read()
            .chain(removed_mesh_gradient_wireframe_query.read())
            .filter_map(|entity| unfilitered_gradients_query.get(entity).ok()),
    ) {
        let main_entity = MainEntity::from(entity);

        // If there were any previous gradients for this entity, despawn them.
        for (render_entity, _) in extracted_gradients
            .items
            .get_mut(&main_entity)
            .iter_mut()
            .flat_map(|(_, gradients)| gradients.drain(..))
        {
            commands.entity(render_entity).despawn();
        }

        // Skip invisible images
        if !inherited_visibility.get() {
            continue;
        }

        let Some(extracted_camera_entity) = camera_mapper.map(camera) else {
            continue;
        };
        if let Some((camera_entity, _)) = extracted_gradients.items.get_mut(&main_entity) {
            *camera_entity = extracted_camera_entity;
        }

        for (gradients, node_type, border_layer) in [
            (gradient.map(|g| &g.0), NodeType::Rect, false),
            (
                gradient_border.map(|g| &g.0),
                NodeType::Border(BORDER_ALL),
                true,
            ),
        ]
        .iter()
        .filter_map(|(g, n, border_layer)| g.map(|g| (g, *n, *border_layer)))
        {
            for (layer, gradient) in gradients.iter().enumerate() {
                if gradient.is_empty() {
                    continue;
                }

                nodes_processed_this_frame.insert(main_entity);

                if let Some(color) = gradient.get_single() {
                    // With a single color stop there's no gradient, fill the node with the color
                    let length = compute_gradient_line_length(0.0, uinode.size);
                    let extracted_stops = compute_color_stops(
                        &[
                            ColorStop::new(color, Val::Percent(0.0)),
                            ColorStop::new(color, Val::Percent(100.0)),
                        ],
                        target.scale_factor(),
                        length,
                        target.physical_size().as_vec2(),
                        &mut sorted_stops,
                        uinode.em_size,
                        uinode.rem_size,
                    );
                    extracted_gradients
                        .items
                        .entry(main_entity)
                        .or_insert_with(|| (extracted_camera_entity, Default::default()))
                        .1
                        .insert(
                            commands.spawn_empty().id(),
                            ExtractedGradient {
                                stack_index: stack_index.0,
                                transform: transform.into(),
                                stops: extracted_stops,
                                rect: Rect {
                                    min: Vec2::ZERO,
                                    max: uinode.size,
                                },
                                clip: clip.cloned(),
                                node_type,
                                border_radius: uinode.border_radius,
                                border: uinode.border,
                                resolved_gradient: ResolvedGradient::Linear { angle: 0.0 },
                                color_space: gradient.get_color_space(),
                            },
                        );
                    continue;
                }
                match gradient {
                    Gradient::Linear(LinearGradient {
                        color_space,
                        angle,
                        stops,
                    }) => {
                        let length = compute_gradient_line_length(*angle, uinode.size);

                        let extracted_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: extracted_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.cloned(),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Linear { angle: *angle },
                                    color_space: *color_space,
                                },
                            );
                    }
                    Gradient::Radial(RadialGradient {
                        color_space,
                        position: center,
                        shape,
                        stops,
                    }) => {
                        let c = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let size = shape.resolve(
                            c,
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        let length = size.x;

                        let computed_stops = compute_color_stops(
                            stops,
                            target.scale_factor(),
                            length,
                            target.physical_size().as_vec2(),
                            &mut sorted_stops,
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: computed_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.cloned(),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Radial { center: c, size },
                                    color_space: *color_space,
                                },
                            );
                    }
                    Gradient::Conic(ConicGradient {
                        color_space,
                        start,
                        position: center,
                        stops,
                    }) => {
                        let g_start = center.resolve(
                            target.scale_factor(),
                            uinode.size,
                            target.physical_size().as_vec2(),
                            uinode.em_size,
                            uinode.rem_size,
                        );

                        // sort the explicit stops
                        sorted_stops.extend(stops.iter().filter_map(|stop| {
                            stop.angle.map(|angle| {
                                (stop.color.to_linear(), angle.clamp(0., TAU), stop.hint)
                            })
                        }));
                        sorted_stops.sort_by_key(|(_, angle, _)| FloatOrd(*angle));
                        let mut sorted_stops_drain = sorted_stops.drain(..);

                        // fill the extracted stops buffer
                        let mut extracted_color_stops: Vec<_> = stops
                            .iter()
                            .map(|stop| {
                                if stop.angle.is_none() {
                                    (stop.color.to_linear(), f32::NAN, stop.hint)
                                } else {
                                    sorted_stops_drain.next().unwrap()
                                }
                            })
                            .collect();

                        interpolate_color_stops(&mut extracted_color_stops, 0., TAU);

                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: extracted_color_stops,
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.cloned(),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Conic {
                                        start: *start,
                                        center: g_start,
                                    },
                                    color_space: *color_space,
                                },
                            );
                    }
                    Gradient::Mesh(mesh) => {
                        let id = MeshGradientId {
                            main_entity,
                            layer: layer as u32,
                            border: border_layer,
                        };
                        extracted_gradients
                            .items
                            .entry(main_entity)
                            .or_insert_with(|| (extracted_camera_entity, Default::default()))
                            .1
                            .insert(
                                commands.spawn_empty().id(),
                                ExtractedGradient {
                                    stack_index: stack_index.0,
                                    transform: transform.into(),
                                    stops: Vec::new(),
                                    rect: Rect {
                                        min: Vec2::ZERO,
                                        max: uinode.size,
                                    },
                                    clip: clip.cloned(),
                                    node_type,
                                    border_radius: uinode.border_radius,
                                    border: uinode.border,
                                    resolved_gradient: ResolvedGradient::Mesh(
                                        ResolvedMeshGradient {
                                            mesh: mesh.clone(),
                                            bounds: surface_cache.get(id, mesh),
                                            id,
                                            display_scale: target.scale_factor(),
                                            wireframe: wireframe.is_some(),
                                        },
                                    ),
                                    color_space: mesh.color_space().into(),
                                },
                            );
                    }
                }
            }
        }
    }

    // Only remove the render-world data if we didn't handle the node above.
    // It's possible that a relevant component was removed and added in the same
    // frame.
    for main_entity in removed_computed_node_query
        .read()
        .chain(removed_computed_stack_index_query.read())
        .chain(removed_computed_ui_target_camera_query.read())
        .chain(removed_computed_ui_render_target_info_query.read())
        .chain(removed_ui_global_transform_query.read())
        .chain(removed_inherited_visibility_query.read())
        .chain(removed_background_gradient_query.read())
        .chain(removed_border_gradient_query.read())
    {
        let main_entity = MainEntity::from(main_entity);
        if nodes_processed_this_frame.contains(&main_entity) {
            continue;
        }
        let Some((_, mut extracted_nodes)) = extracted_gradients.items.remove(&main_entity) else {
            continue;
        };
        for (render_entity, _) in extracted_nodes.drain(..) {
            commands.entity(render_entity).despawn();
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "it's a system that needs a lot of them"
)]
pub fn queue_gradient(
    extracted_gradients: ResMut<ExtractedGradients>,
    gradients_pipeline: Res<GradientPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<GradientPipeline>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    render_views: Query<(&UiCameraView, Option<&UiAntiAlias>), With<ExtractedView>>,
    camera_views: Query<&ExtractedView>,
    pipeline_cache: Res<PipelineCache>,
    draw_functions: Res<DrawFunctions<TransparentUi>>,
) {
    let draw_function = draw_functions.read().id::<DrawGradientFns>();
    let mesh_draw_function = draw_functions.read().id::<DrawMeshGradientFns>();
    let mut current_camera_entity = Entity::PLACEHOLDER;
    let mut current_phase = None;

    for (main_entity, (extracted_camera_entity, sub_gradients)) in extracted_gradients.items.iter()
    {
        if current_camera_entity != *extracted_camera_entity {
            current_phase = render_views.get(*extracted_camera_entity).ok().and_then(
                |(default_camera_view, ui_anti_alias)| {
                    camera_views
                        .get(default_camera_view.0)
                        .ok()
                        .and_then(|view| {
                            transparent_render_phases
                                .get_mut(&view.retained_view_entity)
                                .map(|transparent_phase| {
                                    (view.target_format, ui_anti_alias, transparent_phase)
                                })
                        })
                },
            );
            current_camera_entity = *extracted_camera_entity;
        }

        let Some((target_format, ui_anti_alias, transparent_phase)) = current_phase.as_mut() else {
            continue;
        };
        for (render_entity, gradient) in sub_gradients.iter() {
            let mesh_color_interpolation = match &gradient.resolved_gradient {
                ResolvedGradient::Mesh(mesh) => mesh.mesh.color_interpolation(),
                _ => MeshGradientColorInterpolation::Vertex,
            };
            let mesh_wireframe = matches!(
                &gradient.resolved_gradient,
                ResolvedGradient::Mesh(mesh) if mesh.wireframe
            );
            let mesh_border = gradient.resolved_gradient.is_mesh()
                && matches!(gradient.node_type, NodeType::Border(_));
            let mesh_clipped = gradient.resolved_gradient.is_mesh() && gradient.clip.is_some();
            let mesh_cull_folds = matches!(
                &gradient.resolved_gradient,
                ResolvedGradient::Mesh(mesh)
                    if mesh.mesh.geometry() == MeshGradientGeometry::AllowFolds
            );
            let pipeline = pipelines.specialize(
                &pipeline_cache,
                &gradients_pipeline,
                UiGradientPipelineKey {
                    anti_alias: matches!(ui_anti_alias, None | Some(UiAntiAlias::On)),
                    color_space: gradient.color_space,
                    mesh: gradient.resolved_gradient.is_mesh(),
                    mesh_color_interpolation,
                    mesh_wireframe,
                    mesh_border,
                    mesh_clipped,
                    mesh_cull_folds,
                    mesh_flipped: mesh_cull_folds
                        && gradient.transform.matrix2.determinant().is_sign_negative(),
                    target_format: *target_format,
                },
            );

            transparent_phase.add_transient(TransparentUi {
                draw_function: if gradient.resolved_gradient.is_mesh() {
                    mesh_draw_function
                } else {
                    draw_function
                },
                pipeline,
                entity: (*render_entity, *main_entity),
                sort_key: FloatOrd(
                    gradient.stack_index as f32
                        + match gradient.node_type {
                            NodeType::Rect | NodeType::Inverted => stack_z_offsets::GRADIENT,
                            NodeType::Border(_) => stack_z_offsets::BORDER_GRADIENT,
                        },
                ),
                batch_range: 0..0,
                extra_index: PhaseItemExtraIndex::None,
                indexed: true,
            });
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UiGradientVertex {
    position: [f32; 3],
    uv: [f32; 2],
    flags: u32,
    radius: [[f32; 4]; 2],
    border: [f32; 4],
    size: [f32; 2],
    point: [f32; 2],
    g_start: [f32; 2],
    g_dir: [f32; 2],
    start_color: [f32; 4],
    start_len: f32,
    end_len: f32,
    end_color: [f32; 4],
    hint: f32,
}

fn convert_color_to_space(color: LinearRgba, space: InterpolationColorSpace) -> [f32; 4] {
    match space {
        InterpolationColorSpace::Oklaba => {
            let oklaba: Oklaba = color.into();
            [oklaba.lightness, oklaba.a, oklaba.b, oklaba.alpha]
        }
        InterpolationColorSpace::Oklcha | InterpolationColorSpace::OklchaLong => {
            let oklcha: Oklcha = color.into();
            [
                oklcha.lightness,
                oklcha.chroma,
                // The shader expects normalized hues
                oklcha.hue / 360.,
                oklcha.alpha,
            ]
        }
        InterpolationColorSpace::Okhsla | InterpolationColorSpace::OkhslaLong => {
            let okhsla: Okhsla = color.into();
            [
                okhsla.hue / 360.,
                okhsla.saturation,
                okhsla.lightness,
                okhsla.alpha,
            ]
        }
        InterpolationColorSpace::Srgba => {
            let srgba: Srgba = color.into();
            [srgba.red, srgba.green, srgba.blue, srgba.alpha]
        }
        InterpolationColorSpace::LinearRgba => color.to_f32_array(),
        InterpolationColorSpace::Hsla | InterpolationColorSpace::HslaLong => {
            let hsla: Hsla = color.into();
            // The shader expects normalized hues
            [hsla.hue / 360., hsla.saturation, hsla.lightness, hsla.alpha]
        }
        InterpolationColorSpace::Hsva | InterpolationColorSpace::HsvaLong => {
            let hsva: Hsva = color.into();
            // The shader expects normalized hues
            [hsva.hue / 360., hsva.saturation, hsva.value, hsva.alpha]
        }
    }
}

pub fn prepare_gradient(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    mut ui_meta: ResMut<GradientMeta>,
    extracted_gradients: Res<ExtractedGradients>,
    view_uniforms: Res<ViewUniforms>,
    gradients_pipeline: Res<GradientPipeline>,
    mut phases: ResMut<ViewSortedRenderPhases<TransparentUi>>,
    mut previous_len: Local<usize>,
) {
    if let Some(view_binding) = view_uniforms.uniforms.binding() {
        let mut batches: Vec<(Entity, GradientBatch)> = Vec::with_capacity(*previous_len);

        ui_meta.vertices.clear();
        ui_meta.indices.clear();
        ui_meta.view_bind_group = Some(render_device.create_bind_group(
            "gradient_view_bind_group",
            &pipeline_cache.get_bind_group_layout(&gradients_pipeline.view_layout),
            &BindGroupEntries::single(view_binding),
        ));

        // Buffer indexes
        let mut vertices_index = 0;
        let mut indices_index = 0;

        for ui_phase in phases.values_mut() {
            for item_index in 0..ui_phase.items.len() {
                let item = &mut ui_phase.items[item_index];
                if let Some(gradient) = extracted_gradients
                    .items
                    .get(&item.main_entity())
                    .and_then(|(_, subgradients)| subgradients.get(&item.entity()))
                {
                    *item.batch_range_mut() = item_index as u32..item_index as u32 + 1;
                    if gradient.resolved_gradient.is_mesh() {
                        continue;
                    }
                    let uinode_rect = gradient.rect;

                    let rect_size = uinode_rect.size();

                    // Specify the corners of the node
                    let corner_points = QUAD_VERTEX_POSITIONS.map(|pos| pos * rect_size);
                    let positions =
                        corner_points.map(|pos| gradient.transform.transform_point2(pos));

                    let uvs = { [Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::Y] };

                    let mut flags = if let NodeType::Border(borders) = gradient.node_type {
                        borders
                    } else {
                        0
                    };

                    let (g_start, g_dir, g_flags) = match gradient.resolved_gradient {
                        ResolvedGradient::Linear { angle } => {
                            let corner_index = (angle - FRAC_PI_2).rem_euclid(TAU) / FRAC_PI_2;
                            (
                                corner_points[corner_index as usize].into(),
                                // CSS angles increase in a clockwise direction
                                [sin(angle), -cos(angle)],
                                0,
                            )
                        }
                        ResolvedGradient::Conic { center, start } => {
                            (center.into(), [start, 0.], shader_flags::CONIC)
                        }
                        ResolvedGradient::Radial { center, size } => (
                            center.into(),
                            Vec2::splat(if size.y != 0. { size.x / size.y } else { 1. }).into(),
                            shader_flags::RADIAL,
                        ),
                        ResolvedGradient::Mesh(_) => unreachable!(),
                    };

                    flags |= g_flags;

                    let vertices = clip_polygon(
                        gradient.clip.as_ref(),
                        &[
                            (positions[0], (uvs[0], corner_points[0])),
                            (positions[1], (uvs[1], corner_points[1])),
                            (positions[2], (uvs[2], corner_points[2])),
                            (positions[3], (uvs[3], corner_points[3])),
                        ],
                        |a, b, t| (a.0.lerp(b.0, t), a.1.lerp(b.1, t)),
                    );
                    if vertices.is_empty() {
                        continue;
                    }
                    let segment_index_count = 3 * (vertices.len() as u32 - 2);

                    let range = 0..gradient.stops.len() - 1;
                    let mut segment_count = 0;

                    for stop_index in range {
                        let mut start_stop = gradient.stops[stop_index];
                        let end_stop = gradient.stops[stop_index + 1];
                        if start_stop.1 == end_stop.1 {
                            if stop_index == gradient.stops.len() - 2 {
                                if 0 < segment_count {
                                    start_stop.0 = LinearRgba::NONE;
                                }
                            } else {
                                continue;
                            }
                        }
                        let start_color =
                            convert_color_to_space(start_stop.0, gradient.color_space);
                        let end_color = convert_color_to_space(end_stop.0, gradient.color_space);
                        let mut stop_flags = flags;
                        if 0. < start_stop.1 && (stop_index == 0 || segment_count == 0) {
                            stop_flags |= shader_flags::FILL_START;
                        }
                        if stop_index == gradient.stops.len() - 2 {
                            stop_flags |= shader_flags::FILL_END;
                        }

                        for &(position, (uv, point)) in &vertices {
                            ui_meta.vertices.push(UiGradientVertex {
                                position: position.extend(0.).into(),
                                uv: uv.into(),
                                flags: stop_flags,
                                radius: gradient.border_radius.into(),
                                border: [
                                    gradient.border.min_inset.x,
                                    gradient.border.min_inset.y,
                                    gradient.border.max_inset.x,
                                    gradient.border.max_inset.y,
                                ],
                                size: rect_size.xy().into(),
                                g_start,
                                g_dir,
                                point: point.into(),
                                start_color,
                                start_len: start_stop.1,
                                end_len: end_stop.1,
                                end_color,
                                hint: start_stop.2,
                            });
                        }

                        for i in 1..vertices.len() as u32 - 1 {
                            ui_meta.indices.push(indices_index);
                            ui_meta.indices.push(indices_index + i);
                            ui_meta.indices.push(indices_index + i + 1);
                        }
                        indices_index += vertices.len() as u32;
                        segment_count += 1;
                    }

                    if 0 < segment_count {
                        let vertices_count = segment_index_count * segment_count;

                        batches.push((
                            item.entity(),
                            GradientBatch {
                                range: vertices_index..(vertices_index + vertices_count),
                            },
                        ));

                        vertices_index += vertices_count;
                    }
                }
            }
        }
        ui_meta.vertices.write_buffer(&render_device, &render_queue);
        ui_meta.indices.write_buffer(&render_device, &render_queue);
        *previous_len = batches.len();
        commands.try_insert_batch(batches);
    }
}

const MAX_MESH_GRADIENT_POINTS: usize = MAX_MESH_GRADIENT_DIMENSION * MAX_MESH_GRADIENT_DIMENSION;
// A clip entry occupies three vec4 values. The fixed bound keeps the style
// block finite while supporting deeply nested UI clipping on WebGL2.
const MAX_MESH_GRADIENT_CLIPS: usize = 128;

#[derive(Clone, PartialEq, ShaderType)]
struct MeshGradientPointsUniform {
    positions: [Vec4; MAX_MESH_GRADIENT_POINTS],
    colors: [Vec4; MAX_MESH_GRADIENT_POINTS],
}

#[derive(Clone, PartialEq, ShaderType)]
struct MeshGradientStyleUniform {
    transform_x: Vec4,
    transform_y: Vec4,
    radius_x: Vec4,
    radius_y: Vec4,
    border: Vec4,
    size: Vec4,
    tessellation: Vec4,
    metadata: UVec4,
}

#[derive(Clone, PartialEq, ShaderType)]
struct MeshGradientClipUniform {
    metadata: UVec4,
    clip_rects: [Vec4; MAX_MESH_GRADIENT_CLIPS],
    clip_transform_x: [Vec4; MAX_MESH_GRADIENT_CLIPS],
    clip_transform_y: [Vec4; MAX_MESH_GRADIENT_CLIPS],
}

const _: () = assert!(MeshGradientPointsUniform::SHADER_SIZE.get() <= 16 * 1024);
const _: () = assert!(MeshGradientStyleUniform::SHADER_SIZE.get() <= 16 * 1024);
const _: () = assert!(MeshGradientClipUniform::SHADER_SIZE.get() <= 16 * 1024);

struct GpuParameterTopology {
    vertices: Buffer,
    indices: Buffer,
    index_format: IndexFormat,
    index_count: u32,
}

#[derive(Component)]
pub(crate) struct MeshGradientGpu {
    bind_group: Arc<BindGroup>,
    topology: Arc<GpuParameterTopology>,
    topology_key: TopologyKey,
}

#[derive(Resource, Default)]
struct MeshGradientQualityCache {
    states: HashMap<MeshGradientId, QualityState>,
}

struct CachedMeshGradientBindings {
    points: UniformBuffer<MeshGradientPointsUniform>,
    style: UniformBuffer<MeshGradientStyleUniform>,
    clip: Option<UniformBuffer<MeshGradientClipUniform>>,
    bind_group: Arc<BindGroup>,
}

#[derive(Resource, Default)]
struct MeshGradientBindingCache {
    entries: HashMap<MeshGradientId, CachedMeshGradientBindings>,
}

impl MeshGradientBindingCache {
    #[expect(
        clippy::too_many_arguments,
        reason = "the cache owns and updates all mesh-gradient bindings"
    )]
    fn get(
        &mut self,
        id: MeshGradientId,
        points_value: MeshGradientPointsUniform,
        style_value: MeshGradientStyleUniform,
        clip_value: Option<MeshGradientClipUniform>,
        layout: &BindGroupLayoutDescriptor,
        pipeline_cache: &PipelineCache,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) -> Arc<BindGroup> {
        if let Some(entry) = self.entries.get_mut(&id)
            && entry.clip.is_some() == clip_value.is_some()
        {
            if entry.points.get() != &points_value {
                entry.points.set(points_value);
                entry.points.write_buffer(render_device, render_queue);
            }
            if entry.style.get() != &style_value {
                entry.style.set(style_value);
                entry.style.write_buffer(render_device, render_queue);
            }
            if let (Some(clip), Some(clip_value)) = (&mut entry.clip, clip_value)
                && clip.get() != &clip_value
            {
                clip.set(clip_value);
                clip.write_buffer(render_device, render_queue);
            }
            return entry.bind_group.clone();
        }

        let mut points = UniformBuffer::from(points_value);
        points.set_label(Some("ui_mesh_gradient_points"));
        points.write_buffer(render_device, render_queue);
        let mut style = UniformBuffer::from(style_value);
        style.set_label(Some("ui_mesh_gradient_style"));
        style.write_buffer(render_device, render_queue);
        let clip = clip_value.map(|value| {
            let mut uniform = UniformBuffer::from(value);
            uniform.set_label(Some("ui_mesh_gradient_clips"));
            uniform.write_buffer(render_device, render_queue);
            uniform
        });
        let bind_group = Arc::new(match clip.as_ref() {
            Some(clip) => render_device.create_bind_group(
                "ui_mesh_gradient_clipped_bind_group",
                &pipeline_cache.get_bind_group_layout(layout),
                &BindGroupEntries::sequential((
                    points.binding().unwrap(),
                    style.binding().unwrap(),
                    clip.binding().unwrap(),
                )),
            ),
            None => render_device.create_bind_group(
                "ui_mesh_gradient_bind_group",
                &pipeline_cache.get_bind_group_layout(layout),
                &BindGroupEntries::sequential((
                    points.binding().unwrap(),
                    style.binding().unwrap(),
                )),
            ),
        });
        self.entries.insert(
            id,
            CachedMeshGradientBindings {
                points,
                style,
                clip,
                bind_group: bind_group.clone(),
            },
        );
        bind_group
    }

    fn retain(&mut self, seen: &HashSet<MeshGradientId>) {
        self.entries.retain(|id, _| seen.contains(id));
    }
}

#[derive(Resource, Default)]
struct GpuMeshTopologyCache {
    cpu: TopologyCache,
    entries: HashMap<TopologyKey, Arc<GpuParameterTopology>>,
}

impl GpuMeshTopologyCache {
    fn get(
        &mut self,
        key: &TopologyKey,
        render_device: &RenderDevice,
    ) -> Arc<GpuParameterTopology> {
        if let Some(topology) = self.entries.get(key) {
            return topology.clone();
        }
        let topology = self.cpu.get(key);
        let (indices, index_format) = if topology.vertices.len() <= u16::MAX as usize + 1 {
            let indices: Vec<u16> = topology
                .indices
                .iter()
                .map(|&index| {
                    u16::try_from(index)
                        .expect("topology with at most 65536 vertices uses u16 indices")
                })
                .collect();
            (
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("ui_mesh_gradient_parameter_indices"),
                    contents: cast_slice::<u16, u8>(&indices),
                    usage: BufferUsages::INDEX,
                }),
                IndexFormat::Uint16,
            )
        } else {
            (
                render_device.create_buffer_with_data(&BufferInitDescriptor {
                    label: Some("ui_mesh_gradient_parameter_indices"),
                    contents: cast_slice::<u32, u8>(&topology.indices),
                    usage: BufferUsages::INDEX,
                }),
                IndexFormat::Uint32,
            )
        };
        let gpu = Arc::new(GpuParameterTopology {
            vertices: render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("ui_mesh_gradient_parameter_vertices"),
                contents: cast_slice::<ParameterVertex, u8>(&topology.vertices),
                usage: BufferUsages::VERTEX,
            }),
            indices,
            index_format,
            index_count: topology.indices.len() as u32,
        });
        self.entries.insert(key.clone(), gpu.clone());
        gpu
    }

    fn prune(&mut self) {
        self.entries
            .retain(|_, topology| Arc::strong_count(topology) > 1);
        self.cpu.prune();
    }
}

#[derive(Resource, Default)]
struct MeshGradientDiagnostics {
    invalid: HashSet<MeshGradientId>,
}

#[derive(Debug)]
enum MeshUniformError {
    FullyClipped,
    TooManyClips(usize),
    Invalid,
}

fn affine_is_finite(transform: Affine2) -> bool {
    transform.matrix2.x_axis.is_finite()
        && transform.matrix2.y_axis.is_finite()
        && transform.translation.is_finite()
}

fn mesh_gradient_points_uniform(mesh: &MeshGradient) -> MeshGradientPointsUniform {
    let mut positions = [Vec4::ZERO; MAX_MESH_GRADIENT_POINTS];
    let mut colors = [Vec4::ZERO; MAX_MESH_GRADIENT_POINTS];
    for (index, point) in mesh.points().iter().enumerate() {
        positions[index] = point.position.extend(0.0).extend(0.0);
        colors[index] = Vec4::from_array(interpolation_color(mesh, point.color));
    }
    MeshGradientPointsUniform { positions, colors }
}

fn mesh_gradient_style_uniform(
    gradient: &ExtractedGradient,
    mesh: &MeshGradient,
    wireframe: bool,
) -> Result<(MeshGradientStyleUniform, Option<MeshGradientClipUniform>), MeshUniformError> {
    let size = gradient.rect.size();
    if !size.is_finite()
        || size.x <= 0.0
        || size.y <= 0.0
        || !affine_is_finite(gradient.transform)
        || mesh.points().len() != mesh.width() * mesh.height()
        || mesh.width() > MAX_MESH_GRADIENT_DIMENSION
        || mesh.height() > MAX_MESH_GRADIENT_DIMENSION
    {
        return Err(MeshUniformError::Invalid);
    }

    let clip_rects = match gradient.clip.as_ref() {
        Some(clip) => clip.rects().ok_or(MeshUniformError::FullyClipped)?,
        None => &[],
    };
    if clip_rects.len() > MAX_MESH_GRADIENT_CLIPS {
        return Err(MeshUniformError::TooManyClips(clip_rects.len()));
    }
    if clip_rects.iter().any(|clip| {
        !clip.rect.min.is_finite()
            || !clip.rect.max.is_finite()
            || !affine_is_finite(clip.world_to_clip_local)
    }) {
        return Err(MeshUniformError::Invalid);
    }

    let clip_uniform = gradient.clip.as_ref().map(|_| {
        let mut uniform_clip_rects = [Vec4::ZERO; MAX_MESH_GRADIENT_CLIPS];
        let mut clip_transform_x = [Vec4::ZERO; MAX_MESH_GRADIENT_CLIPS];
        let mut clip_transform_y = [Vec4::ZERO; MAX_MESH_GRADIENT_CLIPS];
        for (index, clip) in clip_rects.iter().enumerate() {
            uniform_clip_rects[index] = Vec4::new(
                clip.rect.min.x,
                clip.rect.min.y,
                clip.rect.max.x,
                clip.rect.max.y,
            );
            clip_transform_x[index] = Vec4::new(
                clip.world_to_clip_local.matrix2.x_axis.x,
                clip.world_to_clip_local.matrix2.y_axis.x,
                clip.world_to_clip_local.translation.x,
                0.0,
            );
            clip_transform_y[index] = Vec4::new(
                clip.world_to_clip_local.matrix2.x_axis.y,
                clip.world_to_clip_local.matrix2.y_axis.y,
                clip.world_to_clip_local.translation.y,
                0.0,
            );
        }
        MeshGradientClipUniform {
            metadata: UVec4::new(clip_rects.len() as u32, 0, 0, 0),
            clip_rects: uniform_clip_rects,
            clip_transform_x,
            clip_transform_y,
        }
    });

    let radius: [[f32; 4]; 2] = gradient.border_radius.into();
    let flags = match gradient.node_type {
        NodeType::Border(flags) => flags,
        NodeType::Rect | NodeType::Inverted => 0,
    };
    let style = MeshGradientStyleUniform {
        transform_x: Vec4::new(
            gradient.transform.matrix2.x_axis.x,
            gradient.transform.matrix2.y_axis.x,
            gradient.transform.translation.x,
            0.0,
        ),
        transform_y: Vec4::new(
            gradient.transform.matrix2.x_axis.y,
            gradient.transform.matrix2.y_axis.y,
            gradient.transform.translation.y,
            0.0,
        ),
        radius_x: Vec4::from_array(radius[0]),
        radius_y: Vec4::from_array(radius[1]),
        border: Vec4::new(
            gradient.border.min_inset.x,
            gradient.border.min_inset.y,
            gradient.border.max_inset.x,
            gradient.border.max_inset.y,
        ),
        size: size.extend(0.0).extend(0.0),
        tessellation: Vec4::new(u32::from(wireframe) as f32, 1.0, 0.68, 0.0),
        metadata: UVec4::new(mesh.width() as u32, mesh.height() as u32, 0, flags),
    };
    Ok((style, clip_uniform))
}

#[expect(
    clippy::too_many_arguments,
    reason = "it's a render preparation system"
)]
fn prepare_mesh_gradients(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    pipeline_cache: Res<PipelineCache>,
    gradients_pipeline: Res<GradientPipeline>,
    extracted_gradients: Res<ExtractedGradients>,
    phases: Res<ViewSortedRenderPhases<TransparentUi>>,
    mut quality_cache: ResMut<MeshGradientQualityCache>,
    mut surface_cache: ResMut<MeshGradientSurfaceCache>,
    mut binding_cache: ResMut<MeshGradientBindingCache>,
    mut topology_cache: ResMut<GpuMeshTopologyCache>,
    mut diagnostics: ResMut<MeshGradientDiagnostics>,
    mut prepared: Query<&mut MeshGradientGpu>,
) {
    topology_cache.prune();
    let active: HashSet<_> = extracted_gradients
        .items
        .values()
        .flat_map(|(_, gradients)| gradients.values())
        .filter_map(|gradient| match &gradient.resolved_gradient {
            ResolvedGradient::Mesh(mesh) => Some(mesh.id),
            _ => None,
        })
        .collect();

    for phase in phases.values() {
        for (_, item) in &phase.items {
            let Some(gradient) = extracted_gradients
                .items
                .get(&item.main_entity())
                .and_then(|(_, gradients)| gradients.get(&item.entity()))
            else {
                continue;
            };
            let ResolvedGradient::Mesh(mesh) = &gradient.resolved_gradient else {
                continue;
            };
            if gradient
                .clip
                .as_ref()
                .is_some_and(CalculatedClip::is_fully_clipped)
            {
                continue;
            }
            let axes = physical_axes(
                gradient.rect.size(),
                gradient.transform.matrix2,
                mesh.display_scale,
            );
            let selection = quality_cache
                .states
                .entry(mesh.id)
                .or_default()
                .update(&mesh.bounds, axes);
            if selection.capped && selection.report_cap {
                warn!(
                    "mesh gradient reached its adaptive tessellation cap: {} triangles, maximum axis subdivision {}, geometry error {:.3}px, color error {:.4}",
                    selection.key.triangles(),
                    selection.key.maximum_subdivisions(),
                    selection.error.geometry,
                    selection.error.color,
                );
            }

            if let Ok(mut gpu) = prepared.get_mut(item.entity()) {
                if gpu.topology_key != selection.key {
                    gpu.topology = topology_cache.get(&selection.key, &render_device);
                    gpu.topology_key = selection.key.clone();
                }
                diagnostics.invalid.remove(&mesh.id);
                continue;
            }

            let (style_value, clip_value) = match mesh_gradient_style_uniform(
                gradient,
                &mesh.mesh,
                mesh.wireframe,
            ) {
                Ok(uniform) => uniform,
                Err(MeshUniformError::FullyClipped) => continue,
                Err(error) => {
                    if diagnostics.invalid.insert(mesh.id) {
                        match error {
                            MeshUniformError::TooManyClips(count) => warn!(
                                "skipping mesh gradient with {count} inherited clip rectangles; the WebGL2-safe maximum is {MAX_MESH_GRADIENT_CLIPS}"
                            ),
                            MeshUniformError::Invalid => warn!(
                                "skipping malformed mesh gradient render data"
                            ),
                            MeshUniformError::FullyClipped => unreachable!(),
                        }
                    }
                    continue;
                }
            };

            let layout = if clip_value.is_some() {
                &gradients_pipeline.mesh_clipped_layout
            } else {
                &gradients_pipeline.mesh_layout
            };
            let bind_group = binding_cache.get(
                mesh.id,
                mesh_gradient_points_uniform(&mesh.mesh),
                style_value,
                clip_value,
                layout,
                &pipeline_cache,
                &render_device,
                &render_queue,
            );
            let topology = topology_cache.get(&selection.key, &render_device);
            commands.entity(item.entity()).insert(MeshGradientGpu {
                bind_group,
                topology,
                topology_key: selection.key,
            });
            diagnostics.invalid.remove(&mesh.id);
        }
    }

    quality_cache.states.retain(|id, _| active.contains(id));
    surface_cache.retain(&active);
    binding_cache.retain(&active);
    diagnostics.invalid.retain(|id| active.contains(id));
}

pub type DrawGradientFns = (SetItemPipeline, SetGradientViewBindGroup<0>, DrawGradient);
pub(crate) type DrawMeshGradientFns = (
    SetItemPipeline,
    SetGradientViewBindGroup<0>,
    SetMeshGradientBindGroup<1>,
    DrawMeshGradient,
);

pub struct SetGradientViewBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetGradientViewBindGroup<I> {
    type Param = SRes<GradientMeta>;
    type ViewQuery = Read<ViewUniformOffset>;
    type ItemQuery = ();

    fn render<'w>(
        _item: &P,
        view_uniform: &'w ViewUniformOffset,
        _entity: Option<()>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(view_bind_group) = ui_meta.into_inner().view_bind_group.as_ref() else {
            return RenderCommandResult::Failure("view_bind_group not available");
        };
        pass.set_bind_group(I, view_bind_group, &[view_uniform.offset]);
        RenderCommandResult::Success
    }
}

pub(crate) struct SetMeshGradientBindGroup<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetMeshGradientBindGroup<I> {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<MeshGradientGpu>;

    fn render<'w>(
        _item: &P,
        _view: (),
        mesh: Option<&'w MeshGradientGpu>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(mesh) = mesh else {
            return RenderCommandResult::Skip;
        };
        pass.set_bind_group(I, &mesh.bind_group, &[]);
        RenderCommandResult::Success
    }
}

pub struct DrawGradient;
impl<P: PhaseItem> RenderCommand<P> for DrawGradient {
    type Param = SRes<GradientMeta>;
    type ViewQuery = ();
    type ItemQuery = Read<GradientBatch>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        batch: Option<&'w GradientBatch>,
        ui_meta: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(batch) = batch else {
            return RenderCommandResult::Skip;
        };
        let ui_meta = ui_meta.into_inner();
        let Some(vertices) = ui_meta.vertices.buffer() else {
            return RenderCommandResult::Failure("missing vertices to draw ui");
        };
        let Some(indices) = ui_meta.indices.buffer() else {
            return RenderCommandResult::Failure("missing indices to draw ui");
        };

        // Store the vertices
        pass.set_vertex_buffer(0, vertices.slice(..));
        // Define how to "connect" the vertices
        pass.set_index_buffer(indices.slice(..), IndexFormat::Uint32);
        // Draw the vertices
        pass.draw_indexed(batch.range.clone(), 0, 0..1);
        RenderCommandResult::Success
    }
}

pub(crate) struct DrawMeshGradient;
impl<P: PhaseItem> RenderCommand<P> for DrawMeshGradient {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<MeshGradientGpu>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        mesh: Option<&'w MeshGradientGpu>,
        _param: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let Some(mesh) = mesh else {
            return RenderCommandResult::Skip;
        };
        pass.set_vertex_buffer(0, mesh.topology.vertices.slice(..));
        pass.set_index_buffer(mesh.topology.indices.slice(..), mesh.topology.index_format);
        pass.draw_indexed(0..mesh.topology.index_count, 0, 0..1);
        RenderCommandResult::Success
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::Color;
    use bevy_ui::{CalculatedClipRect, MeshGradientColorSpace, MeshGradientPoint};
    use smallvec::SmallVec;

    fn full_capacity_mesh() -> MeshGradient {
        let mut points = Vec::with_capacity(MAX_MESH_GRADIENT_POINTS);
        for y in 0..16 {
            for x in 0..16 {
                let position = Vec2::new(x as f32, y as f32) / 15.0;
                points.push(MeshGradientPoint::new(
                    position,
                    Color::linear_rgba(position.x * 4.0, position.y * 2.0, -0.5, 0.75),
                ));
            }
        }
        MeshGradient::new_in_color_space(16, 16, points, MeshGradientColorSpace::LinearRgba)
            .unwrap()
    }

    #[test]
    fn surface_bounds_cache_uses_only_inputs_that_affect_bounds() {
        let mut left = full_capacity_mesh();
        let mut right = left.clone();
        right
            .try_set_color(0, Color::linear_rgba(8.0, -2.0, 1.0, 0.5))
            .unwrap();
        assert!(!surface_bounds_inputs_match(&left, &right));

        left.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        right.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        assert!(surface_bounds_inputs_match(&left, &right));
    }

    #[test]
    fn folded_meshes_cull_only_the_locally_reversed_surface() {
        let ordinary = mesh_gradient_primitive_state(false, false);
        assert_eq!(ordinary.cull_mode, None);

        let forward = mesh_gradient_primitive_state(true, false);
        assert_eq!(forward.cull_mode, Some(Face::Back));
        assert_eq!(forward.front_face, FrontFace::Cw);

        let mirrored = mesh_gradient_primitive_state(true, true);
        assert_eq!(mirrored.cull_mode, Some(Face::Back));
        assert_eq!(mirrored.front_face, FrontFace::Ccw);
    }

    fn extracted(clip: Option<CalculatedClip>) -> ExtractedGradient {
        ExtractedGradient {
            stack_index: 0,
            transform: Affine2::IDENTITY,
            rect: Rect::from_corners(Vec2::ZERO, Vec2::new(320.0, 180.0)),
            clip,
            stops: Vec::new(),
            node_type: NodeType::Rect,
            border_radius: ResolvedBorderRadius::default(),
            border: BorderRect::default(),
            resolved_gradient: ResolvedGradient::Linear { angle: 0.0 },
            color_space: InterpolationColorSpace::LinearRgba,
        }
    }

    #[test]
    fn full_capacity_uniform_is_webgl2_safe_and_preserves_control_data() {
        assert!(MeshGradientPointsUniform::SHADER_SIZE.get() <= 16 * 1024);
        assert!(MeshGradientStyleUniform::SHADER_SIZE.get() <= 16 * 1024);
        assert!(MeshGradientClipUniform::SHADER_SIZE.get() <= 16 * 1024);
        let mesh = full_capacity_mesh();
        let points = mesh_gradient_points_uniform(&mesh);
        let (style, clip) = mesh_gradient_style_uniform(&extracted(None), &mesh, true).unwrap();
        assert!(clip.is_none());
        assert_eq!(style.metadata, UVec4::new(16, 16, 0, 0));
        assert_eq!(style.tessellation, Vec4::new(1.0, 1.0, 0.68, 0.0));
        assert_eq!(
            Vec2::new(points.positions[0].x, points.positions[0].y),
            Vec2::ZERO
        );
        assert_eq!(
            Vec2::new(points.positions[255].x, points.positions[255].y),
            Vec2::ONE
        );
        assert_eq!(points.colors[255], Vec4::new(4.0, 2.0, -0.5, 0.75));
    }

    #[test]
    fn clipping_uses_its_own_optional_uniform() {
        let clip_rect = CalculatedClipRect {
            rect: Rect::from_corners(Vec2::new(2.0, 3.0), Vec2::new(17.0, 19.0)),
            world_to_clip_local: Affine2::from_translation(Vec2::new(5.0, 7.0)),
        };
        let (style, clip) = mesh_gradient_style_uniform(
            &extracted(Some(CalculatedClip::Rects(SmallVec::from_iter([
                clip_rect,
            ])))),
            &full_capacity_mesh(),
            false,
        )
        .unwrap();
        let clip = clip.unwrap();

        assert_eq!(style.metadata.z, 0);
        assert_eq!(clip.metadata.x, 1);
        assert_eq!(clip.clip_rects[0], Vec4::new(2.0, 3.0, 17.0, 19.0));
        assert_eq!(clip.clip_transform_x[0].z, 5.0);
        assert_eq!(clip.clip_transform_y[0].z, 7.0);
    }

    #[test]
    fn uniform_rejects_clip_data_that_would_exceed_webgl2_limits() {
        let clips: SmallVec<[CalculatedClipRect; 2]> =
            SmallVec::from_iter((0..=MAX_MESH_GRADIENT_CLIPS).map(|_| CalculatedClipRect {
                rect: Rect::from_corners(Vec2::ZERO, Vec2::ONE),
                world_to_clip_local: Affine2::IDENTITY,
            }));
        assert!(matches!(
            mesh_gradient_style_uniform(
                &extracted(Some(CalculatedClip::Rects(clips))),
                &full_capacity_mesh(),
                false,
            ),
            Err(MeshUniformError::TooManyClips(count)) if count == MAX_MESH_GRADIENT_CLIPS + 1
        ));
    }
}
