//! Internal adaptive mesh-gradient preparation. The renderer supplies physical
//! screen axes, including node size, transform and display scale exactly once.
//!
//! Geometry uses patch-local Hessian remainder bounds for linear triangular
//! interpolation. Color is evaluated either at tessellation vertices or from a
//! bicubic surface in the fragment shader. Each patch has independent
//! power-of-two factors; finer boundary vertices snap to the coarser edge
//! approximation so neighboring patches remain crack-free without propagating
//! refinement. Vertex color mode adds an exact bilinear color-error term, so it
//! spends triangles only where the rasterizer would reveal a patch diagonal.

use bevy_color::{Color, ColorToComponents, LinearRgba, Oklaba, Srgba};
use bevy_math::{DVec2, Mat2, Vec2};
use bevy_platform::{collections::HashMap, sync::Arc};
use bevy_ui::{MeshGradient, MeshGradientColorInterpolation, MeshGradientColorSpace};
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;

/// Maximum certified surface displacement in physical pixels.
const GEOMETRY_LIMIT: f64 = 4.0;
/// Maximum interpolation-space color error for the mobile-friendly vertex path.
/// A roughly 1/32 step is small enough to hide triangle diagonals while
/// avoiding the fragment cost of bicubic color evaluation.
const COLOR_LIMIT: f64 = 0.032;
const MIN_SUBDIVISIONS: usize = 2;
const MAX_TRIANGLES: usize = 131_072;
const MAX_SUBDIVISIONS: usize = 64;
const DEMOTION_FRAMES: u8 = 8;

pub(crate) fn interpolation_color(mesh: &MeshGradient, color: Color) -> [f32; 4] {
    match mesh.color_space() {
        MeshGradientColorSpace::LinearRgba => LinearRgba::from(color).to_f32_array(),
        MeshGradientColorSpace::Srgba => Srgba::from(color).to_f32_array(),
        MeshGradientColorSpace::Oklaba => Oklaba::from(color).to_f32_array(),
    }
}

#[cfg(test)]
type Point = [f64; 6];
#[cfg(test)]
type Patch = [Point; 16];

/// `node_size` is logical; `display_scale` converts it to physical pixels.
/// Translation does not affect interpolation error and is deliberately absent.
pub(crate) fn physical_axes(node_size: Vec2, transform: Mat2, display_scale: f32) -> [DVec2; 2] {
    [
        transform.x_axis.as_dvec2() * (node_size.x as f64 * display_scale as f64),
        transform.y_axis.as_dvec2() * (node_size.y as f64 * display_scale as f64),
    ]
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn exact(value: f64) -> Self {
        Self {
            lo: value,
            hi: value,
        }
    }
    fn add(self, other: Self) -> Self {
        Self {
            lo: (self.lo + other.lo).next_down(),
            hi: (self.hi + other.hi).next_up(),
        }
    }
    fn sub(self, other: Self) -> Self {
        Self {
            lo: (self.lo - other.hi).next_down(),
            hi: (self.hi - other.lo).next_up(),
        }
    }
    fn scale(self, scale: f64) -> Self {
        if scale >= 0.0 {
            Self {
                lo: (self.lo * scale).next_down(),
                hi: (self.hi * scale).next_up(),
            }
        } else {
            Self {
                lo: (self.hi * scale).next_down(),
                hi: (self.lo * scale).next_up(),
            }
        }
    }
    fn sixth(self) -> Self {
        Self {
            lo: (self.lo / 6.0).next_down(),
            hi: (self.hi / 6.0).next_up(),
        }
    }
}

type IntervalPoint<const N: usize> = [Interval; N];
type IntervalPatch<const N: usize> = [IntervalPoint<N>; 16];

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TopologyKey {
    pub width: usize,
    pub height: usize,
    /// U is stored in the low byte and V in the high byte.
    factors: SmallVec<[u16; 16]>,
}

impl TopologyKey {
    fn uniform(width: usize, height: usize, subdivisions: usize) -> Self {
        let factors = core::iter::repeat_n(
            Self::pack(subdivisions, subdivisions),
            (width - 1) * (height - 1),
        )
        .collect();
        Self {
            width,
            height,
            factors,
        }
    }

    const fn pack(u: usize, v: usize) -> u16 {
        u as u16 | ((v as u16) << 8)
    }

    fn index(&self, column: usize, row: usize) -> usize {
        row * (self.width - 1) + column
    }

    fn u(&self, column: usize, row: usize) -> usize {
        (self.factors[self.index(column, row)] & 0xff) as usize
    }

    fn v(&self, column: usize, row: usize) -> usize {
        (self.factors[self.index(column, row)] >> 8) as usize
    }

    fn same_dimensions(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    fn doubled_u(&self, column: usize, row: usize) -> Option<Self> {
        let subdivisions = self.u(column, row);
        if subdivisions >= MAX_SUBDIVISIONS {
            return None;
        }
        let mut next = self.clone();
        let index = next.index(column, row);
        next.factors[index] = Self::pack(subdivisions * 2, self.v(column, row));
        Some(next)
    }

    fn doubled_v(&self, column: usize, row: usize) -> Option<Self> {
        let subdivisions = self.v(column, row);
        if subdivisions >= MAX_SUBDIVISIONS {
            return None;
        }
        let mut next = self.clone();
        let index = next.index(column, row);
        next.factors[index] = Self::pack(self.u(column, row), subdivisions * 2);
        Some(next)
    }

    fn within(&self, cap: &Self) -> bool {
        self.same_dimensions(cap)
            && (0..self.height - 1).all(|row| {
                (0..self.width - 1).all(|column| {
                    self.u(column, row) <= cap.u(column, row)
                        && self.v(column, row) <= cap.v(column, row)
                })
            })
    }

    pub fn maximum_subdivisions(&self) -> usize {
        self.factors
            .iter()
            .flat_map(|factor| [factor & 0xff, factor >> 8])
            .max()
            .unwrap_or(1) as usize
    }

    pub fn triangles(&self) -> usize {
        (0..self.height - 1)
            .flat_map(|row| (0..self.width - 1).map(move |column| (column, row)))
            .map(|(column, row)| 2 * self.u(column, row) * self.v(column, row))
            .sum()
    }
}

/// Patch-local UVs stay dyadic, making adjacent patch edges exactly identical.
///
/// Every value fits in one byte: patch coordinates are at most 15, UV
/// numerators and subdivision counts are at most 64. Packing reduces the
/// parameter topology from four 32-bit two-component attributes to one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ParameterVertex {
    pub packed: [u32; 2],
}

impl ParameterVertex {
    fn new(
        patch: [usize; 2],
        numerator: [usize; 2],
        subdivisions: [usize; 2],
        position_subdivisions: [usize; 2],
    ) -> Self {
        debug_assert!(patch.into_iter().all(|value| value <= u8::MAX as usize));
        debug_assert!(numerator.into_iter().all(|value| value <= u8::MAX as usize));
        debug_assert!(subdivisions
            .into_iter()
            .all(|value| value <= u8::MAX as usize));
        debug_assert!(position_subdivisions
            .into_iter()
            .all(|value| value <= u8::MAX as usize));
        Self {
            packed: [
                pack_bytes(patch[0], patch[1], numerator[0], numerator[1]),
                pack_bytes(
                    subdivisions[0],
                    subdivisions[1],
                    position_subdivisions[0],
                    position_subdivisions[1],
                ),
            ],
        }
    }

    #[cfg(test)]
    fn unpack(self) -> ([u32; 2], [f32; 2], [u32; 2], [u32; 2]) {
        let first = unpack_bytes(self.packed[0]);
        let second = unpack_bytes(self.packed[1]);
        let subdivisions = [second[0], second[1]];
        (
            [first[0], first[1]],
            [
                first[2] as f32 / subdivisions[0] as f32,
                first[3] as f32 / subdivisions[1] as f32,
            ],
            subdivisions,
            [second[2], second[3]],
        )
    }
}

const fn pack_bytes(a: usize, b: usize, c: usize, d: usize) -> u32 {
    a as u32 | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24
}

#[cfg(test)]
const fn unpack_bytes(value: u32) -> [u32; 4] {
    [
        value & 0xff,
        (value >> 8) & 0xff,
        (value >> 16) & 0xff,
        value >> 24,
    ]
}

pub(crate) struct ParameterTopology {
    pub vertices: Vec<ParameterVertex>,
    pub indices: Vec<u32>,
}

#[derive(Default)]
pub(crate) struct TopologyCache {
    entries: HashMap<TopologyKey, Arc<ParameterTopology>>,
}

impl TopologyCache {
    pub fn get(&mut self, key: &TopologyKey) -> Arc<ParameterTopology> {
        self.entries
            .entry(key.clone())
            .or_insert_with(|| {
                let mut vertices = Vec::new();
                let mut indices = Vec::with_capacity(key.triangles() * 3);
                for row in 0..key.height - 1 {
                    for column in 0..key.width - 1 {
                        let columns = key.u(column, row);
                        let rows = key.v(column, row);
                        let base = vertices.len() as u32;
                        for y in 0..=rows {
                            for x in 0..=columns {
                                let mut position_columns = columns;
                                let mut position_rows = rows;
                                if y > 0 && y < rows {
                                    if x == 0 && column > 0 {
                                        position_rows = position_rows.min(key.v(column - 1, row));
                                    }
                                    if x == columns && column + 1 < key.width - 1 {
                                        position_rows = position_rows.min(key.v(column + 1, row));
                                    }
                                }
                                if x > 0 && x < columns {
                                    if y == 0 && row > 0 {
                                        position_columns =
                                            position_columns.min(key.u(column, row - 1));
                                    }
                                    if y == rows && row + 1 < key.height - 1 {
                                        position_columns =
                                            position_columns.min(key.u(column, row + 1));
                                    }
                                }
                                vertices.push(ParameterVertex::new(
                                    [column, row],
                                    [x, y],
                                    [columns, rows],
                                    [position_columns, position_rows],
                                ));
                            }
                        }
                        for y in 0..rows {
                            for x in 0..columns {
                                let a = base + (y * (columns + 1) + x) as u32;
                                let b = a + 1;
                                let c = a + (columns + 1) as u32;
                                let d = c + 1;
                                indices.extend_from_slice(&[a, b, d, a, d, c]);
                            }
                        }
                    }
                }
                Arc::new(ParameterTopology { vertices, indices })
            })
            .clone()
    }

    /// Release entries no live gradient uses. Call after all frame lookups.
    pub fn prune(&mut self) {
        self.entries
            .retain(|_, topology| Arc::strong_count(topology) > 1);
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ErrorBound {
    pub geometry: f64,
    pub color: f64,
}

impl ErrorBound {
    fn meets(self, maximum_error: f64, fraction: f64) -> bool {
        self.geometry <= maximum_error * fraction
            && self.color <= upper_product(COLOR_LIMIT, fraction)
    }

    fn severity(self, maximum_error: f64) -> f64 {
        (self.geometry / maximum_error).max(self.color / COLOR_LIMIT)
    }
}

pub(crate) struct QualitySelection {
    pub key: TopologyKey,
    pub error: ErrorBound,
    pub capped: bool,
    /// Emit a diagnostic on entry to the capped state, not on every frame.
    pub report_cap: bool,
}

#[derive(Default)]
pub(crate) struct QualityState {
    key: Option<TopologyKey>,
    below_half: u8,
    demotion_candidate: Option<TopologyKey>,
    capped: bool,
    report_cooldown: u8,
    screen_bounds: ScreenBounds,
}

impl QualityState {
    pub fn update(&mut self, bounds: &SurfaceBounds, screen_axes: [DVec2; 2]) -> QualitySelection {
        self.report_cooldown = self.report_cooldown.saturating_sub(1);
        let mut screen_bounds = core::mem::take(&mut self.screen_bounds);
        bounds.update_screen(screen_axes, &mut screen_bounds);
        let bounds = &screen_bounds;
        let minimum = TopologyKey::uniform(bounds.width, bounds.height, MIN_SUBDIVISIONS);
        let chosen;
        if let Some(previous) = self
            .key
            .as_ref()
            .filter(|key| key.same_dimensions(&minimum))
        {
            if !bounds.error(previous).meets(GEOMETRY_LIMIT, 1.0) {
                chosen = bounds.refine(previous.clone(), GEOMETRY_LIMIT, 1.0, None);
                self.below_half = 0;
                self.demotion_candidate = None;
            } else {
                let candidate = bounds.refine(minimum.clone(), GEOMETRY_LIMIT, 0.5, Some(previous));
                if self.demotion_candidate.as_ref() != Some(&candidate) {
                    self.below_half = 0;
                }
                self.demotion_candidate = Some(candidate.clone());
                if candidate.triangles() < previous.triangles()
                    && bounds.error(&candidate).meets(GEOMETRY_LIMIT, 0.5)
                {
                    self.below_half += 1;
                } else {
                    self.below_half = 0;
                }
                if self.below_half < DEMOTION_FRAMES {
                    chosen = previous.clone();
                } else {
                    chosen = candidate;
                    self.below_half = 0;
                }
            }
        } else {
            chosen = bounds.refine(minimum, GEOMETRY_LIMIT, 1.0, None);
            self.below_half = 0;
            self.demotion_candidate = None;
        }
        let error = bounds.error(&chosen);
        let capped = !error.meets(GEOMETRY_LIMIT, 1.0);
        let report_cap = capped && !self.capped && self.report_cooldown == 0;
        if report_cap {
            self.report_cooldown = 120;
        }
        self.capped = capped;
        self.key = Some(chosen.clone());
        self.screen_bounds = screen_bounds;
        QualitySelection {
            key: chosen,
            error,
            capped,
            report_cap,
        }
    }
}

#[derive(Clone, Copy)]
struct VectorBounds {
    x: Interval,
    y: Interval,
}

impl VectorBounds {
    fn empty() -> Self {
        Self {
            x: Interval {
                lo: f64::INFINITY,
                hi: f64::NEG_INFINITY,
            },
            y: Interval {
                lo: f64::INFINITY,
                hi: f64::NEG_INFINITY,
            },
        }
    }

    #[cfg(test)]
    fn exact(value: DVec2) -> Self {
        Self {
            x: Interval::exact(value.x),
            y: Interval::exact(value.y),
        }
    }

    fn include(&mut self, value: [Interval; 2]) {
        self.x.lo = self.x.lo.min(value[0].lo);
        self.x.hi = self.x.hi.max(value[0].hi);
        self.y.lo = self.y.lo.min(value[1].lo);
        self.y.hi = self.y.hi.max(value[1].hi);
    }

    fn transformed_length(self, axes: [DVec2; 2]) -> f64 {
        [self.x.lo, self.x.hi]
            .into_iter()
            .flat_map(|x| {
                [self.y.lo, self.y.hi].into_iter().flat_map(move |y| {
                    let screen_x = Interval::exact(x)
                        .scale(axes[0].x)
                        .add(Interval::exact(y).scale(axes[1].x));
                    let screen_y = Interval::exact(x)
                        .scale(axes[0].y)
                        .add(Interval::exact(y).scale(axes[1].y));
                    [screen_x.lo, screen_x.hi].into_iter().flat_map(move |x| {
                        [screen_y.lo, screen_y.hi].map(|y| upper_length(DVec2::new(x, y)))
                    })
                })
            })
            .fold(0.0, f64::max)
    }

    fn transformed_extent(self, axes: [DVec2; 2]) -> DVec2 {
        let mut minimum = DVec2::splat(f64::INFINITY);
        let mut maximum = DVec2::splat(f64::NEG_INFINITY);
        for x in [self.x.lo, self.x.hi] {
            for y in [self.y.lo, self.y.hi] {
                let point = axes[0] * x + axes[1] * y;
                minimum = minimum.min(point);
                maximum = maximum.max(point);
            }
        }
        maximum - minimum
    }
}

#[derive(Clone, Copy)]
struct PatchBounds {
    position: VectorBounds,
    uu: VectorBounds,
    uv: VectorBounds,
    vv: VectorBounds,
    /// U curvature on the top and bottom edges.
    edge_uu: [VectorBounds; 2],
    /// V curvature on the left and right edges.
    edge_vv: [VectorBounds; 2],
    /// Mixed derivative of the exact bilinear color field used by vertex mode.
    color_uv: f64,
}

#[derive(Clone, Copy)]
struct ErrorScore {
    bound: ErrorBound,
    maximum: f64,
    sum: f64,
    shape: f64,
    worst_patch: [usize; 2],
}

#[derive(Clone, Copy)]
struct ScreenPatchBounds {
    extent: DVec2,
    uu: f64,
    uv: f64,
    vv: f64,
    edge_uu: [f64; 2],
    edge_vv: [f64; 2],
    color_uv: f64,
}

#[derive(Default)]
struct ScreenBounds {
    width: usize,
    height: usize,
    patches: SmallVec<[ScreenPatchBounds; 9]>,
}

/// Rebuild on point changes only. Resizing only reevaluates the bounds.
pub(crate) struct SurfaceBounds {
    width: usize,
    height: usize,
    patches: Box<[PatchBounds]>,
}

impl SurfaceBounds {
    pub fn new(mesh: &MeshGradient) -> Self {
        let (width, height) = mesh.dimensions();
        let values: SmallVec<[[f64; 2]; 16]> = mesh
            .points()
            .iter()
            .map(|point| point.position.as_dvec2().to_array())
            .collect();
        let colors: Option<SmallVec<[[f64; 4]; 16]>> =
            (mesh.color_interpolation() == MeshGradientColorInterpolation::Vertex).then(|| {
                mesh.points()
                    .iter()
                    .map(|point| interpolation_color(mesh, point.color).map(f64::from))
                    .collect()
            });
        let interval_patches: SmallVec<[IntervalPatch<2>; 9]> =
            build_interval_patches(&values, width, height);
        let patches = interval_patches
            .iter()
            .enumerate()
            .map(|(patch_index, patch)| {
                let mut position = VectorBounds::empty();
                let mut uu = VectorBounds::empty();
                let mut uv = VectorBounds::empty();
                let mut vv = VectorBounds::empty();
                let mut edge_uu = [VectorBounds::empty(); 2];
                let mut edge_vv = [VectorBounds::empty(); 2];
                for y in 0..4 {
                    for x in 0..2 {
                        let derivative_uu = core::array::from_fn(|channel| {
                            patch[y * 4 + x + 2][channel]
                                .sub(patch[y * 4 + x + 1][channel].scale(2.0))
                                .add(patch[y * 4 + x][channel])
                                .scale(6.0)
                        });
                        uu.include(derivative_uu);
                        if y == 0 {
                            edge_uu[0].include(derivative_uu);
                        } else if y == 3 {
                            edge_uu[1].include(derivative_uu);
                        }
                        let derivative_vv = core::array::from_fn(|channel| {
                            patch[(x + 2) * 4 + y][channel]
                                .sub(patch[(x + 1) * 4 + y][channel].scale(2.0))
                                .add(patch[x * 4 + y][channel])
                                .scale(6.0)
                        });
                        vv.include(derivative_vv);
                        if y == 0 {
                            edge_vv[0].include(derivative_vv);
                        } else if y == 3 {
                            edge_vv[1].include(derivative_vv);
                        }
                    }
                }
                for y in 0..3 {
                    for x in 0..3 {
                        uv.include(core::array::from_fn(|channel| {
                            patch[(y + 1) * 4 + x + 1][channel]
                                .sub(patch[(y + 1) * 4 + x][channel])
                                .sub(patch[y * 4 + x + 1][channel])
                                .add(patch[y * 4 + x][channel])
                                .scale(9.0)
                        }));
                    }
                }
                let column = patch_index % (width - 1);
                let row = patch_index / (width - 1);
                let color_uv = colors.as_ref().map_or(0.0, |colors| {
                    let color = |x: usize, y: usize| colors[y * width + x];
                    let top_left = color(column, row);
                    let top_right = color(column + 1, row);
                    let bottom_left = color(column, row + 1);
                    let bottom_right = color(column + 1, row + 1);
                    (0..4)
                        .map(|channel| {
                            let mixed = Interval::exact(top_left[channel])
                                .sub(Interval::exact(top_right[channel]))
                                .sub(Interval::exact(bottom_left[channel]))
                                .add(Interval::exact(bottom_right[channel]));
                            mixed.lo.abs().max(mixed.hi.abs())
                        })
                        .fold(0.0, f64::max)
                });
                if color_uv > 0.0 {
                    for point in patch {
                        position.include(*point);
                    }
                }
                PatchBounds {
                    position,
                    uu,
                    uv,
                    vv,
                    edge_uu,
                    edge_vv,
                    color_uv,
                }
            })
            .collect();
        Self {
            width,
            height,
            patches,
        }
    }

    fn update_screen(&self, axes: [DVec2; 2], screen: &mut ScreenBounds) {
        screen.width = self.width;
        screen.height = self.height;
        screen.patches.clear();
        screen
            .patches
            .extend(self.patches.iter().map(|patch| ScreenPatchBounds {
                extent: if patch.color_uv > 0.0 {
                    patch.position.transformed_extent(axes)
                } else {
                    DVec2::ZERO
                },
                uu: patch.uu.transformed_length(axes),
                uv: patch.uv.transformed_length(axes),
                vv: patch.vv.transformed_length(axes),
                edge_uu: patch.edge_uu.map(|bound| bound.transformed_length(axes)),
                edge_vv: patch.edge_vv.map(|bound| bound.transformed_length(axes)),
                color_uv: patch.color_uv,
            }));
    }

    #[cfg(test)]
    fn error(&self, topology: &TopologyKey, axes: [DVec2; 2]) -> ErrorBound {
        let mut screen = ScreenBounds::default();
        self.update_screen(axes, &mut screen);
        screen.error(topology)
    }
}

impl ScreenBounds {
    fn color_patch_error(&self, patch: ScreenPatchBounds, columns: f64, rows: f64) -> f64 {
        // Once either cell axis is at most one physical pixel, further color
        // subdivision cannot produce a resolvable improvement along both axes.
        if patch.extent.x / columns <= 1.0 || patch.extent.y / rows <= 1.0 {
            return 0.0;
        }
        upper_product(patch.color_uv, 0.25 / (columns * rows))
    }

    fn base_patch_error(&self, column: usize, row: usize, topology: &TopologyKey) -> ErrorBound {
        let patch = self.patches[row * (self.width - 1) + column];
        let columns = topology.u(column, row) as f64;
        let rows = topology.v(column, row) as f64;
        let uu = upper_product(patch.uu, 0.125 / (columns * columns));
        let uv = upper_product(patch.uv, 0.25 / (columns * rows));
        let vv = upper_product(patch.vv, 0.125 / (rows * rows));
        ErrorBound {
            geometry: upper_sum(upper_sum(uu, uv), vv),
            color: self.color_patch_error(patch, columns, rows),
        }
    }

    fn patch_error(&self, column: usize, row: usize, topology: &TopologyKey) -> ErrorBound {
        let base = self.base_patch_error(column, row, topology);
        let mut snap = 0.0_f64;
        let columns = topology.u(column, row);
        let rows = topology.v(column, row);
        if column > 0 && rows > topology.v(column - 1, row) {
            let neighbor = self.patches[row * (self.width - 1) + column - 1];
            let subdivisions = topology.v(column - 1, row) as f64;
            snap = snap.max(upper_product(
                neighbor.edge_vv[1],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if column + 1 < self.width - 1 && rows > topology.v(column + 1, row) {
            let neighbor = self.patches[row * (self.width - 1) + column + 1];
            let subdivisions = topology.v(column + 1, row) as f64;
            snap = snap.max(upper_product(
                neighbor.edge_vv[0],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if row > 0 && columns > topology.u(column, row - 1) {
            let neighbor = self.patches[(row - 1) * (self.width - 1) + column];
            let subdivisions = topology.u(column, row - 1) as f64;
            snap = snap.max(upper_product(
                neighbor.edge_uu[1],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if row + 1 < self.height - 1 && columns > topology.u(column, row + 1) {
            let neighbor = self.patches[(row + 1) * (self.width - 1) + column];
            let subdivisions = topology.u(column, row + 1) as f64;
            snap = snap.max(upper_product(
                neighbor.edge_uu[0],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        ErrorBound {
            geometry: upper_sum(base.geometry, snap),
            color: base.color,
        }
    }

    fn score(&self, topology: &TopologyKey) -> ErrorScore {
        self.score_with(topology, Self::patch_error)
    }

    fn base_score(&self, topology: &TopologyKey) -> ErrorScore {
        self.score_with(topology, Self::base_patch_error)
    }

    fn score_with(
        &self,
        topology: &TopologyKey,
        error: fn(&Self, usize, usize, &TopologyKey) -> ErrorBound,
    ) -> ErrorScore {
        let mut bound = ErrorBound {
            geometry: 0.0,
            color: 0.0,
        };
        let mut maximum_severity = 0.0_f64;
        let mut sum = 0.0_f64;
        let mut shape = 0.0_f64;
        let mut worst_patch = [0, 0];
        for row in 0..self.height - 1 {
            for column in 0..self.width - 1 {
                let patch_error = error(self, column, row, topology);
                let severity = patch_error.severity(GEOMETRY_LIMIT);
                if severity > maximum_severity {
                    maximum_severity = severity;
                    worst_patch = [column, row];
                }
                bound.geometry = bound.geometry.max(patch_error.geometry);
                bound.color = bound.color.max(patch_error.color);
                sum += severity;
                let patch = self.patches[row * (self.width - 1) + column];
                let cell_width = patch.extent.x / topology.u(column, row) as f64;
                let cell_height = patch.extent.y / topology.v(column, row) as f64;
                shape += (cell_width - cell_height).abs();
            }
        }
        ErrorScore {
            bound,
            maximum: maximum_severity,
            sum,
            shape,
            worst_patch,
        }
    }

    fn refine(
        &self,
        topology: TopologyKey,
        maximum_error: f64,
        fraction: f64,
        cap: Option<&TopologyKey>,
    ) -> TopologyKey {
        let mut topology = self.refine_base(topology, maximum_error, fraction, cap);
        loop {
            let current = self.score(&topology);
            if current.bound.meets(maximum_error, fraction) {
                return topology;
            }
            let mut best: Option<(TopologyKey, ErrorScore)> = None;
            let [column, row] = current.worst_patch;
            let mut candidates = SmallVec::<[TopologyKey; 6]>::new();
            candidates.extend(
                [
                    topology.doubled_u(column, row),
                    topology.doubled_v(column, row),
                ]
                .into_iter()
                .flatten(),
            );
            let columns = topology.u(column, row);
            let rows = topology.v(column, row);
            if column > 0 && rows > topology.v(column - 1, row) {
                candidates.extend(topology.doubled_v(column - 1, row));
            }
            if column + 1 < self.width - 1 && rows > topology.v(column + 1, row) {
                candidates.extend(topology.doubled_v(column + 1, row));
            }
            if row > 0 && columns > topology.u(column, row - 1) {
                candidates.extend(topology.doubled_u(column, row - 1));
            }
            if row + 1 < self.height - 1 && columns > topology.u(column, row + 1) {
                candidates.extend(topology.doubled_u(column, row + 1));
            }
            for candidate in candidates {
                if candidate.triangles() > MAX_TRIANGLES
                    || cap.is_some_and(|cap| !candidate.within(cap))
                {
                    continue;
                }
                let score = self.score(&candidate);
                let improves_best = best.as_ref().is_none_or(|(best_topology, best_score)| {
                    score
                        .maximum
                        .total_cmp(&best_score.maximum)
                        .then_with(|| score.sum.total_cmp(&best_score.sum))
                        .then_with(|| score.shape.total_cmp(&best_score.shape))
                        .then_with(|| candidate.triangles().cmp(&best_topology.triangles()))
                        .is_lt()
                });
                if improves_best {
                    best = Some((candidate, score));
                }
            }
            let Some((candidate, _)) = best else {
                return topology;
            };
            topology = candidate;
        }
    }

    fn refine_base(
        &self,
        mut topology: TopologyKey,
        maximum_error: f64,
        fraction: f64,
        cap: Option<&TopologyKey>,
    ) -> TopologyKey {
        loop {
            let current = self.base_score(&topology);
            if current.bound.meets(maximum_error, fraction) {
                return topology;
            }
            let [column, row] = current.worst_patch;
            let mut best: Option<(TopologyKey, ErrorScore)> = None;
            for candidate in [
                topology.doubled_u(column, row),
                topology.doubled_v(column, row),
            ]
            .into_iter()
            .flatten()
            {
                if candidate.triangles() > MAX_TRIANGLES
                    || cap.is_some_and(|cap| !candidate.within(cap))
                {
                    continue;
                }
                let score = self.base_score(&candidate);
                let improves_best = best.as_ref().is_none_or(|(best_topology, best_score)| {
                    score
                        .maximum
                        .total_cmp(&best_score.maximum)
                        .then_with(|| score.sum.total_cmp(&best_score.sum))
                        .then_with(|| score.shape.total_cmp(&best_score.shape))
                        .then_with(|| candidate.triangles().cmp(&best_topology.triangles()))
                        .is_lt()
                });
                if improves_best {
                    best = Some((candidate, score));
                }
            }
            let Some((candidate, _)) = best else {
                return topology;
            };
            topology = candidate;
        }
    }

    fn error(&self, topology: &TopologyKey) -> ErrorBound {
        self.score(topology).bound
    }
}

/// Both operands are nonnegative upper bounds. One outward rounding step makes
/// the result an upper bound of their exact IEEE-754 operation.
fn upper_product(left: f64, right: f64) -> f64 {
    (left * right).next_up()
}

fn upper_sum(left: f64, right: f64) -> f64 {
    (left + right).next_up()
}

fn upper_length(value: DVec2) -> f64 {
    Interval::exact(value.x.abs())
        .scale(value.x.abs())
        .add(Interval::exact(value.y.abs()).scale(value.y.abs()))
        .hi
        .sqrt()
        .next_up()
}

fn build_interval_patches<const N: usize, A>(
    values: &[[f64; N]],
    width: usize,
    height: usize,
) -> SmallVec<A>
where
    A: smallvec::Array<Item = IntervalPatch<N>>,
{
    fn sample<const N: usize>(
        values: &[[f64; N]],
        width: usize,
        height: usize,
        x: isize,
        y: isize,
    ) -> IntervalPoint<N> {
        let row = |y: usize| {
            let exact = |x: usize| values[y * width + x].map(Interval::exact);
            if x < 0 {
                core::array::from_fn(|channel| exact(0)[channel].scale(2.0).sub(exact(1)[channel]))
            } else if x >= width as isize {
                core::array::from_fn(|channel| {
                    exact(width - 1)[channel]
                        .scale(2.0)
                        .sub(exact(width - 2)[channel])
                })
            } else {
                exact(x as usize)
            }
        };
        if y < 0 {
            let first = row(0);
            let second = row(1);
            core::array::from_fn(|channel| first[channel].scale(2.0).sub(second[channel]))
        } else if y >= height as isize {
            let last = row(height - 1);
            let previous = row(height - 2);
            core::array::from_fn(|channel| last[channel].scale(2.0).sub(previous[channel]))
        } else {
            row(y as usize)
        }
    }
    fn bezier<const N: usize>(p: [IntervalPoint<N>; 4]) -> [IntervalPoint<N>; 4] {
        [
            p[1],
            core::array::from_fn(|i| p[1][i].add(p[2][i].sub(p[0][i]).sixth())),
            core::array::from_fn(|i| p[2][i].sub(p[3][i].sub(p[1][i]).sixth())),
            p[2],
        ]
    }
    let mut result = SmallVec::with_capacity((width - 1) * (height - 1));
    for row in 0..height - 1 {
        for column in 0..width - 1 {
            let rows: [[IntervalPoint<N>; 4]; 4] = core::array::from_fn(|y| {
                bezier(core::array::from_fn(|x| {
                    sample(
                        values,
                        width,
                        height,
                        column as isize + x as isize - 1,
                        row as isize + y as isize - 1,
                    )
                }))
            });
            let columns: [[IntervalPoint<N>; 4]; 4] =
                core::array::from_fn(|x| bezier(core::array::from_fn(|y| rows[y][x])));
            result.push(core::array::from_fn(|i| columns[i % 4][i / 4]));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_math::Vec2;
    use bevy_ui::{MeshGradientGeometry, MeshGradientPoint};

    fn interval_patches(mesh: &MeshGradient) -> SmallVec<[IntervalPatch<6>; 9]> {
        let values: SmallVec<[Point; 16]> = mesh
            .points()
            .iter()
            .map(|point| {
                let color = match mesh.color_space() {
                    MeshGradientColorSpace::LinearRgba => {
                        LinearRgba::from(point.color).to_f32_array()
                    }
                    MeshGradientColorSpace::Srgba => Srgba::from(point.color).to_f32_array(),
                    MeshGradientColorSpace::Oklaba => Oklaba::from(point.color).to_f32_array(),
                };
                [
                    point.position.x as f64,
                    point.position.y as f64,
                    color[0] as f64,
                    color[1] as f64,
                    color[2] as f64,
                    color[3] as f64,
                ]
            })
            .collect();
        build_interval_patches(&values, mesh.width(), mesh.height())
    }

    fn patches(mesh: &MeshGradient) -> Vec<Patch> {
        interval_patches(mesh)
            .into_iter()
            .map(|patch| patch.map(|p| p.map(|v| v.lo + (v.hi - v.lo) * 0.5)))
            .collect()
    }

    fn mesh(
        size: usize,
        displacement: f32,
        contrast: f32,
        space: MeshGradientColorSpace,
    ) -> MeshGradient {
        let mut points = Vec::new();
        for y in 0..size {
            for x in 0..size {
                let mut position = Vec2::new(x as f32, y as f32) / (size - 1) as f32;
                if x == size / 2 && y == size / 2 && size > 2 {
                    position.x += displacement;
                }
                points.push(MeshGradientPoint::new(
                    position,
                    Color::linear_rgba(position.x * contrast, position.y * contrast, 0.3, 0.7),
                ));
            }
        }
        MeshGradient::new_in_color_space(size, size, points, space).unwrap()
    }

    fn checkerboard(
        contrast: f32,
        space: MeshGradientColorSpace,
        interpolation: MeshGradientColorInterpolation,
    ) -> MeshGradient {
        let points = [
            (Vec2::new(0.0, 0.0), 0.0),
            (Vec2::new(1.0, 0.0), contrast),
            (Vec2::new(0.0, 1.0), contrast),
            (Vec2::new(1.0, 1.0), 0.0),
        ]
        .map(|(position, red)| {
            MeshGradientPoint::new(position, Color::linear_rgba(red, 0.0, 0.0, 1.0))
        });
        MeshGradient::new_with_geometry(
            2,
            2,
            points.into(),
            space,
            MeshGradientGeometry::NonFolding,
        )
        .map(|mesh| mesh.with_color_interpolation(interpolation))
        .unwrap()
    }

    fn axes(size: f64) -> [DVec2; 2] {
        [DVec2::X * size, DVec2::Y * size]
    }

    fn unit_position_bounds() -> VectorBounds {
        VectorBounds {
            x: Interval { lo: 0.0, hi: 1.0 },
            y: Interval { lo: 0.0, hi: 1.0 },
        }
    }

    type ShaderPoint = [f32; 6];

    fn shader_values(mesh: &MeshGradient) -> Vec<ShaderPoint> {
        mesh.points()
            .iter()
            .map(|point| {
                let color = match mesh.color_space() {
                    MeshGradientColorSpace::LinearRgba => {
                        LinearRgba::from(point.color).to_f32_array()
                    }
                    MeshGradientColorSpace::Srgba => Srgba::from(point.color).to_f32_array(),
                    MeshGradientColorSpace::Oklaba => Oklaba::from(point.color).to_f32_array(),
                };
                [
                    point.position.x,
                    point.position.y,
                    color[0],
                    color[1],
                    color[2],
                    color[3],
                ]
            })
            .collect()
    }

    fn shader_control(
        values: &[ShaderPoint],
        width: usize,
        height: usize,
        x: isize,
        y: isize,
    ) -> ShaderPoint {
        fn row(values: &[ShaderPoint], width: usize, x: isize, y: usize) -> ShaderPoint {
            let at = |x| values[y * width + x];
            if x < 0 {
                core::array::from_fn(|channel| 2.0 * at(0)[channel] - at(1)[channel])
            } else if x >= width as isize {
                core::array::from_fn(|channel| {
                    2.0 * at(width - 1)[channel] - at(width - 2)[channel]
                })
            } else {
                at(x as usize)
            }
        }
        if y < 0 {
            let a = row(values, width, x, 0);
            let b = row(values, width, x, 1);
            core::array::from_fn(|channel| 2.0 * a[channel] - b[channel])
        } else if y >= height as isize {
            let a = row(values, width, x, height - 1);
            let b = row(values, width, x, height - 2);
            core::array::from_fn(|channel| 2.0 * a[channel] - b[channel])
        } else {
            row(values, width, x, y as usize)
        }
    }

    fn shader_cubic(
        p0: ShaderPoint,
        p1: ShaderPoint,
        p2: ShaderPoint,
        p3: ShaderPoint,
        t: f32,
    ) -> ShaderPoint {
        if t == 0.0 {
            return p1;
        }
        if t == 1.0 {
            return p2;
        }
        core::array::from_fn(|channel| {
            0.5 * (2.0 * p1[channel]
                + (-p0[channel] + p2[channel]) * t
                + (2.0 * p0[channel] - 5.0 * p1[channel] + 4.0 * p2[channel] - p3[channel]) * t * t
                + (-p0[channel] + 3.0 * p1[channel] - 3.0 * p2[channel] + p3[channel]) * t * t * t)
        })
    }

    fn shader_surface(mesh: &MeshGradient, column: usize, row: usize, uv: Vec2) -> ShaderPoint {
        let values = shader_values(mesh);
        let rows: [ShaderPoint; 4] = core::array::from_fn(|y| {
            shader_cubic(
                shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize - 1,
                    row as isize + y as isize - 1,
                ),
                shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize,
                    row as isize + y as isize - 1,
                ),
                shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize + 1,
                    row as isize + y as isize - 1,
                ),
                shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize + 2,
                    row as isize + y as isize - 1,
                ),
                uv.x,
            )
        });
        shader_cubic(rows[0], rows[1], rows[2], rows[3], uv.y)
    }

    #[test]
    fn shader_f32_surface_agrees_with_cpu_reference_and_shares_exact_edges() {
        for space in [
            MeshGradientColorSpace::LinearRgba,
            MeshGradientColorSpace::Srgba,
            MeshGradientColorSpace::Oklaba,
        ] {
            for size in [2, 3, 16] {
                let grid = mesh(size, 0.02 / (size - 1) as f32, 4.0, space);
                let reference = patches(&grid);
                for row in 0..size - 1 {
                    for column in 0..size - 1 {
                        let patch = &reference[row * (size - 1) + column];
                        for uv in [
                            Vec2::ZERO,
                            Vec2::new(0.125, 0.75),
                            Vec2::new(0.5, 0.5),
                            Vec2::new(0.875, 0.25),
                            Vec2::ONE,
                        ] {
                            let gpu = shader_surface(&grid, column, row, uv);
                            let cpu = evaluate(patch, uv.as_dvec2());
                            for channel in 0..6 {
                                let tolerance = 2e-5 * (1.0 + cpu[channel].abs());
                                assert!(
                                    (f64::from(gpu[channel]) - cpu[channel]).abs() <= tolerance,
                                    "{space:?} {size}x{size} patch ({column},{row}) {uv:?} channel {channel}: gpu={} cpu={}",
                                    gpu[channel],
                                    cpu[channel],
                                );
                            }
                        }
                        if column + 1 < size - 1 {
                            let left = shader_surface(&grid, column, row, Vec2::new(1.0, 0.375));
                            let right =
                                shader_surface(&grid, column + 1, row, Vec2::new(0.0, 0.375));
                            assert_eq!(left.map(f32::to_bits), right.map(f32::to_bits));
                        }
                        if row + 1 < size - 1 {
                            let top = shader_surface(&grid, column, row, Vec2::new(0.625, 1.0));
                            let bottom =
                                shader_surface(&grid, column, row + 1, Vec2::new(0.625, 0.0));
                            assert_eq!(top.map(f32::to_bits), bottom.map(f32::to_bits));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn affine_minimum_and_curvature_resize_promotion() {
        let affine = SurfaceBounds::new(&mesh(2, 0.0, 1.0, MeshGradientColorSpace::LinearRgba));
        assert_eq!(
            QualityState::default()
                .update(&affine, axes(4096.0))
                .key
                .maximum_subdivisions(),
            MIN_SUBDIVISIONS
        );
        let curved = SurfaceBounds::new(&mesh(3, 0.04, 1.0, MeshGradientColorSpace::LinearRgba));
        let mut state = QualityState::default();
        let small = state.update(&curved, axes(256.0));
        let large = state.update(&curved, axes(4096.0));
        assert!(small.key.maximum_subdivisions() > 1);
        assert!(large.key.triangles() > small.key.triangles());
        assert!(!large.capped);
    }

    #[test]
    fn vertex_color_error_refines_only_resolvable_mixed_color() {
        for space in [
            MeshGradientColorSpace::LinearRgba,
            MeshGradientColorSpace::Srgba,
            MeshGradientColorSpace::Oklaba,
        ] {
            let vivid = SurfaceBounds::new(&checkerboard(
                1.0,
                space,
                MeshGradientColorInterpolation::Vertex,
            ));
            let chosen = QualityState::default().update(&vivid, axes(512.0));
            assert_eq!(chosen.key.u(0, 0), 4);
            assert_eq!(chosen.key.v(0, 0), 4);
            assert!(chosen.error.color <= COLOR_LIMIT.next_up());

            let subpixel = QualityState::default().update(&vivid, axes(1.0));
            assert_eq!(subpixel.key.maximum_subdivisions(), MIN_SUBDIVISIONS);

            let quiet = SurfaceBounds::new(&checkerboard(
                0.001,
                space,
                MeshGradientColorInterpolation::Vertex,
            ));
            assert_eq!(
                QualityState::default()
                    .update(&quiet, axes(512.0))
                    .key
                    .maximum_subdivisions(),
                MIN_SUBDIVISIONS
            );

            let bicubic = SurfaceBounds::new(&checkerboard(
                1.0,
                space,
                MeshGradientColorInterpolation::Bicubic,
            ));
            assert_eq!(
                QualityState::default()
                    .update(&bicubic, axes(512.0))
                    .key
                    .maximum_subdivisions(),
                MIN_SUBDIVISIONS
            );
        }
    }

    #[test]
    fn bicubic_color_does_not_drive_geometry_tessellation_and_scale_is_physical() {
        assert_eq!(
            physical_axes(Vec2::splat(256.0), Mat2::IDENTITY, 2.0),
            axes(512.0)
        );
        let mut quiet_mesh = mesh(3, 0.04, 0.0, MeshGradientColorSpace::LinearRgba);
        quiet_mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        let mut vivid_mesh = mesh(3, 0.04, 8.0, MeshGradientColorSpace::LinearRgba);
        vivid_mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        let quiet = SurfaceBounds::new(&quiet_mesh);
        let vivid = SurfaceBounds::new(&vivid_mesh);
        let a = QualityState::default().update(&quiet, axes(1.0));
        let b = QualityState::default().update(&vivid, axes(1.0));
        assert_eq!(b.key, a.key);
        let topology = TopologyKey::uniform(3, 3, 8);
        let rotated = [DVec2::Y * 512.0, -DVec2::X * 512.0];
        assert_eq!(
            vivid.error(&topology, axes(512.0)).geometry,
            vivid.error(&topology, rotated).geometry
        );
        assert_eq!(
            vivid.error(&topology, axes(256.0)).geometry * 2.0,
            vivid.error(&topology, axes(512.0)).geometry
        );
        assert!(
            vivid
                .error(&topology, [DVec2::new(512.0, 512.0), DVec2::Y * 512.0])
                .geometry
                > vivid.error(&topology, axes(512.0)).geometry
        );
    }

    #[test]
    fn demotion_waits_eight_consecutive_frames() {
        let curved = SurfaceBounds::new(&mesh(3, 0.04, 0.0, MeshGradientColorSpace::LinearRgba));
        let flat = SurfaceBounds::new(&mesh(3, 0.0, 0.0, MeshGradientColorSpace::LinearRgba));
        let mut state = QualityState::default();
        let high = state.update(&curved, axes(4096.0)).key;
        for _ in 0..7 {
            assert_eq!(state.update(&flat, axes(256.0)).key, high);
        }
        state.update(&curved, axes(4096.0));
        for _ in 0..7 {
            assert_eq!(state.update(&flat, axes(256.0)).key, high);
        }
        assert_eq!(
            state.update(&flat, axes(256.0)).key.maximum_subdivisions(),
            MIN_SUBDIVISIONS
        );
    }

    #[test]
    fn demotion_can_choose_an_intermediate_safe_tier() {
        let bounds = SurfaceBounds {
            width: 2,
            height: 2,
            patches: Box::new([PatchBounds {
                position: unit_position_bounds(),
                uu: VectorBounds::exact(DVec2::new(8.0, 0.0)),
                uv: VectorBounds::exact(DVec2::ZERO),
                vv: VectorBounds::exact(DVec2::ZERO),
                edge_uu: [VectorBounds::exact(DVec2::new(8.0, 0.0)); 2],
                edge_vv: [VectorBounds::exact(DVec2::ZERO); 2],
                color_uv: 0.0,
            }]),
        };
        let mut state = QualityState::default();
        assert_eq!(
            state
                .update(&bounds, axes(500.0))
                .key
                .maximum_subdivisions(),
            16
        );
        // Four subdivisions meets the full limit, but only eight meets half.
        for _ in 0..7 {
            assert_eq!(
                state.update(&bounds, axes(50.0)).key.maximum_subdivisions(),
                16
            );
        }
        assert_eq!(
            state.update(&bounds, axes(50.0)).key.maximum_subdivisions(),
            8
        );
    }

    #[test]
    fn alpha_alone_does_not_promote_geometry_quality() {
        let mut grid = mesh(3, 0.02, 0.0, MeshGradientColorSpace::LinearRgba);
        let quiet = QualityState::default().update(&SurfaceBounds::new(&grid), axes(1.0));
        grid.try_edit_points(|points| {
            for point in points {
                point.color = Color::linear_rgba(0.0, 0.0, 0.0, point.position.x);
            }
            Ok(())
        })
        .unwrap();
        let alpha = QualityState::default().update(&SurfaceBounds::new(&grid), axes(1.0));
        assert_eq!(alpha.key, quiet.key);
    }

    #[test]
    fn cap_is_visible_bounded_and_reported_once() {
        let bounds = SurfaceBounds::new(&mesh(16, 0.005, 1.0, MeshGradientColorSpace::LinearRgba));
        let mut state = QualityState::default();
        let selection = state.update(&bounds, axes(1e10));
        assert!(selection.capped && selection.report_cap);
        assert!(selection.key.triangles() <= MAX_TRIANGLES);
        assert!(selection.key.maximum_subdivisions() <= MAX_SUBDIVISIONS);
        assert!(!state.update(&bounds, axes(1e10)).report_cap);
        state.update(&bounds, axes(1.0));
        assert!(!state.update(&bounds, axes(1e10)).report_cap);
    }

    #[test]
    fn hdr_color_changes_reuse_geometry_topology() {
        let mut hdr = mesh(3, 0.02, 1.0, MeshGradientColorSpace::LinearRgba);
        hdr.try_edit_points(|points| {
            for point in points {
                let p = point.position;
                point.color = Color::linear_rgba(100.0 + p.x, -100.0 - p.y, 0.5, 1.0);
            }
            Ok(())
        })
        .unwrap();
        let hdr_key = QualityState::default()
            .update(&SurfaceBounds::new(&hdr), axes(4096.0))
            .key;
        let ordinary_key = QualityState::default()
            .update(
                &SurfaceBounds::new(&mesh(3, 0.02, 1.0, MeshGradientColorSpace::LinearRgba)),
                axes(4096.0),
            )
            .key;
        assert_eq!(hdr_key, ordinary_key);
    }

    #[test]
    fn point_edits_reuse_selected_topology_and_grid_changes_reset_state() {
        let mut grid = mesh(3, 0.02, 1.0, MeshGradientColorSpace::LinearRgba);
        let mut state = QualityState::default();
        let mut cache = TopologyCache::default();
        let first = state.update(&SurfaceBounds::new(&grid), axes(512.0));
        let topology = cache.get(&first.key);
        grid.try_set_position(4, Vec2::new(0.52001, 0.5)).unwrap();
        let next = state.update(&SurfaceBounds::new(&grid), axes(512.0));
        assert!(Arc::ptr_eq(&topology, &cache.get(&next.key)));
        let replacement = mesh(16, 0.0, 1.0, MeshGradientColorSpace::LinearRgba);
        let changed = state.update(&SurfaceBounds::new(&replacement), axes(512.0));
        assert_eq!(changed.key.width, 16);
        assert_eq!(changed.key.maximum_subdivisions(), MIN_SUBDIVISIONS);
    }

    #[test]
    fn topology_reuses_point_edits_and_releases_unused_entries() {
        assert_eq!(size_of::<ParameterVertex>(), 8);
        let mut cache = TopologyCache::default();
        let key = TopologyKey::uniform(3, 3, 8);
        let first = cache.get(&key);
        let second = cache.get(&key);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.indices.len() / 3, key.triangles());
        let stride = 9 * 9;
        for y in 0..=8 {
            let left = first.vertices[y * 9 + 8];
            let right = first.vertices[stride + y * 9];
            let (left_patch, left_uv, _, _) = left.unpack();
            let (right_patch, right_uv, _, _) = right.unpack();
            assert_eq!(
                left_patch[0] as f32 + left_uv[0],
                right_patch[0] as f32 + right_uv[0]
            );
            assert_eq!(left_uv[1].to_bits(), right_uv[1].to_bits());
        }
        cache.prune();
        assert_eq!(cache.entries.len(), 1);
        drop(first);
        drop(second);
        cache.prune();
        assert!(cache.entries.is_empty());
    }

    #[test]
    fn localized_curvature_refines_only_the_axes_that_need_it() {
        let bounds = SurfaceBounds {
            width: 3,
            height: 2,
            patches: Box::new([
                PatchBounds {
                    position: unit_position_bounds(),
                    uu: VectorBounds::exact(DVec2::new(1.0, 0.0)),
                    uv: VectorBounds::exact(DVec2::ZERO),
                    vv: VectorBounds::exact(DVec2::ZERO),
                    edge_uu: [VectorBounds::exact(DVec2::new(1.0, 0.0)); 2],
                    edge_vv: [VectorBounds::exact(DVec2::ZERO); 2],
                    color_uv: 0.0,
                },
                PatchBounds {
                    position: unit_position_bounds(),
                    uu: VectorBounds::exact(DVec2::ZERO),
                    uv: VectorBounds::exact(DVec2::ZERO),
                    vv: VectorBounds::exact(DVec2::ZERO),
                    edge_uu: [VectorBounds::exact(DVec2::ZERO); 2],
                    edge_vv: [VectorBounds::exact(DVec2::ZERO); 2],
                    color_uv: 0.0,
                },
            ]),
        };
        let selection = QualityState::default().update(&bounds, axes(1024.0));

        assert_eq!(selection.key.u(0, 0), 8);
        assert_eq!(selection.key.u(1, 0), MIN_SUBDIVISIONS);
        assert_eq!(selection.key.v(0, 0), MIN_SUBDIVISIONS);
        assert_eq!(selection.key.triangles(), 40);
        assert_eq!(TopologyKey::uniform(3, 2, 8).triangles(), 256);
        assert!(!selection.capped);
    }

    #[test]
    fn nonuniform_topology_snaps_fine_edges_to_coarse_chords() {
        let key = TopologyKey::uniform(3, 3, 1)
            .doubled_u(0, 0)
            .unwrap()
            .doubled_u(0, 0)
            .unwrap()
            .doubled_v(0, 0)
            .unwrap()
            .doubled_v(0, 0)
            .unwrap();
        let topology = TopologyCache::default().get(&key);
        let fine_right: Vec<_> = topology
            .vertices
            .iter()
            .filter(|vertex| {
                let (patch, uv, _, _) = vertex.unpack();
                patch == [0, 0] && uv[0] == 1.0 && uv[1] > 0.0 && uv[1] < 1.0
            })
            .collect();
        assert_eq!(fine_right.len(), 3);
        assert!(fine_right.iter().all(|vertex| {
            let (_, _, subdivisions, position_subdivisions) = vertex.unpack();
            subdivisions == [4, 4] && position_subdivisions == [4, 1]
        }));

        let fine_bottom: Vec<_> = topology
            .vertices
            .iter()
            .filter(|vertex| {
                let (patch, uv, _, _) = vertex.unpack();
                patch == [0, 0] && uv[1] == 1.0 && uv[0] > 0.0 && uv[0] < 1.0
            })
            .collect();
        assert_eq!(fine_bottom.len(), 3);
        assert!(fine_bottom.iter().all(|vertex| {
            let (_, _, subdivisions, position_subdivisions) = vertex.unpack();
            subdivisions == [4, 4] && position_subdivisions == [1, 4]
        }));

        assert!(topology
            .vertices
            .iter()
            .filter(|vertex| {
                let (patch, uv, _, _) = vertex.unpack();
                (patch == [1, 0] && uv[0] == 0.0) || (patch == [0, 1] && uv[1] == 0.0)
            })
            .all(|vertex| {
                let (_, _, subdivisions, position_subdivisions) = vertex.unpack();
                position_subdivisions == subdivisions
            }));
    }

    #[test]
    fn folded_editor_shape_uses_less_than_a_global_maximum_grid() {
        let mut points: Vec<_> = (0..20)
            .map(|index| {
                let column = index % 5;
                let row = index / 5;
                let position = Vec2::new(column as f32 / 4.0, row as f32 / 3.0);
                MeshGradientPoint::new(position, Color::WHITE)
            })
            .collect();
        points[7].position = Vec2::new(0.57, 0.405);
        points[8].position = Vec2::new(0.64, 0.233);
        points[12].position = Vec2::new(0.393, 0.49);
        let mesh = MeshGradient::new_with_geometry(
            5,
            4,
            points,
            MeshGradientColorSpace::Oklaba,
            MeshGradientGeometry::AllowFolds,
        )
        .unwrap();
        let selection = QualityState::default().update(
            &SurfaceBounds::new(&mesh),
            [DVec2::X * 1_125.0, DVec2::Y * 866.0],
        );
        let global = TopologyKey::uniform(5, 4, selection.key.maximum_subdivisions());

        assert!(
            selection.key.triangles() * 4 <= global.triangles() * 3,
            "adaptive={} global={} max={} error={:?}",
            selection.key.triangles(),
            global.triangles(),
            selection.key.maximum_subdivisions(),
            selection.error,
        );
        assert!(selection.key.triangles() <= MAX_TRIANGLES);
        assert!(selection.error.geometry <= GEOMETRY_LIMIT);
        assert!(!selection.capped);
    }

    fn evaluate(patch: &Patch, uv: DVec2) -> Point {
        fn basis(t: f64) -> [f64; 4] {
            [
                (1.0 - t).powi(3),
                3.0 * t * (1.0 - t).powi(2),
                3.0 * t * t * (1.0 - t),
                t.powi(3),
            ]
        }
        let x = basis(uv.x);
        let y = basis(uv.y);
        core::array::from_fn(|channel| {
            (0..16)
                .map(|i| patch[i][channel] * x[i % 4] * y[i / 4])
                .sum()
        })
    }

    fn evaluate_topology_vertex(
        patch: &Patch,
        uv: DVec2,
        column: usize,
        row: usize,
        topology: &TopologyKey,
    ) -> Point {
        let columns = topology.u(column, row);
        let rows = topology.v(column, row);
        let mut position_columns = columns;
        let mut position_rows = rows;
        if uv.y > 0.0 && uv.y < 1.0 {
            if uv.x == 0.0 && column > 0 {
                position_rows = position_rows.min(topology.v(column - 1, row));
            }
            if uv.x == 1.0 && column + 1 < topology.width - 1 {
                position_rows = position_rows.min(topology.v(column + 1, row));
            }
        }
        if uv.x > 0.0 && uv.x < 1.0 {
            if uv.y == 0.0 && row > 0 {
                position_columns = position_columns.min(topology.u(column, row - 1));
            }
            if uv.y == 1.0 && row + 1 < topology.height - 1 {
                position_columns = position_columns.min(topology.u(column, row + 1));
            }
        }
        let (start, end, weight) = if position_columns < columns {
            let grid = uv.x * position_columns as f64;
            let segment = grid.floor();
            (
                uv.with_x(segment / position_columns as f64),
                uv.with_x((segment + 1.0) / position_columns as f64),
                grid.fract(),
            )
        } else if position_rows < rows {
            let grid = uv.y * position_rows as f64;
            let segment = grid.floor();
            (
                uv.with_y(segment / position_rows as f64),
                uv.with_y((segment + 1.0) / position_rows as f64),
                grid.fract(),
            )
        } else {
            return evaluate(patch, uv);
        };
        let start = evaluate(patch, start);
        let end = evaluate(patch, end);
        core::array::from_fn(|channel| start[channel] + (end[channel] - start[channel]) * weight)
    }

    #[test]
    fn dense_geometry_reference_is_below_estimator() {
        for size in [2, 3, 4, 16] {
            let grid = mesh(
                size,
                0.04 / (size - 1) as f32,
                1.0,
                MeshGradientColorSpace::LinearRgba,
            );
            let bounds = SurfaceBounds::new(&grid);
            for screen in [256.0, 1024.0, 4096.0] {
                for screen_axes in [
                    axes(screen),
                    [
                        DVec2::new(screen, screen * 0.25),
                        DVec2::new(-screen * 0.2, screen * 0.8),
                    ],
                ] {
                    let chosen = QualityState::default().update(&bounds, screen_axes);
                    assert!(
                        !chosen.capped,
                        "size={size}, screen={screen}, error={:?}",
                        chosen.error
                    );
                    for (patch_index, patch) in patches(&grid).into_iter().enumerate() {
                        let column = patch_index % (size - 1);
                        let row = patch_index / (size - 1);
                        let columns = chosen.key.u(column, row);
                        let rows = chosen.key.v(column, row);
                        let step = DVec2::new(1.0 / columns as f64, 1.0 / rows as f64);
                        for y in 0..rows {
                            for x in 0..columns {
                                for offset in [
                                    DVec2::new(0.2, 0.7),
                                    DVec2::new(0.7, 0.2),
                                    DVec2::splat(0.5),
                                ] {
                                    let a = DVec2::new(
                                        x as f64 / columns as f64,
                                        y as f64 / rows as f64,
                                    );
                                    let p = a + offset * step;
                                    let q = evaluate(&patch, p);
                                    let corners = if offset.x >= offset.y {
                                        [
                                            (a, 1.0 - offset.x),
                                            (a + DVec2::X * step.x, offset.x - offset.y),
                                            (a + step, offset.y),
                                        ]
                                    } else {
                                        [
                                            (a, 1.0 - offset.y),
                                            (a + step, offset.x),
                                            (a + DVec2::Y * step.y, offset.y - offset.x),
                                        ]
                                    };
                                    let approximate: DVec2 = corners
                                        .into_iter()
                                        .map(|(uv, w)| {
                                            let q = evaluate_topology_vertex(
                                                &patch,
                                                uv,
                                                column,
                                                row,
                                                &chosen.key,
                                            );
                                            DVec2::new(q[0], q[1]) * w
                                        })
                                        .sum();
                                    let delta = DVec2::new(q[0], q[1]) - approximate;
                                    let error = (screen_axes[0] * delta.x
                                        + screen_axes[1] * delta.y)
                                        .length();
                                    assert!(error <= chosen.error.geometry + 1e-9);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
