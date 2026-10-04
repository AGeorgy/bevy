//! Internal adaptive mesh-gradient preparation. The renderer supplies physical
//! screen axes from the already-physical node size and its transform.
//!
//! Geometry uses patch-local Hessian remainder bounds for linear triangular
//! interpolation. Cartesian colors are evaluated at tessellation vertices;
//! hue-based bilinear colors and bicubic colors are evaluated per fragment.
//! Each patch has independent
//! power-of-two factors; finer boundary vertices snap to the coarser edge
//! approximation so neighboring patches remain crack-free without propagating
//! refinement. Vertex color mode adds an exact bilinear color-error term, so it
//! spends triangles only where the rasterizer would reveal a patch diagonal.

use bevy_math::{DVec2, Mat2, Vec2};
use bevy_platform::sync::Arc;
use bevy_ui::{MeshGradient, MeshGradientColorInterpolation};
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

#[cfg(test)]
type Point = [f64; 6];
#[cfg(test)]
type Patch = [Point; 16];

/// `node_size` is already measured in physical pixels by UI layout.
/// Translation does not affect interpolation error and is deliberately absent.
pub(crate) fn compute_physical_axes(node_size: Vec2, transform: Mat2) -> [DVec2; 2] {
    [
        transform.x_axis.as_dvec2() * node_size.x as f64,
        transform.y_axis.as_dvec2() * node_size.y as f64,
    ]
}

#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn from_exact_value(value: f64) -> Self {
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
    fn divide_by_six(self) -> Self {
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
    fn new_uniform(width: usize, height: usize, subdivisions: usize) -> Self {
        let factors = core::iter::repeat_n(
            Self::pack_subdivisions(subdivisions, subdivisions),
            (width - 1) * (height - 1),
        )
        .collect();
        Self {
            width,
            height,
            factors,
        }
    }

    const fn pack_subdivisions(u: usize, v: usize) -> u16 {
        u as u16 | ((v as u16) << 8)
    }

    fn compute_patch_index(&self, column: usize, row: usize) -> usize {
        row * (self.width - 1) + column
    }

    fn u_subdivisions(&self, column: usize, row: usize) -> usize {
        (self.factors[self.compute_patch_index(column, row)] & 0xff) as usize
    }

    fn v_subdivisions(&self, column: usize, row: usize) -> usize {
        (self.factors[self.compute_patch_index(column, row)] >> 8) as usize
    }

    fn has_same_dimensions(&self, other: &Self) -> bool {
        self.width == other.width && self.height == other.height
    }

    fn try_double_u_subdivisions(&self, column: usize, row: usize) -> Option<Self> {
        let subdivisions = self.u_subdivisions(column, row);
        if subdivisions >= MAX_SUBDIVISIONS {
            return None;
        }
        let mut next = self.clone();
        let index = next.compute_patch_index(column, row);
        next.factors[index] =
            Self::pack_subdivisions(subdivisions * 2, self.v_subdivisions(column, row));
        Some(next)
    }

    fn try_double_v_subdivisions(&self, column: usize, row: usize) -> Option<Self> {
        let subdivisions = self.v_subdivisions(column, row);
        if subdivisions >= MAX_SUBDIVISIONS {
            return None;
        }
        let mut next = self.clone();
        let index = next.compute_patch_index(column, row);
        next.factors[index] =
            Self::pack_subdivisions(self.u_subdivisions(column, row), subdivisions * 2);
        Some(next)
    }

    fn fits_within(&self, cap: &Self) -> bool {
        self.has_same_dimensions(cap)
            && self
                .factors
                .iter()
                .zip(&cap.factors)
                .all(|(factor, cap)| (factor & 0xff) <= (cap & 0xff) && (factor >> 8) <= (cap >> 8))
    }

    pub fn find_maximum_subdivisions(&self) -> usize {
        self.factors
            .iter()
            .flat_map(|factor| [factor & 0xff, factor >> 8])
            .max()
            .unwrap_or(1) as usize
    }

    pub fn count_triangles(&self) -> usize {
        self.factors
            .iter()
            .map(|factor| 2 * usize::from(factor & 0xff) * usize::from(factor >> 8))
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
                u32::from_le_bytes([
                    patch[0] as u8,
                    patch[1] as u8,
                    numerator[0] as u8,
                    numerator[1] as u8,
                ]),
                u32::from_le_bytes([
                    subdivisions[0] as u8,
                    subdivisions[1] as u8,
                    position_subdivisions[0] as u8,
                    position_subdivisions[1] as u8,
                ]),
            ],
        }
    }

    #[cfg(test)]
    fn unpack(self) -> ([u32; 2], [f32; 2], [u32; 2], [u32; 2]) {
        let first = self.packed[0].to_le_bytes().map(u32::from);
        let second = self.packed[1].to_le_bytes().map(u32::from);
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

pub(crate) struct ParameterTopology {
    pub vertices: Vec<ParameterVertex>,
    pub indices: Vec<u32>,
}

impl ParameterTopology {
    pub fn new(key: &TopologyKey) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::with_capacity(key.count_triangles() * 3);
        for row in 0..key.height - 1 {
            for column in 0..key.width - 1 {
                let columns = key.u_subdivisions(column, row);
                let rows = key.v_subdivisions(column, row);
                let base = vertices.len() as u32;
                for y in 0..=rows {
                    for x in 0..=columns {
                        let mut position_columns = columns;
                        let mut position_rows = rows;
                        if y > 0 && y < rows {
                            if x == 0 && column > 0 {
                                position_rows =
                                    position_rows.min(key.v_subdivisions(column - 1, row));
                            }
                            if x == columns && column + 1 < key.width - 1 {
                                position_rows =
                                    position_rows.min(key.v_subdivisions(column + 1, row));
                            }
                        }
                        if x > 0 && x < columns {
                            if y == 0 && row > 0 {
                                position_columns =
                                    position_columns.min(key.u_subdivisions(column, row - 1));
                            }
                            if y == rows && row + 1 < key.height - 1 {
                                position_columns =
                                    position_columns.min(key.u_subdivisions(column, row + 1));
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
        Self { vertices, indices }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ErrorBound {
    pub geometry: f64,
    pub color: f64,
}

impl ErrorBound {
    fn meets_tolerance(self, fraction: f64) -> bool {
        self.geometry <= GEOMETRY_LIMIT * fraction
            && self.color <= multiply_round_up(COLOR_LIMIT, fraction)
    }

    fn compute_severity(self) -> f64 {
        (self.geometry / GEOMETRY_LIMIT).max(self.color / COLOR_LIMIT)
    }
}

pub(crate) struct QualitySelection {
    pub key: TopologyKey,
    pub error: ErrorBound,
    /// Emit a diagnostic on entry to the capped state, not on every frame.
    pub report_cap: bool,
}

#[derive(Default)]
pub(crate) struct QualityState {
    key: Option<TopologyKey>,
    below_half: u8,
    capped: bool,
    report_cooldown: u8,
    screen_bounds: ScreenBounds,
    cached_input: Option<(Arc<SurfaceBounds>, [DVec2; 2])>,
    cached_error: Option<ErrorBound>,
    cached_demotion: Option<(TopologyKey, ErrorBound)>,
}

impl QualityState {
    pub fn update(
        &mut self,
        bounds: &Arc<SurfaceBounds>,
        screen_axes: [DVec2; 2],
    ) -> QualitySelection {
        self.report_cooldown = self.report_cooldown.saturating_sub(1);
        let unchanged = self
            .cached_input
            .as_ref()
            .is_some_and(|(previous, axes)| Arc::ptr_eq(previous, bounds) && *axes == screen_axes);
        if !unchanged {
            bounds.update_screen(screen_axes, &mut self.screen_bounds);
            self.cached_input = Some((Arc::clone(bounds), screen_axes));
            self.cached_error = None;
        }
        let bounds = &self.screen_bounds;
        let (chosen, error);
        if let Some(previous) = self
            .key
            .as_ref()
            .filter(|key| key.width == bounds.width && key.height == bounds.height)
        {
            let previous_error = self
                .cached_error
                .unwrap_or_else(|| bounds.estimate_error_bound(previous));
            if !previous_error.meets_tolerance(1.0) {
                // An unchanged capped selection has already exhausted its
                // permitted refinements. Its diagnostic still advances below.
                chosen = if unchanged {
                    previous.clone()
                } else {
                    bounds.refine(previous.clone(), 1.0, None)
                };
                error = if chosen == *previous {
                    previous_error
                } else {
                    bounds.estimate_error_bound(&chosen)
                };
                self.below_half = 0;
                self.cached_demotion = None;
            } else {
                let (candidate, candidate_error) = match self.cached_demotion.take() {
                    Some(cached) if unchanged => cached,
                    cached => {
                        let minimum =
                            TopologyKey::new_uniform(bounds.width, bounds.height, MIN_SUBDIVISIONS);
                        let candidate = bounds.refine(minimum, 0.5, Some(previous));
                        let error = bounds.estimate_error_bound(&candidate);
                        if cached.as_ref().map(|(previous, _)| previous) != Some(&candidate) {
                            self.below_half = 0;
                        }
                        (candidate, error)
                    }
                };
                if candidate.count_triangles() < previous.count_triangles()
                    && candidate_error.meets_tolerance(0.5)
                {
                    self.below_half += 1;
                } else {
                    self.below_half = 0;
                }
                if self.below_half < DEMOTION_FRAMES {
                    chosen = previous.clone();
                    error = previous_error;
                } else {
                    chosen = candidate.clone();
                    error = candidate_error;
                    self.below_half = 0;
                }
                self.cached_demotion = Some((candidate, candidate_error));
            }
        } else {
            let minimum = TopologyKey::new_uniform(bounds.width, bounds.height, MIN_SUBDIVISIONS);
            chosen = bounds.refine(minimum, 1.0, None);
            error = bounds.estimate_error_bound(&chosen);
            self.below_half = 0;
        }
        let capped = !error.meets_tolerance(1.0);
        let report_cap = capped && !self.capped && self.report_cooldown == 0;
        if report_cap {
            self.report_cooldown = 120;
        }
        self.capped = capped;
        if self.key.as_ref() != Some(&chosen) {
            self.cached_demotion = None;
        }
        self.key = Some(chosen.clone());
        self.cached_error = Some(error);
        QualitySelection {
            key: chosen,
            error,
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
    fn new_empty() -> Self {
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
    fn from_exact_value(value: DVec2) -> Self {
        Self {
            x: Interval::from_exact_value(value.x),
            y: Interval::from_exact_value(value.y),
        }
    }

    fn include(&mut self, value: [Interval; 2]) {
        self.x.lo = self.x.lo.min(value[0].lo);
        self.x.hi = self.x.hi.max(value[0].hi);
        self.y.lo = self.y.lo.min(value[1].lo);
        self.y.hi = self.y.hi.max(value[1].hi);
    }

    fn compute_transformed_length_bound(self, axes: [DVec2; 2]) -> f64 {
        let mut maximum = 0.0_f64;
        for x in [self.x.lo, self.x.hi] {
            for y in [self.y.lo, self.y.hi] {
                let screen_x = Interval::from_exact_value(x)
                    .scale(axes[0].x)
                    .add(Interval::from_exact_value(y).scale(axes[1].x));
                let screen_y = Interval::from_exact_value(x)
                    .scale(axes[0].y)
                    .add(Interval::from_exact_value(y).scale(axes[1].y));
                for x in [screen_x.lo, screen_x.hi] {
                    for y in [screen_y.lo, screen_y.hi] {
                        maximum = maximum.max(compute_length_upper_bound(DVec2::new(x, y)));
                    }
                }
            }
        }
        maximum
    }

    fn compute_transformed_extent(self, axes: [DVec2; 2]) -> DVec2 {
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
        let colors: Option<SmallVec<[[f64; 4]; 16]>> = (mesh.color_interpolation()
            == MeshGradientColorInterpolation::Vertex
            && !mesh.color_space().is_hue_based())
        .then(|| {
            mesh.points()
                .iter()
                .map(|point| mesh.color_space().to_components(point.color).map(f64::from))
                .collect()
        });
        let interval_patches: SmallVec<[IntervalPatch<2>; 9]> =
            build_interval_patches(&values, width, height);
        let patches = interval_patches
            .iter()
            .enumerate()
            .map(|(patch_index, patch)| {
                let mut position = VectorBounds::new_empty();
                let mut uu = VectorBounds::new_empty();
                let mut uv = VectorBounds::new_empty();
                let mut vv = VectorBounds::new_empty();
                let mut edge_uu = [VectorBounds::new_empty(); 2];
                let mut edge_vv = [VectorBounds::new_empty(); 2];
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
                            let mixed = Interval::from_exact_value(top_left[channel])
                                .sub(Interval::from_exact_value(top_right[channel]))
                                .sub(Interval::from_exact_value(bottom_left[channel]))
                                .add(Interval::from_exact_value(bottom_right[channel]));
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
        screen.patches.extend(self.patches.iter().map(|patch| {
            ScreenPatchBounds {
                extent: if patch.color_uv > 0.0 {
                    patch.position.compute_transformed_extent(axes)
                } else {
                    DVec2::ZERO
                },
                uu: patch.uu.compute_transformed_length_bound(axes),
                uv: patch.uv.compute_transformed_length_bound(axes),
                vv: patch.vv.compute_transformed_length_bound(axes),
                edge_uu: patch
                    .edge_uu
                    .map(|bound| bound.compute_transformed_length_bound(axes)),
                edge_vv: patch
                    .edge_vv
                    .map(|bound| bound.compute_transformed_length_bound(axes)),
                color_uv: patch.color_uv,
            }
        }));
    }

    #[cfg(test)]
    fn estimate_error_bound(&self, topology: &TopologyKey, axes: [DVec2; 2]) -> ErrorBound {
        let mut screen = ScreenBounds::default();
        self.update_screen(axes, &mut screen);
        screen.estimate_error_bound(topology)
    }
}

impl ScreenBounds {
    fn estimate_patch_color_error(&self, patch: ScreenPatchBounds, columns: f64, rows: f64) -> f64 {
        // Once either cell axis is at most one physical pixel, further color
        // subdivision cannot produce a resolvable improvement along both axes.
        if patch.extent.x / columns <= 1.0 || patch.extent.y / rows <= 1.0 {
            return 0.0;
        }
        multiply_round_up(patch.color_uv, 0.25 / (columns * rows))
    }

    fn estimate_base_patch_error(
        &self,
        column: usize,
        row: usize,
        topology: &TopologyKey,
    ) -> ErrorBound {
        let patch = self.patches[row * (self.width - 1) + column];
        let columns = topology.u_subdivisions(column, row) as f64;
        let rows = topology.v_subdivisions(column, row) as f64;
        let uu = multiply_round_up(patch.uu, 0.125 / (columns * columns));
        let uv = multiply_round_up(patch.uv, 0.25 / (columns * rows));
        let vv = multiply_round_up(patch.vv, 0.125 / (rows * rows));
        ErrorBound {
            geometry: add_round_up(add_round_up(uu, uv), vv),
            color: self.estimate_patch_color_error(patch, columns, rows),
        }
    }

    fn estimate_patch_error(
        &self,
        column: usize,
        row: usize,
        topology: &TopologyKey,
    ) -> ErrorBound {
        let base = self.estimate_base_patch_error(column, row, topology);
        let mut snap = 0.0_f64;
        let columns = topology.u_subdivisions(column, row);
        let rows = topology.v_subdivisions(column, row);
        if column > 0 && rows > topology.v_subdivisions(column - 1, row) {
            let neighbor = self.patches[row * (self.width - 1) + column - 1];
            let subdivisions = topology.v_subdivisions(column - 1, row) as f64;
            snap = snap.max(multiply_round_up(
                neighbor.edge_vv[1],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if column + 1 < self.width - 1 && rows > topology.v_subdivisions(column + 1, row) {
            let neighbor = self.patches[row * (self.width - 1) + column + 1];
            let subdivisions = topology.v_subdivisions(column + 1, row) as f64;
            snap = snap.max(multiply_round_up(
                neighbor.edge_vv[0],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if row > 0 && columns > topology.u_subdivisions(column, row - 1) {
            let neighbor = self.patches[(row - 1) * (self.width - 1) + column];
            let subdivisions = topology.u_subdivisions(column, row - 1) as f64;
            snap = snap.max(multiply_round_up(
                neighbor.edge_uu[1],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        if row + 1 < self.height - 1 && columns > topology.u_subdivisions(column, row + 1) {
            let neighbor = self.patches[(row + 1) * (self.width - 1) + column];
            let subdivisions = topology.u_subdivisions(column, row + 1) as f64;
            snap = snap.max(multiply_round_up(
                neighbor.edge_uu[0],
                0.125 / (subdivisions * subdivisions),
            ));
        }
        ErrorBound {
            geometry: add_round_up(base.geometry, snap),
            color: base.color,
        }
    }

    fn compute_error_score(&self, topology: &TopologyKey) -> ErrorScore {
        self.compute_error_score_with(topology, Self::estimate_patch_error)
    }

    fn compute_base_error_score(&self, topology: &TopologyKey) -> ErrorScore {
        self.compute_error_score_with(topology, Self::estimate_base_patch_error)
    }

    fn compute_error_score_with(
        &self,
        topology: &TopologyKey,
        estimate_error: fn(&Self, usize, usize, &TopologyKey) -> ErrorBound,
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
                let patch_error = estimate_error(self, column, row, topology);
                let severity = patch_error.compute_severity();
                if severity > maximum_severity {
                    maximum_severity = severity;
                    worst_patch = [column, row];
                }
                bound.geometry = bound.geometry.max(patch_error.geometry);
                bound.color = bound.color.max(patch_error.color);
                sum += severity;
                let patch = self.patches[row * (self.width - 1) + column];
                let cell_width = patch.extent.x / topology.u_subdivisions(column, row) as f64;
                let cell_height = patch.extent.y / topology.v_subdivisions(column, row) as f64;
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
        fraction: f64,
        cap: Option<&TopologyKey>,
    ) -> TopologyKey {
        let mut topology = self.refine_base(topology, fraction, cap);
        loop {
            let current = self.compute_error_score(&topology);
            if current.bound.meets_tolerance(fraction) {
                return topology;
            }
            let [column, row] = current.worst_patch;
            let mut candidates = SmallVec::<[TopologyKey; 6]>::new();
            candidates.extend(
                [
                    topology.try_double_u_subdivisions(column, row),
                    topology.try_double_v_subdivisions(column, row),
                ]
                .into_iter()
                .flatten(),
            );
            let columns = topology.u_subdivisions(column, row);
            let rows = topology.v_subdivisions(column, row);
            if column > 0 && rows > topology.v_subdivisions(column - 1, row) {
                candidates.extend(topology.try_double_v_subdivisions(column - 1, row));
            }
            if column + 1 < self.width - 1 && rows > topology.v_subdivisions(column + 1, row) {
                candidates.extend(topology.try_double_v_subdivisions(column + 1, row));
            }
            if row > 0 && columns > topology.u_subdivisions(column, row - 1) {
                candidates.extend(topology.try_double_u_subdivisions(column, row - 1));
            }
            if row + 1 < self.height - 1 && columns > topology.u_subdivisions(column, row + 1) {
                candidates.extend(topology.try_double_u_subdivisions(column, row + 1));
            }
            let Some(candidate) =
                self.find_best_refinement(candidates, cap, Self::compute_error_score)
            else {
                return topology;
            };
            topology = candidate;
        }
    }

    fn refine_base(
        &self,
        mut topology: TopologyKey,
        fraction: f64,
        cap: Option<&TopologyKey>,
    ) -> TopologyKey {
        loop {
            let current = self.compute_base_error_score(&topology);
            if current.bound.meets_tolerance(fraction) {
                return topology;
            }
            let [column, row] = current.worst_patch;
            let candidates = [
                topology.try_double_u_subdivisions(column, row),
                topology.try_double_v_subdivisions(column, row),
            ]
            .into_iter()
            .flatten();
            let Some(candidate) =
                self.find_best_refinement(candidates, cap, Self::compute_base_error_score)
            else {
                return topology;
            };
            topology = candidate;
        }
    }

    fn find_best_refinement(
        &self,
        candidates: impl IntoIterator<Item = TopologyKey>,
        cap: Option<&TopologyKey>,
        compute_score: impl Fn(&Self, &TopologyKey) -> ErrorScore,
    ) -> Option<TopologyKey> {
        let mut best: Option<(TopologyKey, ErrorScore)> = None;
        for candidate in candidates {
            if candidate.count_triangles() > MAX_TRIANGLES
                || cap.is_some_and(|cap| !candidate.fits_within(cap))
            {
                continue;
            }
            let score = compute_score(self, &candidate);
            let improves_best = best.as_ref().is_none_or(|(best_topology, best_score)| {
                score
                    .maximum
                    .total_cmp(&best_score.maximum)
                    .then_with(|| score.sum.total_cmp(&best_score.sum))
                    .then_with(|| score.shape.total_cmp(&best_score.shape))
                    .then_with(|| {
                        candidate
                            .count_triangles()
                            .cmp(&best_topology.count_triangles())
                    })
                    .is_lt()
            });
            if improves_best {
                best = Some((candidate, score));
            }
        }
        best.map(|(topology, _)| topology)
    }

    fn estimate_error_bound(&self, topology: &TopologyKey) -> ErrorBound {
        self.compute_error_score(topology).bound
    }
}

/// Both operands are nonnegative upper bounds. One outward rounding step makes
/// the result an upper bound of their exact IEEE-754 operation.
fn multiply_round_up(left: f64, right: f64) -> f64 {
    (left * right).next_up()
}

fn add_round_up(left: f64, right: f64) -> f64 {
    (left + right).next_up()
}

fn compute_length_upper_bound(value: DVec2) -> f64 {
    Interval::from_exact_value(value.x.abs())
        .scale(value.x.abs())
        .add(Interval::from_exact_value(value.y.abs()).scale(value.y.abs()))
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
        let sample_row = |y: usize| {
            let sample_control_interval =
                |x: usize| values[y * width + x].map(Interval::from_exact_value);
            if x < 0 {
                core::array::from_fn(|channel| {
                    sample_control_interval(0)[channel]
                        .scale(2.0)
                        .sub(sample_control_interval(1)[channel])
                })
            } else if x >= width as isize {
                core::array::from_fn(|channel| {
                    sample_control_interval(width - 1)[channel]
                        .scale(2.0)
                        .sub(sample_control_interval(width - 2)[channel])
                })
            } else {
                sample_control_interval(x as usize)
            }
        };
        if y < 0 {
            let first = sample_row(0);
            let second = sample_row(1);
            core::array::from_fn(|channel| first[channel].scale(2.0).sub(second[channel]))
        } else if y >= height as isize {
            let last = sample_row(height - 1);
            let previous = sample_row(height - 2);
            core::array::from_fn(|channel| last[channel].scale(2.0).sub(previous[channel]))
        } else {
            sample_row(y as usize)
        }
    }
    fn convert_catmull_rom_to_bezier<const N: usize>(
        p: [IntervalPoint<N>; 4],
    ) -> [IntervalPoint<N>; 4] {
        [
            p[1],
            core::array::from_fn(|i| p[1][i].add(p[2][i].sub(p[0][i]).divide_by_six())),
            core::array::from_fn(|i| p[2][i].sub(p[3][i].sub(p[1][i]).divide_by_six())),
            p[2],
        ]
    }
    let mut result = SmallVec::with_capacity((width - 1) * (height - 1));
    for row in 0..height - 1 {
        for column in 0..width - 1 {
            let rows: [[IntervalPoint<N>; 4]; 4] = core::array::from_fn(|y| {
                convert_catmull_rom_to_bezier(core::array::from_fn(|x| {
                    sample(
                        values,
                        width,
                        height,
                        column as isize + x as isize - 1,
                        row as isize + y as isize - 1,
                    )
                }))
            });
            let columns: [[IntervalPoint<N>; 4]; 4] = core::array::from_fn(|x| {
                convert_catmull_rom_to_bezier(core::array::from_fn(|y| rows[y][x]))
            });
            result.push(core::array::from_fn(|i| columns[i % 4][i / 4]));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::Color;
    use bevy_math::Vec2;
    use bevy_ui::{InterpolationColorSpace, MeshGradientGeometry, MeshGradientPoint};

    fn create_surface_bounds(mesh: &MeshGradient) -> Arc<SurfaceBounds> {
        Arc::new(SurfaceBounds::new(mesh))
    }

    fn build_reference_interval_patches(mesh: &MeshGradient) -> SmallVec<[IntervalPatch<6>; 9]> {
        let values: SmallVec<[Point; 16]> = mesh
            .points()
            .iter()
            .map(|point| {
                let color = mesh.color_space().to_components(point.color);
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

    fn build_reference_patches(mesh: &MeshGradient) -> Vec<Patch> {
        build_reference_interval_patches(mesh)
            .into_iter()
            .map(|patch| patch.map(|p| p.map(|v| v.lo + (v.hi - v.lo) * 0.5)))
            .collect()
    }

    fn create_test_mesh(
        size: usize,
        displacement: f32,
        contrast: f32,
        space: InterpolationColorSpace,
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

    fn create_checkerboard_mesh(
        contrast: f32,
        space: InterpolationColorSpace,
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

    fn create_screen_axes(size: f64) -> [DVec2; 2] {
        [DVec2::X * size, DVec2::Y * size]
    }

    fn create_unit_position_bounds() -> VectorBounds {
        VectorBounds {
            x: Interval { lo: 0.0, hi: 1.0 },
            y: Interval { lo: 0.0, hi: 1.0 },
        }
    }

    type ShaderPoint = [f32; 6];

    fn build_shader_control_values(mesh: &MeshGradient) -> Vec<ShaderPoint> {
        mesh.points()
            .iter()
            .map(|point| {
                let color = mesh.color_space().to_components(point.color);
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

    fn sample_shader_control(
        values: &[ShaderPoint],
        width: usize,
        height: usize,
        x: isize,
        y: isize,
    ) -> ShaderPoint {
        fn sample_shader_row(
            values: &[ShaderPoint],
            width: usize,
            x: isize,
            y: usize,
        ) -> ShaderPoint {
            let read_point = |x| values[y * width + x];
            if x < 0 {
                core::array::from_fn(|channel| {
                    2.0 * read_point(0)[channel] - read_point(1)[channel]
                })
            } else if x >= width as isize {
                core::array::from_fn(|channel| {
                    2.0 * read_point(width - 1)[channel] - read_point(width - 2)[channel]
                })
            } else {
                read_point(x as usize)
            }
        }
        if y < 0 {
            let a = sample_shader_row(values, width, x, 0);
            let b = sample_shader_row(values, width, x, 1);
            core::array::from_fn(|channel| 2.0 * a[channel] - b[channel])
        } else if y >= height as isize {
            let a = sample_shader_row(values, width, x, height - 1);
            let b = sample_shader_row(values, width, x, height - 2);
            core::array::from_fn(|channel| 2.0 * a[channel] - b[channel])
        } else {
            sample_shader_row(values, width, x, y as usize)
        }
    }

    fn evaluate_shader_cubic(
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

    fn evaluate_shader_surface(
        mesh: &MeshGradient,
        column: usize,
        row: usize,
        uv: Vec2,
    ) -> ShaderPoint {
        let values = build_shader_control_values(mesh);
        let rows: [ShaderPoint; 4] = core::array::from_fn(|y| {
            evaluate_shader_cubic(
                sample_shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize - 1,
                    row as isize + y as isize - 1,
                ),
                sample_shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize,
                    row as isize + y as isize - 1,
                ),
                sample_shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize + 1,
                    row as isize + y as isize - 1,
                ),
                sample_shader_control(
                    &values,
                    mesh.width(),
                    mesh.height(),
                    column as isize + 2,
                    row as isize + y as isize - 1,
                ),
                uv.x,
            )
        });
        evaluate_shader_cubic(rows[0], rows[1], rows[2], rows[3], uv.y)
    }

    #[test]
    fn shader_f32_surface_agrees_with_cpu_reference_and_shares_exact_edges() {
        for space in [
            InterpolationColorSpace::LinearRgba,
            InterpolationColorSpace::Srgba,
            InterpolationColorSpace::Oklaba,
        ] {
            for size in [2, 3, 16] {
                let grid = create_test_mesh(size, 0.02 / (size - 1) as f32, 4.0, space);
                let reference = build_reference_patches(&grid);
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
                            let gpu = evaluate_shader_surface(&grid, column, row, uv);
                            let cpu = evaluate_reference_patch(patch, uv.as_dvec2());
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
                            let left =
                                evaluate_shader_surface(&grid, column, row, Vec2::new(1.0, 0.375));
                            let right = evaluate_shader_surface(
                                &grid,
                                column + 1,
                                row,
                                Vec2::new(0.0, 0.375),
                            );
                            assert_eq!(left.map(f32::to_bits), right.map(f32::to_bits));
                        }
                        if row + 1 < size - 1 {
                            let top =
                                evaluate_shader_surface(&grid, column, row, Vec2::new(0.625, 1.0));
                            let bottom = evaluate_shader_surface(
                                &grid,
                                column,
                                row + 1,
                                Vec2::new(0.625, 0.0),
                            );
                            assert_eq!(top.map(f32::to_bits), bottom.map(f32::to_bits));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn affine_minimum_and_curvature_resize_promotion() {
        let affine = create_surface_bounds(&create_test_mesh(
            2,
            0.0,
            1.0,
            InterpolationColorSpace::LinearRgba,
        ));
        assert_eq!(
            QualityState::default()
                .update(&affine, create_screen_axes(4096.0))
                .key
                .find_maximum_subdivisions(),
            MIN_SUBDIVISIONS
        );
        let curved = create_surface_bounds(&create_test_mesh(
            3,
            0.04,
            1.0,
            InterpolationColorSpace::LinearRgba,
        ));
        let mut state = QualityState::default();
        let small = state.update(&curved, create_screen_axes(256.0));
        let large = state.update(&curved, create_screen_axes(4096.0));
        assert!(small.key.find_maximum_subdivisions() > 1);
        assert!(large.key.count_triangles() > small.key.count_triangles());
        assert!(large.error.meets_tolerance(1.0));
    }

    #[test]
    fn vertex_color_error_refines_only_resolvable_mixed_color() {
        for space in [
            InterpolationColorSpace::LinearRgba,
            InterpolationColorSpace::Srgba,
            InterpolationColorSpace::Oklaba,
        ] {
            let vivid = create_surface_bounds(&create_checkerboard_mesh(
                1.0,
                space,
                MeshGradientColorInterpolation::Vertex,
            ));
            let chosen = QualityState::default().update(&vivid, create_screen_axes(512.0));
            assert_eq!(chosen.key.u_subdivisions(0, 0), 4);
            assert_eq!(chosen.key.v_subdivisions(0, 0), 4);
            assert!(chosen.error.color <= COLOR_LIMIT.next_up());

            let subpixel = QualityState::default().update(&vivid, create_screen_axes(1.0));
            assert_eq!(subpixel.key.find_maximum_subdivisions(), MIN_SUBDIVISIONS);

            let quiet = create_surface_bounds(&create_checkerboard_mesh(
                0.001,
                space,
                MeshGradientColorInterpolation::Vertex,
            ));
            assert_eq!(
                QualityState::default()
                    .update(&quiet, create_screen_axes(512.0))
                    .key
                    .find_maximum_subdivisions(),
                MIN_SUBDIVISIONS
            );

            let bicubic = create_surface_bounds(&create_checkerboard_mesh(
                1.0,
                space,
                MeshGradientColorInterpolation::Bicubic,
            ));
            assert_eq!(
                QualityState::default()
                    .update(&bicubic, create_screen_axes(512.0))
                    .key
                    .find_maximum_subdivisions(),
                MIN_SUBDIVISIONS
            );
        }
    }

    #[test]
    fn hue_color_paths_use_normalized_transport_and_geometry_only_tessellation() {
        for space in [
            InterpolationColorSpace::Hsva,
            InterpolationColorSpace::HsvaLong,
        ] {
            let mut grid = create_test_mesh(2, 0.0, 1.0, space);
            for hue in [10.0, 350.0] {
                let color = grid
                    .color_space()
                    .to_components(Color::hsva(hue, 0.7, 0.8, 0.5));
                for (actual, expected) in color.into_iter().zip([hue / 360.0, 0.7, 0.8, 0.5]) {
                    assert!((actual - expected).abs() < 1e-5);
                }
            }
            let before = create_surface_bounds(&grid);
            let before = QualityState::default().update(&before, create_screen_axes(4096.0));
            grid.try_set_color(0, Color::hsva(350.0, 1.0, 1.0, 1.0))
                .unwrap();
            let after = create_surface_bounds(&grid);
            let after = QualityState::default().update(&after, create_screen_axes(4096.0));
            assert_eq!(before.key, after.key);
            assert_eq!(after.key.find_maximum_subdivisions(), MIN_SUBDIVISIONS);
            assert_eq!(after.error.color, 0.0);
        }
    }

    #[test]
    fn bicubic_color_does_not_drive_geometry_tessellation_and_scale_is_physical() {
        assert_eq!(
            compute_physical_axes(Vec2::splat(256.0), Mat2::IDENTITY),
            create_screen_axes(256.0)
        );
        let mut quiet_mesh = create_test_mesh(3, 0.04, 0.0, InterpolationColorSpace::LinearRgba);
        quiet_mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        let mut vivid_mesh = create_test_mesh(3, 0.04, 8.0, InterpolationColorSpace::LinearRgba);
        vivid_mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        let quiet = create_surface_bounds(&quiet_mesh);
        let vivid = create_surface_bounds(&vivid_mesh);
        let a = QualityState::default().update(&quiet, create_screen_axes(1.0));
        let b = QualityState::default().update(&vivid, create_screen_axes(1.0));
        assert_eq!(b.key, a.key);
        let topology = TopologyKey::new_uniform(3, 3, 8);
        let rotated = [DVec2::Y * 512.0, -DVec2::X * 512.0];
        assert_eq!(
            vivid
                .estimate_error_bound(&topology, create_screen_axes(512.0))
                .geometry,
            vivid.estimate_error_bound(&topology, rotated).geometry
        );
        assert_eq!(
            vivid
                .estimate_error_bound(&topology, create_screen_axes(256.0))
                .geometry
                * 2.0,
            vivid
                .estimate_error_bound(&topology, create_screen_axes(512.0))
                .geometry
        );
        assert!(
            vivid
                .estimate_error_bound(&topology, [DVec2::new(512.0, 512.0), DVec2::Y * 512.0])
                .geometry
                > vivid
                    .estimate_error_bound(&topology, create_screen_axes(512.0))
                    .geometry
        );
    }

    #[test]
    fn demotion_waits_eight_consecutive_frames() {
        let curved = create_surface_bounds(&create_test_mesh(
            3,
            0.04,
            0.0,
            InterpolationColorSpace::LinearRgba,
        ));
        let flat = create_surface_bounds(&create_test_mesh(
            3,
            0.0,
            0.0,
            InterpolationColorSpace::LinearRgba,
        ));
        let mut state = QualityState::default();
        let high = state.update(&curved, create_screen_axes(4096.0)).key;
        for _ in 0..7 {
            assert_eq!(state.update(&flat, create_screen_axes(256.0)).key, high);
        }
        state.update(&curved, create_screen_axes(4096.0));
        for _ in 0..7 {
            assert_eq!(state.update(&flat, create_screen_axes(256.0)).key, high);
        }
        assert_eq!(
            state
                .update(&flat, create_screen_axes(256.0))
                .key
                .find_maximum_subdivisions(),
            MIN_SUBDIVISIONS
        );
    }

    #[test]
    fn demotion_accumulates_matching_candidates_across_input_changes() {
        let curved = create_surface_bounds(&create_test_mesh(
            3,
            0.04,
            0.0,
            InterpolationColorSpace::LinearRgba,
        ));
        let flat = create_test_mesh(3, 0.0, 0.0, InterpolationColorSpace::LinearRgba);
        let mut state = QualityState::default();
        let high = state.update(&curved, create_screen_axes(4096.0)).key;
        let minimum = TopologyKey::new_uniform(3, 3, MIN_SUBDIVISIONS);
        assert!(high.count_triangles() > minimum.count_triangles());

        // Rebuilding bounds and resizing invalidate the numerical cache each
        // frame, but the same safe candidate must retain its demotion delay.
        for frame in 1..DEMOTION_FRAMES {
            let bounds = create_surface_bounds(&flat);
            let axes = create_screen_axes(256.0 + f64::from(frame));
            assert_eq!(state.update(&bounds, axes).key, high);
            assert_eq!(state.below_half, frame);
        }
        assert_eq!(
            state
                .update(
                    &create_surface_bounds(&flat),
                    create_screen_axes(256.0 + f64::from(DEMOTION_FRAMES)),
                )
                .key,
            minimum
        );
        assert_eq!(state.below_half, 0);
    }

    #[test]
    fn cached_quality_matches_rebuilt_bounds_through_frame_transitions() {
        fn compare_frames(
            mesh: &MeshGradient,
            axes: [DVec2; 2],
            frames: usize,
            cached: &mut QualityState,
            rebuilt: &mut QualityState,
        ) -> QualitySelection {
            let bounds = create_surface_bounds(mesh);
            let mut last = None;
            for frame in 0..frames {
                let actual = cached.update(&bounds, axes);
                // A new immutable input on each frame forces all numerical
                // work, while preserving the same frame-history decisions.
                let expected = rebuilt.update(&create_surface_bounds(mesh), axes);
                assert_eq!(actual.key, expected.key, "frame {frame}");
                assert_eq!(
                    actual.error.geometry.to_bits(),
                    expected.error.geometry.to_bits()
                );
                assert_eq!(actual.error.color.to_bits(), expected.error.color.to_bits());
                assert_eq!(actual.report_cap, expected.report_cap, "frame {frame}");
                last = Some(actual);
            }
            last.unwrap()
        }

        let mut cached = QualityState::default();
        let mut rebuilt = QualityState::default();
        let mut checker = create_test_mesh(16, 0.0, 0.0, InterpolationColorSpace::LinearRgba);
        checker
            .try_edit_points(|points| {
                for (index, point) in points.iter_mut().enumerate() {
                    let red = ((index % 16 + index / 16) % 2) as f32;
                    point.color = Color::linear_rgba(red, 0.0, 0.0, 1.0);
                }
                Ok(())
            })
            .unwrap();
        for (axes, frames) in [
            (create_screen_axes(512.0), 10),
            (create_screen_axes(4096.0), 3),
            ([-DVec2::X * 4096.0, DVec2::Y * 4096.0], 3),
            (create_screen_axes(16.0), 10),
        ] {
            compare_frames(&checker, axes, frames, &mut cached, &mut rebuilt);
        }
        let center = checker.points()[136].position;
        checker
            .try_set_position(136, center + Vec2::new(0.01, 0.0))
            .unwrap();
        compare_frames(
            &checker,
            create_screen_axes(512.0),
            10,
            &mut cached,
            &mut rebuilt,
        );
        checker.try_set_color(136, Color::WHITE).unwrap();
        compare_frames(
            &checker,
            create_screen_axes(512.0),
            10,
            &mut cached,
            &mut rebuilt,
        );

        let curved = create_test_mesh(3, 0.1, 0.0, InterpolationColorSpace::LinearRgba);
        compare_frames(
            &curved,
            create_screen_axes(4096.0),
            3,
            &mut cached,
            &mut rebuilt,
        );
        let demoted = compare_frames(
            &curved,
            create_screen_axes(1.0),
            8,
            &mut cached,
            &mut rebuilt,
        );
        assert_eq!(demoted.key.find_maximum_subdivisions(), MIN_SUBDIVISIONS);

        let cap = create_test_mesh(16, 0.005, 1.0, InterpolationColorSpace::LinearRgba);
        let capped = compare_frames(&cap, create_screen_axes(1e10), 1, &mut cached, &mut rebuilt);
        assert!(capped.report_cap);
        assert!(!capped.error.meets_tolerance(1.0));
        let stable_cap =
            compare_frames(&cap, create_screen_axes(1e10), 3, &mut cached, &mut rebuilt);
        assert!(!stable_cap.report_cap);
        let flat = create_test_mesh(2, 0.0, 0.0, InterpolationColorSpace::LinearRgba);
        compare_frames(&flat, create_screen_axes(1.0), 1, &mut cached, &mut rebuilt);
        let early = compare_frames(&cap, create_screen_axes(1e10), 1, &mut cached, &mut rebuilt);
        assert!(!early.report_cap);
        compare_frames(
            &flat,
            create_screen_axes(1.0),
            120,
            &mut cached,
            &mut rebuilt,
        );
        let later = compare_frames(&cap, create_screen_axes(1e10), 1, &mut cached, &mut rebuilt);
        assert!(later.report_cap);
    }

    #[test]
    fn demotion_can_choose_an_intermediate_safe_tier() {
        let bounds = Arc::new(SurfaceBounds {
            width: 2,
            height: 2,
            patches: Box::new([PatchBounds {
                position: create_unit_position_bounds(),
                uu: VectorBounds::from_exact_value(DVec2::new(8.0, 0.0)),
                uv: VectorBounds::from_exact_value(DVec2::ZERO),
                vv: VectorBounds::from_exact_value(DVec2::ZERO),
                edge_uu: [VectorBounds::from_exact_value(DVec2::new(8.0, 0.0)); 2],
                edge_vv: [VectorBounds::from_exact_value(DVec2::ZERO); 2],
                color_uv: 0.0,
            }]),
        });
        let mut state = QualityState::default();
        assert_eq!(
            state
                .update(&bounds, create_screen_axes(500.0))
                .key
                .find_maximum_subdivisions(),
            16
        );
        // Four subdivisions meet the full limit, but only eight meet half.
        for _ in 0..7 {
            assert_eq!(
                state
                    .update(&bounds, create_screen_axes(50.0))
                    .key
                    .find_maximum_subdivisions(),
                16
            );
        }
        assert_eq!(
            state
                .update(&bounds, create_screen_axes(50.0))
                .key
                .find_maximum_subdivisions(),
            8
        );
    }

    #[test]
    fn alpha_alone_does_not_promote_geometry_quality() {
        let mut grid = create_test_mesh(3, 0.02, 0.0, InterpolationColorSpace::LinearRgba);
        let quiet =
            QualityState::default().update(&create_surface_bounds(&grid), create_screen_axes(1.0));
        grid.try_edit_points(|points| {
            for point in points {
                point.color = Color::linear_rgba(0.0, 0.0, 0.0, point.position.x);
            }
            Ok(())
        })
        .unwrap();
        let alpha =
            QualityState::default().update(&create_surface_bounds(&grid), create_screen_axes(1.0));
        assert_eq!(alpha.key, quiet.key);
    }

    #[test]
    fn cap_is_visible_bounded_and_reported_once() {
        let bounds = create_surface_bounds(&create_test_mesh(
            16,
            0.005,
            1.0,
            InterpolationColorSpace::LinearRgba,
        ));
        let mut state = QualityState::default();
        let selection = state.update(&bounds, create_screen_axes(1e10));
        assert!(!selection.error.meets_tolerance(1.0));
        assert!(selection.report_cap);
        assert!(selection.key.count_triangles() <= MAX_TRIANGLES);
        assert!(selection.key.find_maximum_subdivisions() <= MAX_SUBDIVISIONS);
        assert!(!state.update(&bounds, create_screen_axes(1e10)).report_cap);
        state.update(&bounds, create_screen_axes(1.0));
        assert!(!state.update(&bounds, create_screen_axes(1e10)).report_cap);
    }

    #[test]
    fn hdr_color_changes_reuse_geometry_topology() {
        let mut hdr = create_test_mesh(3, 0.02, 1.0, InterpolationColorSpace::LinearRgba);
        hdr.try_edit_points(|points| {
            for point in points {
                let p = point.position;
                point.color = Color::linear_rgba(100.0 + p.x, -100.0 - p.y, 0.5, 1.0);
            }
            Ok(())
        })
        .unwrap();
        let hdr_key = QualityState::default()
            .update(&create_surface_bounds(&hdr), create_screen_axes(4096.0))
            .key;
        let ordinary_key = QualityState::default()
            .update(
                &create_surface_bounds(&create_test_mesh(
                    3,
                    0.02,
                    1.0,
                    InterpolationColorSpace::LinearRgba,
                )),
                create_screen_axes(4096.0),
            )
            .key;
        assert_eq!(hdr_key, ordinary_key);
    }

    #[test]
    fn point_edits_reuse_selected_topology_and_grid_changes_reset_state() {
        let mut grid = create_test_mesh(3, 0.02, 1.0, InterpolationColorSpace::LinearRgba);
        let mut state = QualityState::default();
        let first = state.update(&create_surface_bounds(&grid), create_screen_axes(512.0));
        grid.try_set_position(4, Vec2::new(0.52001, 0.5)).unwrap();
        let next = state.update(&create_surface_bounds(&grid), create_screen_axes(512.0));
        assert_eq!(first.key, next.key);
        let replacement = create_test_mesh(16, 0.0, 1.0, InterpolationColorSpace::LinearRgba);
        let changed = state.update(
            &create_surface_bounds(&replacement),
            create_screen_axes(512.0),
        );
        assert_eq!(changed.key.width, 16);
        assert_eq!(changed.key.find_maximum_subdivisions(), MIN_SUBDIVISIONS);
    }

    #[test]
    fn topology_packs_vertices_and_preserves_shared_edges() {
        assert_eq!(size_of::<ParameterVertex>(), 8);
        let key = TopologyKey::new_uniform(3, 3, 8);
        let topology = ParameterTopology::new(&key);
        assert_eq!(topology.indices.len() / 3, key.count_triangles());
        let stride = 9 * 9;
        for y in 0..=8 {
            let left = topology.vertices[y * 9 + 8];
            let right = topology.vertices[stride + y * 9];
            let (left_patch, left_uv, _, _) = left.unpack();
            let (right_patch, right_uv, _, _) = right.unpack();
            assert_eq!(
                left_patch[0] as f32 + left_uv[0],
                right_patch[0] as f32 + right_uv[0]
            );
            assert_eq!(left_uv[1].to_bits(), right_uv[1].to_bits());
        }
    }

    #[test]
    fn localized_curvature_refines_only_the_axes_that_need_it() {
        let bounds = Arc::new(SurfaceBounds {
            width: 3,
            height: 2,
            patches: Box::new([
                PatchBounds {
                    position: create_unit_position_bounds(),
                    uu: VectorBounds::from_exact_value(DVec2::new(1.0, 0.0)),
                    uv: VectorBounds::from_exact_value(DVec2::ZERO),
                    vv: VectorBounds::from_exact_value(DVec2::ZERO),
                    edge_uu: [VectorBounds::from_exact_value(DVec2::new(1.0, 0.0)); 2],
                    edge_vv: [VectorBounds::from_exact_value(DVec2::ZERO); 2],
                    color_uv: 0.0,
                },
                PatchBounds {
                    position: create_unit_position_bounds(),
                    uu: VectorBounds::from_exact_value(DVec2::ZERO),
                    uv: VectorBounds::from_exact_value(DVec2::ZERO),
                    vv: VectorBounds::from_exact_value(DVec2::ZERO),
                    edge_uu: [VectorBounds::from_exact_value(DVec2::ZERO); 2],
                    edge_vv: [VectorBounds::from_exact_value(DVec2::ZERO); 2],
                    color_uv: 0.0,
                },
            ]),
        });
        let selection = QualityState::default().update(&bounds, create_screen_axes(1024.0));

        assert_eq!(selection.key.u_subdivisions(0, 0), 8);
        assert_eq!(selection.key.u_subdivisions(1, 0), MIN_SUBDIVISIONS);
        assert_eq!(selection.key.v_subdivisions(0, 0), MIN_SUBDIVISIONS);
        assert_eq!(selection.key.count_triangles(), 40);
        assert_eq!(TopologyKey::new_uniform(3, 2, 8).count_triangles(), 256);
        assert!(selection.error.meets_tolerance(1.0));
    }

    #[test]
    fn nonuniform_topology_snaps_fine_edges_to_coarse_chords() {
        let key = TopologyKey::new_uniform(3, 3, 1)
            .try_double_u_subdivisions(0, 0)
            .unwrap()
            .try_double_u_subdivisions(0, 0)
            .unwrap()
            .try_double_v_subdivisions(0, 0)
            .unwrap()
            .try_double_v_subdivisions(0, 0)
            .unwrap();
        let topology = ParameterTopology::new(&key);
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
            InterpolationColorSpace::Oklaba,
            MeshGradientGeometry::AllowFolds,
        )
        .unwrap();
        let selection = QualityState::default().update(
            &create_surface_bounds(&mesh),
            [DVec2::X * 1_125.0, DVec2::Y * 866.0],
        );
        let global = TopologyKey::new_uniform(5, 4, selection.key.find_maximum_subdivisions());

        assert!(
            selection.key.count_triangles() * 4 <= global.count_triangles() * 3,
            "adaptive={} global={} max={} error={:?}",
            selection.key.count_triangles(),
            global.count_triangles(),
            selection.key.find_maximum_subdivisions(),
            selection.error,
        );
        assert!(selection.key.count_triangles() <= MAX_TRIANGLES);
        assert!(selection.error.geometry <= GEOMETRY_LIMIT);
        assert!(selection.error.meets_tolerance(1.0));
    }

    fn evaluate_reference_patch(patch: &Patch, uv: DVec2) -> Point {
        fn compute_bernstein_weights(t: f64) -> [f64; 4] {
            [
                (1.0 - t).powi(3),
                3.0 * t * (1.0 - t).powi(2),
                3.0 * t * t * (1.0 - t),
                t.powi(3),
            ]
        }
        let x = compute_bernstein_weights(uv.x);
        let y = compute_bernstein_weights(uv.y);
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
        let columns = topology.u_subdivisions(column, row);
        let rows = topology.v_subdivisions(column, row);
        let mut position_columns = columns;
        let mut position_rows = rows;
        if uv.y > 0.0 && uv.y < 1.0 {
            if uv.x == 0.0 && column > 0 {
                position_rows = position_rows.min(topology.v_subdivisions(column - 1, row));
            }
            if uv.x == 1.0 && column + 1 < topology.width - 1 {
                position_rows = position_rows.min(topology.v_subdivisions(column + 1, row));
            }
        }
        if uv.x > 0.0 && uv.x < 1.0 {
            if uv.y == 0.0 && row > 0 {
                position_columns = position_columns.min(topology.u_subdivisions(column, row - 1));
            }
            if uv.y == 1.0 && row + 1 < topology.height - 1 {
                position_columns = position_columns.min(topology.u_subdivisions(column, row + 1));
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
            return evaluate_reference_patch(patch, uv);
        };
        let start = evaluate_reference_patch(patch, start);
        let end = evaluate_reference_patch(patch, end);
        core::array::from_fn(|channel| start[channel] + (end[channel] - start[channel]) * weight)
    }

    fn measure_dense_geometry_error(
        grid: &MeshGradient,
        topology: &TopologyKey,
        screen_axes: [DVec2; 2],
        offsets: &[DVec2],
    ) -> f64 {
        let mut maximum = 0.0_f64;
        for (patch_index, patch) in build_reference_patches(grid).into_iter().enumerate() {
            let column = patch_index % (grid.width() - 1);
            let row = patch_index / (grid.width() - 1);
            let columns = topology.u_subdivisions(column, row);
            let rows = topology.v_subdivisions(column, row);
            let step = DVec2::new(1.0 / columns as f64, 1.0 / rows as f64);
            for y in 0..rows {
                for x in 0..columns {
                    for &offset in offsets {
                        let a = DVec2::new(x as f64 / columns as f64, y as f64 / rows as f64);
                        let p = a + offset * step;
                        let q = evaluate_reference_patch(&patch, p);
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
                                let q = evaluate_topology_vertex(&patch, uv, column, row, topology);
                                DVec2::new(q[0], q[1]) * w
                            })
                            .sum();
                        let delta = DVec2::new(q[0], q[1]) - approximate;
                        let error = (screen_axes[0] * delta.x + screen_axes[1] * delta.y).length();
                        maximum = maximum.max(error);
                    }
                }
            }
        }
        maximum
    }

    #[test]
    fn center_displacement_matches_independent_cardinal_basis() {
        // A regular 3x3 grid with only its center displaced is the affine
        // surface plus one separable cardinal basis function. These closed
        // forms are independent of the interval/Bezier conversion above.
        fn center_influence(patch: usize, t: f64) -> f64 {
            if patch == 0 {
                t + t * t - t * t * t
            } else {
                1.0 - 2.0 * t * t + t * t * t
            }
        }
        let mesh = create_test_mesh(3, 0.1, 0.0, InterpolationColorSpace::LinearRgba);
        let displacement = f64::from(mesh.points()[4].position.x) - 0.5;
        for (index, patch) in build_reference_patches(&mesh).into_iter().enumerate() {
            let column = index % 2;
            let row = index / 2;
            for y in 0..=16 {
                for x in 0..=16 {
                    let uv = DVec2::new(x as f64 / 16.0, y as f64 / 16.0);
                    let expected = DVec2::new(
                        (column as f64 + uv.x) * 0.5
                            + displacement
                                * center_influence(column, uv.x)
                                * center_influence(row, uv.y),
                        (row as f64 + uv.y) * 0.5,
                    );
                    let actual = evaluate_reference_patch(&patch, uv);
                    assert!((DVec2::new(actual[0], actual[1]) - expected).length() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn physical_size_is_independent_of_display_scale_and_respects_pixel_error() {
        let mesh = create_test_mesh(3, 0.1, 0.0, InterpolationColorSpace::LinearRgba);
        let bounds = create_surface_bounds(&mesh);
        let offsets: Vec<_> = (1..20)
            .flat_map(|y| (1..20).map(move |x| DVec2::new(x as f64 / 20.0, y as f64 / 20.0)))
            .collect();
        let mut expected_key = None;
        for display_scale in [0.5, 1.0, 2.0] {
            // UI layout applies display scale before populating ComputedNode.
            let logical_size = Vec2::splat(256.0 / display_scale);
            let physical_size = logical_size * display_scale;
            let axes = compute_physical_axes(physical_size, Mat2::IDENTITY);
            assert_eq!(axes, create_screen_axes(256.0));
            let selected = QualityState::default().update(&bounds, axes);
            if let Some(key) = &expected_key {
                assert_eq!(&selected.key, key);
            }
            expected_key = Some(selected.key.clone());
            let measured = measure_dense_geometry_error(&mesh, &selected.key, axes, &offsets);
            assert!(measured <= selected.error.geometry + 1e-9);
            assert!(measured <= GEOMETRY_LIMIT);
        }
        let transform = Mat2::from_cols(Vec2::new(-2.0, 0.25), Vec2::new(0.5, 1.0));
        assert_eq!(
            compute_physical_axes(Vec2::new(256.0, 128.0), transform),
            [DVec2::new(-512.0, 64.0), DVec2::new(64.0, 128.0)]
        );
    }

    #[test]
    fn dense_geometry_reference_is_below_estimator() {
        for size in [2, 3, 4, 16] {
            let grid = create_test_mesh(
                size,
                0.04 / (size - 1) as f32,
                1.0,
                InterpolationColorSpace::LinearRgba,
            );
            let bounds = create_surface_bounds(&grid);
            for screen in [256.0, 1024.0, 4096.0] {
                for screen_axes in [
                    create_screen_axes(screen),
                    [
                        DVec2::new(screen, screen * 0.25),
                        DVec2::new(-screen * 0.2, screen * 0.8),
                    ],
                ] {
                    let chosen = QualityState::default().update(&bounds, screen_axes);
                    assert!(
                        chosen.error.meets_tolerance(1.0),
                        "size={size}, screen={screen}, error={:?}",
                        chosen.error
                    );
                    let measured = measure_dense_geometry_error(
                        &grid,
                        &chosen.key,
                        screen_axes,
                        &[
                            DVec2::new(0.2, 0.7),
                            DVec2::new(0.7, 0.2),
                            DVec2::splat(0.5),
                        ],
                    );
                    assert!(measured <= chosen.error.geometry + 1e-9);
                }
            }
        }
    }
}
