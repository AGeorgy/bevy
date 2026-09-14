//! Internal adaptive mesh-gradient preparation. The renderer supplies physical
//! screen axes, including node size, transform and display scale exactly once.
//!
//! Geometry uses a Hessian remainder bound for linear triangular interpolation.
//! Color is evaluated in the fragment shader, so its error is bounded through
//! the inverse surface's Lipschitz constant rather than vertex-color error.

use bevy_color::{ColorToComponents, LinearRgba, Oklaba, Srgba};
use bevy_math::{DVec2, Mat2, Vec2};
use bevy_platform::{collections::HashMap, sync::Arc};
use bevy_ui::{MeshGradient, MeshGradientColorSpace};
use bytemuck::{Pod, Zeroable};
use smallvec::SmallVec;

const GEOMETRY_LIMIT: f64 = 0.25;
const COLOR_LIMIT: f64 = 1.0 / 255.0;
const HDR_RELATIVE_LIMIT: f64 = 0.005;
const MAX_TRIANGLES: usize = 131_072;
const MAX_SUBDIVISIONS: usize = 64;
const DEMOTION_FRAMES: u8 = 8;

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
    fn magnitude(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
    }
}

type IntervalPoint = [Interval; 6];
type IntervalPatch = [IntervalPoint; 16];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TopologyKey {
    pub width: usize,
    pub height: usize,
    pub subdivisions: usize,
}

impl TopologyKey {
    pub fn triangles(self) -> usize {
        2 * (self.width - 1) * (self.height - 1) * self.subdivisions.pow(2)
    }
}

/// Patch-local UVs stay dyadic, making adjacent patch edges exactly identical.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ParameterVertex {
    pub patch: [u32; 2],
    pub uv: [f32; 2],
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
    pub fn get(&mut self, key: TopologyKey) -> Arc<ParameterTopology> {
        self.entries
            .entry(key)
            .or_insert_with(|| {
                let n = key.subdivisions;
                let mut vertices = Vec::new();
                let mut indices = Vec::with_capacity(key.triangles() * 3);
                for row in 0..key.height - 1 {
                    for column in 0..key.width - 1 {
                        let base = vertices.len() as u32;
                        for y in 0..=n {
                            for x in 0..=n {
                                vertices.push(ParameterVertex {
                                    patch: [column as u32, row as u32],
                                    uv: [x as f32 / n as f32, y as f32 / n as f32],
                                });
                            }
                        }
                        for y in 0..n {
                            for x in 0..n {
                                let a = base + (y * (n + 1) + x) as u32;
                                let b = a + 1;
                                let c = a + (n + 1) as u32;
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
    /// Color error in SDR-equivalent units; HDR channels are normalized by
    /// their certified relative allowance before taking the maximum.
    pub color: f64,
}

impl ErrorBound {
    fn meets(self, fraction: f64) -> bool {
        self.geometry <= GEOMETRY_LIMIT * fraction && self.color <= COLOR_LIMIT * fraction
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
    demotion_candidate: Option<usize>,
    capped: bool,
    report_cooldown: u8,
}

impl QualityState {
    pub fn update(&mut self, bounds: &SurfaceBounds, screen_axes: [DVec2; 2]) -> QualitySelection {
        self.report_cooldown = self.report_cooldown.saturating_sub(1);
        let mut required = 1;
        let maximum = bounds.maximum_subdivisions();
        while required < maximum && !bounds.error(required, screen_axes).meets(1.0) {
            required *= 2;
        }
        let mut chosen = required;
        if let Some(previous) = self
            .key
            .filter(|key| key.width == bounds.width && key.height == bounds.height)
        {
            if required < previous.subdivisions {
                let mut candidate = required;
                while candidate < previous.subdivisions
                    && !bounds.error(candidate, screen_axes).meets(0.5)
                {
                    candidate *= 2;
                }
                if self.demotion_candidate != Some(candidate) {
                    self.below_half = 0;
                }
                self.demotion_candidate = Some(candidate);
                if candidate < previous.subdivisions {
                    self.below_half += 1;
                } else {
                    self.below_half = 0;
                }
                if self.below_half < DEMOTION_FRAMES {
                    chosen = previous.subdivisions;
                } else {
                    chosen = candidate;
                    self.below_half = 0;
                }
            } else {
                self.below_half = 0;
                self.demotion_candidate = None;
            }
        } else {
            self.below_half = 0;
            self.demotion_candidate = None;
        }
        let error = bounds.error(chosen, screen_axes);
        let capped = !error.meets(1.0);
        let report_cap = capped && !self.capped && self.report_cooldown == 0;
        if report_cap {
            self.report_cooldown = 120;
        }
        self.capped = capped;
        let key = TopologyKey {
            width: bounds.width,
            height: bounds.height,
            subdivisions: chosen,
        };
        self.key = Some(key);
        QualitySelection {
            key,
            error,
            capped,
            report_cap,
        }
    }
}

/// Rebuild on point/color changes only. Resizing only reevaluates the bounds.
pub(crate) struct SurfaceBounds {
    width: usize,
    height: usize,
    remainder: DVec2,
    color_per_position: f64,
    conversion_error: f64,
}

impl SurfaceBounds {
    pub fn new(mesh: &MeshGradient) -> Self {
        let (width, height) = mesh.dimensions();
        let patches = interval_patches(mesh);
        let tolerances = color_tolerances(&patches, mesh.color_space());
        let mut remainder = DVec2::ZERO;
        let mut inverse_margin = f64::INFINITY;
        let mut color_lipschitz = 0.0_f64;
        for patch in &patches {
            let du = derivative_bound(patch, true, (width - 1) as f64);
            let dv = derivative_bound(patch, false, (height - 1) as f64);
            // Minimum eigenvalue of the symmetric Jacobian. The model uses
            // this same sufficient global injectivity certificate.
            let a = du.0[0];
            let d = dv.0[1];
            let b = ((abs_bound(du, 1) + abs_bound(dv, 0)).next_up() * 0.5).next_up();
            // det/trace is a lower bound for the smaller eigenvalue and
            // avoids subtracting nearly equal eigenvalues.
            let determinant = (a * d).next_down() - (b * b).next_up();
            let margin = if a > 0.0 && d > 0.0 && determinant > 0.0 {
                (determinant.next_down() / (a + d).next_up()).next_down()
            } else {
                0.0
            };
            inverse_margin = inverse_margin.min(margin);
            let mut color_derivative = [0.0; 4];
            for (channel, derivative) in color_derivative.iter_mut().enumerate() {
                // L1 bounds L2, avoiding a platform-dependent square root
                // in this upper bound.
                *derivative = (abs_bound(du, channel + 2) + abs_bound(dv, channel + 2)).next_up();
            }
            let output_factor = output_lipschitz(patch, mesh.color_space());
            color_lipschitz = color_lipschitz.max(color_derivative[3]);
            for channel in 0..3 {
                let derivative = if mesh.color_space() == MeshGradientColorSpace::LinearRgba {
                    color_derivative[channel]
                } else {
                    (output_factor * color_derivative[..3].iter().copied().fold(0.0, f64::max))
                        .next_up()
                };
                color_lipschitz = color_lipschitz
                    .max(((derivative / tolerances[channel]).next_up() * COLOR_LIMIT).next_up());
            }
            for channel in 0..2 {
                let mut uu = 0.0_f64;
                let mut vv = 0.0_f64;
                let mut uv = 0.0_f64;
                for y in 0..4 {
                    for x in 0..2 {
                        uu = uu.max(
                            patch[y * 4 + x + 2][channel]
                                .sub(patch[y * 4 + x + 1][channel].scale(2.0))
                                .add(patch[y * 4 + x][channel])
                                .scale(6.0)
                                .magnitude(),
                        );
                        vv = vv.max(
                            patch[(x + 2) * 4 + y][channel]
                                .sub(patch[(x + 1) * 4 + y][channel].scale(2.0))
                                .add(patch[x * 4 + y][channel])
                                .scale(6.0)
                                .magnitude(),
                        );
                    }
                }
                for y in 0..3 {
                    for x in 0..3 {
                        uv = uv.max(
                            patch[(y + 1) * 4 + x + 1][channel]
                                .sub(patch[(y + 1) * 4 + x][channel])
                                .sub(patch[y * 4 + x + 1][channel])
                                .add(patch[y * 4 + x][channel])
                                .scale(9.0)
                                .magnitude(),
                        );
                    }
                }
                // Barycentric first-order terms cancel. Each coordinate's
                // weighted variance is <= h²/4; Cauchy-Schwarz gives the same
                // bound for the mixed term. Taylor's factor 1/2 yields 1/8.
                remainder[channel] = remainder[channel].max(
                    Interval::exact(uu)
                        .add(Interval::exact(uv).scale(2.0))
                        .add(Interval::exact(vv))
                        .scale(0.125)
                        .hi,
                );
            }
        }
        // Near-singular proofs fail closed. Preparation uses outward rounding;
        // GPU f32 arithmetic is a separate renderer agreement test.
        let color_per_position = if inverse_margin > 0.0 {
            (color_lipschitz / inverse_margin).next_up()
        } else {
            f64::INFINITY
        };
        Self {
            width,
            height,
            remainder,
            color_per_position,
            // Rounded sRGB transfer constants leave a tiny discontinuity at
            // the linear/power branch boundary. Include it separately from
            // the derivative bound, even as subdivision tends to infinity.
            conversion_error: if mesh.color_space() == MeshGradientColorSpace::Srgba {
                1e-7
            } else {
                0.0
            },
        }
    }

    fn maximum_subdivisions(&self) -> usize {
        let mut n: usize = 1;
        while n < MAX_SUBDIVISIONS
            && 2 * (self.width - 1) * (self.height - 1) * (2 * n).pow(2) <= MAX_TRIANGLES
        {
            n *= 2;
        }
        n
    }

    fn error(&self, subdivisions: usize, axes: [DVec2; 2]) -> ErrorBound {
        let residual = self.remainder / (subdivisions * subdivisions) as f64;
        ErrorBound {
            geometry: Interval::exact(upper_length(axes[0]))
                .scale(residual.x)
                .add(Interval::exact(upper_length(axes[1])).scale(residual.y))
                .hi,
            color: ((upper_length(residual) * self.color_per_position).next_up()
                + self.conversion_error)
                .next_up(),
        }
    }
}

fn upper_length(value: DVec2) -> f64 {
    Interval::exact(value.x.abs())
        .scale(value.x.abs())
        .add(Interval::exact(value.y.abs()).scale(value.y.abs()))
        .hi
        .sqrt()
        .next_up()
}

fn color_tolerances(patches: &[IntervalPatch], space: MeshGradientColorSpace) -> [f64; 3] {
    if space != MeshGradientColorSpace::LinearRgba {
        // Nonlinear conversions can cross zero inside the control hull. Until
        // a tighter output range is certified, absolute tolerance is the safe
        // (stricter) bound, including for HDR.
        return [COLOR_LIMIT; 3];
    }
    core::array::from_fn(|channel| {
        let lo = patches
            .iter()
            .flatten()
            .map(|p| p[channel + 2].lo)
            .fold(f64::INFINITY, f64::min);
        let hi = patches
            .iter()
            .flatten()
            .map(|p| p[channel + 2].hi)
            .fold(f64::NEG_INFINITY, f64::max);
        let minimum_magnitude = if lo > 0.0 {
            lo
        } else if hi < 0.0 {
            -hi
        } else {
            0.0
        };
        if minimum_magnitude > 1.0 {
            (minimum_magnitude * HDR_RELATIVE_LIMIT)
                .next_down()
                .max(COLOR_LIMIT)
        } else {
            COLOR_LIMIT
        }
    })
}

fn abs_bound(bounds: (Point, Point), channel: usize) -> f64 {
    bounds.0[channel].abs().max(bounds.1[channel].abs())
}

fn derivative_bound(patch: &IntervalPatch, horizontal: bool, scale: f64) -> (Point, Point) {
    let mut lo = [f64::INFINITY; 6];
    let mut hi = [f64::NEG_INFINITY; 6];
    for outer in 0..4 {
        for inner in 0..3 {
            let index = if horizontal {
                outer * 4 + inner
            } else {
                inner * 4 + outer
            };
            let next = index + if horizontal { 1 } else { 4 };
            for channel in 0..6 {
                let value = patch[next][channel]
                    .sub(patch[index][channel])
                    .scale(3.0 * scale);
                lo[channel] = lo[channel].min(value.lo);
                hi[channel] = hi[channel].max(value.hi);
            }
        }
    }
    (lo, hi)
}

fn output_lipschitz(patch: &IntervalPatch, space: MeshGradientColorSpace) -> f64 {
    match space {
        MeshGradientColorSpace::LinearRgba => 1.0,
        MeshGradientColorSpace::Srgba => {
            let maximum = patch
                .iter()
                .flat_map(|p| p[2..5].iter())
                .map(|v| v.hi)
                .fold(0.0, f64::max);
            // x^1.4 <= max(1,x²) for x>=0. This polynomial envelope avoids
            // relying on an unbounded libm pow rounding error.
            let base = ((maximum + 0.055).next_up() / 1.055).next_up();
            ((2.4 / 1.055_f64).next_up() * (base * base).next_up().max(1.0)).next_up()
        }
        MeshGradientColorSpace::Oklaba => {
            // Infinity norm of the Jacobian of OKLab -> LMS cubing -> linear RGB.
            let lms = [
                [1.0, 0.39633778_f32 as f64, 0.21580376_f32 as f64],
                [1.0, -0.105561346_f32 as f64, -0.06385417_f32 as f64],
                [1.0, -0.08948418_f32 as f64, -1.2914855_f32 as f64],
            ];
            let rgb: [[f64; 3]; 3] = [
                [
                    4.0767417_f32 as f64,
                    -3.3077116_f32 as f64,
                    0.23096994_f32 as f64,
                ],
                [
                    -1.268438_f32 as f64,
                    2.6097574_f32 as f64,
                    -0.34131938_f32 as f64,
                ],
                [
                    -0.0041960863_f32 as f64,
                    -0.7034186_f32 as f64,
                    1.7076147_f32 as f64,
                ],
            ];
            let slopes: [f64; 3] = core::array::from_fn(|i| {
                let bound = patch
                    .iter()
                    .map(|p| {
                        (0..3)
                            .fold(Interval::exact(0.0), |sum, j| {
                                sum.add(p[j + 2].scale(lms[i][j]))
                            })
                            .magnitude()
                    })
                    .fold(0.0, f64::max);
                let norm = lms[i]
                    .iter()
                    .fold(Interval::exact(0.0), |sum, v| {
                        sum.add(Interval::exact(v.abs()))
                    })
                    .hi;
                Interval::exact(bound)
                    .scale(bound)
                    .scale(3.0)
                    .scale(norm)
                    .hi
            });
            rgb.iter()
                .map(|row| {
                    (0..3)
                        .fold(Interval::exact(0.0), |sum, i| {
                            sum.add(Interval::exact(slopes[i]).scale(row[i].abs()))
                        })
                        .hi
                })
                .fold(0.0, f64::max)
        }
    }
}

fn interval_patches(mesh: &MeshGradient) -> SmallVec<[IntervalPatch; 9]> {
    let (width, height) = mesh.dimensions();
    let values: SmallVec<[Point; 16]> = mesh
        .points()
        .iter()
        .map(|point| {
            let color = match mesh.color_space() {
                MeshGradientColorSpace::LinearRgba => LinearRgba::from(point.color).to_f32_array(),
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
    fn sample(values: &[Point], width: usize, height: usize, x: isize, y: isize) -> IntervalPoint {
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
    fn bezier(p: [IntervalPoint; 4]) -> [IntervalPoint; 4] {
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
            let rows: [[IntervalPoint; 4]; 4] = core::array::from_fn(|y| {
                bezier(core::array::from_fn(|x| {
                    sample(
                        &values,
                        width,
                        height,
                        column as isize + x as isize - 1,
                        row as isize + y as isize - 1,
                    )
                }))
            });
            let columns: [[IntervalPoint; 4]; 4] =
                core::array::from_fn(|x| bezier(core::array::from_fn(|y| rows[y][x])));
            result.push(core::array::from_fn(|i| columns[i % 4][i / 4]));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_color::Color;
    use bevy_math::{ops, Vec2};
    use bevy_ui::MeshGradientPoint;

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

    fn axes(size: f64) -> [DVec2; 2] {
        [DVec2::X * size, DVec2::Y * size]
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
                .subdivisions,
            1
        );
        let curved = SurfaceBounds::new(&mesh(3, 0.04, 1.0, MeshGradientColorSpace::LinearRgba));
        let mut state = QualityState::default();
        let small = state.update(&curved, axes(256.0));
        let large = state.update(&curved, axes(4096.0));
        assert!(small.key.subdivisions > 1);
        assert!(large.key.subdivisions > small.key.subdivisions);
        assert!(!large.capped);
    }

    #[test]
    fn color_drives_refinement_and_transform_scale_is_physical() {
        assert_eq!(
            physical_axes(Vec2::splat(256.0), Mat2::IDENTITY, 2.0),
            axes(512.0)
        );
        let quiet = SurfaceBounds::new(&mesh(3, 0.04, 0.0, MeshGradientColorSpace::LinearRgba));
        let vivid = SurfaceBounds::new(&mesh(3, 0.04, 8.0, MeshGradientColorSpace::LinearRgba));
        let a = QualityState::default().update(&quiet, axes(1.0));
        let b = QualityState::default().update(&vivid, axes(1.0));
        assert!(b.key.subdivisions > a.key.subdivisions);
        let rotated = [DVec2::Y * 512.0, -DVec2::X * 512.0];
        assert_eq!(
            vivid.error(8, axes(512.0)).geometry,
            vivid.error(8, rotated).geometry
        );
        assert_eq!(
            vivid.error(8, axes(256.0)).geometry * 2.0,
            vivid.error(8, axes(512.0)).geometry
        );
        assert!(
            vivid
                .error(8, [DVec2::new(512.0, 512.0), DVec2::Y * 512.0])
                .geometry
                > vivid.error(8, axes(512.0)).geometry
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
        assert_eq!(state.update(&flat, axes(256.0)).key.subdivisions, 1);
    }

    #[test]
    fn demotion_can_choose_an_intermediate_safe_tier() {
        let bounds = SurfaceBounds {
            width: 2,
            height: 2,
            remainder: DVec2::new(1.0, 0.0),
            color_per_position: 0.0,
            conversion_error: 0.0,
        };
        let mut state = QualityState::default();
        assert_eq!(state.update(&bounds, axes(500.0)).key.subdivisions, 64);
        // 16 subdivisions meets the full limit, but only 32 meets half.
        for _ in 0..7 {
            assert_eq!(state.update(&bounds, axes(50.0)).key.subdivisions, 64);
        }
        assert_eq!(state.update(&bounds, axes(50.0)).key.subdivisions, 32);
    }

    #[test]
    fn alpha_alone_can_promote_quality() {
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
        assert!(alpha.key.subdivisions > quiet.key.subdivisions);
    }

    #[test]
    fn cap_is_visible_bounded_and_reported_once() {
        let bounds = SurfaceBounds::new(&mesh(16, 0.005, 1.0, MeshGradientColorSpace::LinearRgba));
        let mut state = QualityState::default();
        let selection = state.update(&bounds, axes(1e10));
        assert!(selection.capped && selection.report_cap);
        assert!(selection.key.triangles() <= MAX_TRIANGLES);
        assert_eq!(selection.key.subdivisions, 16);
        assert!(!state.update(&bounds, axes(1e10)).report_cap);
        state.update(&bounds, axes(1.0));
        assert!(!state.update(&bounds, axes(1e10)).report_cap);
    }

    #[test]
    fn hdr_relative_allowance_is_certified_per_channel() {
        let mut hdr = mesh(3, 0.02, 1.0, MeshGradientColorSpace::LinearRgba);
        hdr.try_edit_points(|points| {
            for point in points {
                let p = point.position;
                point.color = Color::linear_rgba(100.0 + p.x, -100.0 - p.y, 0.5, 1.0);
            }
            Ok(())
        })
        .unwrap();
        let tolerances = color_tolerances(&interval_patches(&hdr), hdr.color_space());
        assert!(tolerances[0] > 0.49 && tolerances[1] > 0.49);
        assert_eq!(tolerances[2], COLOR_LIMIT);
        let bounds = SurfaceBounds::new(&hdr);
        assert!(!QualityState::default().update(&bounds, axes(4096.0)).capped);
    }

    #[test]
    fn point_edits_reuse_selected_topology_and_grid_changes_reset_state() {
        let mut grid = mesh(3, 0.02, 1.0, MeshGradientColorSpace::LinearRgba);
        let mut state = QualityState::default();
        let mut cache = TopologyCache::default();
        let first = state.update(&SurfaceBounds::new(&grid), axes(512.0));
        let topology = cache.get(first.key);
        grid.try_set_position(4, Vec2::new(0.52001, 0.5)).unwrap();
        let next = state.update(&SurfaceBounds::new(&grid), axes(512.0));
        assert!(Arc::ptr_eq(&topology, &cache.get(next.key)));
        let replacement = mesh(16, 0.0, 1.0, MeshGradientColorSpace::LinearRgba);
        let changed = state.update(&SurfaceBounds::new(&replacement), axes(512.0));
        assert_eq!(changed.key.width, 16);
        assert_eq!(changed.key.subdivisions, 1);
    }

    #[test]
    fn topology_reuses_point_edits_and_releases_unused_entries() {
        let mut cache = TopologyCache::default();
        let key = TopologyKey {
            width: 3,
            height: 3,
            subdivisions: 8,
        };
        let first = cache.get(key);
        let second = cache.get(key);
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(first.indices.len() / 3, key.triangles());
        let stride = 9 * 9;
        for y in 0..=8 {
            let left = first.vertices[y * 9 + 8];
            let right = first.vertices[stride + y * 9];
            assert_eq!(
                left.patch[0] as f32 + left.uv[0],
                right.patch[0] as f32 + right.uv[0]
            );
            assert_eq!(left.uv[1].to_bits(), right.uv[1].to_bits());
        }
        cache.prune();
        assert_eq!(cache.entries.len(), 1);
        drop(first);
        drop(second);
        cache.prune();
        assert!(cache.entries.is_empty());
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

    fn output(value: Point, space: MeshGradientColorSpace) -> [f64; 4] {
        let color = match space {
            MeshGradientColorSpace::LinearRgba => LinearRgba::new(
                value[2] as f32,
                value[3] as f32,
                value[4] as f32,
                value[5] as f32,
            ),
            MeshGradientColorSpace::Srgba => Srgba::new(
                value[2] as f32,
                value[3] as f32,
                value[4] as f32,
                value[5] as f32,
            )
            .into(),
            MeshGradientColorSpace::Oklaba => Oklaba::new(
                value[2] as f32,
                value[3] as f32,
                value[4] as f32,
                value[5] as f32,
            )
            .into(),
        };
        let mut result = color.to_f32_array().map(f64::from);
        result[3] = result[3].clamp(0.0, 1.0);
        result
    }

    #[test]
    fn dense_rendered_color_reference_is_below_estimator() {
        for space in [
            MeshGradientColorSpace::LinearRgba,
            MeshGradientColorSpace::Srgba,
            MeshGradientColorSpace::Oklaba,
        ] {
            for size in [2, 3, 4, 16] {
                let grid = mesh(size, 0.02 / (size - 1) as f32, 1.0, space);
                let bounds = SurfaceBounds::new(&grid);
                for screen in [256.0, 4096.0] {
                    let selection = QualityState::default().update(&bounds, axes(screen));
                    assert!(
                        !selection.capped,
                        "{space:?}, size={size}, error={:?}",
                        selection.error
                    );
                    let n = selection.key.subdivisions;
                    for patch in patches(&grid) {
                        for y in 0..n {
                            for x in 0..n {
                                let base = DVec2::new(x as f64, y as f64) / n as f64;
                                let uv = base + DVec2::new(0.7, 0.3) / n as f64;
                                let a = evaluate(&patch, base);
                                let b = evaluate(&patch, base + DVec2::X / n as f64);
                                let d = evaluate(&patch, base + DVec2::ONE / n as f64);
                                let target = DVec2::new(
                                    a[0] * 0.3 + b[0] * 0.4 + d[0] * 0.3,
                                    a[1] * 0.3 + b[1] * 0.4 + d[1] * 0.3,
                                );
                                // Independently invert the exact surface at the
                                // rasterized pixel using a numerical Jacobian.
                                let mut exact_uv = uv;
                                for _ in 0..8 {
                                    let p = evaluate(&patch, exact_uv);
                                    let px = evaluate(&patch, exact_uv + DVec2::X * 1e-6);
                                    let py = evaluate(&patch, exact_uv + DVec2::Y * 1e-6);
                                    let dx = DVec2::new(px[0] - p[0], px[1] - p[1]) / 1e-6;
                                    let dy = DVec2::new(py[0] - p[0], py[1] - p[1]) / 1e-6;
                                    let residual = DVec2::new(p[0], p[1]) - target;
                                    let determinant = dx.perp_dot(dy);
                                    exact_uv -=
                                        DVec2::new(residual.perp_dot(dy), dx.perp_dot(residual))
                                            / determinant;
                                }
                                assert!(
                                    exact_uv.min_element() >= 0.0 && exact_uv.max_element() <= 1.0
                                );
                                let raster = output(evaluate(&patch, uv), space);
                                let reference = output(evaluate(&patch, exact_uv), space);
                                for channel in 0..4 {
                                    // Bevy's reference conversions operate in
                                    // f32; allow their rounding noise separately.
                                    assert!(
                                        (raster[channel] - reference[channel]).abs()
                                            <= selection.error.color + 2e-6
                                    );
                                    assert!(
                                        (raster[channel] - reference[channel]).abs() <= COLOR_LIMIT
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
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
                let chosen = QualityState::default().update(&bounds, axes(screen));
                assert!(
                    !chosen.capped,
                    "size={size}, screen={screen}, error={:?}",
                    chosen.error
                );
                let n = chosen.key.subdivisions;
                for patch in patches(&grid) {
                    for y in 0..n {
                        for x in 0..n {
                            for offset in [
                                DVec2::new(0.2, 0.7),
                                DVec2::new(0.7, 0.2),
                                DVec2::splat(0.5),
                            ] {
                                let a = DVec2::new(x as f64, y as f64) / n as f64;
                                let p = a + offset / n as f64;
                                let q = evaluate(&patch, p);
                                let corners = if offset.x >= offset.y {
                                    [
                                        (a, 1.0 - offset.x),
                                        (a + DVec2::X / n as f64, offset.x - offset.y),
                                        (a + DVec2::ONE / n as f64, offset.y),
                                    ]
                                } else {
                                    [
                                        (a, 1.0 - offset.y),
                                        (a + DVec2::ONE / n as f64, offset.x),
                                        (a + DVec2::Y / n as f64, offset.y - offset.x),
                                    ]
                                };
                                let approximate: DVec2 = corners
                                    .into_iter()
                                    .map(|(uv, w)| {
                                        let q = evaluate(&patch, uv);
                                        DVec2::new(q[0], q[1]) * w
                                    })
                                    .sum();
                                let error =
                                    (DVec2::new(q[0], q[1]) - approximate).length() * screen;
                                assert!(error <= chosen.error.geometry + 1e-9);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "manual release-mode performance and quality-characteristic gate"]
    #[expect(clippy::print_stderr, reason = "Allowed in a manual performance test.")]
    fn animated_workload_performance_and_tiers() {
        use std::{hint::black_box, time::Instant};

        let mut gradients: Vec<_> = (0..64).map(stress_mesh).collect();
        let mut quality: Vec<_> = (0..64).map(|_| QualityState::default()).collect();
        let mut samples = Vec::with_capacity(1_000);
        let mut tiers = [0usize; MAX_SUBDIVISIONS + 1];
        let mut triangles = 0;

        for frame in 0..1_100 {
            let started = Instant::now();
            tiers.fill(0);
            triangles = 0;
            for (index, (mesh, state)) in gradients.iter_mut().zip(&mut quality).enumerate() {
                let phase = frame as f32 * 0.03 + index as f32 * 0.07;
                mesh.try_edit_points(|points| {
                    for row in 1..3 {
                        for column in 1..3 {
                            let point = &mut points[row * 4 + column];
                            let offset = phase + row as f32 * 0.8 + column as f32 * 0.6;
                            point.position = Vec2::new(
                                column as f32 / 3.0 + ops::sin(offset) * 0.022,
                                row as f32 / 3.0 + ops::sin(offset * 0.83) * 0.018,
                            );
                        }
                    }
                    Ok(())
                })
                .unwrap();
                let selection = state.update(&SurfaceBounds::new(mesh), axes(256.0));
                tiers[selection.key.subdivisions] += 1;
                triangles += selection.key.triangles();
                black_box(selection);
            }
            if frame >= 100 {
                samples.push(started.elapsed());
            }
        }

        samples.sort_unstable();
        let median = samples[samples.len() / 2];
        let p95 = samples[samples.len() * 95 / 100];
        let editor = editor_mesh();
        let editor_selection = QualityState::default().update(
            &SurfaceBounds::new(&editor),
            [DVec2::X * 1_120.0, DVec2::Y * 956.0],
        );
        let full_capacity = QualityState::default().update(
            &SurfaceBounds::new(&mesh(16, 0.0, 1.0, MeshGradientColorSpace::LinearRgba)),
            axes(256.0),
        );
        let tier_counts: Vec<_> = tiers
            .iter()
            .enumerate()
            .filter_map(|(tier, count)| (*count > 0).then_some((tier, *count)))
            .collect();
        eprintln!(
            "mesh-gradient workload: CPU median={median:?} p95={p95:?}; stress tiers={tier_counts:?} triangles={triangles}; editor tier={} triangles={}; 16x16 tier={} triangles={}",
            editor_selection.key.subdivisions,
            editor_selection.key.triangles(),
            full_capacity.key.subdivisions,
            full_capacity.key.triangles(),
        );
        assert!(p95 <= core::time::Duration::from_millis(4));
        assert!(!editor_selection.capped);
        assert!(!full_capacity.capped);
    }

    fn stress_mesh(index: usize) -> MeshGradient {
        let points = (0..16)
            .map(|point| {
                let column = point % 4;
                let row = point / 4;
                let x = column as f32 / 3.0;
                let y = row as f32 / 3.0;
                let phase = index as f32 * 0.13;
                MeshGradientPoint::new(
                    Vec2::new(x, y),
                    Color::oklaba(
                        0.62 + 0.18 * x - 0.06 * y,
                        0.14 * ops::sin(phase + x * 2.1) - 0.08 * y,
                        0.14 * ops::sin(phase * 0.7 + y * 2.3) - 0.08 * x,
                        0.72 + 0.28 * (1.0 - x * y),
                    ),
                )
            })
            .collect();
        MeshGradient::new(4, 4, points).unwrap()
    }

    fn editor_mesh() -> MeshGradient {
        let points = (0..9)
            .map(|index| {
                let column = index % 3;
                let row = index / 3;
                let x = column as f32 / 2.0;
                let y = row as f32 / 2.0;
                let mut position = Vec2::new(x, y);
                if column == 1 && row == 1 {
                    position += Vec2::new(0.045, -0.03);
                }
                MeshGradientPoint::new(
                    position,
                    Color::oklaba(
                        0.68 + 0.12 * x - 0.08 * y,
                        0.15 - 0.27 * x + 0.04 * y,
                        0.13 + 0.04 * x - 0.25 * y,
                        1.0 - 0.22 * x * y,
                    ),
                )
            })
            .collect();
        MeshGradient::new(3, 3, points).unwrap()
    }
}
