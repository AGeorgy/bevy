//! Throwaway bicubic mesh prototype; intentionally independent of Bevy.
//!
//! Each cell is a tensor Catmull–Rom interpolant with extrapolated boundary
//! samples, represented as a Bezier patch. Colors use the same cubic basis.
//! A sufficient (not necessary) geometry certificate proves the symmetric part
//! of the Jacobian positive definite on the WHOLE logical rectangle. Integrating
//! (F(p)-F(q))·(p-q) along its straight segment then proves global injectivity.
//! Bezier derivative coefficients bound derivatives everywhere by convexity.
//! Outward-rounded f64 intervals enclose coefficient construction and the test.
//! This rejects some perfectly valid surfaces (notably rotations >=90 degrees
//! and strongly skewed grids). It certifies the mathematical patch surface,
//! not rasterization or f32 vertex conversion. `try_tessellate` separately
//! certifies the actual f32 triangles before subsequent GPU operations.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub position: [f32; 2],
    /// Coordinates in the caller's chosen interpolation color space, plus alpha.
    pub color: [f32; 4],
}
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 2],
    pub color: [f32; 4],
}
#[derive(Clone, Debug, PartialEq)]
pub enum MeshError {
    Dimensions,
    PointCount { expected: usize, actual: usize },
    NonFinite { point: usize },
    AlphaOutOfRange { point: usize },
    UncertifiedGeometry { cell: [usize; 2] },
    TessellationBudget,
    UncertifiedTriangle { triangle: usize },
    DiscontinuousEdge,
}
#[derive(Clone, Debug)]
pub struct MeshSurface {
    width: usize,
    height: usize,
    points: Vec<Point>,
    cells: Vec<[[f64; 6]; 16]>,
    margin: f64,
}

#[derive(Clone, Copy, Debug)]
struct I {
    lo: f64,
    hi: f64,
}
fn down(x: f64) -> f64 {
    if x == f64::NEG_INFINITY {
        x
    } else if x == 0.0 {
        -f64::from_bits(1)
    } else {
        f64::from_bits(if x > 0.0 {
            x.to_bits() - 1
        } else {
            x.to_bits() + 1
        })
    }
}
fn up(x: f64) -> f64 {
    -down(-x)
}
impl I {
    fn new(x: f64) -> Self {
        Self { lo: x, hi: x }
    }
    fn add(self, b: Self) -> Self {
        Self {
            lo: down(self.lo + b.lo),
            hi: up(self.hi + b.hi),
        }
    }
    fn sub(self, b: Self) -> Self {
        Self {
            lo: down(self.lo - b.hi),
            hi: up(self.hi - b.lo),
        }
    }
    fn scale(self, x: f64) -> Self {
        if x >= 0.0 {
            Self {
                lo: down(self.lo * x),
                hi: up(self.hi * x),
            }
        } else {
            Self {
                lo: down(self.hi * x),
                hi: up(self.lo * x),
            }
        }
    }
    fn div(self, x: f64) -> Self {
        Self {
            lo: down(self.lo / x),
            hi: up(self.hi / x),
        }
    }
    fn mid(self) -> f64 {
        self.lo + (self.hi - self.lo) * 0.5
    }
}
type C = [I; 6];
fn plus(a: C, b: C) -> C {
    std::array::from_fn(|k| a[k].add(b[k]))
}
fn minus(a: C, b: C) -> C {
    std::array::from_fn(|k| a[k].sub(b[k]))
}
fn scale(a: C, s: f64) -> C {
    a.map(|v| v.scale(s))
}
fn cubic(p: [C; 4]) -> [C; 4] {
    [
        p[1],
        plus(p[1], minus(p[2], p[0]).map(|v| v.div(6.0))),
        minus(p[2], minus(p[3], p[1]).map(|v| v.div(6.0))),
        p[2],
    ]
}
impl MeshSurface {
    pub fn try_new(width: usize, height: usize, points: Vec<Point>) -> Result<Self, MeshError> {
        if width < 2 || height < 2 || width > i32::MAX as usize || height > i32::MAX as usize {
            return Err(MeshError::Dimensions);
        }
        let expected = width.checked_mul(height).ok_or(MeshError::Dimensions)?;
        if points.len() != expected {
            return Err(MeshError::PointCount {
                expected,
                actual: points.len(),
            });
        }
        for (point, p) in points.iter().enumerate() {
            if p.position
                .iter()
                .chain(p.color.iter())
                .any(|v| !v.is_finite())
            {
                return Err(MeshError::NonFinite { point });
            }
            if !(0.0..=1.0).contains(&p.color[3]) {
                return Err(MeshError::AlphaOutOfRange { point });
            }
        }
        let mut mesh = Self {
            width,
            height,
            points,
            cells: Vec::new(),
            margin: f64::INFINITY,
        };
        for y in 0..height - 1 {
            for x in 0..width - 1 {
                let rows: [[C; 4]; 4] = std::array::from_fn(|j| {
                    cubic(std::array::from_fn(|i| {
                        mesh.extended(x as i64 + i as i64 - 1, y as i64 + j as i64 - 1)
                    }))
                });
                let columns: [[C; 4]; 4] =
                    std::array::from_fn(|i| cubic(std::array::from_fn(|j| rows[j][i])));
                let controls: [C; 16] = std::array::from_fn(|k| columns[k % 4][k / 4]);
                let mut a = f64::INFINITY;
                let mut d = f64::INFINITY;
                let mut b = 0.0_f64;
                let mut c = 0.0_f64;
                for j in 0..4 {
                    for i in 0..3 {
                        let dx = minus(controls[j * 4 + i + 1], controls[j * 4 + i]);
                        a = a.min(dx[0].scale(3.0 * (width - 1) as f64).lo);
                        let cy = dx[1].scale(3.0 * (width - 1) as f64);
                        c = c.max(cy.lo.abs().max(cy.hi.abs()));
                    }
                }
                for j in 0..3 {
                    for i in 0..4 {
                        let dy = minus(controls[(j + 1) * 4 + i], controls[j * 4 + i]);
                        d = d.min(dy[1].scale(3.0 * (height - 1) as f64).lo);
                        let bx = dy[0].scale(3.0 * (height - 1) as f64);
                        b = b.max(bx.lo.abs().max(bx.hi.abs()));
                    }
                }
                let off = up(up(b + c) * 0.5);
                let margin = down(down(a * d) - up(off * off));
                if !(a > 0.0 && d > 0.0 && margin > 0.0 && margin.is_finite()) {
                    return Err(MeshError::UncertifiedGeometry { cell: [x, y] });
                }
                mesh.margin = mesh.margin.min(margin);
                mesh.cells.push(controls.map(|p| p.map(I::mid)));
            }
        }
        Ok(mesh)
    }
    fn extended(&self, x: i64, y: i64) -> C {
        fn indices(v: i64, n: usize) -> [(usize, f64); 2] {
            if v < 0 {
                [(0, 2.0), (1, -1.0)]
            } else if v >= n as i64 {
                [(n - 1, 2.0), (n - 2, -1.0)]
            } else {
                [(v as usize, 1.0), (v as usize, 0.0)]
            }
        }
        let mut result = [I::new(0.0); 6];
        for (iy, wy) in indices(y, self.height) {
            for (ix, wx) in indices(x, self.width) {
                if wx * wy == 0.0 {
                    continue;
                }
                let p = self.points[iy * self.width + ix];
                let q = [
                    p.position[0],
                    p.position[1],
                    p.color[0],
                    p.color[1],
                    p.color[2],
                    p.color[3],
                ]
                .map(|v| I::new(v as f64));
                result = plus(result, scale(q, wx * wy));
            }
        }
        result
    }
    /// Failure leaves every previous point and cached patch unchanged.
    pub fn try_replace_points(&mut self, points: Vec<Point>) -> Result<(), MeshError> {
        let next = Self::try_new(self.width, self.height, points)?;
        *self = next;
        Ok(())
    }
    pub fn points(&self) -> &[Point] {
        &self.points
    }
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }
    /// Minimum conservative determinant bound for the symmetric Jacobian.
    pub fn certificate(&self) -> f64 {
        self.margin
    }
    pub fn sample(&self, cell_x: usize, cell_y: usize, u: f32, v: f32) -> ([f32; 2], [f32; 4]) {
        assert!(cell_x < self.width - 1 && cell_y < self.height - 1);
        assert!(u.is_finite() && v.is_finite());
        fn basis(t: f64) -> [f64; 4] {
            let s = 1.0 - t;
            [s * s * s, 3.0 * t * s * s, 3.0 * t * t * s, t * t * t]
        }
        let bu = basis(u.clamp(0.0, 1.0) as f64);
        let bv = basis(v.clamp(0.0, 1.0) as f64);
        let cell = &self.cells[cell_y * (self.width - 1) + cell_x];
        let mut out = [0.0; 6];
        for j in 0..4 {
            for i in 0..4 {
                for k in 0..6 {
                    out[k] += cell[j * 4 + i][k] * bu[i] * bv[j];
                }
            }
        }
        let out = out.map(|v| v.clamp(-(f32::MAX as f64), f32::MAX as f64) as f32);
        (
            [out[0], out[1]],
            [out[2], out[3], out[4], out[5].clamp(0.0, 1.0)],
        )
    }
    /// Uniform tessellation: duplicate shared-edge vertices but identical samples.
    /// This is a cost/appearance probe, not a production adaptive tessellator.
    pub fn tessellate(&self, subdivisions: usize) -> (Vec<Vertex>, Vec<u32>) {
        self.try_tessellate(subdivisions)
            .expect("prototype tessellation must certify")
    }
    /// Certifies the actual f32 vertex positions as a continuous piecewise-affine
    /// strongly monotone map of the logical rectangle. Shared patch edges must
    /// be bit-identical, and each triangle's constant symmetric Jacobian must be
    /// positive definite. The same segment-integral injectivity proof applies.
    /// This does not cover subsequent GPU transforms, clipping, or rasterization.
    pub fn try_tessellate(
        &self,
        subdivisions: usize,
    ) -> Result<(Vec<Vertex>, Vec<u32>), MeshError> {
        if !(1..=1024).contains(&subdivisions) {
            return Err(MeshError::TessellationBudget);
        }
        let count = (self.width - 1)
            .checked_mul(self.height - 1)
            .and_then(|n| n.checked_mul((subdivisions + 1) * (subdivisions + 1)))
            .ok_or(MeshError::TessellationBudget)?;
        if count > 1_000_000 {
            return Err(MeshError::TessellationBudget);
        }
        let mut vertices = Vec::with_capacity(count);
        let mut indices = Vec::new();
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let base = vertices.len() as u32;
                let stride = (subdivisions + 1) as u32;
                for j in 0..=subdivisions {
                    for i in 0..=subdivisions {
                        let (position, color) = self.sample(
                            x,
                            y,
                            i as f32 / subdivisions as f32,
                            j as f32 / subdivisions as f32,
                        );
                        vertices.push(Vertex { position, color });
                    }
                }
                for j in 0..subdivisions {
                    for i in 0..subdivisions {
                        let a = base + j as u32 * stride + i as u32;
                        indices.extend([a, a + 1, a + stride, a + 1, a + stride + 1, a + stride]);
                    }
                }
            }
        }
        let stride = subdivisions + 1;
        let patch_size = stride * stride;
        for y in 0..self.height - 1 {
            for x in 0..self.width - 1 {
                let base = (y * (self.width - 1) + x) * patch_size;
                for k in 0..=subdivisions {
                    let pair = if x > 0 {
                        Some((
                            base + k * stride,
                            base - patch_size + k * stride + subdivisions,
                        ))
                    } else {
                        None
                    };
                    let pair2 = if y > 0 {
                        Some((
                            base + k,
                            base - (self.width - 1) * patch_size + subdivisions * stride + k,
                        ))
                    } else {
                        None
                    };
                    for (a, b) in pair.into_iter().chain(pair2) {
                        if vertices[a].position.map(f32::to_bits)
                            != vertices[b].position.map(f32::to_bits)
                        {
                            return Err(MeshError::DiscontinuousEdge);
                        }
                    }
                }
            }
        }
        for (triangle, ids) in indices.chunks_exact(3).enumerate() {
            let p: [[I; 2]; 3] = std::array::from_fn(|i| {
                vertices[ids[i] as usize].position.map(|v| I::new(v as f64))
            });
            let (dx, dy): ([I; 2], [I; 2]) = if triangle % 2 == 0 {
                (
                    std::array::from_fn(|k| p[1][k].sub(p[0][k])),
                    std::array::from_fn(|k| p[2][k].sub(p[0][k])),
                )
            } else {
                (
                    std::array::from_fn(|k| p[1][k].sub(p[2][k])),
                    std::array::from_fn(|k| p[1][k].sub(p[0][k])),
                )
            };
            // Integer factors are exact: the vertex budget bounds them below 2^53.
            let dx = dx.map(|v| v.scale(((self.width - 1) * subdivisions) as f64));
            let dy = dy.map(|v| v.scale(((self.height - 1) * subdivisions) as f64));
            let off = dx[1].add(dy[0]).scale(0.5);
            let max_off = off.lo.abs().max(off.hi.abs());
            let margin = down(down(dx[0].lo * dy[1].lo) - up(max_off * max_off));
            if !(dx[0].lo > 0.0 && dy[1].lo > 0.0 && margin > 0.0 && margin.is_finite()) {
                return Err(MeshError::UncertifiedTriangle { triangle });
            }
        }
        Ok((vertices, indices))
    }
}
