use super::InterpolationColorSpace;
use alloc::vec::Vec;
use bevy_color::{
    Alpha, Color, ColorToComponents, Hsla, Hsva, LinearRgba, Okhsla, Oklaba, Oklcha, Srgba,
};
use bevy_math::Vec2;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
#[cfg(feature = "serialize")]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};
use thiserror::Error;

/// The smallest supported mesh-gradient dimension.
pub const MIN_MESH_GRADIENT_DIMENSION: usize = 2;

/// The largest supported mesh-gradient dimension.
///
/// A mesh gradient can therefore contain at most 256 colored points. This bound
/// keeps validation work and the WebGL2-compatible GPU transport finite.
pub const MAX_MESH_GRADIENT_DIMENSION: usize = 16;

/// A position and color in a [`MeshGradient`] control grid.
///
/// Positions use node-relative coordinates. `(0, 0)` is the node's top-left
/// corner and `(1, 1)` is its bottom-right corner. Finite positions outside that
/// range are supported when the resulting surface can still be certified.
#[derive(Clone, Copy, Debug, PartialEq, Reflect)]
#[reflect(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    reflect(Serialize, Deserialize)
)]
pub struct MeshGradientPoint {
    /// The point's node-relative position.
    pub position: Vec2,
    /// The point's color.
    pub color: Color,
}

impl MeshGradientPoint {
    /// Creates a colored mesh-gradient point.
    pub const fn new(position: Vec2, color: Color) -> Self {
        Self { position, color }
    }
}

/// A color space supported by mesh-gradient interpolation.
///
/// Supports the same color spaces and hue paths as other UI gradients. All
/// variants interpolate alpha separately from the color coordinates.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
#[reflect(Default, Clone, PartialEq, Debug, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    reflect(Serialize, Deserialize)
)]
pub enum MeshGradientColorSpace {
    /// Interpolate in `OKLab` for perceptually smoother transitions.
    Oklaba,
    /// Interpolate in OKLCH along the shorter hue path.
    Oklcha,
    /// Interpolate in OKLCH along the longer hue path.
    OklchaLong,
    /// Interpolate in HSL along the shorter hue path.
    Hsla,
    /// Interpolate in HSL along the longer hue path.
    HslaLong,
    /// Interpolate in HSV along the shorter hue path.
    Hsva,
    /// Interpolate in HSV along the longer hue path.
    HsvaLong,
    /// Interpolate in OKHSL along the shorter hue path.
    Okhsla,
    /// Interpolate in OKHSL along the longer hue path.
    OkhslaLong,
    /// Interpolate in sRGB.
    Srgba,
    /// Interpolate in linear RGB. This is the fastest option because the
    /// fragment shader does not need a color-space conversion. This is the
    /// default.
    #[default]
    LinearRgba,
}

/// Controls how mesh-gradient colors are evaluated.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
#[reflect(Default, Clone, PartialEq, Debug, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    reflect(Serialize, Deserialize)
)]
pub enum MeshGradientColorInterpolation {
    /// Evaluate bilinear colors at tessellation vertices and let the rasterizer
    /// interpolate across each triangle. This is the mobile-friendly default.
    /// Hue-based spaces evaluate bilinear colors per fragment to preserve the
    /// selected hue path across the circular hue boundary.
    #[default]
    Vertex,
    /// Evaluate a tensor-product Catmull-Rom color surface per fragment. This
    /// gives smooth derivatives across cells at a higher fragment cost.
    /// Hue coordinates use the same continuous winding field as bilinear mode.
    Bicubic,
}

/// Controls whether a mesh gradient accepts folded surface geometry.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq, Hash, Reflect)]
#[reflect(Default, Clone, PartialEq, Debug, Hash)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize, serde::Deserialize),
    reflect(Serialize, Deserialize)
)]
pub enum MeshGradientGeometry {
    /// Require a conservative proof that the complete surface does not fold or
    /// overlap.
    #[default]
    NonFolding,
    /// Accept any finite point arrangement, including coincident and crossing
    /// points that create sharp transitions or overlapping folds. The renderer
    /// omits locally reversed triangles so a folded layer does not cover the
    /// forward-facing surface with a narrow overlap artifact.
    AllowFolds,
}

impl MeshGradientColorSpace {
    /// Whether interpolation follows a circular hue coordinate.
    pub const fn is_hue_based(self) -> bool {
        !matches!(self, Self::Oklaba | Self::Srgba | Self::LinearRgba)
    }
}

impl From<MeshGradientColorSpace> for InterpolationColorSpace {
    fn from(value: MeshGradientColorSpace) -> Self {
        match value {
            MeshGradientColorSpace::Oklaba => Self::Oklaba,
            MeshGradientColorSpace::Oklcha => Self::Oklcha,
            MeshGradientColorSpace::OklchaLong => Self::OklchaLong,
            MeshGradientColorSpace::Hsla => Self::Hsla,
            MeshGradientColorSpace::HslaLong => Self::HslaLong,
            MeshGradientColorSpace::Hsva => Self::Hsva,
            MeshGradientColorSpace::HsvaLong => Self::HsvaLong,
            MeshGradientColorSpace::Okhsla => Self::Okhsla,
            MeshGradientColorSpace::OkhslaLong => Self::OkhslaLong,
            MeshGradientColorSpace::Srgba => Self::Srgba,
            MeshGradientColorSpace::LinearRgba => Self::LinearRgba,
        }
    }
}

impl From<InterpolationColorSpace> for MeshGradientColorSpace {
    fn from(value: InterpolationColorSpace) -> Self {
        match value {
            InterpolationColorSpace::Oklaba => Self::Oklaba,
            InterpolationColorSpace::Oklcha => Self::Oklcha,
            InterpolationColorSpace::OklchaLong => Self::OklchaLong,
            InterpolationColorSpace::Hsla => Self::Hsla,
            InterpolationColorSpace::HslaLong => Self::HslaLong,
            InterpolationColorSpace::Hsva => Self::Hsva,
            InterpolationColorSpace::HsvaLong => Self::HsvaLong,
            InterpolationColorSpace::Okhsla => Self::Okhsla,
            InterpolationColorSpace::OkhslaLong => Self::OkhslaLong,
            InterpolationColorSpace::Srgba => Self::Srgba,
            InterpolationColorSpace::LinearRgba => Self::LinearRgba,
        }
    }
}

/// An error returned when constructing or editing a [`MeshGradient`].
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum MeshGradientError {
    /// A dimension is smaller than two.
    #[error("mesh-gradient dimensions must both be at least 2 (got {width}x{height})")]
    DimensionsTooSmall {
        /// Number of columns supplied by the caller.
        width: usize,
        /// Number of rows supplied by the caller.
        height: usize,
    },
    /// Multiplying the dimensions overflowed `usize`.
    #[error("mesh-gradient dimensions overflow their point count ({width}x{height})")]
    DimensionsOverflow {
        /// Number of columns supplied by the caller.
        width: usize,
        /// Number of rows supplied by the caller.
        height: usize,
    },
    /// A dimension exceeds [`MAX_MESH_GRADIENT_DIMENSION`].
    #[error(
        "mesh-gradient dimensions exceed the {maximum}x{maximum} capacity (got {width}x{height})"
    )]
    CapacityExceeded {
        /// Number of columns supplied by the caller.
        width: usize,
        /// Number of rows supplied by the caller.
        height: usize,
        /// Largest supported dimension.
        maximum: usize,
    },
    /// The point count does not match the declared dimensions.
    #[error("mesh-gradient expected {expected} points but received {actual}")]
    PointCount {
        /// Required row-major point count.
        expected: usize,
        /// Supplied point count.
        actual: usize,
    },
    /// A point index is outside the grid.
    #[error("mesh-gradient point index {index} is outside its {point_count}-point grid")]
    PointIndexOutOfBounds {
        /// Supplied point index.
        index: usize,
        /// Number of points in the current grid.
        point_count: usize,
    },
    /// A point position contains a non-finite component.
    #[error("mesh-gradient point {point} has a non-finite position")]
    NonFinitePosition {
        /// Index of the invalid point.
        point: usize,
    },
    /// A point color contains a non-finite component or cannot be converted to
    /// finite coordinates in the selected interpolation space.
    #[error("mesh-gradient point {point} has a non-finite color")]
    NonFiniteColor {
        /// Index of the invalid point.
        point: usize,
    },
    /// A point's input alpha lies outside `[0, 1]`.
    #[error("mesh-gradient point {point} has alpha outside 0..=1")]
    AlphaOutOfRange {
        /// Index of the invalid point.
        point: usize,
    },
    /// Inferred bicubic control data cannot be represented finitely by the GPU.
    #[error("mesh-gradient patch ({column}, {row}) derives non-finite control data")]
    NonFiniteDerived {
        /// Patch column.
        column: usize,
        /// Patch row.
        row: usize,
    },
    /// Under [`MeshGradientGeometry::NonFolding`], a patch has collapsed to an
    /// area with no usable two-dimensional extent.
    #[error("mesh-gradient patch ({column}, {row}) is degenerate")]
    DegenerateGeometry {
        /// Patch column.
        column: usize,
        /// Patch row.
        row: usize,
    },
    /// Under [`MeshGradientGeometry::NonFolding`], the conservative
    /// continuous-surface proof could not establish that a patch participates
    /// in a globally non-folding surface.
    #[error("mesh-gradient patch ({column}, {row}) could not be certified")]
    UncertifiedGeometry {
        /// Patch column.
        column: usize,
        /// Patch row.
        row: usize,
    },
}

/// A checked, smooth two-dimensional UI gradient controlled by a colored grid.
///
/// `MeshGradient` stores only values that pass its dimensional and numeric
/// invariants, plus the selected [`MeshGradientGeometry`] policy. Its fields are
/// private, reflection is opaque, and deserialization re-enters the checked
/// constructor. Use the checked edit methods for animation; a rejected edit
/// leaves the previous gradient intact.
/// Grids contain between 2 and 16 columns and rows, inclusive, and points are
/// supplied in row-major order. Positions are relative to the UI node: `(0, 0)`
/// is its top-left and `(1, 1)` is its bottom-right. The surface may cover only
/// part of the node, and any uncovered area remains transparent.
///
/// Geometry is a tensor-product Catmull-Rom surface represented as bicubic
/// Bézier patches. Boundary points are inferred by linear extrapolation. The
/// validator uses outward-rounded interval arithmetic to prove a sufficient
/// strong-monotonicity condition over the complete curved surface. This proves
/// that accepted surfaces do not fold or overlap, but it is conservative and
/// can reject otherwise valid rotations or strongly skewed grids. Use
/// [`MeshGradientGeometry::AllowFolds`] for deliberate collapsed or folded
/// geometry such as sharp transitions.
///
/// RGB coordinates may be HDR. Colors are evaluated at tessellation vertices
/// from the four cell colors by default, then interpolated by the rasterizer.
/// For Cartesian spaces, this mode stays within the convex hull of each cell's
/// interpolation-space coordinates. Hue-based spaces choose a consistent
/// winding along the first grid column and then each row, using the selected
/// short or long hue path. Independent path choices around a two-dimensional
/// loop can conflict, so other vertical edges follow that continuous field.
/// Achromatic points borrow the nearest chromatic grid point's hue. Colors are
/// evaluated per fragment, and hue wraps only during final RGB conversion.
/// [`MeshGradientColorInterpolation::Bicubic`] instead evaluates the inferred
/// Catmull-Rom color surface per fragment for smooth derivatives;
/// it can overshoot the neighboring color coordinates. Derived alpha is
/// clamped by the renderer. Input alpha must be in `[0, 1]`, and all input and
/// derived control values must remain finite.
/// Colors interpolate in linear RGB by default; all UI gradient spaces are
/// available through [`MeshGradientColorSpace`]. Near saturated blue, OKHSL
/// meshes blend toward an Oklab surface derived from the same colored points
/// to avoid a gamut discontinuity while keeping control-point colors intact.
/// Alpha always interpolates
/// separately from the color coordinates. The linear RGB default avoids
/// fragment color conversion and is the lowest-cost option.
///
/// The renderer chooses crack-free adaptive tessellation from patch-local
/// curvature, the transform, and physical on-screen size. Each bicubic patch
/// receives independent power-of-two subdivision counts for its two axes, so a
/// curved region can refine without increasing tessellation across a complete
/// row or column. Where adjacent patches use different counts, vertices on the
/// finer edge are evaluated along the coarser edge's piecewise-linear chords.
/// This keeps the surface connected while preserving local refinement.
///
/// The CPU selects and caches this parameter-space triangle topology. The
/// vertex shader evaluates surface positions and, in the default mode, colors
/// from the control points. Every patch starts with a 2x2 subdivision baseline.
/// The default vertex-color path selectively refines patches whose bilinear
/// color field would reveal the triangle split, while cells that are already
/// subpixel stop refining. Adaptive tessellation allows up to four physical
/// pixels of geometric approximation error. Authors do not supply subdivision
/// counts. If those requirements exceed the renderer's bounded triangle budget,
/// Bevy renders the finest supported topology and emits a rate-limited
/// diagnostic.
#[derive(Clone, Debug, PartialEq, Reflect)]
#[reflect(opaque)]
#[reflect(Clone, PartialEq, Debug)]
#[cfg_attr(
    feature = "serialize",
    derive(serde::Serialize),
    serde(rename(serialize = "SerializedMeshGradientRef")),
    reflect(Serialize, Deserialize)
)]
pub struct MeshGradient {
    width: usize,
    height: usize,
    points: Vec<MeshGradientPoint>,
    color_space: MeshGradientColorSpace,
    color_interpolation: MeshGradientColorInterpolation,
    geometry: MeshGradientGeometry,
}

impl MeshGradient {
    /// Creates a checked mesh gradient in the default mesh-gradient color
    /// space.
    ///
    /// `points` must contain `width * height` values in row-major order. This
    /// uses [`MeshGradientGeometry::NonFolding`].
    pub fn new(
        width: usize,
        height: usize,
        points: Vec<MeshGradientPoint>,
    ) -> Result<Self, MeshGradientError> {
        Self::new_in_color_space(width, height, points, MeshGradientColorSpace::default())
    }

    /// Creates a checked mesh gradient in the selected interpolation color
    /// space.
    ///
    /// `points` must contain `width * height` values in row-major order. This
    /// uses [`MeshGradientGeometry::NonFolding`].
    pub fn new_in_color_space(
        width: usize,
        height: usize,
        points: Vec<MeshGradientPoint>,
        color_space: MeshGradientColorSpace,
    ) -> Result<Self, MeshGradientError> {
        Self::new_with_geometry(
            width,
            height,
            points,
            color_space,
            MeshGradientGeometry::NonFolding,
        )
    }

    /// Creates a checked mesh gradient with the selected color space and
    /// geometry policy.
    ///
    /// `points` must contain `width * height` values in row-major order.
    /// [`MeshGradientGeometry::AllowFolds`] still validates dimensions, point
    /// count, numeric finiteness, alpha, and derived GPU control values.
    pub fn new_with_geometry(
        width: usize,
        height: usize,
        points: Vec<MeshGradientPoint>,
        color_space: MeshGradientColorSpace,
        geometry: MeshGradientGeometry,
    ) -> Result<Self, MeshGradientError> {
        Self::validate(width, height, &points, color_space, geometry)?;
        Ok(Self {
            width,
            height,
            points,
            color_space,
            color_interpolation: MeshGradientColorInterpolation::default(),
            geometry,
        })
    }

    /// Returns the number of point columns.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns the number of point rows.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Returns `(columns, rows)` for the point grid.
    pub const fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Returns the row-major colored points.
    pub fn points(&self) -> &[MeshGradientPoint] {
        &self.points
    }

    /// Returns a point by row-major index.
    pub fn point(&self, index: usize) -> Option<&MeshGradientPoint> {
        self.points.get(index)
    }

    /// Returns a point by `(column, row)`.
    pub fn point_at(&self, column: usize, row: usize) -> Option<&MeshGradientPoint> {
        (column < self.width && row < self.height).then(|| &self.points[row * self.width + column])
    }

    /// Returns the mesh's interpolation color space.
    pub const fn color_space(&self) -> MeshGradientColorSpace {
        self.color_space
    }

    /// Returns the mesh's color interpolation mode.
    pub const fn color_interpolation(&self) -> MeshGradientColorInterpolation {
        self.color_interpolation
    }

    /// Changes color interpolation without rebuilding the checked control grid.
    pub fn set_color_interpolation(&mut self, interpolation: MeshGradientColorInterpolation) {
        self.color_interpolation = interpolation;
    }

    /// Returns this mesh gradient with the requested color interpolation mode.
    pub const fn with_color_interpolation(
        mut self,
        interpolation: MeshGradientColorInterpolation,
    ) -> Self {
        self.color_interpolation = interpolation;
        self
    }

    /// Returns the mesh's geometry validation policy.
    pub const fn geometry(&self) -> MeshGradientGeometry {
        self.geometry
    }

    /// Replaces one colored point after validating the complete candidate.
    pub fn try_set_point(
        &mut self,
        index: usize,
        point: MeshGradientPoint,
    ) -> Result<(), MeshGradientError> {
        let point_count = self.points.len();
        let Some(current) = self.points.get_mut(index) else {
            return Err(MeshGradientError::PointIndexOutOfBounds { index, point_count });
        };
        let previous = core::mem::replace(current, point);
        if let Err(error) = Self::validate(
            self.width,
            self.height,
            &self.points,
            self.color_space,
            self.geometry,
        ) {
            self.points[index] = previous;
            return Err(error);
        }
        Ok(())
    }

    /// Replaces one point position after validating the complete candidate.
    pub fn try_set_position(
        &mut self,
        index: usize,
        position: Vec2,
    ) -> Result<(), MeshGradientError> {
        let Some(point) = self.point(index).copied() else {
            return Err(MeshGradientError::PointIndexOutOfBounds {
                index,
                point_count: self.points.len(),
            });
        };
        self.try_set_point(index, MeshGradientPoint { position, ..point })
    }

    /// Replaces one point color after validating the complete candidate.
    pub fn try_set_color(&mut self, index: usize, color: Color) -> Result<(), MeshGradientError> {
        let Some(point) = self.point(index).copied() else {
            return Err(MeshGradientError::PointIndexOutOfBounds {
                index,
                point_count: self.points.len(),
            });
        };
        self.try_set_point(index, MeshGradientPoint { color, ..point })
    }

    /// Applies an atomic multi-point edit.
    ///
    /// The closure edits a temporary copy. The copy replaces the stored points
    /// only after the complete mesh validates successfully. Returning an error
    /// from the closure or failing validation leaves `self` unchanged.
    pub fn try_edit_points<F>(&mut self, edit: F) -> Result<(), MeshGradientError>
    where
        F: FnOnce(&mut [MeshGradientPoint]) -> Result<(), MeshGradientError>,
    {
        let mut candidate = self.points.clone();
        edit(&mut candidate)?;
        self.try_replace_points(candidate)
    }

    /// Replaces all points without changing the grid dimensions.
    pub fn try_replace_points(
        &mut self,
        points: Vec<MeshGradientPoint>,
    ) -> Result<(), MeshGradientError> {
        Self::validate(
            self.width,
            self.height,
            &points,
            self.color_space,
            self.geometry,
        )?;
        self.points = points;
        Ok(())
    }

    /// Replaces the entire grid atomically while retaining the color space.
    pub fn try_replace_grid(
        &mut self,
        width: usize,
        height: usize,
        points: Vec<MeshGradientPoint>,
    ) -> Result<(), MeshGradientError> {
        Self::validate(width, height, &points, self.color_space, self.geometry)?;
        self.width = width;
        self.height = height;
        self.points = points;
        Ok(())
    }

    /// Changes the interpolation space after revalidating all derived color
    /// controls.
    pub fn try_set_color_space(
        &mut self,
        color_space: MeshGradientColorSpace,
    ) -> Result<(), MeshGradientError> {
        Self::validate(
            self.width,
            self.height,
            &self.points,
            color_space,
            self.geometry,
        )?;
        self.color_space = color_space;
        Ok(())
    }

    /// Changes the geometry policy after revalidating the complete surface.
    pub fn try_set_geometry(
        &mut self,
        geometry: MeshGradientGeometry,
    ) -> Result<(), MeshGradientError> {
        Self::validate(
            self.width,
            self.height,
            &self.points,
            self.color_space,
            geometry,
        )?;
        self.geometry = geometry;
        Ok(())
    }

    fn validate(
        width: usize,
        height: usize,
        points: &[MeshGradientPoint],
        color_space: MeshGradientColorSpace,
        geometry: MeshGradientGeometry,
    ) -> Result<(), MeshGradientError> {
        let expected = width
            .checked_mul(height)
            .ok_or(MeshGradientError::DimensionsOverflow { width, height })?;
        if width < MIN_MESH_GRADIENT_DIMENSION || height < MIN_MESH_GRADIENT_DIMENSION {
            return Err(MeshGradientError::DimensionsTooSmall { width, height });
        }
        if width > MAX_MESH_GRADIENT_DIMENSION || height > MAX_MESH_GRADIENT_DIMENSION {
            return Err(MeshGradientError::CapacityExceeded {
                width,
                height,
                maximum: MAX_MESH_GRADIENT_DIMENSION,
            });
        }
        if points.len() != expected {
            return Err(MeshGradientError::PointCount {
                expected,
                actual: points.len(),
            });
        }

        let mut values = Vec::with_capacity(points.len());
        for (index, point) in points.iter().enumerate() {
            if !point.position.is_finite() {
                return Err(MeshGradientError::NonFinitePosition { point: index });
            }
            let raw = read_color_components(point.color);
            if raw.iter().any(|component| !component.is_finite()) {
                return Err(MeshGradientError::NonFiniteColor { point: index });
            }
            if !(0.0..=1.0).contains(&point.color.alpha()) {
                return Err(MeshGradientError::AlphaOutOfRange { point: index });
            }
            let color = convert_color_to_interpolation_components(point.color, color_space);
            if color.iter().any(|component| !component.is_finite()) {
                return Err(MeshGradientError::NonFiniteColor { point: index });
            }
            values.push([
                point.position.x as f64,
                point.position.y as f64,
                color[0] as f64,
                color[1] as f64,
                color[2] as f64,
                color[3] as f64,
            ]);
        }

        let gpu_component_range = -(f32::MAX as f64)..=f32::MAX as f64;
        for row in 0..height - 1 {
            for column in 0..width - 1 {
                if geometry == MeshGradientGeometry::NonFolding {
                    validate_cell_corners(width, &values, column, row)?;
                }
                let controls = build_patch_controls(width, height, &values, column, row);
                if controls.iter().flatten().any(|component| {
                    !gpu_component_range.contains(&component.lo)
                        || !gpu_component_range.contains(&component.hi)
                }) {
                    return Err(MeshGradientError::NonFiniteDerived { column, row });
                }
                if geometry == MeshGradientGeometry::NonFolding {
                    certify_patch(width, height, &controls, column, row)?;
                }
            }
        }
        Ok(())
    }
}

fn read_color_components(color: Color) -> [f32; 4] {
    match color {
        Color::Srgba(color) => color.to_f32_array(),
        Color::LinearRgba(color) => color.to_f32_array(),
        Color::Hsla(color) => color.to_f32_array(),
        Color::Hsva(color) => color.to_f32_array(),
        Color::Hwba(color) => color.to_f32_array(),
        Color::Laba(color) => color.to_f32_array(),
        Color::Lcha(color) => color.to_f32_array(),
        Color::Oklaba(color) => color.to_f32_array(),
        Color::Oklcha(color) => color.to_f32_array(),
        Color::Xyza(color) => color.to_f32_array(),
        Color::Okhsla(color) => color.to_f32_array(),
        Color::Okhsva(color) => color.to_f32_array(),
        Color::Okhwba(color) => color.to_f32_array(),
    }
}

fn convert_color_to_interpolation_components(
    color: Color,
    color_space: MeshGradientColorSpace,
) -> [f32; 4] {
    match color_space {
        MeshGradientColorSpace::Oklaba => Oklaba::from(color).to_f32_array(),
        MeshGradientColorSpace::Srgba => Srgba::from(color).to_f32_array(),
        MeshGradientColorSpace::LinearRgba => LinearRgba::from(color).to_f32_array(),
        MeshGradientColorSpace::Oklcha | MeshGradientColorSpace::OklchaLong => {
            let color = Oklcha::from(color);
            [
                color.lightness,
                color.chroma,
                color.hue / 360.0,
                color.alpha,
            ]
        }
        MeshGradientColorSpace::Hsla | MeshGradientColorSpace::HslaLong => {
            let color = Hsla::from(color);
            [
                color.hue / 360.0,
                color.saturation,
                color.lightness,
                color.alpha,
            ]
        }
        MeshGradientColorSpace::Hsva | MeshGradientColorSpace::HsvaLong => {
            let color = Hsva::from(color);
            [
                color.hue / 360.0,
                color.saturation,
                color.value,
                color.alpha,
            ]
        }
        MeshGradientColorSpace::Okhsla | MeshGradientColorSpace::OkhslaLong => {
            let color = Okhsla::from(color);
            [
                color.hue / 360.0,
                color.saturation,
                color.lightness,
                color.alpha,
            ]
        }
    }
}

fn validate_cell_corners(
    width: usize,
    points: &[[f64; 6]],
    column: usize,
    row: usize,
) -> Result<(), MeshGradientError> {
    let indices = [
        row * width + column,
        row * width + column + 1,
        (row + 1) * width + column + 1,
        (row + 1) * width + column,
    ];
    let corners = indices.map(|index| [points[index][0], points[index][1]]);
    let area_twice = (0..4).fold(0.0, |area, index| {
        let next = (index + 1) % 4;
        area + corners[index][0] * corners[next][1] - corners[index][1] * corners[next][0]
    });
    let has_collapsed_edge = (0..4).any(|index| corners[index] == corners[(index + 1) % 4]);
    if area_twice == 0.0 || has_collapsed_edge {
        return Err(MeshGradientError::DegenerateGeometry { column, row });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
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

    fn scale(self, factor: f64) -> Self {
        if factor >= 0.0 {
            Self {
                lo: (self.lo * factor).next_down(),
                hi: (self.hi * factor).next_up(),
            }
        } else {
            Self {
                lo: (self.hi * factor).next_down(),
                hi: (self.lo * factor).next_up(),
            }
        }
    }

    fn divide(self, divisor: f64) -> Self {
        Self {
            lo: (self.lo / divisor).next_down(),
            hi: (self.hi / divisor).next_up(),
        }
    }
}

type Control = [Interval; 6];

fn add_controls(left: Control, right: Control) -> Control {
    core::array::from_fn(|index| left[index].add(right[index]))
}

fn subtract_controls(left: Control, right: Control) -> Control {
    core::array::from_fn(|index| left[index].sub(right[index]))
}

fn scale_control(control: Control, factor: f64) -> Control {
    control.map(|component| component.scale(factor))
}

fn convert_catmull_rom_to_bezier(points: [Control; 4]) -> [Control; 4] {
    [
        points[1],
        add_controls(
            points[1],
            subtract_controls(points[2], points[0]).map(|value| value.divide(6.0)),
        ),
        subtract_controls(
            points[2],
            subtract_controls(points[3], points[1]).map(|value| value.divide(6.0)),
        ),
        points[2],
    ]
}

fn build_patch_controls(
    width: usize,
    height: usize,
    points: &[[f64; 6]],
    column: usize,
    row: usize,
) -> [Control; 16] {
    let rows: [[Control; 4]; 4] = core::array::from_fn(|y| {
        convert_catmull_rom_to_bezier(core::array::from_fn(|x| {
            sample_extended_point(
                width,
                height,
                points,
                column as i64 + x as i64 - 1,
                row as i64 + y as i64 - 1,
            )
        }))
    });
    let columns: [[Control; 4]; 4] = core::array::from_fn(|x| {
        convert_catmull_rom_to_bezier(core::array::from_fn(|y| rows[y][x]))
    });
    core::array::from_fn(|index| columns[index % 4][index / 4])
}

fn sample_extended_point(
    width: usize,
    height: usize,
    points: &[[f64; 6]],
    column: i64,
    row: i64,
) -> Control {
    fn compute_extrapolation_weights(index: i64, length: usize) -> [(usize, f64); 2] {
        if index < 0 {
            [(0, 2.0), (1, -1.0)]
        } else if index >= length as i64 {
            [(length - 1, 2.0), (length - 2, -1.0)]
        } else {
            [(index as usize, 1.0), (index as usize, 0.0)]
        }
    }

    let mut result = [Interval::from_exact_value(0.0); 6];
    for (y, y_weight) in compute_extrapolation_weights(row, height) {
        for (x, x_weight) in compute_extrapolation_weights(column, width) {
            let weight = x_weight * y_weight;
            if weight == 0.0 {
                continue;
            }
            let point = points[y * width + x].map(Interval::from_exact_value);
            result = add_controls(result, scale_control(point, weight));
        }
    }
    result
}

fn certify_patch(
    width: usize,
    height: usize,
    controls: &[Control; 16],
    column: usize,
    row: usize,
) -> Result<(), MeshGradientError> {
    let mut xx = f64::INFINITY;
    let mut yy = f64::INFINITY;
    let mut xy = 0.0_f64;
    let mut yx = 0.0_f64;

    for y in 0..4 {
        for x in 0..3 {
            let derivative = subtract_controls(controls[y * 4 + x + 1], controls[y * 4 + x]);
            xx = xx.min(derivative[0].scale(3.0 * (width - 1) as f64).lo);
            let cross = derivative[1].scale(3.0 * (width - 1) as f64);
            yx = yx.max(cross.lo.abs().max(cross.hi.abs()));
        }
    }
    for y in 0..3 {
        for x in 0..4 {
            let derivative = subtract_controls(controls[(y + 1) * 4 + x], controls[y * 4 + x]);
            yy = yy.min(derivative[1].scale(3.0 * (height - 1) as f64).lo);
            let cross = derivative[0].scale(3.0 * (height - 1) as f64);
            xy = xy.max(cross.lo.abs().max(cross.hi.abs()));
        }
    }

    let off_diagonal = ((xy + yx).next_up() * 0.5).next_up();
    let determinant = ((xx * yy).next_down() - (off_diagonal * off_diagonal).next_up()).next_down();
    if xx > 0.0 && yy > 0.0 && determinant > 0.0 && determinant.is_finite() {
        Ok(())
    } else {
        Err(MeshGradientError::UncertifiedGeometry { column, row })
    }
}

#[cfg(feature = "serialize")]
#[derive(serde::Deserialize)]
#[cfg_attr(test, derive(serde::Serialize))]
struct SerializedMeshGradient {
    width: usize,
    height: usize,
    points: Vec<MeshGradientPoint>,
    color_space: MeshGradientColorSpace,
    #[serde(default)]
    color_interpolation: MeshGradientColorInterpolation,
    #[serde(default)]
    geometry: MeshGradientGeometry,
}

#[cfg(feature = "serialize")]
impl<'de> serde::Deserialize<'de> for MeshGradient {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = SerializedMeshGradient::deserialize(deserializer)?;
        Self::new_with_geometry(
            raw.width,
            raw.height,
            raw.points,
            raw.color_space,
            raw.geometry,
        )
        .map(|mesh| mesh.with_color_interpolation(raw.color_interpolation))
        .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_reflect::{PartialReflect, ReflectRef};
    use proptest::{collection, prelude::*};

    fn create_regular_grid(width: usize, height: usize) -> Vec<MeshGradientPoint> {
        (0..height)
            .flat_map(|row| {
                (0..width).map(move |column| {
                    let x = column as f32 / (width - 1) as f32;
                    let y = row as f32 / (height - 1) as f32;
                    MeshGradientPoint::new(Vec2::new(x, y), Color::linear_rgba(x, y, 0.5, 1.0))
                })
            })
            .collect()
    }

    fn create_regular_mesh(width: usize, height: usize) -> MeshGradient {
        MeshGradient::new_in_color_space(
            width,
            height,
            create_regular_grid(width, height),
            MeshGradientColorSpace::LinearRgba,
        )
        .unwrap()
    }

    fn read_point_bits(mesh: &MeshGradient) -> Vec<[u32; 6]> {
        mesh.points()
            .iter()
            .map(|point| {
                let color = read_color_components(point.color);
                [
                    point.position.x.to_bits(),
                    point.position.y.to_bits(),
                    color[0].to_bits(),
                    color[1].to_bits(),
                    color[2].to_bits(),
                    color[3].to_bits(),
                ]
            })
            .collect()
    }

    #[test]
    fn defaults_to_the_mobile_friendly_color_path() {
        let mesh = MeshGradient::new(2, 2, create_regular_grid(2, 2)).unwrap();

        assert_eq!(mesh.color_space(), MeshGradientColorSpace::LinearRgba);
        assert_eq!(
            mesh.color_interpolation(),
            MeshGradientColorInterpolation::Vertex
        );
    }

    #[test]
    fn accepts_every_supported_regular_dimension() {
        for width in MIN_MESH_GRADIENT_DIMENSION..=MAX_MESH_GRADIENT_DIMENSION {
            for height in MIN_MESH_GRADIENT_DIMENSION..=MAX_MESH_GRADIENT_DIMENSION {
                let mesh = create_regular_mesh(width, height);
                assert_eq!(mesh.dimensions(), (width, height));
                assert_eq!(mesh.points().len(), width * height);
            }
        }
    }

    #[test]
    fn dimensions_and_cardinality_are_checked() {
        assert!(matches!(
            MeshGradient::new(1, 2, create_regular_grid(2, 2)),
            Err(MeshGradientError::DimensionsTooSmall { .. })
        ));
        assert!(matches!(
            MeshGradient::new(usize::MAX, 2, Vec::new()),
            Err(MeshGradientError::DimensionsOverflow { .. })
        ));
        assert!(matches!(
            MeshGradient::new(17, 2, Vec::new()),
            Err(MeshGradientError::CapacityExceeded { .. })
        ));
        assert!(matches!(
            MeshGradient::new(2, 2, create_regular_grid(2, 2)[..3].to_vec()),
            Err(MeshGradientError::PointCount {
                expected: 4,
                actual: 3
            })
        ));
    }

    #[test]
    fn validates_input_and_derived_numeric_values() {
        let mut points = create_regular_grid(2, 2);
        points[0].position.x = f32::NAN;
        assert!(matches!(
            MeshGradient::new(2, 2, points),
            Err(MeshGradientError::NonFinitePosition { point: 0 })
        ));

        let mut points = create_regular_grid(2, 2);
        points[0].color = Color::linear_rgba(f32::NAN, 0.0, 0.0, 1.0);
        assert!(matches!(
            MeshGradient::new_in_color_space(2, 2, points, MeshGradientColorSpace::LinearRgba),
            Err(MeshGradientError::NonFiniteColor { point: 0 })
        ));

        let mut points = create_regular_grid(2, 2);
        points[0].color = Color::linear_rgba(0.0, 0.0, 0.0, 1.1);
        assert!(matches!(
            MeshGradient::new_in_color_space(2, 2, points, MeshGradientColorSpace::LinearRgba),
            Err(MeshGradientError::AlphaOutOfRange { point: 0 })
        ));
    }

    #[test]
    fn accepts_outside_coordinates_and_hdr_colors() {
        let points = create_regular_grid(2, 2)
            .into_iter()
            .map(|mut point| {
                point.position = point.position * 3.0 - Vec2::splat(1.0);
                point.color = Color::linear_rgba(8.0, -4.0, 2.0, 0.5);
                point
            })
            .collect();
        let mesh =
            MeshGradient::new_in_color_space(2, 2, points, MeshGradientColorSpace::LinearRgba)
                .unwrap();
        assert_eq!(mesh.point_at(0, 0).unwrap().position, Vec2::splat(-1.0));

        let mut unrepresentable = create_regular_grid(2, 2);
        unrepresentable[0].color = Color::linear_rgba(f32::MAX, 0.0, 0.0, 1.0);
        assert!(matches!(
            MeshGradient::new_in_color_space(
                2,
                2,
                unrepresentable,
                MeshGradientColorSpace::LinearRgba
            ),
            Err(MeshGradientError::NonFiniteDerived { .. })
        ));
    }

    #[test]
    fn rejects_degenerate_folded_and_curved_folded_surfaces() {
        let mut degenerate = create_regular_grid(2, 2);
        degenerate[2].position = degenerate[0].position;
        degenerate[3].position = degenerate[1].position;
        assert!(matches!(
            MeshGradient::new(2, 2, degenerate),
            Err(MeshGradientError::DegenerateGeometry { .. })
        ));

        let rotated = create_regular_grid(3, 3)
            .into_iter()
            .map(|mut point| {
                point.position = Vec2::ONE - point.position;
                point
            })
            .collect();
        assert!(matches!(
            MeshGradient::new(3, 3, rotated),
            Err(MeshGradientError::UncertifiedGeometry { .. })
        ));

        let mut curved_fold = create_regular_grid(3, 3);
        for point in &mut curved_fold {
            if point.position.x == 0.5 {
                point.position.x = 0.01;
            }
        }
        assert!(matches!(
            MeshGradient::new(3, 3, curved_fold),
            Err(MeshGradientError::UncertifiedGeometry { .. })
        ));
    }

    #[test]
    fn fold_enabled_geometry_accepts_editor_drag_rejected_by_non_folding_policy() {
        let mut points = create_regular_grid(5, 4);
        points[7].position.x = 0.589;

        assert!(matches!(
            MeshGradient::new(5, 4, points.clone()),
            Err(MeshGradientError::UncertifiedGeometry { .. })
        ));
        let mesh = MeshGradient::new_with_geometry(
            5,
            4,
            points,
            MeshGradientColorSpace::Oklaba,
            MeshGradientGeometry::AllowFolds,
        )
        .unwrap();
        assert_eq!(mesh.geometry(), MeshGradientGeometry::AllowFolds);
    }

    #[test]
    fn fold_enabled_geometry_accepts_collapsed_and_crossing_points() {
        let mut points = create_regular_grid(3, 3);
        points[4].position = points[5].position;
        points[7].position = Vec2::new(1.25, -0.25);

        let mut mesh = MeshGradient::new_with_geometry(
            3,
            3,
            points,
            MeshGradientColorSpace::Oklaba,
            MeshGradientGeometry::AllowFolds,
        )
        .unwrap();
        assert!(mesh
            .try_set_geometry(MeshGradientGeometry::NonFolding)
            .is_err());
        assert_eq!(mesh.geometry(), MeshGradientGeometry::AllowFolds);
    }

    #[test]
    fn color_interpolation_defaults_to_vertex_and_survives_checked_edits() {
        let mut mesh = create_regular_mesh(3, 3);
        assert_eq!(
            mesh.color_interpolation(),
            MeshGradientColorInterpolation::Vertex
        );
        mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        mesh.try_set_position(4, Vec2::new(0.52, 0.48)).unwrap();
        assert_eq!(
            mesh.color_interpolation(),
            MeshGradientColorInterpolation::Bicubic
        );
    }

    #[test]
    fn rejects_distant_surface_overlap() {
        let width = 10;
        let height = 2;
        let mut points = create_regular_grid(width, height);
        for row in 0..height {
            let radius = 1.0 + row as f32 * 0.25;
            for column in 0..width {
                let angle = column as f32 * core::f32::consts::FRAC_PI_4;
                let (sin, cos) = bevy_math::ops::sin_cos(angle);
                points[row * width + column].position = Vec2::new(cos, sin) * radius;
            }
        }

        let values: Vec<_> = points
            .iter()
            .map(|point| {
                let color = convert_color_to_interpolation_components(
                    point.color,
                    MeshGradientColorSpace::LinearRgba,
                );
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
        for column in 0..width - 1 {
            assert!(validate_cell_corners(width, &values, column, 0).is_ok());
        }

        assert!(matches!(
            MeshGradient::new_in_color_space(
                width,
                height,
                points,
                MeshGradientColorSpace::LinearRgba
            ),
            Err(MeshGradientError::UncertifiedGeometry { .. })
        ));
    }

    #[test]
    fn rejected_edits_preserve_every_stored_bit() {
        let mut mesh = create_regular_mesh(3, 3);
        let dimensions = mesh.dimensions();
        let color_space = mesh.color_space();
        let before = read_point_bits(&mesh);

        assert!(mesh.try_set_position(4, Vec2::new(1.2, 0.5)).is_err());
        assert_eq!(mesh.dimensions(), dimensions);
        assert_eq!(mesh.color_space(), color_space);
        assert_eq!(read_point_bits(&mesh), before);

        assert!(mesh
            .try_edit_points(|points| {
                points[1].position.x = 0.9;
                points[4].position.x = -0.5;
                Ok(())
            })
            .is_err());
        assert_eq!(read_point_bits(&mesh), before);

        assert!(matches!(
            mesh.try_set_color(mesh.points().len(), Color::WHITE),
            Err(MeshGradientError::PointIndexOutOfBounds { .. })
        ));
        assert_eq!(read_point_bits(&mesh), before);

        assert!(mesh
            .try_set_point(
                0,
                MeshGradientPoint::new(Vec2::new(f32::NAN, 0.0), Color::WHITE)
            )
            .is_err());
        assert_eq!(read_point_bits(&mesh), before);

        assert!(mesh
            .try_set_color(0, Color::linear_rgba(0.0, 0.0, 0.0, 2.0))
            .is_err());
        assert_eq!(read_point_bits(&mesh), before);

        let mut incomplete = mesh.points().to_vec();
        incomplete.pop();
        assert!(mesh.try_replace_points(incomplete).is_err());
        assert_eq!(read_point_bits(&mesh), before);

        assert!(mesh
            .try_replace_grid(1, 2, create_regular_grid(2, 2))
            .is_err());
        assert_eq!(mesh.dimensions(), dimensions);
        assert_eq!(mesh.color_space(), color_space);
        assert_eq!(read_point_bits(&mesh), before);
    }

    #[test]
    fn batch_edits_validate_only_the_complete_candidate() {
        let mut mesh = create_regular_mesh(3, 3);
        mesh.try_edit_points(|points| {
            points.swap(3, 4);
            points.swap(4, 3);
            points[4].position = Vec2::new(0.55, 0.45);
            Ok(())
        })
        .unwrap();
        assert_eq!(mesh.point(4).unwrap().position, Vec2::new(0.55, 0.45));
    }

    #[test]
    fn shared_patch_controls_are_identical() {
        let width = 3;
        let height = 3;
        let points = create_regular_grid(width, height);
        let mut values = Vec::new();
        for point in points {
            let color = convert_color_to_interpolation_components(
                point.color,
                MeshGradientColorSpace::LinearRgba,
            );
            values.push([
                point.position.x as f64,
                point.position.y as f64,
                color[0] as f64,
                color[1] as f64,
                color[2] as f64,
                color[3] as f64,
            ]);
        }
        let left = build_patch_controls(width, height, &values, 0, 0);
        let right = build_patch_controls(width, height, &values, 1, 0);
        for row in 0..4 {
            for component in 0..6 {
                assert_eq!(
                    left[row * 4 + 3][component].lo.to_bits(),
                    right[row * 4][component].lo.to_bits()
                );
                assert_eq!(
                    left[row * 4 + 3][component].hi.to_bits(),
                    right[row * 4][component].hi.to_bits()
                );
            }
        }
    }

    #[test]
    fn every_ui_color_space_is_supported_and_validated() {
        for space in [
            InterpolationColorSpace::Oklaba,
            InterpolationColorSpace::Oklcha,
            InterpolationColorSpace::OklchaLong,
            InterpolationColorSpace::Hsla,
            InterpolationColorSpace::HslaLong,
            InterpolationColorSpace::Hsva,
            InterpolationColorSpace::HsvaLong,
            InterpolationColorSpace::Okhsla,
            InterpolationColorSpace::OkhslaLong,
            InterpolationColorSpace::Srgba,
            InterpolationColorSpace::LinearRgba,
        ] {
            let mesh_space = MeshGradientColorSpace::from(space);
            assert_eq!(InterpolationColorSpace::from(mesh_space), space);
            let mut mesh = create_regular_mesh(2, 2);
            mesh.try_set_color_space(mesh_space).unwrap();
            assert_eq!(mesh.color_space(), mesh_space);
            #[cfg(feature = "serialize")]
            {
                let encoded = ron::to_string(&mesh).unwrap();
                assert_eq!(ron::from_str::<MeshGradient>(&encoded).unwrap(), mesh);
            }
            assert!(convert_color_to_interpolation_components(
                Color::hsva(350.0, 0.7, 0.8, 0.5),
                mesh_space
            )
            .iter()
            .all(|component| component.is_finite()));
        }
    }

    #[test]
    fn reflection_is_opaque() {
        let mesh = create_regular_mesh(2, 2);
        assert!(matches!(mesh.reflect_ref(), ReflectRef::Opaque(_)));
    }

    #[cfg(feature = "serialize")]
    #[test]
    fn serialization_round_trips_and_revalidates() {
        let mut points = create_regular_grid(3, 3);
        points[4].position = points[5].position;
        let mesh = MeshGradient::new_with_geometry(
            3,
            3,
            points,
            MeshGradientColorSpace::LinearRgba,
            MeshGradientGeometry::AllowFolds,
        )
        .unwrap();
        let mut mesh = mesh;
        mesh.set_color_interpolation(MeshGradientColorInterpolation::Bicubic);
        let serialized = ron::to_string(&mesh).unwrap();
        assert_eq!(ron::from_str::<MeshGradient>(&serialized).unwrap(), mesh);

        let malformed = SerializedMeshGradient {
            width: 1,
            height: 2,
            points: create_regular_grid(2, 2),
            color_space: MeshGradientColorSpace::LinearRgba,
            color_interpolation: MeshGradientColorInterpolation::Vertex,
            geometry: MeshGradientGeometry::NonFolding,
        };
        let serialized = ron::to_string(&malformed).unwrap();
        assert!(ron::from_str::<MeshGradient>(&serialized).is_err());
    }

    proptest! {
        #[test]
        fn arbitrary_raw_inputs_never_bypass_validation(
            width in 0usize..20,
            height in 0usize..20,
            raw in collection::vec(
                (-20i16..20, -20i16..20, -20i16..20, -20i16..20, -20i16..20, -5i16..15),
                0..300,
            ),
        ) {
            let points = raw.into_iter().map(|(x, y, red, green, blue, alpha)| {
                MeshGradientPoint::new(
                    Vec2::new(x as f32 / 10.0, y as f32 / 10.0),
                    Color::linear_rgba(
                        red as f32 / 10.0,
                        green as f32 / 10.0,
                        blue as f32 / 10.0,
                        alpha as f32 / 10.0,
                    ),
                )
            }).collect();
            if let Ok(mesh) = MeshGradient::new_in_color_space(
                width,
                height,
                points,
                MeshGradientColorSpace::LinearRgba,
            ) {
                prop_assert!(MeshGradient::validate(
                    mesh.width,
                    mesh.height,
                    &mesh.points,
                    mesh.color_space,
                    mesh.geometry,
                ).is_ok());
            }
        }

        #[test]
        fn arbitrary_edit_sequences_preserve_the_invariant(
            edits in collection::vec((0usize..16, -8i16..8, -8i16..8), 0..64),
        ) {
            let mut mesh = create_regular_mesh(4, 4);
            for (index, x_offset, y_offset) in edits {
                let before = read_point_bits(&mesh);
                let result = mesh.try_edit_points(|points| {
                    points[index].position += Vec2::new(
                        x_offset as f32 / 100.0,
                        y_offset as f32 / 100.0,
                    );
                    Ok(())
                });
                if result.is_err() {
                    prop_assert_eq!(read_point_bits(&mesh), before);
                } else {
                    prop_assert!(MeshGradient::validate(
                        mesh.width,
                        mesh.height,
                        &mesh.points,
                        mesh.color_space,
                        mesh.geometry,
                    ).is_ok());
                }
            }
        }
    }
}
