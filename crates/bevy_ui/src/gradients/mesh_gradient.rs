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

// Patch corner controls contain every source component. The GPU's two boundary
// extrapolations magnify it by at most 3 * 3, and each Catmull-Rom weighted sum
// has an absolute weight sum below 2 for 0 <= t <= 1.5. Snapping can evaluate
// t = 1.5 at a zero-weight endpoint because its coarsest subdivision count is 2.
// Thus each surface evaluation is bounded by 36 times the source magnitude.
// Reserving a factor of 128 also covers a difference of two evaluated endpoints
// in a snapped mix, with room for f32 rounding. Validate both color modes so
// changing interpolation cannot make a previously checked grid overflow.
const MAX_GPU_COMPONENT_MAGNITUDE: f64 = f32::MAX as f64 / 128.0;

// Color UVs stay in [0, 1], so the Bezier control hull bounds the interpolated
// coordinates. Oklab's LMS transform has an absolute row sum below 2.4 and its
// RGB transform below 8: components bounded by C produce RGB below 111 * C^3.
// The sRGB decoder raises positive coordinates to 2.4, while HSL/HSV first
// multiply saturation by lightness/value. These limits leave several orders of
// magnitude for f32 rounding in the shared shader conversions.
const MAX_GPU_OKLAB_COMPONENT: f64 = 1.0e11;
const MAX_GPU_SRGB_COMPONENT: f64 = 1.0e14;
const MAX_GPU_HSL_HSV_COMPONENT: f64 = 1.0e6;

// OKHSL also uploads a separate Oklab lightness/chroma surface. With each
// source L/a/b bounded by C/16, its chroma is at most sqrt(2) * C/16. Even the
// conservative 36-fold parameter evaluation bound keeps its final Oklab RGB
// conversion below 4e36, leaving substantial room below f32::MAX.
const MAX_GPU_OKHSL_FALLBACK_COMPONENT: f64 = MAX_GPU_OKLAB_COMPONENT / 16.0;

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
    /// Derived OKHSL saturation is clamped to `[0, 1]` before RGB conversion.
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
    /// finite linear RGB or safe coordinates in the selected interpolation
    /// space, including OKHSL's Oklab fallback surface.
    #[error("mesh-gradient point {point} has a color outside the finite GPU evaluation range")]
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
    /// Inferred bicubic controls exceed the range that keeps GPU boundary
    /// extrapolation, interpolation, and color conversion finite, even if each
    /// input is finite.
    #[error("mesh-gradient patch ({column}, {row}) exceeds the finite GPU evaluation range")]
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
/// derived control values must remain finite and leave room for GPU boundary
/// extrapolation and interpolation in parameter space. Extremely large finite
/// coordinates can therefore be rejected in either color interpolation mode.
/// Colors must also convert to finite linear RGB at the supplied points.
/// Interpolation-space limits reserve room for nonlinear RGB conversion after
/// interpolation; ordinary HDR colors remain supported.
/// Colors interpolate in linear RGB by default; all UI gradient spaces are
/// available through [`MeshGradientColorSpace`]. Near saturated blue, OKHSL
/// meshes blend toward an Oklab surface derived from the same colored points
/// to avoid a gamut discontinuity while keeping colors in OKHSL's unit domain
/// intact. Derived OKHSL saturation is clamped to `[0, 1]` before RGB conversion.
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
    ///
    /// # Errors
    ///
    /// Returns a [`MeshGradientError`] for invalid dimensions, point count,
    /// numeric values, or a surface that cannot be certified as non-folding.
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
    ///
    /// # Errors
    ///
    /// Returns a [`MeshGradientError`] for invalid dimensions, point count,
    /// numeric values in the selected color space, or uncertified geometry.
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
    ///
    /// # Errors
    ///
    /// Returns a [`MeshGradientError`] for dimensions outside the supported
    /// range, a mismatched point count, invalid position/color/alpha values,
    /// controls outside the finite GPU evaluation range, or geometry that
    /// fails the selected policy.
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
    ///
    /// # Errors
    ///
    /// Returns [`MeshGradientError::PointIndexOutOfBounds`] for an invalid
    /// index, or a validation error for the candidate grid. Rejected edits
    /// leave the mesh unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns [`MeshGradientError::PointIndexOutOfBounds`] for an invalid
    /// index, or a numeric or geometry validation error for the new position.
    /// Rejected edits leave the mesh unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns [`MeshGradientError::PointIndexOutOfBounds`] for an invalid
    /// index, or a color, alpha, or derived-control validation error. Rejected
    /// edits leave the mesh unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns the closure's error or a validation error for the edited grid.
    pub fn try_edit_points<F>(&mut self, edit: F) -> Result<(), MeshGradientError>
    where
        F: FnOnce(&mut [MeshGradientPoint]) -> Result<(), MeshGradientError>,
    {
        let mut candidate = self.points.clone();
        edit(&mut candidate)?;
        self.try_replace_points(candidate)
    }

    /// Replaces all points without changing the grid dimensions.
    ///
    /// # Errors
    ///
    /// Returns a [`MeshGradientError`] for a mismatched point count, invalid
    /// numeric values, or geometry that fails the current policy. Rejected
    /// edits leave the mesh unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns a [`MeshGradientError`] for invalid dimensions, point count,
    /// numeric values, or geometry. Rejected edits leave the mesh unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns a color or derived-control validation error if the selected
    /// space cannot represent the grid safely. Rejected edits leave the mesh
    /// unchanged.
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
    ///
    /// # Errors
    ///
    /// Returns a geometry validation error if the surface fails the requested
    /// policy. Rejected edits leave the mesh unchanged.
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
            if LinearRgba::from(point.color)
                .to_f32_array()
                .iter()
                .any(|component| !component.is_finite())
            {
                return Err(MeshGradientError::NonFiniteColor { point: index });
            }
            let color = convert_color_to_interpolation_components(point.color, color_space);
            if color.iter().any(|component| !component.is_finite()) {
                return Err(MeshGradientError::NonFiniteColor { point: index });
            }
            if matches!(
                color_space,
                MeshGradientColorSpace::Okhsla | MeshGradientColorSpace::OkhslaLong
            ) {
                let lab = Oklaba::from(point.color);
                let fallback_range =
                    -MAX_GPU_OKHSL_FALLBACK_COMPONENT..=MAX_GPU_OKHSL_FALLBACK_COMPONENT;
                if [lab.lightness, lab.a, lab.b]
                    .iter()
                    .any(|component| !fallback_range.contains(&(*component as f64)))
                {
                    return Err(MeshGradientError::NonFiniteColor { point: index });
                }
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

        let mut gpu_component_limits = [MAX_GPU_COMPONENT_MAGNITUDE; 6];
        match color_space {
            MeshGradientColorSpace::Oklaba => {
                gpu_component_limits[2..5].fill(MAX_GPU_OKLAB_COMPONENT);
            }
            MeshGradientColorSpace::Oklcha | MeshGradientColorSpace::OklchaLong => {
                gpu_component_limits[2..4].fill(MAX_GPU_OKLAB_COMPONENT);
            }
            MeshGradientColorSpace::Srgba => {
                gpu_component_limits[2..5].fill(MAX_GPU_SRGB_COMPONENT);
            }
            MeshGradientColorSpace::Hsla
            | MeshGradientColorSpace::HslaLong
            | MeshGradientColorSpace::Hsva
            | MeshGradientColorSpace::HsvaLong => {
                gpu_component_limits[3..5].fill(MAX_GPU_HSL_HSV_COMPONENT);
            }
            MeshGradientColorSpace::Okhsla
            | MeshGradientColorSpace::OkhslaLong
            | MeshGradientColorSpace::LinearRgba => {}
        }
        for row in 0..height - 1 {
            for column in 0..width - 1 {
                if geometry == MeshGradientGeometry::NonFolding {
                    validate_cell_corners(width, &values, column, row)?;
                }
                let controls = build_patch_controls(width, height, &values, column, row);
                if controls.iter().any(|control| {
                    control
                        .iter()
                        .zip(gpu_component_limits)
                        .any(|(component, limit)| {
                            let range = -limit..=limit;
                            !range.contains(&component.lo) || !range.contains(&component.hi)
                        })
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

        let mut points = create_regular_grid(2, 2);
        points[0].color = Color::from(Oklaba::new(1.0e20, 0.0, 0.0, 1.0));
        assert!(matches!(
            MeshGradient::new_in_color_space(2, 2, points, MeshGradientColorSpace::Oklaba),
            Err(MeshGradientError::NonFiniteColor { point: 0 })
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
    fn rejects_finite_inputs_that_overflow_gpu_extrapolation() {
        for geometry in [
            MeshGradientGeometry::NonFolding,
            MeshGradientGeometry::AllowFolds,
        ] {
            let mut points = create_regular_grid(2, 2);
            for point in &mut points {
                point.color = Color::linear_rgba(2.0e38, 0.0, 0.0, 1.0);
            }
            assert!(matches!(
                MeshGradient::new_with_geometry(
                    2,
                    2,
                    points,
                    MeshGradientColorSpace::LinearRgba,
                    geometry
                ),
                Err(MeshGradientError::NonFiniteDerived { column: 0, row: 0 })
            ));

            let mut points = create_regular_grid(2, 2);
            for point in &mut points {
                point.position *= 2.0e38;
            }
            assert!(matches!(
                MeshGradient::new_with_geometry(
                    2,
                    2,
                    points,
                    MeshGradientColorSpace::LinearRgba,
                    geometry
                ),
                Err(MeshGradientError::NonFiniteDerived { column: 0, row: 0 })
            ));
        }

        let mut mesh = create_regular_mesh(2, 2);
        for interpolation in [
            MeshGradientColorInterpolation::Vertex,
            MeshGradientColorInterpolation::Bicubic,
        ] {
            mesh.set_color_interpolation(interpolation);
            let before = read_point_bits(&mesh);
            assert!(matches!(
                mesh.try_set_color(0, Color::linear_rgba(2.0e38, 0.0, 0.0, 1.0)),
                Err(MeshGradientError::NonFiniteDerived { .. })
            ));
            assert_eq!(read_point_bits(&mesh), before);
            assert_eq!(mesh.color_interpolation(), interpolation);
        }
    }

    #[test]
    fn rejects_color_controls_that_can_overflow_nonlinear_conversion() {
        let lightness = core::hint::black_box(4.0e12_f32);
        let midpoint = 1.125 * lightness;
        assert!(LinearRgba::from(Oklaba::new(lightness, 0.0, 0.0, 1.0))
            .to_f32_array()
            .iter()
            .all(|component| component.is_finite()));
        // The RGB result is mathematically finite, but this intermediate in
        // the shared f32 Oklab shader overflows at the plateau's midpoint.
        assert!(!(4.0767417 * midpoint * midpoint * midpoint).is_finite());
        let mut points = create_regular_grid(4, 2);
        for (index, point) in points.iter_mut().enumerate() {
            let l = if matches!(index % 4, 1 | 2) {
                lightness
            } else {
                0.0
            };
            point.color = Color::from(Oklaba::new(l, 0.0, 0.0, 1.0));
        }
        assert!(matches!(
            MeshGradient::new_in_color_space(4, 2, points, MeshGradientColorSpace::Oklaba),
            Err(MeshGradientError::NonFiniteDerived { .. })
        ));

        // Supplied points fit their limit; only the plateau's Bezier handles
        // exceed it. This verifies that checking inputs alone is insufficient.
        for space in [
            MeshGradientColorSpace::Oklaba,
            MeshGradientColorSpace::Oklcha,
            MeshGradientColorSpace::OklchaLong,
            MeshGradientColorSpace::Srgba,
            MeshGradientColorSpace::Hsla,
            MeshGradientColorSpace::HslaLong,
            MeshGradientColorSpace::Hsva,
            MeshGradientColorSpace::HsvaLong,
        ] {
            let create_color = |high| {
                let fraction = if high { 0.95 } else { 0.0 };
                match space {
                    MeshGradientColorSpace::Oklaba => Color::from(Oklaba::new(
                        fraction * MAX_GPU_OKLAB_COMPONENT as f32,
                        0.0,
                        0.0,
                        1.0,
                    )),
                    MeshGradientColorSpace::Oklcha | MeshGradientColorSpace::OklchaLong => {
                        Color::from(Oklcha::new(
                            0.5,
                            fraction * MAX_GPU_OKLAB_COMPONENT as f32,
                            0.0,
                            1.0,
                        ))
                    }
                    MeshGradientColorSpace::Srgba => Color::from(Srgba::new(
                        fraction * MAX_GPU_SRGB_COMPONENT as f32,
                        0.0,
                        0.0,
                        1.0,
                    )),
                    MeshGradientColorSpace::Hsla | MeshGradientColorSpace::HslaLong => Color::from(
                        Hsla::new(0.0, fraction * MAX_GPU_HSL_HSV_COMPONENT as f32, 0.25, 1.0),
                    ),
                    MeshGradientColorSpace::Hsva | MeshGradientColorSpace::HsvaLong => Color::from(
                        Hsva::new(0.0, fraction * MAX_GPU_HSL_HSV_COMPONENT as f32, 0.5, 1.0),
                    ),
                    _ => unreachable!(),
                }
            };
            let mut points = create_regular_grid(4, 2);
            for (index, point) in points.iter_mut().enumerate() {
                point.color = create_color(matches!(index % 4, 1 | 2));
                assert!(LinearRgba::from(point.color)
                    .to_f32_array()
                    .iter()
                    .all(|component| component.is_finite()));
            }
            assert!(
                matches!(
                    MeshGradient::new_in_color_space(4, 2, points, space),
                    Err(MeshGradientError::NonFiniteDerived { .. })
                ),
                "{space:?}"
            );
        }
        for space in [
            MeshGradientColorSpace::Okhsla,
            MeshGradientColorSpace::OkhslaLong,
        ] {
            let mut points = create_regular_grid(2, 2);
            for point in &mut points {
                point.color = Color::from(Oklaba::new(
                    2.0 * MAX_GPU_OKHSL_FALLBACK_COMPONENT as f32,
                    0.0,
                    0.0,
                    1.0,
                ));
                assert!(LinearRgba::from(point.color)
                    .to_f32_array()
                    .iter()
                    .all(|component| component.is_finite()));
            }
            assert!(
                matches!(
                    MeshGradient::new_in_color_space(2, 2, points, space),
                    Err(MeshGradientError::NonFiniteColor { .. })
                ),
                "{space:?}"
            );
        }
    }

    #[test]
    fn conversion_budgets_preserve_safe_hdr_in_both_color_modes() {
        for space in [
            MeshGradientColorSpace::Oklaba,
            MeshGradientColorSpace::Oklcha,
            MeshGradientColorSpace::OklchaLong,
            MeshGradientColorSpace::Hsla,
            MeshGradientColorSpace::HslaLong,
            MeshGradientColorSpace::Hsva,
            MeshGradientColorSpace::HsvaLong,
            MeshGradientColorSpace::Okhsla,
            MeshGradientColorSpace::OkhslaLong,
            MeshGradientColorSpace::Srgba,
            MeshGradientColorSpace::LinearRgba,
        ] {
            let mut points = create_regular_grid(2, 2);
            for point in &mut points {
                point.color = Color::linear_rgba(8.0, 4.0, 2.0, 0.5);
            }
            let mut mesh = MeshGradient::new_in_color_space(2, 2, points, space).unwrap();
            for mode in [
                MeshGradientColorInterpolation::Vertex,
                MeshGradientColorInterpolation::Bicubic,
            ] {
                mesh.set_color_interpolation(mode);
                assert_eq!(mesh.color_interpolation(), mode);
                assert_eq!(mesh.color_space(), space);
            }
        }
        let cases = [
            (
                MeshGradientColorSpace::Oklaba,
                Color::from(Oklaba::new(1.0e10, -1.0e10, 1.0e10, 1.0)),
            ),
            (
                MeshGradientColorSpace::Oklcha,
                Color::from(Oklcha::new(1.0e10, 1.0e10, 30.0, 1.0)),
            ),
            (
                MeshGradientColorSpace::Srgba,
                Color::from(Srgba::new(1.0e13, -1.0e13, 1.0e13, 1.0)),
            ),
            (
                MeshGradientColorSpace::Hsla,
                Color::from(Hsla::new(30.0, 1.0e5, 1.0e5, 1.0)),
            ),
            (
                MeshGradientColorSpace::Hsva,
                Color::from(Hsva::new(30.0, 1.0e5, 1.0e5, 1.0)),
            ),
            (
                MeshGradientColorSpace::Okhsla,
                Color::from(Oklaba::new(1.0e8, 0.0, 0.0, 1.0)),
            ),
        ];
        for (space, color) in cases {
            let mut points = create_regular_grid(2, 2);
            for point in &mut points {
                point.color = color;
            }
            let mut mesh = MeshGradient::new_in_color_space(2, 2, points, space).unwrap();
            for mode in [
                MeshGradientColorInterpolation::Vertex,
                MeshGradientColorInterpolation::Bicubic,
            ] {
                mesh.set_color_interpolation(mode);
                assert_eq!(mesh.color_interpolation(), mode);
            }
        }
    }

    #[test]
    fn nonlinear_conversion_budgets_keep_f32_intermediates_finite() {
        fn check_oklab(lightness: f32, a: f32, b: f32) {
            let l = lightness + 0.39633778 * a + 0.21580376 * b;
            let m = lightness - 0.105561346 * a - 0.06385417 * b;
            let s = lightness - 0.08948418 * a - 1.2914855 * b;
            let cubes = [l * l * l, m * m * m, s * s * s];
            assert!(cubes.iter().all(|value| value.is_finite()));
            for coefficients in [
                [4.0767417, -3.3077116, 0.23096994],
                [-1.268438, 2.6097574, -0.34131938],
                [-0.0041960863, -0.7034186, 1.7076147],
            ] {
                let mut rgb = 0.0;
                for (coefficient, cube) in coefficients.into_iter().zip(cubes) {
                    let term = coefficient * cube;
                    assert!(term.is_finite());
                    rgb += term;
                    assert!(rgb.is_finite());
                }
            }
        }
        // The larger cube covers OKHSL's lightness/chroma fallback bound:
        // 36 * sqrt(2) / 16 < 3.2, including conversion of chroma to a/b.
        for limit in [
            MAX_GPU_OKLAB_COMPONENT as f32,
            3.2 * MAX_GPU_OKLAB_COMPONENT as f32,
        ] {
            for lightness in [-limit, limit] {
                for a in [-limit, limit] {
                    for b in [-limit, limit] {
                        check_oklab(lightness, a, b);
                    }
                }
            }
        }
        let srgb_limit = MAX_GPU_SRGB_COMPONENT as f32;
        let hsl_hsv_rgb_bound =
            4.0 * MAX_GPU_HSL_HSV_COMPONENT as f32 * (MAX_GPU_HSL_HSV_COMPONENT as f32 + 1.0);
        for value in [srgb_limit, hsl_hsv_rgb_bound] {
            assert!(bevy_math::ops::powf((value + 0.055) / 1.055, 2.4).is_finite());
        }
    }

    #[test]
    fn safe_hdr_controls_keep_f32_shader_intermediates_finite() {
        // Exercise the GPU arithmetic rather than the validator's f64 controls.
        // Keep zero-weight samples: evaluating infinity * 0 caused the original
        // constant-HDR endpoint regression.
        fn sample(values: &[[f32; 6]], width: usize, height: usize, x: i64, y: i64) -> [f32; 6] {
            let sample_row = |row: usize| {
                let point = |column: usize| values[row * width + column];
                let result = if x < 0 {
                    core::array::from_fn(|channel| 2.0 * point(0)[channel] - point(1)[channel])
                } else if x >= width as i64 {
                    core::array::from_fn(|channel| {
                        2.0 * point(width - 1)[channel] - point(width - 2)[channel]
                    })
                } else {
                    point(x as usize)
                };
                assert!(result.iter().all(|component| component.is_finite()));
                result
            };
            let result = if y < 0 {
                let first = sample_row(0);
                let next = sample_row(1);
                core::array::from_fn(|channel| 2.0 * first[channel] - next[channel])
            } else if y >= height as i64 {
                let last = sample_row(height - 1);
                let previous = sample_row(height - 2);
                core::array::from_fn(|channel| 2.0 * last[channel] - previous[channel])
            } else {
                sample_row(y as usize)
            };
            assert!(result.iter().all(|component| component.is_finite()));
            result
        }

        fn evaluate(points: [[f32; 6]; 4], t: f32) -> [f32; 6] {
            let t2 = t * t;
            let t3 = t2 * t;
            let weights = [
                0.5 * (-t + 2.0 * t2 - t3),
                0.5 * (2.0 - 5.0 * t2 + 3.0 * t3),
                0.5 * (t + 4.0 * t2 - 3.0 * t3),
                0.5 * (-t2 + t3),
            ];
            core::array::from_fn(|channel| {
                let mut value = 0.0;
                for (point, weight) in points.iter().zip(weights) {
                    let term = point[channel] * weight;
                    assert!(term.is_finite());
                    value += term;
                    assert!(value.is_finite());
                }
                value
            })
        }

        for (width, height, alternating) in [(2, 2, false), (3, 3, true), (5, 4, true)] {
            let magnitude = (MAX_GPU_COMPONENT_MAGNITUDE / 4.0) as f32;
            let mut points = create_regular_grid(width, height);
            for (index, point) in points.iter_mut().enumerate() {
                let sign = if alternating && index % 2 == 0 {
                    -1.0
                } else {
                    1.0
                };
                point.color = Color::linear_rgba(sign * magnitude, -sign * magnitude, 1.0e30, 0.75);
                if alternating {
                    point.position = Vec2::new(sign * magnitude, -sign * magnitude);
                }
            }
            let mut mesh = MeshGradient::new_with_geometry(
                width,
                height,
                points,
                MeshGradientColorSpace::LinearRgba,
                MeshGradientGeometry::AllowFolds,
            )
            .unwrap();
            let values: Vec<_> = mesh
                .points()
                .iter()
                .map(|point| {
                    let color = LinearRgba::from(point.color);
                    [
                        point.position.x,
                        point.position.y,
                        color.red,
                        color.green,
                        color.blue,
                        color.alpha,
                    ]
                })
                .collect();
            for interpolation in [
                MeshGradientColorInterpolation::Vertex,
                MeshGradientColorInterpolation::Bicubic,
            ] {
                mesh.set_color_interpolation(interpolation);
                for row in 0..height - 1 {
                    for column in 0..width - 1 {
                        for u in [0.0, 0.125, 0.5, 0.875, 1.0, 1.5] {
                            for v in [0.0, 0.125, 0.5, 0.875, 1.0, 1.5] {
                                let rows = core::array::from_fn(|y| {
                                    evaluate(
                                        core::array::from_fn(|x| {
                                            sample(
                                                &values,
                                                width,
                                                height,
                                                column as i64 + x as i64 - 1,
                                                row as i64 + y as i64 - 1,
                                            )
                                        }),
                                        u,
                                    )
                                });
                                let end = evaluate(rows, v);
                                let start = evaluate(rows, 0.0);
                                for channel in 0..6 {
                                    assert!((end[channel] - start[channel]).is_finite());
                                    assert!((start[channel]
                                        + (end[channel] - start[channel]) * 0.5)
                                        .is_finite());
                                }
                            }
                        }
                    }
                }
            }
        }
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
    fn patch_controls_contain_independent_affine_and_center_impulse_references() {
        let origin = [1.0, 2.0, 3.0, 4.0, 5.0, 0.5];
        let horizontal = [2.0, 0.25, -1.0, 0.5, 3.0, 0.1];
        let vertical = [-0.5, 3.0, 2.0, -0.25, -4.0, 0.2];
        let affine_values: Vec<_> = (0..4)
            .map(|index| {
                core::array::from_fn(|channel| {
                    origin[channel]
                        + horizontal[channel] * (index % 2) as f64
                        + vertical[channel] * (index / 2) as f64
                })
            })
            .collect();
        let controls = build_patch_controls(2, 2, &affine_values, 0, 0);
        for (index, control) in controls.iter().enumerate() {
            for (channel, component) in control.iter().enumerate() {
                // A cardinal spline through affine samples is the same affine
                // function; its Bezier controls are uniformly spaced thirds.
                let expected = origin[channel]
                    + horizontal[channel] * (index % 4) as f64 / 3.0
                    + vertical[channel] * (index / 4) as f64 / 3.0;
                assert!(component.lo <= expected && expected <= component.hi);
            }
        }

        let delta = 0.6_f32 as f64 - 0.5;
        let mut values: Vec<[f64; 6]> = (0..9)
            .map(|index| {
                let x = (index % 3) as f64 * 0.5;
                let y = (index / 3) as f64 * 0.5;
                [x, y, x, y, 0.5, 1.0]
            })
            .collect();
        values[4][0] += delta;
        // Independently derived rational influence of a unit middle-sample
        // impulse on the left and right cardinal segments' Bezier controls.
        // Taking the tensor product gives the 2D center point's influence.
        let influence = [[0.0, 1.0 / 3.0, 1.0, 1.0], [1.0, 1.0, 1.0 / 3.0, 0.0]];
        for row in 0..2 {
            for column in 0..2 {
                let controls = build_patch_controls(3, 3, &values, column, row);
                for (index, control) in controls.iter().enumerate() {
                    let x = index % 4;
                    let y = index / 4;
                    let u = (column as f64 + x as f64 / 3.0) * 0.5;
                    let v = (row as f64 + y as f64 / 3.0) * 0.5;
                    let expected = [
                        u + delta * influence[column][x] * influence[row][y],
                        v,
                        u,
                        v,
                        0.5,
                        1.0,
                    ];
                    for (component, expected) in control.iter().zip(expected) {
                        assert!(component.lo <= expected && expected <= component.hi);
                    }
                }
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
        fn generated_valid_grids_accept_and_controlled_mutations_reject(
            width in 2usize..=8,
            height in 2usize..=8,
            x_origin in -100i16..100,
            y_origin in -100i16..100,
            x_step in 1u16..16,
            y_step in 1u16..16,
            color in (-128i16..128, -128i16..128, -128i16..128, 0u8..=16),
            point_seed in any::<usize>(),
            mutation in 0u8..9,
        ) {
            // Positive axis steps make this an independently known non-folding
            // affine surface. Binary fractions keep the grid exactly uniform.
            let mut points: Vec<_> = (0..width * height).map(|index| {
                let x = (x_origin as f32 + (index % width) as f32 * x_step as f32) / 16.0;
                let y = (y_origin as f32 + (index / width) as f32 * y_step as f32) / 16.0;
                MeshGradientPoint::new(Vec2::new(x, y), Color::linear_rgba(
                    color.0 as f32 / 16.0,
                    color.1 as f32 / 16.0,
                    color.2 as f32 / 16.0,
                    color.3 as f32 / 16.0,
                ))
            }).collect();
            prop_assert!(MeshGradient::new(
                width,
                height,
                points.clone(),
            ).is_ok());

            let index = point_seed % points.len();
            let (invalid_width, invalid_height) = match mutation {
                0 => {
                    points[index].position.x = f32::NAN;
                    (width, height)
                }
                1 => {
                    points[index].color = Color::linear_rgba(f32::NAN, 0.0, 0.0, 1.0);
                    (width, height)
                }
                2 => {
                    points[index].color = Color::linear_rgba(0.0, 0.0, 0.0, 1.25);
                    (width, height)
                }
                3 => {
                    points.pop();
                    (width, height)
                }
                4 => (1, height),
                5 => (MAX_MESH_GRADIENT_DIMENSION + 1, height),
                6 => (usize::MAX, 2),
                7 => {
                    for point in &mut points {
                        point.color = Color::linear_rgba(2.0e38, 0.0, 0.0, 1.0);
                    }
                    (width, height)
                }
                _ => {
                    points[1].position = points[0].position;
                    (width, height)
                }
            };
            let result = MeshGradient::new(invalid_width, invalid_height, points);
            let expected_error = match (mutation, result) {
                (0, Err(MeshGradientError::NonFinitePosition { point }))
                | (1, Err(MeshGradientError::NonFiniteColor { point }))
                | (2, Err(MeshGradientError::AlphaOutOfRange { point })) => point == index,
                (3, Err(MeshGradientError::PointCount { expected, actual })) => expected == width * height && actual + 1 == expected,
                (4, Err(MeshGradientError::DimensionsTooSmall { .. }))
                | (5, Err(MeshGradientError::CapacityExceeded { .. }))
                | (6, Err(MeshGradientError::DimensionsOverflow { .. }))
                | (7, Err(MeshGradientError::NonFiniteDerived { .. }))
                | (8, Err(MeshGradientError::DegenerateGeometry { column: 0, row: 0 })) => true,
                _ => false,
            };
            prop_assert!(expected_error, "mutation {} did not produce its expected error class", mutation);
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
