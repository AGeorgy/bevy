# SwiftUI mesh-gradient semantics and Bevy compatibility baseline

Research for [Establish SwiftUI mesh-gradient semantics and a Bevy compatibility baseline](https://github.com/AGeorgy/bevy/issues/2), inspected 2026-09-11. Base checkout: d0518456863130e887b51fbc09af0ddbf722ff32.

## Finding

A movable colored grid with inferred curved geometry is a coherent SwiftUI-inspired target. Pixel equivalence is not established by Apple's public documentation. Geometry, color smoothing, alpha, degeneracy, and rendering quality need an explicit Bevy contract. This report supplies evidence and candidate acceptance cases; it does not settle remaining human fidelity or renderer decisions.

## Documented model

**Topology and sizes.** Width counts vertices in a row; height counts vertices in a column. Point and color arrays each contain width times height entries. Defaults are clear uncovered-area color, enabled color smoothing, and device color-space interpolation. The initializer documents counts but does not state its response to mismatches or a minimum dimension. [Point initializer](https://developer.apple.com/documentation/swiftui/meshgradient/init(width:height:points:colors:background:smoothscolors:colorspace:))

**Ordering.** Apple's first-party 3-by-3 example lists x across each row, then increases y for the next row; its parallel color array associates one color with each position. This is evidence for row-major authoring, index = row * width + column, rather than an independently published indexing formula. It demonstrates an interior point displaced from a regular lattice with a fixed perimeter. [What's new in SwiftUI, 4:42 sample](https://developer.apple.com/videos/play/wwdc2024/10144/?time=282)

**Coordinates and motion.** Apple's visual-effects session describes 0-to-1 x/y coordinates when used as a view. It demonstrates moving an interior point and says closer points sharpen color transitions. It presents animation as a use case, but supplies no mesh-specific temporal interpolation contract in that segment. Its 7:30 example moves the center to (0.9, 0.3). This supports normalized UI-space authoring and position updates; it does not prove arbitrary out-of-range inputs or animated topology changes are supported. [Create custom visual effects, 6:49–8:50 and 7:30 code](https://developer.apple.com/videos/play/wwdc2024/10151/?time=409)

**Geometry and color are separate.** Each vertex has a position, a color, and four neighboring-edge Bézier control points. Unused edge/corner handles are ignored. Handles can be supplied or inferred. Rendering tessellates Bézier patches; vertex colors interpolate linearly or with neighbor-derived cubic curves. The overview does not publish tangent inference, patch-interior construction, color cubic coefficients, tessellation tolerance, or continuity order. Calling the implementation Catmull–Rom, Coons, or a particular bicubic tensor construction would exceed this source. [MeshGradient overview](https://developer.apple.com/documentation/swiftui/meshgradient)

**Explicit controls.** BezierPoint exposes a vertex position plus leading, top, trailing, and bottom control-point positions as SIMD2<Float>. Its initializer describes the vertex position in the gradient's interpreted coordinate space. It does not describe controls as offsets from the vertex; do not borrow that convention from another framework without verification. [Bézier-point initializer](https://developer.apple.com/documentation/swiftui/meshgradient/bezierpoint/init(position:leadingcontrolpoint:topcontrolpoint:trailingcontrolpoint:bottomcontrolpoint:))

**Smoothing and uncovered area.** smoothsColors selects cubic color interpolation; disabling it does not disable curved geometry. background fills points outside the defined vertex mesh, not a documented clamp-to-nearest-edge rule. Its default is clear. The initializer does not specify how background participates beneath partially transparent covered pixels. [Explicit-geometry initializer](https://developer.apple.com/documentation/swiftui/meshgradient/init(width:height:bezierpoints:colors:background:smoothscolors:colorspace:))

**Color representation.** Device interpolation uses the output color space; perceptual interpolation uses a perceptual space. The page does not name a concrete transform such as Oklab, so matching a Bevy mode by name would be inference. [Gradient.ColorSpace](https://developer.apple.com/documentation/swiftui/gradient/colorspace). Resolved-color overloads accept already-resolved sRGB colors; input representation and interpolation space are distinct concepts. [Resolved-color initializer](https://developer.apple.com/documentation/swiftui/meshgradient/init(width:height:points:resolvedcolors:background:smoothscolors:colorspace:))

**Alpha.** SwiftUI colors can carry opacity. [Color.opacity](https://developer.apple.com/documentation/swiftui/color/opacity(_:)). However, the mesh pages above do not specify whether interpolation operates on premultiplied channels, how cubic alpha overshoot is handled, or when clamping occurs. Apple's public Shader API requires premultiplied fill output, but that is a different contract and insufficient evidence for MeshGradient's internal interpolation. [Shader](https://developer.apple.com/documentation/swiftui/shader)

## Unspecified behavior and evidence limits

No behavior guarantee was found in the inspected Apple mesh pages or WWDC samples for zero/one-sized dimensions, count mismatch, negative dimensions, NaN/infinite coordinates, coincident vertices, crossed rows, folded patches, self-intersection, off-view vertices, gamut overshoot, or patch overlap ordering. Absence of documentation is not evidence of rejection or support. None was experimentally tested against a native SwiftUI renderer in this investigation.

Animation support establishes intended use, not automatic interpolation of every field. Fixed-grid updates, transitions between dimensions, replacement of color arrays, and changing inferred to explicit geometry must be distinguished. No claim is made about exact SwiftUI frame timing or performance.

## Proposed minimum coherent subset, pending decisions

These are recommendations for subsequent API and renderer tickets, not adopted specifications:

- A rectangular topology with at least two vertices per axis, row-major colored points, finite normalized UI coordinates, and checked cardinality. Define resource limits rather than inferring Apple's.
- Inferred smooth curved geometry and smooth color transitions between neighboring patches. Choose and document actual formulas. Shared-edge positions and colors must agree during edits; flat triangles with a visible diagonal would not meet the agreed smooth experience.
- Per-point RGBA values, a documented interpolation space and alpha rule, and predictable uncovered-area behavior. Reuse appropriate Bevy conventions after inspecting them.
- Fixed-topology position/color updates suitable for animation. Grid presets can replace topology discretely; animated row/column count changes need not be implied.
- A deliberate policy for invalid/folded meshes. Constraining editor drags helps UX but cannot be the engine's only defense because callers construct inputs directly.

Deferring explicit handles retains point-driven authoring but removes independent tangent editing: users cannot reproduce every explicitly controlled SwiftUI shape while holding vertices fixed. Preserve a plausible extension point without exposing an unchosen handle API. Deferring exact interpolation equivalence means matching palettes and positions may look different, particularly at high contrast or transparency. Advertise the experience as SwiftUI-inspired and validate Bevy's chosen equations, not alleged Apple golden values.

## Candidate acceptance cases

Expected outcomes below are proposed Bevy requirements, not measured SwiftUI behavior. Numeric tolerances and performance thresholds remain to be decided.

| Case | Setup | What to assess |
| --- | --- | --- |
| Ordering and resize | 2-by-3 grid with distinct colors; square, wide, tall nodes | Correct point/color association; normalized geometry follows bounds |
| Interior drag | Fixed-perimeter 3-by-3 grid, center moves from (0.5, 0.5) to (0.9, 0.3) | Continuous deformation; no internal cracks or triangulation diagonals |
| Shared-edge continuity | Irregular 4-by-4 spacing and contrasting colors | No patch seams during motion; close points compress transitions |
| Uncovered area | Inset perimeter, transparent then opaque background | Defined outside fill and clean boundaries |
| RGBA | Opaque red beside transparent blue over checkerboard | Selected premultiplication/clamping rule; no unexplained halos |
| Color smoothing | Alternating high-contrast rows; all-identical colors | Smooth-mode behavior, constant-color invariance, overshoot policy |
| Integration | Background and border with rounded corners, clipping, stacking | Established UI coverage/compositing; no double blend at shared edges |
| Animation | Loop interior motion and color changes at fixed topology | No tearing, stale frames, or allocation growth |
| Invalid inputs | Wrong counts, zero axes, NaN/infinity, coincident points, folded cell | Deterministic documented outcome; no unbounded GPU work |
| Example workflow | Select, drag, edit RGBA, presets, reset, animation toggle | Selection survives valid edits; reliable reset; previews agree |

Before implementation, resolve inference formulas and continuity target; interpolation/alpha/gamut policy; boundary and invalid/fold behavior; explicit-handle scope; grid limits; and platform quality/performance budgets. Source research is complete; these are design decisions for the existing map.
