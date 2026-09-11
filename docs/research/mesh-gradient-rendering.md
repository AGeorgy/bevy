# Mesh-gradient rendering approaches for Bevy UI

Research for [Compare mesh-gradient rendering approaches across Bevy UI platforms](https://github.com/AGeorgy/bevy/issues/3), 2026-09-11. This report resolves the comparison, not the renderer selection. No benchmarks or implementation tests were run.

## Result

All three families can fit the UI feature, but they answer different questions. Tessellation maps a parameterized surface forward into triangles; fragment evaluation must recover a parameter coordinate for each covered pixel; an intermediate texture separates surface generation from UI composition and can use either of the first two approaches. They are not three mutually exclusive interpolation models.

**Engineering inference:** The first useful comparison prototype is a tessellated reference surface versus bounded inverse evaluation on the same patch model. Add offscreen composition if fold/transparency semantics or caching justify its cost. Do not select an architecture before defining folds, boundary coverage, smoothness, and alpha interpolation. WebGL2 is a baseline requirement for this effort, not an optional fallback left until the end.

## Current Bevy integration facts

Source baseline: commit `d0518456863130e887b51fbc09af0ddbf722ff32`. Links below are immutable. The working checkout's tracked sources match this baseline; setup documents are untracked and were not included in the research commit.

- `Gradient` currently represents linear, radial and conic gradients. The abstraction is shared by background and border components. Mesh data therefore needs a new representation; converting colored points into an ordinary ordered list of stops loses the two-dimensional domain. [Gradient API](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui/src/gradients.rs)
- Extraction stores transform, clip, node rectangle, border widths/radii, stack index and color space. Relevant component changes trigger extraction. Queueing uses `TransparentUi` and separate gradient/background versus border-gradient stack offsets. [Extraction and queue](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui_render/src/gradient.rs#L343)
- Preparation transforms the node corners, calls `clip_polygon` while preserving local coordinates, generates geometry for adjacent stop pairs, and uploads shared vertex/index buffers. Each extracted gradient gets its own range; shared buffers do not imply all gradients become one draw. Preparation clears and rebuilds these buffers even though extraction is change-filtered. [Preparation](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui_render/src/gradient.rs#L827)
- The shader computes one scalar gradient coordinate and interpolates one stop pair. The pipeline selects color-space and antialias variants, uses straight-alpha blending, and has fifteen vertex attributes. Replacing the stop evaluator alone cannot make it represent arbitrary deformed patches. [Pipeline](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui_render/src/gradient.rs#L144), [shader](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui_render/src/gradient.wesl)
- `draw_uinode_background` masks the inner rounded rectangle; `draw_uinode_border` masks the border ring and enabled sides. Both return straight RGB with coverage applied to alpha. Reusing these functions preserves their shape semantics only if mesh rendering supplies the same node-local `point`, `size`, border and radius values. These masks do not implement arbitrary ancestor clipping by themselves. [UI masking](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_ui_render/src/ui.wesl#L191)

## Platform constraints

Bevy explicitly chooses `Limits::downlevel_webgl2_defaults()` for its WebGL configuration. Its tracked render manifest requests wgpu 30. The ignored local lockfile contains an older wgpu resolution and is not evidence that this source targets that older version. [Settings](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_render/src/settings.rs#L89), [manifest](https://github.com/AGeorgy/bevy/blob/d0518456863130e887b51fbc09af0ddbf722ff32/crates/bevy_render/Cargo.toml#L90)

wgpu 30 documents these WebGL2 defaults: zero storage buffers/textures per shader stage, zero compute invocations, a 16 KiB uniform binding size, sixteen vertex attributes, a 255-byte vertex stride and fifteen inter-stage variables. These are baseline limits, not measured device performance or a complete feature matrix. Query the actual enabled device limits and format capabilities. [wgpu 30 limits](https://docs.rs/wgpu/30.0.0/wgpu/struct.Limits.html#method.downlevel_webgl2_defaults)

WebGL2 follows GLES 3.0 and exposes vertex/fragment rendering, uniform buffers, indexed triangles, textures and framebuffers. A design requiring compute shaders, storage buffers or hardware tessellation stages cannot be the sole WebGL2 path. Float texture availability must not be confused with float render-target or filtering support; check formats/extensions. The cited current WebGL document is an editor's draft. [WebGL2 specification](https://registry.khronos.org/webgl/specs/latest/2.0/)

**Engineering implications:** Ordinary triangle buffers plus vertex/fragment shaders are the most direct common transport. A uniform patch block can work within a documented capacity; many patches require chunking or another data layout. Sampled textures can carry control data, with exact fetch coordinates and suitable formats. Do not append many control points to the current fifteen-attribute vertex format. A WebGPU-only storage-buffer optimization is possible later, but would need equivalent baseline behavior.

## A. Tessellated patches

Define a patch position `P(u,v)` and color field `C(u,v)`. Sample them on a parameter grid, then rasterize indexed triangles. This forward map requires no pixel-to-patch inverse. CPU evaluation is portable; a fixed parameter grid evaluated by a vertex shader can reduce animated vertex uploads if its controls fit a portable binding scheme. Adaptive CPU tessellation trades more generation/management work for fewer triangles.

**Mathematical consequences, not benchmark claims:** For `p` independent patches with `s` subdivisions on each axis, a simple uniform implementation produces `2*p*s*s` triangles and at most `p*(s+1)^2` sampled vertices before edge sharing. Linear triangle color interpolation approximates a curved color field; more triangles can reduce that approximation error but do not change the underlying patch model. A hybrid can carry parameter coordinates and evaluate the color polynomial per fragment; the position-to-parameter relation is still only the tessellated approximation.

Watertight geometry needs identical edge samples, compatible edge subdivision and consistent winding. Adaptive neighbors with different edge splits need shared edge schedules or stitching. Independently antialiasing every patch boundary creates false internal seams: internal joins need ordinary shared coverage, while exterior coverage must be antialiased deliberately. Screen-space error controls are preferable to an unexplained fixed subdivision count, but error estimates must consider color variation as well as geometric curvature. NVIDIA's original tessellation chapter demonstrates screen-space flatness tests and discusses cracks when adjacent patches differ in numerical evaluation. It is precedent for these techniques, not a directly portable Bevy implementation or a Bevy performance result. [Bunnell, GPU Gems 2, sections 7.1.3 and 7.1.6](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-7-adaptive-tessellation-subdivision-surfaces)

**Bevy integration inference:** Clip the generated triangles/polygons with equivalent ancestor clipping, retaining interpolated node-local attributes, then use the existing shape masks. Preserve the parent gradient's UI ordering across all its patches. A transformed node can reuse local samples when its error budget permits; zoom/DPI changes may require refinement. Color-only edits can avoid geometric resampling if geometry and color caches are separate. A dirty control may affect several neighboring patches when automatic tangents use neighboring points.

**Main risks:** CPU generation/upload cost during animation; excessive tiny triangles; approximation error on magnification; T-junction cracks; topology changes causing visible popping; and unintended repeated blending where folded triangles overlap.

## B. Per-fragment evaluation/inversion

Rasterize a conservative patch bound or a node polygon. For each fragment at local position `x`, locate a patch and solve `P(u,v)=x`, then evaluate `C(u,v)`. An undeformed axis-aligned grid can locate cells directly; moving points removes that shortcut. A general curved mapping needs an inverse strategy. This is a different task from merely evaluating a bicubic polynomial at known parameters.

**Mathematical analysis:** Newton iteration updates `(u,v)` using the inverse of the 2-by-2 Jacobian `[dP/du, dP/dv]`. Near a zero determinant it is ill-conditioned; a guessed seed can diverge, leave the patch, or find one of several roots. A bounded iteration count needs a residual threshold and defined fallback. Boundary ownership and root selection must be deterministic. These consequences follow from solving the displayed equation, and are not a claim about SwiftUI's private renderer.

Direct surface intersection is a substantial graphics problem: the cited primary research uses bounding and recursive subdivision with explicit GPU work-distribution strategies for Bézier/Gregory patches. This supports treating inverse coverage as an algorithm to validate, not assuming that a short Newton loop is universally robust. The paper addresses 3D ray tracing, so its throughput and full machinery should not be transferred to 2D UI. [Binder and Keller, 2018](https://arxiv.org/abs/1811.03510)

**Engineering inference:** Control updates can be small compared with regenerated tessellation. Cost instead scales with shaded candidate pixels, overlapping bounds, tested patches and iterations. A full-node loop over every patch scales poorly in the worst case; per-patch bounds or a spatial index can reduce candidates but add data preparation and binding work. Bounds must include curved interiors, not just four corners. Derivative-based antialias code around divergent search must obey backend shader rules; shader compilation and runtime validation belong in the prototype.

This path can preserve exact chosen color interpolation at a successfully recovered parameter without tessellation color error. It still needs outer coverage antialiasing and precision policy. Near folds it cannot silently discard nonconvergent pixels. A one-pass whole-gradient solver can resolve overlap before blending, but must actually find and order the relevant roots. Drawing one bound per patch with normal alpha blending instead produces patch compositing semantics.

## C. Intermediate texture

Generate the gradient into a texture, then draw a clipped node polygon sampling it with the shared background/border masks. The generator can use tessellation, inverse fragments or CPU rasterization plus upload. Compute generation may accelerate a WebGPU variant, but is not a common baseline.

**Engineering analysis:** This decouples UI composition from patch coverage and allows reuse across unchanged frames. A single final compositing draw can help enforce gradient-level opacity, provided the texture generator first implements the intended overlap policy. An offscreen target alone does not fix folds. A `W` by `H` RGBA8 texture requires `4*W*H` bytes for its pixel storage, excluding allocation overhead, mipmaps and extra buffers; multisampling/resolves or higher precision add cost. Resolution, maximum allocation, reuse and eviction need explicit policy.

Animation invalidates affected texture regions or the entire texture. Screen-size changes and magnification can reveal blur; minification can alias unless sampling/mips are adequate. Baking at physical node resolution is a policy with memory and regeneration consequences. An atlas can reduce texture bind changes but introduces padding/filter bleed, allocation and invalidation complexity. Separate per-node textures can disrupt batching. Neither improvement should be assumed faster without measurement.

For transparent output, distinguish interpolation representation from storage representation and final blending. Filtering straight RGB across transparent texels can expose hidden colors. Premultiplied storage avoids that class of halo, but the final draw must match it: directly feeding premultiplied RGB into the current straight-alpha path multiplies opacity again. A dedicated compositing blend/mask path or safe conversion is required. Test zero alpha explicitly. Avoid applying border coverage twice by baking it and masking again.

## Shared semantic decisions that rendering cannot hide

**Continuity, mathematical analysis:** Shared corner colors guarantee agreement only at those corners. Matching the complete color function along a shared edge gives value continuity; matching appropriate derivatives is needed for smooth slopes. Likewise, sharing a geometric edge gives positional continuity but does not automatically give matching transverse derivatives. Cubic geometry with bilinear colors can still show abrupt color-slope changes. Bicubic color interpolation needs tangent/boundary rules and a policy for overshoot, gamut and alpha outside the valid range. Perceptual and hue spaces introduce additional conversion and wrapping choices beyond four independent scalar interpolations.

**Folds and alpha:** A mesh with moved points may leave uncovered regions, overlap another patch or fold onto itself. Decide whether to reject/constrain such inputs, show a background, choose one visible surface sample, or composite overlapping samples. Cairo explicitly documents patch order and parameter-based self-fold ordering; it demonstrates that this is a public semantic choice, not an inevitable property of mesh gradients. It is not evidence that SwiftUI uses Cairo's rules. [Cairo mesh pattern semantics](https://www.cairographics.org/manual/cairo-cairo-pattern-t.html#cairo-pattern-create-mesh)

**Bevy compatibility inference:** Keep mesh gradients in the same ordered background/border feature, including mixed ordinary-gradient layers, inherited clipping, transforms and camera target format. Existing interpolation uses RGB/hue mixing and alpha mixing separately. Selecting premultiplied interpolation for the mesh is a possible API decision, not an automatic compatibility rule. A renderer should consume validated/resolved mesh data without forcing shader layout or tessellation resolution into the public colored-point API.

## Prototype measurements and acceptance evidence

Use the same geometry/color definition in all candidates and compare screenshots against a high-precision, densely sampled reference whose own error is checked by further refinement. Record CPU generation/extraction/preparation time, uploaded bytes, draw calls/bind changes, GPU duration where supported, total frame time, allocations and texture memory. Separate warm pipeline/cache runs from cold creation. Record hardware, browser, adapter, backend, resolution and enabled limits. No numeric pass threshold is selected here.

Run at least native rendering, browser WebGPU and actual browser WebGL2. Compiling Rust to wasm does not demonstrate WebGL2 shader or render-target compatibility. Include minimum-limit validation and allocation/size failure behavior.

Use small and larger grids; straight and strongly curved cells; coincident points, zero-area cells and near-singular folds; opaque and translucent colors over a checkerboard; edge/corner edits and fully transparent colored points; color-only versus geometry animation; many small nodes versus one large node; DPI/zoom/rotation and nonuniform scaling; narrow asymmetric rounded borders; nested overflow clipping; mixed gradient/image/text layers; resize, hidden nodes and removal. Inspect internal seams separately from exterior antialiasing and temporal flicker.

Report inverse residuals/failure counts and candidate counts; report tessellation error, vertex counts and edge agreement; report texture resolution, regeneration frequency, cache hit rate and filtering artifacts. Set performance budgets and quality thresholds in the acceptance decision after these observations.

## Decisions left to the map

1. Patch geometry and color continuity model, automatic control inference and explicit Bézier scope.
2. Degenerate/folded input and uncovered-region semantics, plus alpha/color-space policy.
3. Renderer choice informed by equivalent prototype measurements, including any WebGPU optimization with baseline equivalence.
4. Quality bounds, capacities and resource-failure behavior; caching ownership and invalidation.
5. Visual acceptance fixtures and editor constraints that match those semantics.

The research is complete as a comparison. These are deliberate downstream decisions, not unresolved factual claims or a reason to declare one renderer already chosen.
