---
title: Mesh gradients
authors: ["@AGeorgy"]
pull_requests: []
---

Bevy UI now supports mesh gradients: smooth two-dimensional color surfaces controlled by a grid of colored points. Moving a point changes the shape of the color transitions, making mesh gradients useful for animated backgrounds and borders.

Create a `MeshGradient` from row-major `MeshGradientPoint` values and use it in a `BackgroundGradient` or `BorderGradient`. Mesh gradients compose with existing gradient layers and support rounded corners, clipping, transparency, and UI transforms.

The checked API supports grids from 2×2 to 16×16. Constructors and edit methods validate the complete surface, and rejected edits preserve the previous gradient. `MeshGradientGeometry::AllowFolds` enables deliberate collapsed or crossing geometry for sharp color transitions. Serialization and reflection preserve the checked model's invariants.

Surface positions are evaluated on the GPU from the control points. Adaptive tessellation automatically adjusts each patch to its curvature and physical size on screen. The default color path interpolates linear RGB at vertices to keep fragment work small; `MeshGradientColorInterpolation::Bicubic` evaluates a smoother color surface per fragment. All existing UI gradient color spaces, including their short and long hue paths, are supported. The renderer uses uniform buffers and ordinary vertex and fragment shaders, including on WebGL2.

The `mesh_gradient` example provides draggable colored points, RGBA editing, grid presets, reset and animation controls, and synchronized background and border previews. The `gradients` example includes a compact mesh preview, and `many_gradients --mesh --animate` exercises animated mesh gradients at scale.
