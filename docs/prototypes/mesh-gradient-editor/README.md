# Mesh-gradient editor interaction prototype

This throwaway prototype supports [Validate the mesh-gradient editor example interaction](https://github.com/AGeorgy/bevy/issues/6). It builds on the approved GPU renderer prototype and compares three editor layouts over one checked mesh-gradient state.

## Question

Which interaction design is clearest and most useful as an idiomatic Bevy example while demonstrating the real checked editing model?

## Variants

- **A / Canvas first** gives direct manipulation most of the window and keeps a compact inspector on the right. This favors learning by dragging.
- **B / Inspector first** leads with presets, animation, exact RGBA controls, and state feedback. This favors discoverability and precise editing.
- **C / Dual preview** shows editable background and synchronized border results together, with a horizontal control strip below. This favors demonstrating integration coverage.

The layouts are intentionally structurally different. The selected variant is shareable through the `--variant 0|1|2` argument, and the floating bottom switcher plus Left/Right keys cycle variants without resetting the shared editor state.

## Shared interaction model

- Clicking a colored point selects it. The selected handle grows and receives a white outline.
- Dragging converts pointer movement through the current computed canvas size into normalized mesh coordinates and clamps the candidate point to the `[0, 1]` domain.
- Every drag submits a complete candidate point grid to the checked surface. A rejected candidate leaves the last valid surface visible and reports the validation error in the inspector.
- RGBA uses Bevy's headless `Slider` widget. A color edit also submits a complete candidate grid through the checked API.
- The 2x2, 3x3, and 4x4 presets replace the whole grid atomically. Reset restores the current preset.
- Animation submits one complete batch edit per frame. Starting a manual drag or color edit pauses animation at the current valid state. If an animated candidate becomes invalid, animation stops before displaying it.
- Background and border modes use the same gradient value. Variant C keeps both synchronized and visible at once.
- The inspector always shows the selected point index, normalized position, RGBA values, animation and preview modes, and the last accepted or rejected action.

Editor constraints remain interaction aids. The checked surface is the authority for curved-surface validity.

## Renderer boundary

This interaction prototype uses the renderer prototype's private `PrototypeMeshGradient` transport with 24 parameter subdivisions per patch. That setting is not part of the proposed public API. The approved production feature requires automatic adaptive tessellation driven by screen-space geometry and color error, with shared edges and cached topology.

## Run

```sh
cargo +1.97.1 run --offline --example mesh_gradient_editor_prototype \
  --no-default-features --features ui -- --variant 0
```

Controls:

- Bottom `<` and `>` buttons or Left/Right: change layout.
- Point drag: checked position edit.
- RGBA sliders: checked color edit.
- `2`, `3`, `4`: replace the grid preset.
- Space: toggle animation.
- `B`: switch the single preview between background and border.
- `R`: reset.
- `S`: save `/tmp/mesh-gradient-editor-prototype.png`.

For automated capture and invalid-edit evidence, use `--capture <path> --frames <n>` and `--attempt-fold`.

## Verification before live review

- Native compile and runtime startup pass on Apple M1 Pro / Metal.
- Layout A computes one 866x664 logical-pixel preview at a 2x display scale.
- Layout B computes one 798x664 logical-pixel preview.
- Layout C computes two synchronized 609x366 logical-pixel previews.
- The scripted folded edit is rejected with `UncertifiedGeometry`, and the checked model retains the last valid surface.
- The `wasm32-unknown-unknown` target compiles with Bevy's `ui,webgl2` features.

The host Mac was locked during the final automated captures, so those images contained no desktop pixels and are not included. Visual layout, pointer behavior, slider behavior, resizing, and the preferred variant remain pending the required live review.
