//! THROWAWAY: three mesh-gradient editor layouts for interaction review.
//!
//! Run with `cargo run --example mesh_gradient_editor_prototype --no-default-features --features ui`.
//! Switch variants with the bottom bar, Left/Right, or `--variant 0|1|2`.

mod mesh_gradient_prototype_math;

use bevy::{
    picking::hover::Hovered,
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    ui::{Pressed, PrototypeMeshGradient},
    ui_widgets::{Activate, Button, Slider, SliderRange, SliderThumb, SliderValue, ValueChange},
};
use mesh_gradient_prototype_math::{MeshSurface, Point};

const VARIANTS: [&str; 3] = [
    "A / Canvas first",
    "B / Inspector first",
    "C / Dual preview",
];
const PANEL: Color = Color::srgb(0.075, 0.09, 0.125);
const PANEL_ALT: Color = Color::srgb(0.105, 0.12, 0.16);
const TEXT: Color = Color::srgb(0.88, 0.91, 0.96);
const MUTED: Color = Color::srgb(0.56, 0.63, 0.73);
const ACCENT: Color = Color::srgb(0.25, 0.72, 0.92);

#[derive(Resource)]
struct EditorState {
    surface: MeshSurface,
    rest: Vec<Point>,
    width: usize,
    height: usize,
    selected: usize,
    variant: usize,
    preview_border: bool,
    animate: bool,
    elapsed: f32,
    rebuild: bool,
    status: String,
}

impl EditorState {
    fn new(variant: usize) -> Self {
        let points = preset(3);
        Self {
            surface: MeshSurface::try_new(3, 3, points.clone()).expect("preset must be valid"),
            rest: points,
            width: 3,
            height: 3,
            selected: 4,
            variant: variant.min(VARIANTS.len() - 1),
            preview_border: false,
            animate: false,
            elapsed: 0.,
            rebuild: true,
            status: "Ready: drag a point or select it to edit RGBA".into(),
        }
    }

    fn replace_grid(&mut self, size: usize) {
        let points = preset(size);
        self.surface =
            MeshSurface::try_new(size, size, points.clone()).expect("preset must be valid");
        self.rest = points;
        self.width = size;
        self.height = size;
        self.selected = (size * size) / 2;
        self.animate = false;
        self.status = format!("Replaced the complete grid atomically with {size}x{size}");
        self.rebuild = true;
    }

    fn reset(&mut self) {
        self.replace_grid(self.width);
        self.status = "Reset positions and colors".into();
    }

    fn accept(&mut self, candidate: Vec<Point>, action: &str) {
        match self.surface.try_replace_points(candidate) {
            Ok(()) => {
                self.animate = false;
                self.rest = self.surface.points().to_vec();
                self.status = format!("Accepted {action}");
            }
            Err(error) => {
                self.status = format!("Rejected {action}; retained last valid surface: {error:?}");
            }
        }
    }
}

#[derive(Resource)]
struct Capture {
    path: Option<String>,
    frame: u32,
    at: u32,
    attempt_fold: bool,
}

#[derive(Component)]
struct EditorRoot;
#[derive(Component)]
struct EditorCanvas;
#[derive(Component)]
struct MeshPreview {
    fixed_border: Option<bool>,
}
#[derive(Component)]
struct ControlPoint(usize);
#[derive(Component)]
struct StateReadout;
#[derive(Component)]
struct VariantReadout;
#[derive(Component)]
struct SelectedSwatch;
#[derive(Component)]
struct ChannelSlider(usize);
#[derive(Component)]
struct ChannelValue(usize);
#[derive(Component)]
struct SliderVisual;
#[derive(Component, Clone, Copy)]
enum EditorAction {
    PreviousVariant,
    NextVariant,
    Grid(usize),
    Reset,
    ToggleAnimation,
    ShowBackground,
    ShowBorder,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.025, 0.033, 0.05)))
        .insert_resource(EditorState::new(
            value("--variant")
                .and_then(|value| value.parse().ok())
                .unwrap_or(0),
        ))
        .insert_resource(Capture {
            path: value("--capture"),
            frame: 0,
            at: value("--frames")
                .and_then(|value| value.parse().ok())
                .unwrap_or(90),
            attempt_fold: args.iter().any(|arg| arg == "--attempt-fold"),
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mesh gradient editor prototype".into(),
                resolution: (1280, 800).into(),
                ..default()
            }),
            ..default()
        }))
        .add_observer(handle_action)
        .add_observer(select_point)
        .add_observer(drag_point)
        .add_observer(edit_channel)
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                keyboard,
                animate,
                scripted_invalid_edit,
                rebuild_editor,
                sync_editor,
                style_buttons,
                capture,
            )
                .chain(),
        )
        .run();
}

fn preset(size: usize) -> Vec<Point> {
    let palette = [
        [0.96, 0.18, 0.30, 1.0],
        [1.00, 0.70, 0.12, 1.0],
        [0.25, 0.82, 0.78, 1.0],
        [0.58, 0.13, 0.91, 1.0],
        [0.12, 0.31, 0.94, 1.0],
        [0.96, 0.39, 0.64, 0.72],
        [0.18, 0.63, 0.44, 1.0],
        [0.69, 0.32, 0.98, 0.78],
        [0.96, 0.90, 0.52, 1.0],
    ];
    (0..size * size)
        .map(|index| {
            let x = index % size;
            let y = index / size;
            let mut position = [x as f32 / (size - 1) as f32, y as f32 / (size - 1) as f32];
            if x > 0 && x + 1 < size && y > 0 && y + 1 < size {
                position[0] += 0.055;
                position[1] -= 0.035;
            }
            Point {
                position,
                color: palette[(x * 2 + y * 3) % palette.len()],
            }
        })
        .collect()
}

fn gradient(surface: &MeshSurface) -> Gradient {
    let (width, height) = surface.dimensions();
    PrototypeMeshGradient {
        points: surface
            .points()
            .iter()
            .map(|point| (Vec2::from_array(point.position), point.color))
            .collect(),
        width: width as u32,
        height: height as u32,
        subdivisions: 24,
        color_space: InterpolationColorSpace::Srgba,
    }
    .into()
}

fn setup(mut commands: Commands, mut state: ResMut<EditorState>) {
    commands.spawn(Camera2d);
    spawn_editor(&mut commands, &state);
    state.rebuild = false;
}

fn rebuild_editor(
    mut commands: Commands,
    mut state: ResMut<EditorState>,
    roots: Query<Entity, With<EditorRoot>>,
) {
    if !state.rebuild {
        return;
    }
    for entity in &roots {
        commands.entity(entity).despawn();
    }
    spawn_editor(&mut commands, &state);
    state.rebuild = false;
}

fn spawn_editor(commands: &mut Commands, state: &EditorState) {
    let root = commands
        .spawn((
            EditorRoot,
            Node {
                width: percent(100),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(22)),
                row_gap: px(14),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    height: px(54),
                    flex_direction: FlexDirection::Column,
                    ..default()
                },
                children![
                    (
                        Text::new("MESH GRADIENT / EDITOR INTERACTION PROTOTYPE"),
                        TextFont {
                            font_size: FontSize::Px(23.),
                            ..default()
                        },
                        TextColor(TEXT),
                    ),
                    (
                        Text::new("Three layouts share one checked state. Drag points; invalid edits keep the last valid surface."),
                        TextFont {
                            font_size: FontSize::Px(13.),
                            ..default()
                        },
                        TextColor(MUTED),
                    ),
                ],
            ));
            match state.variant {
                0 => spawn_canvas_first(parent, state),
                1 => spawn_inspector_first(parent, state),
                _ => spawn_dual_preview(parent, state),
            }
        })
        .id();
    commands.entity(root).with_child(switcher());
}

fn spawn_canvas_first(parent: &mut ChildSpawnerCommands, state: &EditorState) {
    parent
        .spawn(Node {
            flex_grow: 1.,
            width: percent(100),
            flex_direction: FlexDirection::Row,
            column_gap: px(18),
            ..default()
        })
        .with_children(|body| {
            body.spawn(Node {
                flex_grow: 1.,
                height: percent(100),
                flex_direction: FlexDirection::Column,
                row_gap: px(8),
                ..default()
            })
            .with_children(|canvas| {
                section_title(canvas, "CANVAS / direct manipulation");
                spawn_preview(canvas, state, true, None, percent(100), percent(100));
            });
            body.spawn((
                Node {
                    width: px(352),
                    height: percent(100),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(px(18)),
                    row_gap: px(13),
                    ..default()
                },
                BackgroundColor(PANEL),
            ))
            .with_children(|inspector| spawn_inspector(inspector, state, false));
        });
}

fn spawn_inspector_first(parent: &mut ChildSpawnerCommands, state: &EditorState) {
    parent
        .spawn(Node {
            flex_grow: 1.,
            width: percent(100),
            flex_direction: FlexDirection::Row,
            column_gap: px(18),
            ..default()
        })
        .with_children(|body| {
            body.spawn((
                Node {
                    width: px(420),
                    height: percent(100),
                    flex_direction: FlexDirection::Column,
                    padding: UiRect::all(px(20)),
                    row_gap: px(13),
                    ..default()
                },
                BackgroundColor(PANEL_ALT),
            ))
            .with_children(|inspector| spawn_inspector(inspector, state, true));
            body.spawn(Node {
                flex_grow: 1.,
                height: percent(100),
                flex_direction: FlexDirection::Column,
                row_gap: px(8),
                ..default()
            })
            .with_children(|canvas| {
                section_title(canvas, "PREVIEW / inspector-led workflow");
                spawn_preview(canvas, state, true, None, percent(100), percent(100));
            });
        });
}

fn spawn_dual_preview(parent: &mut ChildSpawnerCommands, state: &EditorState) {
    parent
        .spawn(Node {
            flex_grow: 1.,
            width: percent(100),
            flex_direction: FlexDirection::Column,
            row_gap: px(14),
            ..default()
        })
        .with_children(|body| {
            body.spawn(Node {
                height: px(390),
                width: percent(100),
                flex_direction: FlexDirection::Row,
                column_gap: px(18),
                ..default()
            })
            .with_children(|previews| {
                previews
                    .spawn(Node {
                        width: percent(50),
                        height: percent(100),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(8),
                        ..default()
                    })
                    .with_children(|column| {
                        section_title(column, "BACKGROUND / editable");
                        spawn_preview(column, state, true, Some(false), percent(100), percent(100));
                    });
                previews
                    .spawn(Node {
                        width: percent(50),
                        height: percent(100),
                        flex_direction: FlexDirection::Column,
                        row_gap: px(8),
                        ..default()
                    })
                    .with_children(|column| {
                        section_title(column, "BORDER / synchronized");
                        spawn_preview(column, state, false, Some(true), percent(100), percent(100));
                    });
            });
            body.spawn((
                Node {
                    flex_grow: 1.,
                    width: percent(100),
                    flex_direction: FlexDirection::Row,
                    padding: UiRect::axes(px(18), px(12)),
                    column_gap: px(22),
                    ..default()
                },
                BackgroundColor(PANEL),
            ))
            .with_children(|inspector| spawn_inspector(inspector, state, true));
        });
}

fn section_title(parent: &mut ChildSpawnerCommands, label: &str) {
    parent.spawn((
        Text::new(label),
        TextFont {
            font_size: FontSize::Px(13.),
            ..default()
        },
        TextColor(MUTED),
    ));
}

fn spawn_preview(
    parent: &mut ChildSpawnerCommands,
    state: &EditorState,
    editable: bool,
    fixed_border: Option<bool>,
    width: Val,
    height: Val,
) {
    let use_border = fixed_border.unwrap_or(state.preview_border);
    let current = gradient(&state.surface);
    let mut entity = parent.spawn((
        MeshPreview { fixed_border },
        Node {
            position_type: PositionType::Relative,
            width,
            height,
            min_height: px(260),
            border: UiRect::all(px(24)),
            border_radius: BorderRadius::all(px(34)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.055, 0.065, 0.09)),
        BackgroundGradient(if use_border {
            vec![]
        } else {
            vec![current.clone()]
        }),
        BorderGradient(if use_border { vec![current] } else { vec![] }),
    ));
    if editable {
        entity.insert(EditorCanvas);
    }
    entity.with_children(|preview| {
        if editable {
            for (index, point) in state.surface.points().iter().enumerate() {
                preview.spawn(control_point(index, point, index == state.selected));
            }
        }
        preview.spawn((
            Node {
                position_type: PositionType::Absolute,
                left: px(14),
                bottom: px(12),
                padding: UiRect::axes(px(9), px(5)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.02, 0.03, 0.05, 0.72)),
            children![(
                Text::new(if use_border { "BORDER" } else { "BACKGROUND" }),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(TEXT),
            )],
        ));
    });
}

fn control_point(index: usize, point: &Point, selected: bool) -> impl Bundle {
    (
        ControlPoint(index),
        Pickable {
            should_block_lower: true,
            is_hoverable: true,
        },
        Node {
            position_type: PositionType::Absolute,
            left: percent(point.position[0] * 100.),
            top: percent(point.position[1] * 100.),
            width: px(if selected { 22. } else { 17. }),
            height: px(if selected { 22. } else { 17. }),
            border: UiRect::all(px(if selected { 4. } else { 2. })),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        UiTransform::from_translation(Val2::px(
            if selected { -11. } else { -8.5 },
            if selected { -11. } else { -8.5 },
        )),
        BackgroundColor(Color::srgba(
            point.color[0],
            point.color[1],
            point.color[2],
            point.color[3],
        )),
        BorderColor::all(if selected { Color::WHITE } else { Color::BLACK }),
        GlobalZIndex(5),
    )
}

fn spawn_inspector(parent: &mut ChildSpawnerCommands, state: &EditorState, horizontal: bool) {
    let selected = state.surface.points()[state.selected];
    parent.spawn((
        Node {
            min_width: px(if horizontal { 310 } else { 0 }),
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            (
                Text::new("GRID"),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(MUTED),
            ),
            (
                button_row(),
                children![
                    button("2x2", EditorAction::Grid(2)),
                    button("3x3", EditorAction::Grid(3)),
                    button("4x4", EditorAction::Grid(4))
                ]
            ),
            (
                button_row(),
                children![
                    button("Reset", EditorAction::Reset),
                    button(
                        if state.animate { "Pause" } else { "Animate" },
                        EditorAction::ToggleAnimation
                    )
                ]
            ),
        ],
    ));
    parent.spawn((
        Node {
            min_width: px(if horizontal { 390 } else { 0 }),
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            (
                Text::new(format!("POINT {} / RGBA", state.selected)),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(MUTED),
            ),
            (
                Node {
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Center,
                    column_gap: px(10),
                    ..default()
                },
                children![
                    (
                        SelectedSwatch,
                        Node {
                            width: px(34),
                            height: px(34),
                            border: UiRect::all(px(2)),
                            border_radius: BorderRadius::all(px(7)),
                            ..default()
                        },
                        BorderColor::all(Color::WHITE),
                        BackgroundColor(Color::srgba(
                            selected.color[0],
                            selected.color[1],
                            selected.color[2],
                            selected.color[3]
                        ))
                    ),
                    (
                        Node {
                            flex_grow: 1.,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(5),
                            ..default()
                        },
                        children![
                            channel(0, selected.color[0]),
                            channel(1, selected.color[1]),
                            channel(2, selected.color[2]),
                            channel(3, selected.color[3])
                        ]
                    ),
                ]
            ),
        ],
    ));
    parent.spawn((
        Node {
            min_width: px(if horizontal { 340 } else { 0 }),
            flex_grow: 1.,
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            (
                Text::new("PREVIEW"),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(MUTED),
            ),
            (
                button_row(),
                children![
                    button("Background", EditorAction::ShowBackground),
                    button("Border", EditorAction::ShowBorder)
                ]
            ),
            (
                StateReadout,
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(12.),
                    ..default()
                },
                TextColor(TEXT),
            ),
        ],
    ));
}

fn button_row() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        column_gap: px(7),
        ..default()
    }
}

fn button(label: &str, action: EditorAction) -> impl Bundle {
    (
        action,
        Button,
        Hovered::default(),
        Node {
            height: px(31),
            min_width: px(62),
            padding: UiRect::axes(px(11), px(4)),
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::all(px(7)),
            align_items: AlignItems::Center,
            justify_content: JustifyContent::Center,
            ..default()
        },
        BackgroundColor(Color::srgb(0.13, 0.16, 0.21)),
        BorderColor::all(Color::srgb(0.22, 0.27, 0.34)),
        children![(
            Text::new(label),
            TextFont {
                font_size: FontSize::Px(12.),
                ..default()
            },
            TextColor(TEXT),
            Pickable::IGNORE,
        )],
    )
}

fn channel(index: usize, value: f32) -> impl Bundle {
    let label = ["R", "G", "B", "A"][index];
    (
        Node {
            height: px(19),
            flex_direction: FlexDirection::Row,
            align_items: AlignItems::Center,
            column_gap: px(7),
            ..default()
        },
        children![
            (
                Text::new(label),
                TextFont {
                    font_size: FontSize::Px(11.),
                    ..default()
                },
                TextColor(MUTED),
                Node {
                    width: px(11),
                    ..default()
                },
            ),
            (
                ChannelSlider(index),
                Slider::default(),
                SliderValue(value),
                SliderRange::new(0., 1.),
                Hovered::default(),
                Node {
                    width: px(190),
                    height: px(14),
                    align_items: AlignItems::Center,
                    ..default()
                },
                children![
                    (
                        Node {
                            position_type: PositionType::Absolute,
                            width: percent(100),
                            height: px(5),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(Color::srgb(0.18, 0.21, 0.27))
                    ),
                    (
                        SliderVisual,
                        SliderThumb,
                        Node {
                            position_type: PositionType::Absolute,
                            left: percent(value * 100.),
                            width: px(12),
                            height: px(12),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(ACCENT)
                    ),
                ],
            ),
            (
                ChannelValue(index),
                Text::new(format!("{value:.2}")),
                TextFont {
                    font_size: FontSize::Px(11.),
                    ..default()
                },
                TextColor(TEXT),
                Node {
                    width: px(32),
                    ..default()
                },
            ),
        ],
    )
}

fn switcher() -> impl Bundle {
    (
        Node {
            position_type: PositionType::Absolute,
            left: percent(50),
            bottom: px(14),
            height: px(42),
            padding: UiRect::all(px(5)),
            column_gap: px(8),
            align_items: AlignItems::Center,
            border: UiRect::all(px(1)),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        UiTransform::from_translation(Val2::percent(-50., 0.)),
        GlobalZIndex(20),
        BackgroundColor(Color::srgb(0.02, 0.025, 0.035)),
        BorderColor::all(Color::srgb(0.32, 0.39, 0.49)),
        children![
            button("<", EditorAction::PreviousVariant),
            (
                VariantReadout,
                Text::new(""),
                TextFont {
                    font_size: FontSize::Px(13.),
                    ..default()
                },
                TextColor(TEXT),
                Node {
                    width: px(170),
                    justify_content: JustifyContent::Center,
                    ..default()
                },
            ),
            button(">", EditorAction::NextVariant),
        ],
    )
}

fn handle_action(
    event: On<Activate>,
    actions: Query<&EditorAction>,
    mut state: ResMut<EditorState>,
) {
    let Ok(action) = actions.get(event.entity) else {
        return;
    };
    match *action {
        EditorAction::PreviousVariant => {
            state.variant = (state.variant + VARIANTS.len() - 1) % VARIANTS.len();
            state.rebuild = true;
        }
        EditorAction::NextVariant => {
            state.variant = (state.variant + 1) % VARIANTS.len();
            state.rebuild = true;
        }
        EditorAction::Grid(size) => state.replace_grid(size),
        EditorAction::Reset => state.reset(),
        EditorAction::ToggleAnimation => {
            state.animate = !state.animate;
            state.rest = state.surface.points().to_vec();
            state.elapsed = 0.;
            state.status = if state.animate {
                "Animation enabled; direct editing pauses it".into()
            } else {
                "Animation paused at the current valid surface".into()
            };
            state.rebuild = true;
        }
        EditorAction::ShowBackground => state.preview_border = false,
        EditorAction::ShowBorder => state.preview_border = true,
    }
}

fn select_point(
    event: On<PointerPress>,
    points: Query<&ControlPoint>,
    mut state: ResMut<EditorState>,
) {
    let Ok(point) = points.get(event.entity) else {
        return;
    };
    state.selected = point.0;
    state.status = format!("Selected point {}", point.0);
}

fn drag_point(
    mut event: On<PointerDrag>,
    points: Query<&ControlPoint>,
    canvas: Query<&ComputedNode, With<EditorCanvas>>,
    mut state: ResMut<EditorState>,
) {
    let Ok(point) = points.get(event.entity) else {
        return;
    };
    let Ok(canvas) = canvas.single() else {
        return;
    };
    event.propagate(false);
    let size = canvas.size();
    if size.min_element() <= 0. {
        return;
    }
    state.selected = point.0;
    let mut candidate = state.surface.points().to_vec();
    candidate[point.0].position[0] =
        (candidate[point.0].position[0] + event.delta.x / size.x).clamp(0., 1.);
    candidate[point.0].position[1] =
        (candidate[point.0].position[1] + event.delta.y / size.y).clamp(0., 1.);
    state.accept(candidate, "drag; animation paused");
}

fn edit_channel(
    event: On<ValueChange<f32>>,
    channels: Query<&ChannelSlider>,
    mut state: ResMut<EditorState>,
) {
    let Ok(channel) = channels.get(event.source) else {
        return;
    };
    let mut candidate = state.surface.points().to_vec();
    candidate[state.selected].color[channel.0] = event.value.clamp(0., 1.);
    state.accept(candidate, ["red", "green", "blue", "alpha"][channel.0]);
}

fn keyboard(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
    if keys.just_pressed(KeyCode::ArrowLeft) {
        state.variant = (state.variant + VARIANTS.len() - 1) % VARIANTS.len();
        state.rebuild = true;
    }
    if keys.just_pressed(KeyCode::ArrowRight) {
        state.variant = (state.variant + 1) % VARIANTS.len();
        state.rebuild = true;
    }
    for (key, size) in [
        (KeyCode::Digit2, 2),
        (KeyCode::Digit3, 3),
        (KeyCode::Digit4, 4),
    ] {
        if keys.just_pressed(key) {
            state.replace_grid(size);
        }
    }
    if keys.just_pressed(KeyCode::KeyR) {
        state.reset();
    }
    if keys.just_pressed(KeyCode::KeyB) {
        state.preview_border = !state.preview_border;
    }
    if keys.just_pressed(KeyCode::Space) {
        state.animate = !state.animate;
        state.rest = state.surface.points().to_vec();
        state.elapsed = 0.;
        state.rebuild = true;
    }
}

fn animate(time: Res<Time>, mut state: ResMut<EditorState>) {
    if !state.animate {
        return;
    }
    state.elapsed += time.delta_secs();
    let mut candidate = state.rest.clone();
    for y in 1..state.height - 1 {
        for x in 1..state.width - 1 {
            let point = &mut candidate[y * state.width + x];
            point.position[0] += 0.035 * state.elapsed.sin();
            point.position[1] += 0.025 * (state.elapsed * 0.7).cos();
        }
    }
    if let Err(error) = state.surface.try_replace_points(candidate) {
        state.animate = false;
        state.status = format!("Animation stopped before invalid frame: {error:?}");
    }
}

fn scripted_invalid_edit(capture: Res<Capture>, mut state: ResMut<EditorState>) {
    if !capture.attempt_fold || capture.frame != 10 || state.surface.points().len() < 5 {
        return;
    }
    let mut candidate = state.surface.points().to_vec();
    candidate[4].position = [1.6, -0.5];
    state.selected = 4;
    state.accept(candidate, "scripted fold");
}

fn sync_editor(
    mut commands: Commands,
    state: Res<EditorState>,
    mut previews: Query<(&MeshPreview, &mut BackgroundGradient, &mut BorderGradient)>,
    mut points: Query<(
        &ControlPoint,
        &mut Node,
        &mut UiTransform,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
    mut readouts: Query<&mut Text, With<StateReadout>>,
    mut variants: Query<&mut Text, (With<VariantReadout>, Without<StateReadout>)>,
    mut swatches: Query<&mut BackgroundColor, (With<SelectedSwatch>, Without<ControlPoint>)>,
    sliders: Query<(Entity, &ChannelSlider, &SliderValue, &Children)>,
    mut slider_visuals: Query<&mut Node, (With<SliderVisual>, Without<ControlPoint>)>,
    mut channel_values: Query<
        (&ChannelValue, &mut Text),
        (Without<StateReadout>, Without<VariantReadout>),
    >,
) {
    let current = gradient(&state.surface);
    for (preview, mut background, mut border) in &mut previews {
        let use_border = preview.fixed_border.unwrap_or(state.preview_border);
        background.0 = if use_border {
            vec![]
        } else {
            vec![current.clone()]
        };
        border.0 = if use_border {
            vec![current.clone()]
        } else {
            vec![]
        };
    }
    for (marker, mut node, mut transform, mut color, mut border) in &mut points {
        let point = &state.surface.points()[marker.0];
        let selected = marker.0 == state.selected;
        node.left = percent(point.position[0] * 100.);
        node.top = percent(point.position[1] * 100.);
        let diameter = if selected { 22. } else { 17. };
        node.width = px(diameter);
        node.height = px(diameter);
        node.border = UiRect::all(px(if selected { 4. } else { 2. }));
        transform.translation = Val2::px(-diameter / 2., -diameter / 2.);
        color.0 = Color::srgba(
            point.color[0],
            point.color[1],
            point.color[2],
            point.color[3],
        );
        border.set_all(if selected { Color::WHITE } else { Color::BLACK });
    }
    let selected = &state.surface.points()[state.selected];
    for mut text in &mut readouts {
        text.0 = format!(
            "Selected {}  pos [{:.3}, {:.3}]\nRGBA [{:.2}, {:.2}, {:.2}, {:.2}]\nAnimation: {}  preview: {}\n{}",
            state.selected,
            selected.position[0],
            selected.position[1],
            selected.color[0],
            selected.color[1],
            selected.color[2],
            selected.color[3],
            state.animate,
            if state.preview_border { "border" } else { "background" },
            state.status,
        );
    }
    for mut text in &mut variants {
        text.0 = VARIANTS[state.variant].into();
    }
    for mut color in &mut swatches {
        color.0 = Color::srgba(
            selected.color[0],
            selected.color[1],
            selected.color[2],
            selected.color[3],
        );
    }
    for (entity, channel, value, children) in &sliders {
        let desired = selected.color[channel.0];
        if value.0 != desired {
            commands.entity(entity).insert(SliderValue(desired));
        }
        for descendant in children.iter() {
            if let Ok(mut thumb) = slider_visuals.get_mut(descendant) {
                thumb.left = percent(desired * 100.);
            }
        }
    }
    for (channel, mut text) in &mut channel_values {
        text.0 = format!("{:.2}", selected.color[channel.0]);
    }
}

fn style_buttons(
    mut buttons: Query<
        (
            &Hovered,
            Has<Pressed>,
            &mut BackgroundColor,
            &mut BorderColor,
        ),
        With<Button>,
    >,
) {
    for (hovered, pressed, mut background, mut border) in &mut buttons {
        background.0 = match (hovered.0, pressed) {
            (_, true) => Color::srgb(0.16, 0.45, 0.60),
            (true, false) => Color::srgb(0.18, 0.23, 0.30),
            _ => Color::srgb(0.13, 0.16, 0.21),
        };
        border.set_all(if hovered.0 {
            ACCENT
        } else {
            Color::srgb(0.22, 0.27, 0.34)
        });
    }
}

fn capture(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut capture: ResMut<Capture>,
    mut exit: MessageWriter<AppExit>,
    state: Res<EditorState>,
    roots: Query<&ComputedNode, With<EditorRoot>>,
    previews: Query<&ComputedNode, With<MeshPreview>>,
) {
    capture.frame += 1;
    if keys.just_pressed(KeyCode::KeyS) || (capture.path.is_some() && capture.frame == capture.at) {
        let root_size = roots.single().map(ComputedNode::size).unwrap_or_default();
        let preview_sizes: Vec<Vec2> = previews.iter().map(ComputedNode::size).collect();
        println!(
            "EDITOR_PROTOTYPE variant={} root={root_size:?} previews={preview_sizes:?} status={}",
            VARIANTS[state.variant], state.status
        );
        let path = capture
            .path
            .clone()
            .unwrap_or("/tmp/mesh-gradient-editor-prototype.png".into());
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
    }
    if capture.path.is_some() && capture.frame > capture.at + 40 {
        exit.write(AppExit::Success);
    }
}
