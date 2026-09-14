//! Edit a checked mesh gradient while previewing it as a UI background and border.
//!
//! Drag a colored point in the left preview or select it and edit its RGBA channels.
//! Every change is submitted through [`MeshGradient`]'s checked API. If a candidate
//! surface is rejected, both previews retain the last valid mesh.

use bevy::{
    color::ColorToComponents,
    math::ops,
    picking::hover::Hovered,
    prelude::*,
    ui::{MeshGradient, MeshGradientError, MeshGradientPoint, Pressed},
    ui_widgets::{Activate, Button, Slider, SliderRange, SliderThumb, SliderValue, ValueChange},
    window::PrimaryWindow,
};

const PANEL: Color = Color::srgb(0.075, 0.09, 0.125);
const TEXT: Color = Color::srgb(0.88, 0.91, 0.96);
const MUTED: Color = Color::srgb(0.56, 0.63, 0.73);
const ACCENT: Color = Color::srgb(0.25, 0.72, 0.92);

#[derive(Resource)]
struct EditorState {
    mesh: MeshGradient,
    rest: Vec<MeshGradientPoint>,
    selected: usize,
    animate: bool,
    elapsed: f32,
    show_background: bool,
    show_border: bool,
    rebuild: bool,
    status: String,
}

impl EditorState {
    fn new() -> Self {
        let points = preset(3);
        Self {
            mesh: MeshGradient::new(3, 3, points.clone()).expect("preset must be valid"),
            rest: points,
            selected: 4,
            animate: false,
            elapsed: 0.0,
            show_background: true,
            show_border: true,
            rebuild: true,
            status: "Ready: drag a point or select it to edit RGBA".into(),
        }
    }

    fn replace_grid(&mut self, size: usize) {
        let points = preset(size);
        match self.mesh.try_replace_grid(size, size, points.clone()) {
            Ok(()) => {
                self.rest = points;
                self.selected = (size * size) / 2;
                self.animate = false;
                self.elapsed = 0.0;
                self.status = format!("Accepted atomic {size}x{size} grid replacement");
                self.rebuild = true;
            }
            Err(error) => self.reject("grid replacement", error),
        }
    }

    fn reset(&mut self) {
        let size = self.mesh.width();
        self.replace_grid(size);
        self.status = "Reset positions and colors".into();
    }

    fn accept_manual(&mut self, candidate: Vec<MeshGradientPoint>, action: &str) {
        self.animate = false;
        self.elapsed = 0.0;
        match self.mesh.try_replace_points(candidate) {
            Ok(()) => {
                self.rest = self.mesh.points().to_vec();
                self.status = format!("Accepted {action}; animation paused");
            }
            Err(error) => self.reject(action, error),
        }
    }

    fn reject(&mut self, action: &str, error: MeshGradientError) {
        self.animate = false;
        self.status = format!("Rejected {action}; retained last valid surface: {error}");
    }
}

#[derive(Component)]
struct EditorRoot;

#[derive(Component)]
struct EditorCanvas;

#[derive(Component)]
struct PreviewRow;

#[derive(Component)]
struct PreviewColumn;

#[derive(Component, Clone, Copy)]
enum PreviewKind {
    Background,
    Border,
}

#[derive(Component)]
struct ControlPoint(usize);

#[derive(Component)]
struct StateReadout;

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
    Grid(usize),
    Reset,
    ToggleAnimation,
    ToggleBackground,
    ToggleBorder,
}

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.025, 0.033, 0.05)))
        .insert_resource(EditorState::new())
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mesh Gradient".into(),
                resolution: (1280, 800).into(),
                fit_canvas_to_parent: true,
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
                rebuild_editor,
                responsive_layout,
                sync_editor,
                style_buttons,
            )
                .chain(),
        )
        .run();
}

fn preset(size: usize) -> Vec<MeshGradientPoint> {
    (0..size * size)
        .map(|index| {
            let column = index % size;
            let row = index / size;
            let x = column as f32 / (size - 1) as f32;
            let y = row as f32 / (size - 1) as f32;
            let mut position = Vec2::new(x, y);
            if column > 0 && column + 1 < size && row > 0 && row + 1 < size {
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
        .collect()
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
    commands
        .spawn((
            EditorRoot,
            Node {
                width: percent(100),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                padding: UiRect::all(px(20)),
                row_gap: px(14),
                ..default()
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    flex_direction: FlexDirection::Column,
                    flex_shrink: 0.0,
                    ..default()
                },
                children![
                    (
                        Text::new("MESH GRADIENT"),
                        TextFont::from_font_size(23.0),
                        TextColor(TEXT),
                    ),
                    (
                        Text::new(
                            "Drag colored points; checked edits keep both previews synchronized."
                        ),
                        TextFont::from_font_size(13.0),
                        TextColor(MUTED),
                    ),
                ],
            ));

            parent
                .spawn((
                    PreviewRow,
                    Node {
                        flex_grow: 1.0,
                        min_height: px(260),
                        width: percent(100),
                        flex_direction: FlexDirection::Row,
                        column_gap: px(18),
                        row_gap: px(12),
                        ..default()
                    },
                ))
                .with_children(|previews| {
                    preview_column(previews, state, PreviewKind::Background);
                    preview_column(previews, state, PreviewKind::Border);
                });

            parent
                .spawn((
                    Node {
                        flex_shrink: 0.0,
                        width: percent(100),
                        flex_direction: FlexDirection::Row,
                        flex_wrap: FlexWrap::Wrap,
                        padding: UiRect::axes(px(18), px(12)),
                        column_gap: px(24),
                        row_gap: px(12),
                        ..default()
                    },
                    BackgroundColor(PANEL),
                ))
                .with_children(|inspector| spawn_inspector(inspector, state));
        });
}

fn preview_column(parent: &mut ChildSpawnerCommands, state: &EditorState, kind: PreviewKind) {
    let (title, editable) = match kind {
        PreviewKind::Background => ("BACKGROUND / EDITABLE", true),
        PreviewKind::Border => ("BORDER / SYNCHRONIZED", false),
    };
    parent
        .spawn((
            PreviewColumn,
            Node {
                width: percent(50),
                height: percent(100),
                flex_direction: FlexDirection::Column,
                row_gap: px(8),
                ..default()
            },
        ))
        .with_children(|column| {
            column.spawn((
                Text::new(title),
                TextFont::from_font_size(13.0),
                TextColor(MUTED),
            ));
            spawn_preview(column, state, kind, editable);
        });
}

fn spawn_preview(
    parent: &mut ChildSpawnerCommands,
    state: &EditorState,
    kind: PreviewKind,
    editable: bool,
) {
    let show = match kind {
        PreviewKind::Background => state.show_background,
        PreviewKind::Border => state.show_border,
    };
    let gradient: Gradient = state.mesh.clone().into();
    let mut entity = parent.spawn((
        kind,
        Node {
            position_type: PositionType::Relative,
            flex_grow: 1.0,
            width: percent(100),
            min_height: px(180),
            border: UiRect::all(px(24)),
            border_radius: BorderRadius::all(px(34)),
            ..default()
        },
        BackgroundColor(Color::srgb(0.055, 0.065, 0.09)),
        BackgroundGradient(match kind {
            PreviewKind::Background if show => vec![gradient.clone()],
            _ => vec![],
        }),
        BorderGradient(match kind {
            PreviewKind::Border if show => vec![gradient],
            _ => vec![],
        }),
    ));
    if editable {
        entity.insert(EditorCanvas);
    }
    entity.with_children(|preview| {
        if editable {
            for (index, point) in state.mesh.points().iter().enumerate() {
                preview.spawn(control_point(index, point, index == state.selected));
            }
        }
    });
}

fn control_point(index: usize, point: &MeshGradientPoint, selected: bool) -> impl Bundle {
    let diameter = if selected { 22.0 } else { 17.0 };
    (
        ControlPoint(index),
        Pickable {
            should_block_lower: true,
            is_hoverable: true,
        },
        Node {
            position_type: PositionType::Absolute,
            left: percent(point.position.x * 100.0),
            top: percent(point.position.y * 100.0),
            width: px(diameter),
            height: px(diameter),
            border: UiRect::all(px(if selected { 4.0 } else { 2.0 })),
            border_radius: BorderRadius::MAX,
            ..default()
        },
        UiTransform::from_translation(Val2::px(-diameter / 2.0, -diameter / 2.0)),
        BackgroundColor(point.color),
        BorderColor::all(if selected { Color::WHITE } else { Color::BLACK }),
        GlobalZIndex(5),
    )
}

fn spawn_inspector(parent: &mut ChildSpawnerCommands, state: &EditorState) {
    let selected = state.mesh.points()[state.selected];
    let rgba = selected.color.to_srgba().to_f32_array();

    parent.spawn((
        Node {
            min_width: px(265),
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            section_label("GRID + PLAYBACK"),
            (
                button_row(),
                children![
                    button("2x2", EditorAction::Grid(2)),
                    button("3x3", EditorAction::Grid(3)),
                    button("4x4", EditorAction::Grid(4)),
                ]
            ),
            (
                button_row(),
                children![
                    button("Reset", EditorAction::Reset),
                    button(
                        if state.animate { "Pause" } else { "Animate" },
                        EditorAction::ToggleAnimation,
                    ),
                ]
            ),
        ],
    ));

    parent.spawn((
        Node {
            min_width: px(370),
            flex_grow: 1.0,
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            section_label("SELECTED POINT / RGBA"),
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
                        BackgroundColor(selected.color),
                    ),
                    (
                        Node {
                            flex_grow: 1.0,
                            flex_direction: FlexDirection::Column,
                            row_gap: px(5),
                            ..default()
                        },
                        children![
                            channel(0, rgba[0]),
                            channel(1, rgba[1]),
                            channel(2, rgba[2]),
                            channel(3, rgba[3]),
                        ]
                    ),
                ]
            ),
        ],
    ));

    parent.spawn((
        Node {
            min_width: px(300),
            flex_grow: 1.0,
            flex_direction: FlexDirection::Column,
            row_gap: px(7),
            ..default()
        },
        children![
            section_label("PREVIEWS + CHECKED STATE"),
            (
                button_row(),
                children![
                    button("Background", EditorAction::ToggleBackground),
                    button("Border", EditorAction::ToggleBorder),
                ]
            ),
            (
                StateReadout,
                Text::new(""),
                TextFont::from_font_size(12.0),
                TextColor(TEXT),
            ),
        ],
    ));
}

fn section_label(label: &'static str) -> impl Bundle {
    (
        Text::new(label),
        TextFont::from_font_size(12.0),
        TextColor(MUTED),
    )
}

fn button_row() -> Node {
    Node {
        flex_direction: FlexDirection::Row,
        column_gap: px(7),
        ..default()
    }
}

fn button(label: &'static str, action: EditorAction) -> impl Bundle {
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
            TextFont::from_font_size(12.0),
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
                TextFont::from_font_size(11.0),
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
                SliderRange::new(0.0, 1.0),
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
                        BackgroundColor(Color::srgb(0.18, 0.21, 0.27)),
                    ),
                    (
                        SliderVisual,
                        SliderThumb,
                        Node {
                            position_type: PositionType::Absolute,
                            left: percent(value * 100.0),
                            width: px(12),
                            height: px(12),
                            border_radius: BorderRadius::MAX,
                            ..default()
                        },
                        BackgroundColor(ACCENT),
                    ),
                ],
            ),
            (
                ChannelValue(index),
                Text::new(format!("{value:.2}")),
                TextFont::from_font_size(11.0),
                TextColor(TEXT),
                Node {
                    width: px(32),
                    ..default()
                },
            ),
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
        EditorAction::Grid(size) => state.replace_grid(size),
        EditorAction::Reset => state.reset(),
        EditorAction::ToggleAnimation => {
            state.animate = !state.animate;
            state.rest = state.mesh.points().to_vec();
            state.elapsed = 0.0;
            state.status = if state.animate {
                "Animation enabled; direct editing will pause it".into()
            } else {
                "Animation paused at the current valid surface".into()
            };
            state.rebuild = true;
        }
        EditorAction::ToggleBackground => {
            state.show_background = !state.show_background;
            state.status = format!("Background preview visible: {}", state.show_background);
        }
        EditorAction::ToggleBorder => {
            state.show_border = !state.show_border;
            state.status = format!("Border preview visible: {}", state.show_border);
        }
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
    if size.min_element() <= 0.0 {
        return;
    }
    state.selected = point.0;
    let mut candidate = state.mesh.points().to_vec();
    candidate[point.0].position += event.delta / size;
    candidate[point.0].position = candidate[point.0].position.clamp(Vec2::ZERO, Vec2::ONE);
    state.accept_manual(candidate, "point drag");
}

fn edit_channel(
    event: On<ValueChange<f32>>,
    channels: Query<&ChannelSlider>,
    mut state: ResMut<EditorState>,
) {
    let Ok(channel) = channels.get(event.source) else {
        return;
    };
    let mut candidate = state.mesh.points().to_vec();
    let point = &mut candidate[state.selected];
    let mut rgba = point.color.to_srgba().to_f32_array();
    rgba[channel.0] = event.value.clamp(0.0, 1.0);
    point.color = Color::srgba(rgba[0], rgba[1], rgba[2], rgba[3]);
    state.accept_manual(candidate, ["red", "green", "blue", "alpha"][channel.0]);
}

fn keyboard(keys: Res<ButtonInput<KeyCode>>, mut state: ResMut<EditorState>) {
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
    if keys.just_pressed(KeyCode::Space) {
        state.animate = !state.animate;
        state.rest = state.mesh.points().to_vec();
        state.elapsed = 0.0;
        state.status = if state.animate {
            "Animation enabled; direct editing will pause it".into()
        } else {
            "Animation paused at the current valid surface".into()
        };
        state.rebuild = true;
    }
}

fn animate(time: Res<Time>, mut state: ResMut<EditorState>) {
    if !state.animate {
        return;
    }
    state.elapsed += time.delta_secs();
    let mut candidate = state.rest.clone();
    let (width, height) = state.mesh.dimensions();
    for row in 1..height - 1 {
        for column in 1..width - 1 {
            let point = &mut candidate[row * width + column];
            point.position.x += 0.035 * ops::sin(state.elapsed);
            point.position.y += 0.025 * ops::cos(state.elapsed * 0.7);
        }
    }
    if let Err(error) = state.mesh.try_replace_points(candidate) {
        state.reject("animation frame", error);
    }
}

fn responsive_layout(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut roots: Query<
        &mut Node,
        (
            With<EditorRoot>,
            Without<PreviewRow>,
            Without<PreviewColumn>,
        ),
    >,
    mut rows: Query<
        &mut Node,
        (
            With<PreviewRow>,
            Without<EditorRoot>,
            Without<PreviewColumn>,
        ),
    >,
    mut columns: Query<
        &mut Node,
        (
            With<PreviewColumn>,
            Without<EditorRoot>,
            Without<PreviewRow>,
        ),
    >,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let narrow = window.width() < 900.0;
    for mut root in &mut roots {
        root.padding = UiRect::all(px(if narrow { 10 } else { 20 }));
        root.row_gap = px(if narrow { 8 } else { 14 });
    }
    for mut row in &mut rows {
        row.flex_direction = if narrow {
            FlexDirection::Column
        } else {
            FlexDirection::Row
        };
    }
    for mut column in &mut columns {
        column.width = percent(if narrow { 100 } else { 50 });
        column.height = percent(if narrow { 50 } else { 100 });
    }
}

type ControlPointVisuals<'w, 's> = Query<
    'w,
    's,
    (
        &'static ControlPoint,
        &'static mut Node,
        &'static mut UiTransform,
        &'static mut BackgroundColor,
        &'static mut BorderColor,
    ),
>;

fn sync_editor(
    mut commands: Commands,
    state: Res<EditorState>,
    mut previews: Query<(&PreviewKind, &mut BackgroundGradient, &mut BorderGradient)>,
    mut points: ControlPointVisuals,
    mut readouts: Query<&mut Text, With<StateReadout>>,
    mut swatches: Query<&mut BackgroundColor, (With<SelectedSwatch>, Without<ControlPoint>)>,
    sliders: Query<(Entity, &ChannelSlider, &SliderValue, &Children)>,
    mut slider_visuals: Query<&mut Node, (With<SliderVisual>, Without<ControlPoint>)>,
    mut channel_values: Query<(&ChannelValue, &mut Text), Without<StateReadout>>,
) {
    let current: Gradient = state.mesh.clone().into();
    for (kind, mut background, mut border) in &mut previews {
        match kind {
            PreviewKind::Background => {
                background.0 = state
                    .show_background
                    .then(|| current.clone())
                    .into_iter()
                    .collect();
                border.0.clear();
            }
            PreviewKind::Border => {
                background.0.clear();
                border.0 = state
                    .show_border
                    .then(|| current.clone())
                    .into_iter()
                    .collect();
            }
        }
    }

    for (marker, mut node, mut transform, mut color, mut border) in &mut points {
        let point = &state.mesh.points()[marker.0];
        let selected = marker.0 == state.selected;
        let diameter = if selected { 22.0 } else { 17.0 };
        node.left = percent(point.position.x * 100.0);
        node.top = percent(point.position.y * 100.0);
        node.width = px(diameter);
        node.height = px(diameter);
        node.border = UiRect::all(px(if selected { 4.0 } else { 2.0 }));
        transform.translation = Val2::px(-diameter / 2.0, -diameter / 2.0);
        color.0 = point.color;
        border.set_all(if selected { Color::WHITE } else { Color::BLACK });
    }

    let selected = state.mesh.points()[state.selected];
    let rgba = selected.color.to_srgba().to_f32_array();
    for mut text in &mut readouts {
        text.0 = format!(
            "Point {} | pos [{:.3}, {:.3}] | RGBA [{:.2}, {:.2}, {:.2}, {:.2}]\nAnimation: {} | {}",
            state.selected,
            selected.position.x,
            selected.position.y,
            rgba[0],
            rgba[1],
            rgba[2],
            rgba[3],
            if state.animate { "playing" } else { "paused" },
            state.status,
        );
    }
    for mut color in &mut swatches {
        color.0 = selected.color;
    }
    for (entity, channel, value, children) in &sliders {
        let desired = rgba[channel.0];
        if value.0 != desired {
            commands.entity(entity).insert(SliderValue(desired));
        }
        for descendant in children.iter() {
            if let Ok(mut thumb) = slider_visuals.get_mut(descendant) {
                thumb.left = percent(desired * 100.0);
            }
        }
    }
    for (channel, mut text) in &mut channel_values {
        text.0 = format!("{:.2}", rgba[channel.0]);
    }
}

fn style_buttons(
    state: Res<EditorState>,
    mut buttons: Query<(
        &EditorAction,
        &Hovered,
        Has<Pressed>,
        &mut BackgroundColor,
        &mut BorderColor,
    )>,
) {
    for (action, hovered, pressed, mut background, mut border) in &mut buttons {
        let active = match *action {
            EditorAction::Grid(size) => state.mesh.width() == size,
            EditorAction::ToggleAnimation => state.animate,
            EditorAction::ToggleBackground => state.show_background,
            EditorAction::ToggleBorder => state.show_border,
            EditorAction::Reset => false,
        };
        background.0 = match (pressed, hovered.0, active) {
            (true, _, _) => Color::srgb(0.16, 0.45, 0.60),
            (_, _, true) => Color::srgb(0.12, 0.36, 0.49),
            (_, true, _) => Color::srgb(0.18, 0.23, 0.30),
            _ => Color::srgb(0.13, 0.16, 0.21),
        };
        border.set_all(if hovered.0 || active {
            ACCENT
        } else {
            Color::srgb(0.22, 0.27, 0.34)
        });
    }
}
