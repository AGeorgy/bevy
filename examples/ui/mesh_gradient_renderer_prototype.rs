//! THROWAWAY: native UI transport and validity experiment, not the production API.
//! cargo +1.97.1 run --example mesh_gradient_renderer_prototype --no-default-features --features ui
//! Space: animate. C: change color space. F: attempt a fold. R: reset. S: screenshot.
mod mesh_gradient_prototype_math;
use bevy::platform::time::Instant;
use bevy::{
    prelude::*,
    render::view::screenshot::{save_to_disk, Screenshot},
    ui::PrototypeMeshGradient,
};
use mesh_gradient_prototype_math::{MeshSurface, Point};

#[derive(Resource)]
struct Experiment {
    frame: u32,
    animate: bool,
    space: usize,
    elapsed: f32,
    capture: Option<String>,
    attempt_fold: bool,
    frames: u32,
    update_us: Vec<f64>,
    frame_ms: Vec<f64>,
    status: String,
}
#[derive(Component)]
struct Panel {
    surface: MeshSurface,
    base: Vec<Point>,
    n: usize,
    subdivisions: usize,
    kind: usize,
}
#[derive(Component)]
struct Hud;
const SPACES: [InterpolationColorSpace; 3] = [
    InterpolationColorSpace::Oklaba,
    InterpolationColorSpace::Srgba,
    InterpolationColorSpace::LinearRgba,
];
const LABELS: [&str; 6] = [
    "3×3 · 4 subdivisions (coarse)",
    "3×3 · 16 subdivisions",
    "3×3 · 64 subdivisions (reference)",
    "4×4 · alpha over linear gradient",
    "3×3 · asymmetric rounded border",
    "3×3 · transform + ancestor clipping",
];

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.035, 0.045, 0.065)))
        .insert_resource(Experiment {
            frame: 0,
            animate: args.iter().any(|a| a == "--animate"),
            space: 0,
            elapsed: 0.,
            capture: value("--capture"),
            attempt_fold: args.iter().any(|a| a == "--attempt-fold"),
            frames: value("--frames")
                .and_then(|s| s.parse().ok())
                .unwrap_or(150),
            update_us: vec![],
            frame_ms: vec![],
            status: "Validated surface; no attempted edit yet".into(),
        })
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Mesh gradient · renderer prototype".into(),
                resolution: (1160, 820).into(),
                present_mode: bevy::window::PresentMode::AutoVsync,
                ..default()
            }),
            ..default()
        }))
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .run();
}
fn palette(i: usize) -> Color {
    let c = [
        [0.98, 0.15, 0.32],
        [1., 0.72, 0.14],
        [0.28, 0.85, 0.82],
        [0.6, 0.14, 0.92],
        [0.13, 0.34, 0.93],
        [0.98, 0.42, 0.66],
        [0.2, 0.65, 0.46],
        [0.72, 0.35, 0.98],
        [0.95, 0.92, 0.58],
    ][i % 9];
    Color::srgb(c[0], c[1], c[2])
}
fn components(color: Color, space: usize) -> [f32; 4] {
    match space {
        0 => {
            let c = Oklaba::from(color);
            [c.lightness, c.a, c.b, c.alpha]
        }
        1 => color.to_srgba().to_f32_array(),
        _ => color.to_linear().to_f32_array(),
    }
}
fn points(n: usize, kind: usize, space: usize) -> Vec<Point> {
    (0..n * n)
        .map(|i| {
            let (x, y) = (i % n, i / n);
            let mut position = [x as f32 / (n - 1) as f32, y as f32 / (n - 1) as f32];
            if x > 0 && x < n - 1 && y > 0 && y < n - 1 {
                position[0] += 0.06;
                position[1] -= 0.04;
            }
            // Inset boundary demonstrates uncovered area; never replace this with a full-node solid fill.
            if kind == 3 {
                position = position.map(|v| 0.06 + 0.88 * v);
            }
            let mut color = components(palette(i), space);
            if kind == 3 {
                color[3] = if i % 3 == 0 { 0.12 } else { 0.8 };
            }
            Point { position, color }
        })
        .collect()
}
fn transport(surface: &MeshSurface, subdivisions: usize, space: usize) -> Gradient {
    let (vertices, indices) = surface.tessellate(subdivisions);
    PrototypeMeshGradient {
        vertices: vertices
            .into_iter()
            .map(|v| (Vec2::from_array(v.position), v.color))
            .collect(),
        indices,
        color_space: SPACES[space],
    }
    .into()
}
fn underlay() -> Gradient {
    LinearGradient::to_top_right(vec![
        ColorStop::auto(Color::srgb(0.9, 0.9, 0.9)),
        ColorStop::auto(Color::srgb(0.12, 0.18, 0.24)),
    ])
    .into()
}
fn setup(mut commands: Commands) {
    commands.spawn(Camera2d);
    commands.spawn((
        Text::new("MESH GRADIENT / RENDERER PROTOTYPE"),
        TextFont {
            font_size: FontSize::Px(25.),
            ..default()
        },
        TextColor(Color::WHITE),
        Node {
            position_type: PositionType::Absolute,
            left: px(30),
            top: px(20),
            ..default()
        },
    ));
    commands.spawn((Text::new("Same inferred cubic surface · direct UI triangles · 3 color spaces · validity-preserving edits"),TextFont {font_size:FontSize::Px(16.),..default()},TextColor(Color::srgb(0.62,0.69,0.78)),Node {position_type:PositionType::Absolute,left:px(30),top:px(56),..default()}));
    for kind in 0..6 {
        let left = 30. + (kind % 3) as f32 * 374.;
        let top = 108. + (kind / 3) as f32 * 305.;
        commands.spawn((
            Text::new(LABELS[kind]),
            TextFont {
                font_size: FontSize::Px(16.),
                ..default()
            },
            TextColor(Color::srgb(0.85, 0.89, 0.95)),
            Node {
                position_type: PositionType::Absolute,
                left: px(left),
                top: px(top - 25.),
                ..default()
            },
        ));
        let n = if kind == 3 { 4 } else { 3 };
        let subdivisions = match kind {
            0 => 4,
            2 => 64,
            _ => 16,
        };
        let base = points(n, kind, 0);
        let surface = MeshSurface::try_new(n, n, base.clone()).expect("preset must certify");
        let gradient = transport(&surface, subdivisions, 0);
        let parent = commands
            .spawn((
                Node {
                    position_type: PositionType::Absolute,
                    left: px(left),
                    top: px(top),
                    width: px(350),
                    height: px(255),
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.08, 0.10, 0.14)),
            ))
            .id();
        let mut entity = commands.spawn((
            Node {
                width: px(350),
                height: px(255),
                border: if kind == 4 {
                    UiRect {
                        left: px(28),
                        right: px(12),
                        top: px(18),
                        bottom: px(36),
                    }
                } else {
                    UiRect::ZERO
                },
                border_radius: BorderRadius::all(px(30)),
                ..default()
            },
            Panel {
                surface,
                base,
                n,
                subdivisions,
                kind,
            },
        ));
        if kind == 4 {
            entity.insert((
                BackgroundColor(Color::srgb(0.055, 0.07, 0.10)),
                BorderGradient(vec![gradient]),
            ));
        } else {
            entity.insert(BackgroundGradient(if kind == 3 {
                vec![underlay(), gradient]
            } else {
                vec![gradient]
            }));
        }
        if kind == 5 {
            entity.insert(UiTransform {
                rotation: Rot2::radians(0.18),
                scale: Vec2::new(1.1, 1.04),
                ..default()
            });
        }
        let child = entity.id();
        commands.entity(parent).add_child(child);
    }
    commands.spawn((
        Text::new(""),
        TextFont {
            font_size: FontSize::Px(16.),
            ..default()
        },
        TextColor(Color::srgb(0.75, 0.83, 0.91)),
        Node {
            position_type: PositionType::Absolute,
            left: px(30),
            top: px(722),
            ..default()
        },
        Hud,
    ));
}
fn update(
    mut commands: Commands,
    mut ex: ResMut<Experiment>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut panels: Query<(
        &mut Panel,
        Option<&mut BackgroundGradient>,
        Option<&mut BorderGradient>,
    )>,
    mut hud: Query<&mut Text, With<Hud>>,
    mut exit: MessageWriter<AppExit>,
) {
    ex.frame += 1;
    if ex.frame > 30 {
        ex.frame_ms.push(time.delta_secs_f64() * 1000.);
    }
    if keys.just_pressed(KeyCode::Space) {
        ex.animate = !ex.animate;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        ex.space = (ex.space + 1) % 3;
    }
    let reset = keys.just_pressed(KeyCode::KeyR) || keys.just_pressed(KeyCode::KeyC);
    if ex.animate {
        ex.elapsed += time.delta_secs();
    }
    let attempt_fold = keys.just_pressed(KeyCode::KeyF) || (ex.attempt_fold && ex.frame == 10);
    let edit = ex.animate || reset || attempt_fold;
    if edit {
        let start = Instant::now();
        for (mut panel, background, border) in &mut panels {
            if reset {
                panel.base = points(panel.n, panel.kind, ex.space);
            }
            let mut candidate = panel.base.clone();
            for y in 1..panel.n - 1 {
                for x in 1..panel.n - 1 {
                    let p = &mut candidate[y * panel.n + x];
                    if ex.animate {
                        p.position[0] += 0.025 * ex.elapsed.sin();
                        p.position[1] += 0.025 * (ex.elapsed * 0.7).cos();
                    }
                }
            }
            if attempt_fold {
                candidate[panel.n + 1].position = [1.6, -0.5];
            }
            let accepted = match panel.surface.try_replace_points(candidate) {
                Ok(()) => {
                    ex.status = "Accepted atomic edit".into();
                    true
                }
                Err(error) => {
                    ex.status = format!("Rejected edit; last valid surface retained: {error:?}");
                    false
                }
            };
            if !accepted {
                continue;
            }
            let gradient = transport(&panel.surface, panel.subdivisions, ex.space);
            if let Some(mut g) = background {
                g.0 = if panel.kind == 3 {
                    vec![underlay(), gradient]
                } else {
                    vec![gradient]
                };
            } else if let Some(mut g) = border {
                g.0 = vec![gradient];
            }
        }
        ex.update_us.push(start.elapsed().as_secs_f64() * 1e6);
    }
    if let Ok(mut text) = hud.single_mut() {
        text.0=format!("SPACE animate ({})   C color ({:?})   F attempt fold   R reset   S screenshot\n{}\nPrototype transport is raw; the checked surface model owns validation. WebGL2 build passes; browser execution is unverified.",ex.animate,SPACES[ex.space],ex.status);
    }
    let screenshot =
        keys.just_pressed(KeyCode::KeyS) || (ex.capture.is_some() && ex.frame == ex.frames);
    if screenshot {
        let path = ex
            .capture
            .clone()
            .unwrap_or("/tmp/mesh-gradient-prototype.png".into());
        commands
            .spawn(Screenshot::primary_window())
            .observe(save_to_disk(path));
        let mut times = ex.update_us.clone();
        times.sort_by(f64::total_cmp);
        let mut frames = ex.frame_ms.clone();
        frames.sort_by(f64::total_cmp);
        println!("PROTOTYPE_STATS animate={} frames={} update_samples={} update_median_us={:.1} update_p95_us={:.1} frame_median_ms={:.2} frame_p95_ms={:.2}",ex.animate,ex.frame,times.len(),quantile(&times,0.5),quantile(&times,0.95),quantile(&frames,0.5),quantile(&frames,0.95));
    }
    if ex.capture.is_some() && ex.frame > ex.frames + 40 {
        exit.write(AppExit::Success);
    }
}
fn quantile(xs: &[f64], q: f64) -> f64 {
    if xs.is_empty() {
        0.
    } else {
        xs[((xs.len() - 1) as f64 * q) as usize]
    }
}
