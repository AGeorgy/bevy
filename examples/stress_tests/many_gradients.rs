//! Stress test demonstrating gradient performance improvements.
//!
//! This example creates many UI nodes with gradients to measure the performance
//! impact of pre-converting colors to the target color space on the CPU.

use argh::FromArgs;
use bevy::{
    color::palettes::css::*,
    diagnostic::{
        DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin,
    },
    math::ops::sin,
    platform::time::Instant,
    prelude::*,
    render::diagnostic::RenderDiagnosticsPlugin,
    ui::{
        BackgroundGradient, ColorStop, Display, Gradient, InterpolationColorSpace, LinearGradient,
        MeshGradient, MeshGradientPoint, RepeatedGridTrack,
    },
    window::{PresentMode, WindowResolution},
    winit::WinitSettings,
};

const COLS: usize = 30;
const MESH_COLS: usize = 10;
const MESH_NODE_SIZE: f32 = 256.0;
const BENCHMARK_WARMUP_FRAMES: usize = 300;
const BENCHMARK_SAMPLE_FRAMES: usize = 1_000;
const UI_CPU_TIME: DiagnosticPath = DiagnosticPath::const_new("render/ui/elapsed_cpu");
const UI_GPU_TIME: DiagnosticPath = DiagnosticPath::const_new("render/ui/elapsed_gpu");

#[derive(FromArgs, Resource, Debug)]
/// Gradient stress test
struct Args {
    /// how many gradients per group (default: 900)
    #[argh(option, default = "900")]
    gradient_count: usize,

    /// whether to animate gradients by changing colors
    #[argh(switch)]
    animate: bool,

    /// use animated 4x4 mesh gradients
    #[argh(switch)]
    mesh: bool,

    /// record release-mode frame and update percentiles, then exit
    #[argh(switch)]
    benchmark: bool,

    /// use sRGB interpolation
    #[argh(switch)]
    srgb: bool,

    /// use HSL interpolation
    #[argh(switch)]
    hsl: bool,
}

fn main() {
    // `from_env` panics on the web
    #[cfg(not(target_arch = "wasm32"))]
    let mut args: Args = argh::from_env();
    #[cfg(target_arch = "wasm32")]
    let mut args = Args::from_args(&[], &[]).unwrap();

    if args.benchmark {
        args.animate = true;
    }

    let total_gradients = args.gradient_count;

    println!("Gradient stress test with {total_gradients} gradients");
    println!(
        "Gradient mode: {}",
        if args.mesh {
            "4x4 mesh"
        } else if args.srgb {
            "sRGB"
        } else if args.hsl {
            "HSL"
        } else {
            "OkLab (default)"
        }
    );

    let resolution = if args.mesh {
        WindowResolution::new(
            (MESH_COLS as f32 * MESH_NODE_SIZE) as u32,
            (args.gradient_count.div_ceil(MESH_COLS) as f32 * MESH_NODE_SIZE) as u32,
        )
        .with_scale_factor_override(1.0)
    } else {
        WindowResolution::new(1920, 1080).with_scale_factor_override(1.0)
    };
    let benchmark = args.benchmark;
    let mut app = App::new();
    app.add_plugins((
        LogDiagnosticsPlugin::default(),
        FrameTimeDiagnosticsPlugin::default(),
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Gradient Stress Test".to_string(),
                resolution,
                present_mode: PresentMode::AutoNoVsync,
                ..default()
            }),
            ..default()
        }),
    ))
    .insert_resource(WinitSettings::continuous())
    .insert_resource(args)
    .add_systems(Startup, setup)
    .add_systems(Update, animate_gradients);
    if benchmark {
        app.add_plugins(RenderDiagnosticsPlugin)
            .init_resource::<BenchmarkSamples>()
            .add_systems(Last, collect_benchmark);
    }
    app.run();
}

fn setup(mut commands: Commands, args: Res<Args>) {
    warn!(include_str!("warning_string.txt"));

    commands.spawn(Camera2d);

    let columns = if args.mesh { MESH_COLS } else { COLS };
    let rows_to_spawn = args.gradient_count.div_ceil(columns);

    // Create a grid of gradients
    commands
        .spawn(Node {
            width: percent(100),
            height: percent(100),
            display: Display::Grid,
            grid_template_columns: if args.mesh {
                RepeatedGridTrack::px(columns as u16, MESH_NODE_SIZE)
            } else {
                RepeatedGridTrack::flex(columns as u16, 1.0)
            },
            grid_template_rows: if args.mesh {
                RepeatedGridTrack::px(rows_to_spawn as u16, MESH_NODE_SIZE)
            } else {
                RepeatedGridTrack::flex(rows_to_spawn as u16, 1.0)
            },
            ..default()
        })
        .with_children(|parent| {
            for i in 0..args.gradient_count {
                if args.mesh {
                    parent.spawn((
                        Node {
                            width: px(MESH_NODE_SIZE),
                            height: px(MESH_NODE_SIZE),
                            ..default()
                        },
                        BackgroundGradient::from(mesh_gradient(i)),
                        GradientNode { index: i },
                    ));
                    continue;
                }
                let angle = (i as f32 * 10.0) % 360.0;

                let mut gradient = LinearGradient::new(
                    angle,
                    vec![
                        ColorStop::new(RED, percent(0)),
                        ColorStop::new(BLUE, percent(100)),
                        ColorStop::new(GREEN, percent(20)),
                        ColorStop::new(YELLOW, percent(40)),
                        ColorStop::new(ORANGE, percent(60)),
                        ColorStop::new(LIME, percent(80)),
                        ColorStop::new(DARK_CYAN, percent(90)),
                    ],
                );

                gradient.color_space = if args.srgb {
                    InterpolationColorSpace::Srgba
                } else if args.hsl {
                    InterpolationColorSpace::Hsla
                } else {
                    InterpolationColorSpace::Oklaba
                };

                parent.spawn((
                    Node {
                        width: percent(100),
                        height: percent(100),
                        ..default()
                    },
                    BackgroundGradient(vec![Gradient::Linear(gradient)]),
                    GradientNode { index: i },
                ));
            }
        });
}

#[derive(Component)]
struct GradientNode {
    index: usize,
}

#[derive(Resource, Default)]
struct BenchmarkSamples {
    frame: usize,
    last_update_cpu_ms: f64,
    total_ms: Vec<f64>,
    update_cpu_ms: Vec<f64>,
    ui_cpu_ms: Vec<f64>,
    ui_gpu_ms: Vec<f64>,
}

fn animate_gradients(
    mut gradients: Query<(&mut BackgroundGradient, &GradientNode)>,
    args: Res<Args>,
    time: Res<Time>,
    mut benchmark: Option<ResMut<BenchmarkSamples>>,
) {
    if !args.animate {
        return;
    }

    let started = benchmark.as_ref().map(|_| Instant::now());
    let t = time.elapsed_secs();

    for (mut bg_gradient, node) in &mut gradients {
        let offset = node.index as f32 * 0.01;
        let hue_shift = sin(t + offset) * 0.5 + 0.5;

        if let Some(Gradient::Mesh(mesh)) = bg_gradient.0.get_mut(0) {
            let phase = t + node.index as f32 * 0.07;
            mesh.try_edit_points(|points| {
                for row in 1..3 {
                    for column in 1..3 {
                        let point = &mut points[row * 4 + column];
                        let offset = phase + row as f32 * 0.8 + column as f32 * 0.6;
                        point.position = Vec2::new(
                            column as f32 / 3.0 + sin(offset) * 0.022,
                            row as f32 / 3.0 + sin(offset * 0.83) * 0.018,
                        );
                    }
                }
                Ok(())
            })
            .expect("the bounded benchmark animation must remain valid");
        } else if let Some(Gradient::Linear(gradient)) = bg_gradient.0.get_mut(0) {
            let color1 = Color::hsl(hue_shift * 360.0, 1.0, 0.5);
            let color2 = Color::hsl((hue_shift + 0.3) * 360.0 % 360.0, 1.0, 0.5);

            gradient.stops = vec![
                ColorStop::new(color1, percent(0)),
                ColorStop::new(color2, percent(100)),
                ColorStop::new(
                    Color::hsl((hue_shift + 0.1) * 360.0 % 360.0, 1.0, 0.5),
                    percent(20),
                ),
                ColorStop::new(
                    Color::hsl((hue_shift + 0.15) * 360.0 % 360.0, 1.0, 0.5),
                    percent(40),
                ),
                ColorStop::new(
                    Color::hsl((hue_shift + 0.2) * 360.0 % 360.0, 1.0, 0.5),
                    percent(60),
                ),
                ColorStop::new(
                    Color::hsl((hue_shift + 0.25) * 360.0 % 360.0, 1.0, 0.5),
                    percent(80),
                ),
                ColorStop::new(
                    Color::hsl((hue_shift + 0.28) * 360.0 % 360.0, 1.0, 0.5),
                    percent(90),
                ),
            ];
        }
    }
    if let (Some(started), Some(benchmark)) = (started, benchmark.as_deref_mut()) {
        benchmark.last_update_cpu_ms = started.elapsed().as_secs_f64() * 1_000.0;
    }
}

fn mesh_gradient(index: usize) -> MeshGradient {
    let points = (0..16)
        .map(|point| {
            let column = point % 4;
            let row = point / 4;
            let x = column as f32 / 3.0;
            let y = row as f32 / 3.0;
            let phase = index as f32 * 0.13;
            MeshGradientPoint::new(
                Vec2::new(x, y),
                Color::oklaba(
                    0.62 + 0.18 * x - 0.06 * y,
                    0.14 * sin(phase + x * 2.1) - 0.08 * y,
                    0.14 * sin(phase * 0.7 + y * 2.3) - 0.08 * x,
                    0.72 + 0.28 * (1.0 - x * y),
                ),
            )
        })
        .collect();
    MeshGradient::new(4, 4, points).expect("the benchmark mesh must be valid")
}

fn collect_benchmark(
    time: Res<Time<Real>>,
    windows: Query<&Window>,
    diagnostics: Res<DiagnosticsStore>,
    args: Res<Args>,
    mut samples: ResMut<BenchmarkSamples>,
    mut exit: MessageWriter<AppExit>,
) {
    samples.frame += 1;
    if samples.frame <= BENCHMARK_WARMUP_FRAMES {
        return;
    }
    samples.total_ms.push(time.delta_secs_f64() * 1_000.0);
    let update_cpu_ms = samples.last_update_cpu_ms;
    samples.update_cpu_ms.push(update_cpu_ms);
    if let Some(value) = diagnostics
        .get(&UI_CPU_TIME)
        .and_then(|diagnostic| diagnostic.measurement())
        .map(|measurement| measurement.value)
    {
        samples.ui_cpu_ms.push(value);
    }
    if let Some(value) = diagnostics
        .get(&UI_GPU_TIME)
        .and_then(|diagnostic| diagnostic.measurement())
        .map(|measurement| measurement.value)
    {
        samples.ui_gpu_ms.push(value);
    }
    if samples.total_ms.len() < BENCHMARK_SAMPLE_FRAMES {
        return;
    }

    let (total_median, total_p95) = median_and_p95(&mut samples.total_ms);
    let (update_median, update_p95) = median_and_p95(&mut samples.update_cpu_ms);
    let (ui_cpu_median, ui_cpu_p95) = median_and_p95(&mut samples.ui_cpu_ms);
    let gpu = if samples.ui_gpu_ms.is_empty() {
        "unavailable on this backend".to_string()
    } else {
        let (median, p95) = median_and_p95(&mut samples.ui_gpu_ms);
        format!("median={median:.3}ms p95={p95:.3}ms")
    };
    let window = windows.single().ok();
    println!(
        "mesh-gradient stress benchmark: gradients={} node={}x{} physical={}x{} scale={:.2} frames={} total median={total_median:.3}ms p95={total_p95:.3}ms update CPU median={update_median:.3}ms p95={update_p95:.3}ms UI pass CPU median={ui_cpu_median:.3}ms p95={ui_cpu_p95:.3}ms UI pass GPU {gpu}",
        args.gradient_count,
        MESH_NODE_SIZE,
        MESH_NODE_SIZE,
        window.map_or(0, Window::physical_width),
        window.map_or(0, Window::physical_height),
        window.map_or(0.0, Window::scale_factor),
        samples.total_ms.len(),
    );
    exit.write(AppExit::Success);
}

fn median_and_p95(samples: &mut [f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    samples.sort_unstable_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100],
    )
}
