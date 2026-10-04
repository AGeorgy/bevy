//! Stress test for animated UI gradients.
//!
//! Use `--mesh` to exercise checked 4x4 mesh gradients and `--benchmark` to
//! collect frame, update, and UI render-pass timings.

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
                        BackgroundGradient::from(create_mesh_gradient(i)),
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
    last_ui_cpu_measurement: Option<Instant>,
    last_ui_gpu_measurement: Option<Instant>,
}

fn animate_gradients(
    mut gradients: Query<(&mut BackgroundGradient, &GradientNode)>,
    args: Res<Args>,
    time: Res<Time>,
    benchmark: Option<ResMut<BenchmarkSamples>>,
) {
    if !args.animate {
        return;
    }

    let started = benchmark.as_ref().map(|_| Instant::now());
    let t = time.elapsed_secs();

    for (mut bg_gradient, node) in &mut gradients {
        match bg_gradient.0.first_mut() {
            Some(Gradient::Mesh(mesh)) => {
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
            }
            Some(Gradient::Linear(gradient)) => {
                let offset = node.index as f32 * 0.01;
                let hue_shift = sin(t + offset) * 0.5 + 0.5;
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
            _ => {}
        }
    }
    if let (Some(started), Some(mut benchmark)) = (started, benchmark) {
        benchmark.last_update_cpu_ms = started.elapsed().as_secs_f64() * 1_000.0;
    }
}

fn create_mesh_gradient(index: usize) -> MeshGradient {
    let phase = index as f32 * 0.13;
    let points = (0..16)
        .map(|point| {
            let column = point % 4;
            let row = point / 4;
            let x = column as f32 / 3.0;
            let y = row as f32 / 3.0;
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
    let samples = &mut *samples;
    collect_ui_sample(
        &diagnostics,
        &UI_CPU_TIME,
        &mut samples.last_ui_cpu_measurement,
        &mut samples.ui_cpu_ms,
    );
    collect_ui_sample(
        &diagnostics,
        &UI_GPU_TIME,
        &mut samples.last_ui_gpu_measurement,
        &mut samples.ui_gpu_ms,
    );
    if samples.total_ms.len() < BENCHMARK_SAMPLE_FRAMES {
        return;
    }

    let (total_median, total_p95) = compute_latency_percentiles(&mut samples.total_ms);
    let (update_median, update_p95) = compute_latency_percentiles(&mut samples.update_cpu_ms);
    let (ui_cpu_median, ui_cpu_p95) = compute_latency_percentiles(&mut samples.ui_cpu_ms);
    let gpu = if samples.ui_gpu_ms.is_empty() {
        "unavailable on this backend".to_string()
    } else {
        let (median, p95) = compute_latency_percentiles(&mut samples.ui_gpu_ms);
        format!("median={median:.3}ms p95={p95:.3}ms")
    };
    let window = windows.single().ok();
    println!(
        "gradient stress benchmark: mode={} gradients={} node={}x{} physical={}x{} scale={:.2} frames={} total median={total_median:.3}ms p95={total_p95:.3}ms update CPU median={update_median:.3}ms p95={update_p95:.3}ms UI pass CPU median={ui_cpu_median:.3}ms p95={ui_cpu_p95:.3}ms UI pass GPU {gpu}",
        if args.mesh { "mesh" } else { "linear" },
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

fn collect_ui_sample(
    diagnostics: &DiagnosticsStore,
    path: &DiagnosticPath,
    last_measurement: &mut Option<Instant>,
    samples: &mut Vec<f64>,
) {
    let Some(measurement) = diagnostics
        .get(path)
        .and_then(|diagnostic| diagnostic.measurement())
    else {
        return;
    };
    // GPU readback can leave the same render measurement visible for several frames.
    if last_measurement.is_some_and(|time| measurement.time <= time) {
        return;
    }
    *last_measurement = Some(measurement.time);
    samples.push(measurement.value);
}

fn compute_latency_percentiles(samples: &mut [f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    samples.sort_unstable_by(f64::total_cmp);
    (
        samples[samples.len() / 2],
        samples[samples.len() * 95 / 100],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::diagnostic::{Diagnostic, DiagnosticMeasurement};
    use core::time::Duration;

    fn add_measurement(app: &mut App, path: DiagnosticPath, time: Instant, value: f64) {
        let mut diagnostics = app.world_mut().resource_mut::<DiagnosticsStore>();
        if diagnostics.get(&path).is_none() {
            diagnostics.add(Diagnostic::new(path.clone()));
        }
        diagnostics
            .get_mut(&path)
            .unwrap()
            .add_measurement(DiagnosticMeasurement { time, value });
    }

    #[test]
    fn benchmark_records_render_measurements_once_without_skipping_frames() {
        let mut app = App::new();
        app.init_resource::<Time<Real>>()
            .init_resource::<DiagnosticsStore>()
            .insert_resource(BenchmarkSamples {
                frame: BENCHMARK_WARMUP_FRAMES - 1,
                last_update_cpu_ms: 2.0,
                ..default()
            })
            .insert_resource(Args {
                gradient_count: 1,
                animate: true,
                mesh: true,
                benchmark: true,
                srgb: false,
                hsl: false,
            })
            .add_message::<AppExit>()
            .add_systems(Update, collect_benchmark);
        let first_time = Instant::now();
        add_measurement(&mut app, UI_CPU_TIME, first_time, 1.0);

        app.update();
        let samples = app.world().resource::<BenchmarkSamples>();
        assert!(samples.total_ms.is_empty());
        assert!(samples.update_cpu_ms.is_empty());
        assert!(samples.ui_cpu_ms.is_empty());

        app.update();
        app.update();
        let samples = app.world().resource::<BenchmarkSamples>();
        assert_eq!(samples.total_ms.len(), 2);
        assert_eq!(samples.update_cpu_ms, [2.0, 2.0]);
        assert_eq!(samples.ui_cpu_ms, [1.0]);
        assert!(samples.ui_gpu_ms.is_empty());

        let second_time = first_time + Duration::from_millis(1);
        // An equal value with a new timestamp is still a new measurement.
        add_measurement(&mut app, UI_CPU_TIME, second_time, 1.0);
        add_measurement(&mut app, UI_GPU_TIME, second_time, 3.0);
        app.update();
        app.update();
        let samples = app.world().resource::<BenchmarkSamples>();
        assert_eq!(samples.total_ms.len(), 4);
        assert_eq!(samples.update_cpu_ms, [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(samples.ui_cpu_ms, [1.0, 1.0]);
        assert_eq!(samples.ui_gpu_ms, [3.0]);

        let third_time = second_time + Duration::from_millis(1);
        add_measurement(&mut app, UI_GPU_TIME, third_time, 4.0);
        app.update();
        let samples = app.world().resource::<BenchmarkSamples>();
        assert_eq!(samples.ui_cpu_ms, [1.0, 1.0]);
        assert_eq!(samples.ui_gpu_ms, [3.0, 4.0]);
    }
}
