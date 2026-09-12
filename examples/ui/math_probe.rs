#[path = "mesh_gradient_prototype_math.rs"]
mod math;
use math::*;
use std::{hint::black_box, time::Instant};
fn grid(w: usize, h: usize) -> Vec<Point> {
    (0..h)
        .flat_map(|y| {
            (0..w).map(move |x| Point {
                position: [x as f32 / (w - 1) as f32, y as f32 / (h - 1) as f32],
                color: [
                    x as f32 / (w - 1) as f32,
                    y as f32 / (h - 1) as f32,
                    0.5,
                    1.0,
                ],
            })
        })
        .collect()
}
fn main() {
    let mut p = grid(3, 3);
    let mut mesh = MeshSurface::try_new(3, 3, p.clone()).unwrap();
    assert!(mesh.certificate() > 0.99);
    assert!(mesh.try_tessellate(0).is_err());
    assert!(mesh.try_tessellate(1024).is_err());
    for (offset, scale) in [(0.0, f32::from_bits(1)), (16_777_216.0, 2.0)] {
        let tiny: Vec<_> = grid(2, 2)
            .into_iter()
            .map(|mut p| {
                p.position = p.position.map(|v| offset + scale * v);
                p
            })
            .collect();
        let tiny = MeshSurface::try_new(2, 2, tiny).unwrap();
        assert!(matches!(
            tiny.try_tessellate(16),
            Err(MeshError::UncertifiedTriangle { .. })
        ));
    }
    p[4].position = [0.58, 0.46];
    mesh.try_replace_points(p.clone()).unwrap();
    let accepted = mesh.points().to_vec();
    p[4].position = [1.2, 0.5];
    assert!(mesh.try_replace_points(p).is_err());
    assert_eq!(mesh.points(), accepted);
    let mut bad = grid(3, 3);
    bad[4].color[0] = f32::NAN;
    assert!(MeshSurface::try_new(3, 3, bad).is_err());
    let mut bad = grid(3, 3);
    bad[4].color[3] = 1.1;
    assert!(MeshSurface::try_new(3, 3, bad).is_err());
    assert!(MeshSurface::try_new(3, 3, grid(2, 2)).is_err());
    assert!(MeshSurface::try_new(1, 4, grid(2, 2)).is_err());
    let rotated: Vec<_> = grid(3, 3)
        .into_iter()
        .map(|mut p| {
            p.position = [1.0 - p.position[0], 1.0 - p.position[1]];
            p
        })
        .collect();
    assert!(MeshSurface::try_new(3, 3, rotated).is_err());
    // All straight control-grid cells have positive area, yet the automatically
    // inferred curve folds between columns 0 and 1: a node-only test misses it.
    let mut curved_fold = grid(3, 3);
    for p in &mut curved_fold {
        if p.position[0] == 0.5 {
            p.position[0] = 0.01;
        }
    }
    assert!(MeshSurface::try_new(3, 3, curved_fold).is_err());
    let mut hdr = grid(3, 3);
    for p in &mut hdr {
        p.color = [f32::MAX, -f32::MAX, 4.0, 0.5];
    }
    let hdr = MeshSurface::try_new(3, 3, hdr).unwrap();
    assert_eq!(hdr.dimensions(), (3, 3));
    let (vertices, _) = hdr.tessellate(8);
    assert!(vertices.iter().all(|v| v
        .position
        .iter()
        .chain(v.color.iter())
        .all(|x| x.is_finite())));
    // Shared patch edges must be position/color identical after f32 evaluation.
    for i in 0..=100 {
        assert_eq!(
            mesh.sample(0, 0, 1.0, i as f32 / 100.0),
            mesh.sample(1, 0, 0.0, i as f32 / 100.0)
        );
    }
    let inset: Vec<_> = grid(2, 2)
        .into_iter()
        .map(|mut p| {
            p.position = p.position.map(|v| 0.2 + 0.6 * v);
            p.color = [0.5, 0.0, 0.0, 1.0];
            p
        })
        .collect();
    let inset = MeshSurface::try_new(2, 2, inset).unwrap();
    assert_eq!(inset.sample(0, 0, 0.0, 0.0).0, [0.2, 0.2]);
    for n in [3, 5, 9] {
        let p = grid(n, n);
        let start = Instant::now();
        for _ in 0..200 {
            black_box(MeshSurface::try_new(n, n, p.clone()).unwrap());
        }
        let us = start.elapsed().as_secs_f64() * 1e6 / 200.0;
        let m = MeshSurface::try_new(n, n, p).unwrap();
        for sub in [8, 16, 32] {
            let start = Instant::now();
            for _ in 0..20 {
                black_box(m.tessellate(sub));
            }
            let t = start.elapsed().as_secs_f64() * 1e6 / 20.0;
            let (v, i) = m.tessellate(sub);
            println!("{n}x{n}: validation {us:.1} us; subdiv {sub}: tessellation {t:.1} us; {} vertices {} triangles",v.len(),i.len()/3);
        }
    }
    println!("PASS: regular/deformed/inset/HDR, atomic folded rejection, finite/alpha/dimension guards, shared seams, between-node curved fold; 180 degree rotation conservatively rejected; f32 subnormal/large-offset triangle degeneracy and tessellation budget rejected");
}
