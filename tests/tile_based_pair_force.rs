// rq-78f54c5f
//
// Focused invariant tests for the tile-based pair-force architecture
// (`rqm/forces/tile-based-pair-force.md`). These tests verify the
// data-model invariants on a small system without depending on a
// particular pair-force functional form; the JIT-composed pair-force
// kernel is exercised end-to-end by the existing forces_framework /
// pipeline_reproducibility test suites.

use cudarc::driver::{CudaSlice, DeviceSlice};
use std::sync::Arc;

use heddle_md::forces::NeighborListState;
use heddle_md::gpu::{GpuContext, init_device};
use heddle_md::pbc::SimulationBox;
use heddle_md::precision::Real;
use heddle_md::state::ParticleState;
use heddle_md::timings::Timings;

fn small_box(gpu: &GpuContext, side: Real) -> SimulationBox {
    SimulationBox::new(&gpu.device, side, side, side, 0.0, 0.0, 0.0).unwrap()
}

fn lattice_state(n: usize, side: Real) -> ParticleState {
    let m = (n as Real).cbrt().ceil() as usize;
    let mut px: Vec<Real> = Vec::with_capacity(n);
    let mut py: Vec<Real> = Vec::with_capacity(n);
    let mut pz: Vec<Real> = Vec::with_capacity(n);
    'outer: for i in 0..m {
        for j in 0..m {
            for k in 0..m {
                if px.len() == n {
                    break 'outer;
                }
                px.push((i as Real + 0.5) * side / (m as Real));
                py.push((j as Real + 0.5) * side / (m as Real));
                pz.push((k as Real + 0.5) * side / (m as Real));
            }
        }
    }
    let vx = vec![0.0; n];
    let vy = vec![0.0; n];
    let vz = vec![0.0; n];
    let masses = vec![1.0 as Real; n];
    let charges = vec![0.0; n];
    let particle_ids = (0..n as u32).collect();
    ParticleState::new(
        px, py, pz, vx, vy, vz, masses, charges, particle_ids, None, None,
    )
    .unwrap()
}

fn rebuild_for_test(
    gpu: &GpuContext,
    n: usize,
    side: Real,
    r_cut: Real,
    r_skin: Real,
) -> (NeighborListState, SimulationBox, heddle_md::gpu::ParticleBuffers) {
    let state = lattice_state(n, side);
    let sim_box = small_box(gpu, side);
    let buffers = heddle_md::gpu::ParticleBuffers::new(gpu, &state).unwrap();
    let mut nl = NeighborListState::new_cell_list(gpu, &sim_box, n, r_cut, 256, r_skin).unwrap();
    let mut timings = Timings::new(gpu).unwrap();
    nl.rebuild(&sim_box, &buffers, &mut timings).unwrap();
    (nl, sim_box, buffers)
}

fn dtoh(device: &Arc<cudarc::driver::CudaDevice>, slice: &CudaSlice<u32>) -> Vec<u32> {
    let mut host = vec![0u32; slice.len()];
    device.dtoh_sync_copy_into(slice, &mut host).unwrap();
    host
}

fn dtoh_real(device: &Arc<cudarc::driver::CudaDevice>, slice: &CudaSlice<Real>) -> Vec<Real> {
    let mut host = vec![0.0 as Real; slice.len()];
    device.dtoh_sync_copy_into(slice, &mut host).unwrap();
    host
}

// rq-a1b28335 — Tile count derives from ceil(N / 32).
#[test]
fn n_tiles_equals_ceil_n_over_32() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    assert_eq!(nl.n_tiles, ((n as u32) + 31) / 32);
    assert_eq!(nl.n_tiles, 4);
}

// rq-be571c62 — `tile_atom_count` is 32 for every full tile and
// holds the remainder for the partial last tile.
#[test]
fn partial_last_tile_carries_remainder_count_and_truncated_lane_mask() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let counts = dtoh(&gpu.device, &nl.tile_atom_count);
    assert_eq!(counts, vec![32, 32, 32, 4]);
    let masks = dtoh(&gpu.device, &nl.tile_lane_mask);
    assert_eq!(masks, vec![0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000000F]);
}

// rq-69063b1d — `tile_pair_offsets` is monotone non-decreasing and
// the trailing sentinel equals the total pair count.
#[test]
fn tile_pair_offsets_are_monotone_and_terminate_at_tile_pair_count() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let offsets = dtoh(&gpu.device, &nl.tile_pair_offsets);
    assert_eq!(offsets.len(), nl.n_tiles as usize + 1);
    for w in offsets.windows(2) {
        assert!(w[1] >= w[0], "offsets not monotone: {:?}", w);
    }
    assert_eq!(*offsets.last().unwrap(), nl.tile_pair_count);
}

// rq-69063b1d — Each tile's neighbour-tile sublist is ascending in
// j_tile.
#[test]
fn sublists_are_ascending_in_j_tile() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let offsets = dtoh(&gpu.device, &nl.tile_pair_offsets);
    let pair_list = dtoh(&gpu.device, &nl.tile_pair_list);
    for t in 0..nl.n_tiles as usize {
        let start = offsets[t] as usize;
        let end = offsets[t + 1] as usize;
        let sublist = &pair_list[start..end];
        for w in sublist.windows(2) {
            assert!(
                w[1] > w[0],
                "tile {} sublist not strictly ascending: {:?}",
                t,
                sublist
            );
        }
    }
}

// rq-69063b1d — Every tile's sublist includes the self-pair (t, t).
#[test]
fn self_pair_is_always_present() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let offsets = dtoh(&gpu.device, &nl.tile_pair_offsets);
    let pair_list = dtoh(&gpu.device, &nl.tile_pair_list);
    for t in 0..nl.n_tiles as usize {
        let start = offsets[t] as usize;
        let end = offsets[t + 1] as usize;
        let sublist = &pair_list[start..end];
        assert!(
            sublist.contains(&(t as u32)),
            "tile {}'s sublist {:?} missing self-pair",
            t,
            sublist
        );
    }
}

// rq-480ea47d — Two independent rebuilds from byte-identical state
// produce byte-identical tile-pair lists.
#[test]
fn two_rebuilds_produce_byte_identical_tile_pair_data() {
    let gpu = init_device().unwrap();
    let n = 128;
    let side = 2.0e-9 as Real;
    let r_cut = 5.0e-10 as Real;
    let r_skin = 1.0e-10 as Real;

    let (nl_a, _, _) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);
    let (nl_b, _, _) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);

    assert_eq!(nl_a.n_tiles, nl_b.n_tiles);
    assert_eq!(nl_a.tile_pair_count, nl_b.tile_pair_count);
    assert_eq!(
        dtoh(&gpu.device, &nl_a.tile_atom_count),
        dtoh(&gpu.device, &nl_b.tile_atom_count),
    );
    assert_eq!(
        dtoh(&gpu.device, &nl_a.tile_lane_mask),
        dtoh(&gpu.device, &nl_b.tile_lane_mask),
    );
    assert_eq!(
        dtoh(&gpu.device, &nl_a.tile_pair_offsets),
        dtoh(&gpu.device, &nl_b.tile_pair_offsets),
    );
    assert_eq!(
        dtoh(&gpu.device, &nl_a.tile_pair_list),
        dtoh(&gpu.device, &nl_b.tile_pair_list),
    );
}

// rq-d8de5e38 — `compute_tile_bounding_boxes` writes a valid AABB
// for every tile: min ≤ max along each axis.
#[test]
fn tile_bboxes_satisfy_min_leq_max() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let bboxes = dtoh_real(&gpu.device, &nl.tile_bboxes);
    assert_eq!(bboxes.len(), 6 * nl.n_tiles as usize);
    for t in 0..nl.n_tiles as usize {
        let base = t * 6;
        assert!(
            bboxes[base] <= bboxes[base + 3],
            "tile {t} min_x > max_x"
        );
        assert!(
            bboxes[base + 1] <= bboxes[base + 4],
            "tile {t} min_y > max_y"
        );
        assert!(
            bboxes[base + 2] <= bboxes[base + 5],
            "tile {t} min_z > max_z"
        );
    }
}

// rq-d8de5e38 — Self-pair mask equals the i_tile's lane mask
// (every real lane is set; inactive lanes cleared).
#[test]
fn self_pair_mask_equals_tile_lane_mask() {
    let gpu = init_device().unwrap();
    let n = 100;
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let offsets = dtoh(&gpu.device, &nl.tile_pair_offsets);
    let pair_list = dtoh(&gpu.device, &nl.tile_pair_list);
    let pair_masks = dtoh(&gpu.device, &nl.tile_pair_masks);
    let lane_mask = dtoh(&gpu.device, &nl.tile_lane_mask);
    for t in 0..nl.n_tiles as usize {
        let start = offsets[t] as usize;
        let end = offsets[t + 1] as usize;
        let sublist = &pair_list[start..end];
        let masks_for_t = &pair_masks[start..end];
        let self_pos = sublist
            .iter()
            .position(|&j| j as usize == t)
            .expect("self-pair must appear");
        assert_eq!(
            masks_for_t[self_pos], lane_mask[t],
            "tile {t}'s self-pair mask should equal its tile_lane_mask"
        );
    }
}

// rq-d8de5e38 — Inactive j-lanes have their mask bit cleared (for
// a system whose last tile is partial).
#[test]
fn partial_last_tile_inactive_lanes_have_cleared_mask_bits() {
    let gpu = init_device().unwrap();
    let n = 100; // last tile has 4 real atoms; lanes 4..32 inactive
    let (nl, _, _) = rebuild_for_test(&gpu, n, 2.0e-9, 5.0e-10, 1.0e-10);
    let offsets = dtoh(&gpu.device, &nl.tile_pair_offsets);
    let pair_list = dtoh(&gpu.device, &nl.tile_pair_list);
    let pair_masks = dtoh(&gpu.device, &nl.tile_pair_masks);
    let lane_mask = dtoh(&gpu.device, &nl.tile_lane_mask);
    let last = nl.n_tiles as usize - 1;
    // Find every tile-pair entry whose j_tile is the partial last
    // tile and verify high bits are cleared.
    for t in 0..nl.n_tiles as usize {
        let start = offsets[t] as usize;
        let end = offsets[t + 1] as usize;
        for k in start..end {
            let j_tile = pair_list[k] as usize;
            if j_tile == last {
                let mask = pair_masks[k];
                assert_eq!(
                    mask & !lane_mask[last],
                    0,
                    "tile-pair entry {k} (i={t} j={j_tile}) has set bits in inactive lanes"
                );
            }
        }
    }
}

// rq-480ea47d — Two independent rebuilds produce byte-identical
// tile_pair_masks (determinism of the __ballot_sync reduction).
#[test]
fn two_rebuilds_produce_byte_identical_masks() {
    let gpu = init_device().unwrap();
    let n = 128;
    let side = 2.0e-9 as Real;
    let r_cut = 5.0e-10 as Real;
    let r_skin = 1.0e-10 as Real;
    let (nl_a, _, _) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);
    let (nl_b, _, _) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);
    assert_eq!(
        dtoh(&gpu.device, &nl_a.tile_pair_masks),
        dtoh(&gpu.device, &nl_b.tile_pair_masks),
    );
}

// rq-913c85d3 — `tile_sorted_positions[k]` equals
// `positions[sorted_particle_ids[k]]` after `scatter_positions_to_tile_order`
// runs. The scatter is the semantic identity that the JIT pair-force
// kernel relies on; this test pins the relationship without going
// through the full force pipeline.
#[test]
fn scatter_produces_sorted_view_of_positions() {
    let gpu = init_device().unwrap();
    let n = 100;
    let side = 2.0e-9 as Real;
    let r_cut = 5.0e-10 as Real;
    let r_skin = 1.0e-10 as Real;
    let (mut nl, _, buffers) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);

    heddle_md::gpu::scatter_positions_to_tile_order(
        &gpu.kernels,
        &buffers,
        &nl.tile_sorted_particle_ids,
        &mut nl.tile_sorted_positions_x,
        &mut nl.tile_sorted_positions_y,
        &mut nl.tile_sorted_positions_z,
    )
    .unwrap();

    let pos_x = dtoh_real(&gpu.device, &buffers.positions_x);
    let pos_y = dtoh_real(&gpu.device, &buffers.positions_y);
    let pos_z = dtoh_real(&gpu.device, &buffers.positions_z);
    let sorted = dtoh(&gpu.device, &nl.tile_sorted_particle_ids);
    let tsx = dtoh_real(&gpu.device, &nl.tile_sorted_positions_x);
    let tsy = dtoh_real(&gpu.device, &nl.tile_sorted_positions_y);
    let tsz = dtoh_real(&gpu.device, &nl.tile_sorted_positions_z);

    for k in 0..n {
        let pid = sorted[k] as usize;
        assert_eq!(tsx[k].to_bits(), pos_x[pid].to_bits(),
            "x mismatch at k={k}");
        assert_eq!(tsy[k].to_bits(), pos_y[pid].to_bits(),
            "y mismatch at k={k}");
        assert_eq!(tsz[k].to_bits(), pos_z[pid].to_bits(),
            "z mismatch at k={k}");
    }
}

// rq-a46cbf6b — Two independent rebuild+scatter sequences from
// identical initial state produce byte-identical
// `tile_sorted_positions_*` buffers. The scatter is a pure
// permutation, but recording this invariant guards against any
// future race or non-deterministic ordering creeping in.
#[test]
fn two_scatters_produce_byte_identical_tile_sorted_positions() {
    let gpu = init_device().unwrap();
    let n = 128;
    let side = 2.0e-9 as Real;
    let r_cut = 5.0e-10 as Real;
    let r_skin = 1.0e-10 as Real;
    let (mut nl_a, _, buf_a) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);
    let (mut nl_b, _, buf_b) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);

    heddle_md::gpu::scatter_positions_to_tile_order(
        &gpu.kernels,
        &buf_a,
        &nl_a.tile_sorted_particle_ids,
        &mut nl_a.tile_sorted_positions_x,
        &mut nl_a.tile_sorted_positions_y,
        &mut nl_a.tile_sorted_positions_z,
    )
    .unwrap();
    heddle_md::gpu::scatter_positions_to_tile_order(
        &gpu.kernels,
        &buf_b,
        &nl_b.tile_sorted_particle_ids,
        &mut nl_b.tile_sorted_positions_x,
        &mut nl_b.tile_sorted_positions_y,
        &mut nl_b.tile_sorted_positions_z,
    )
    .unwrap();

    assert_eq!(
        dtoh_real(&gpu.device, &nl_a.tile_sorted_positions_x),
        dtoh_real(&gpu.device, &nl_b.tile_sorted_positions_x),
    );
    assert_eq!(
        dtoh_real(&gpu.device, &nl_a.tile_sorted_positions_y),
        dtoh_real(&gpu.device, &nl_b.tile_sorted_positions_y),
    );
    assert_eq!(
        dtoh_real(&gpu.device, &nl_a.tile_sorted_positions_z),
        dtoh_real(&gpu.device, &nl_b.tile_sorted_positions_z),
    );
}

// rq-480ea47d — Bounding-box pruning reduces the candidate count
// when the search radius is shorter than the box diagonal.
//
// A dense cubic lattice with small r_cut should produce a tile-pair
// list shorter than the all-pairs worst case (N_tiles²). This
// verifies that the pruning is actually doing something rather than
// silently admitting every pair.
#[test]
fn bbox_pruning_produces_fewer_pairs_than_all_pairs() {
    let gpu = init_device().unwrap();
    let n = 256;
    let side = 4.0e-9 as Real;
    let r_cut = 5.0e-10 as Real;
    let r_skin = 1.0e-10 as Real;
    let (nl, _, _) = rebuild_for_test(&gpu, n, side, r_cut, r_skin);
    let n_tiles = nl.n_tiles as u64;
    let all_pairs = n_tiles * n_tiles;
    assert!(
        (nl.tile_pair_count as u64) < all_pairs,
        "pruning ineffective: tile_pair_count = {}, all-pairs = {}",
        nl.tile_pair_count,
        all_pairs
    );
}
