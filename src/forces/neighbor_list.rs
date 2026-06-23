// rq-693ce6fa rq-b2d68288
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice};
use cudarc::nvrtc::Ptx;

use crate::gpu::device::get_func;
use crate::gpu::{
    GpuContext, GpuError, Kernels, ParticleBuffers, SPATIAL_HASH_SCAN_BLOCK_SIZE,
    compute_cell_indices_and_histogram, copy_positions_into_reference,
    neighbor_displacement_squared, neighbor_list_build, prefix_scan_cell_counts,
    scatter_atoms_into_cells, sort_cells_by_particle_id,
};
use crate::kernels;
use crate::pbc::SimulationBox;
use crate::timings::{HostStage, KernelStage, Timings};
use crate::precision::Real;

// rq-d8e4407a rq-e1ceb5c0 rq-6cf916af rq-1bbcf3b7
#[derive(Debug, thiserror::Error)]
pub enum NeighborListError {
    #[error("{0}")]
    Gpu(#[from] GpuError),
    #[error("an atom has more than {max} neighbors")]
    NeighborListOverflow { max: u32 },
    #[error("simulation box perpendicular width along lattice direction `{direction}` is {width}, below the required {required}")]
    BoxTooSmallForCells {
        direction: &'static str,
        width: Real,
        required: Real,
    },
    #[error("cell grid has {n_cells_total} cells, exceeding the device limit of {max_supported}")]
    TooManyCells {
        n_cells_total: usize,
        max_supported: usize,
    },
}

// rq-ff424773
#[derive(Debug)]
pub enum NeighborListMode {
    Trivial,
    CellList(CellListData),
    // CellListOnly produces the cell-list output (sorted particle IDs +
    // per-cell offsets) without building a neighbor list. Used by the
    // SPME reciprocal-space slot; see `rqm/forces/spme.md`.
    CellListOnly(CellListData),
}

// rq-ff424773
#[derive(Debug)]
pub struct CellListData {
    pub n_cells: [u32; 3],
    pub n_cells_total: usize,
    pub r_cut: Real,
    pub r_skin: Real,
    pub r_search_sq: Real,
    pub cached_generation: u64,
    pub cell_indices: CudaSlice<u32>,
    pub cell_counts: CudaSlice<u32>,
    pub write_cursors: CudaSlice<u32>,
    pub scan_block_totals: Vec<CudaSlice<u32>>,
    pub sorted_particle_ids: CudaSlice<u32>,
    pub cell_offsets: CudaSlice<u32>,
    pub reference_positions_x: CudaSlice<Real>,
    pub reference_positions_y: CudaSlice<Real>,
    pub reference_positions_z: CudaSlice<Real>,
    pub disp_sq: CudaSlice<Real>,
    pub overflow_flag: CudaSlice<u32>,
    pub needs_rebuild: bool,
}

// rq-b2d68288
#[derive(Debug)]
pub struct NeighborListState {
    pub device: Arc<CudaDevice>,
    pub kernels: Arc<Kernels>,
    pub particle_count: usize,
    pub max_neighbors: u32,
    pub neighbor_list: CudaSlice<u32>,
    pub neighbor_counts: CudaSlice<u32>,
    pub mode: NeighborListMode,
    // Monotonically-increasing counter incremented at the end of every
    // successful `rebuild_impl`. Downstream consumers (e.g. the SPME
    // reciprocal-space slot's atom spatial pre-sort) cache the last
    // observed value and re-run their per-rebuild work when the
    // generation advances.
    rebuild_generation: u64,

    // rq-63755598 — Tile-based pair-force architecture buffers. The
    // active per-step pair-force path still uses
    // `neighbor_list` / `neighbor_counts` / `max_neighbors`; tile
    // buffers are populated alongside on every rebuild so the
    // tile-based composer can be wired through in a follow-up step.
    // See `rqm/forces/tile-based-pair-force.md`.
    pub n_tiles: u32,
    pub tile_atom_count: CudaSlice<u32>,
    pub tile_lane_mask: CudaSlice<u32>,
    /// Per-tile axis-aligned bounding box: 6 × Real per tile in
    /// row-major order `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    /// Populated by `compute_tile_bounding_boxes` during rebuild;
    /// read by `tile_pair_candidate_count` / `tile_pair_emit` for
    /// bounding-box pruning.
    pub tile_bboxes: CudaSlice<Real>,
    pub tile_candidate_counts: CudaSlice<u32>,
    pub tile_pair_offsets: CudaSlice<u32>,
    pub tile_pair_list: CudaSlice<u32>,
    /// Per-tile-pair j-atom interaction mask. Indexed parallel to
    /// `tile_pair_list`: bit `k` of
    /// `tile_pair_masks[tile_pair_offsets[t] + i]` is set iff the
    /// j_atom at tile-lane `k` of `tile_pair_list[...]` is within
    /// `r_cut + r_skin` of at least one atom in i_tile `t`. The
    /// JIT-composed pair-force kernel uses this to short-circuit
    /// its per-pair inner loop over j_atoms; see
    /// `rqm/forces/tile-based-pair-force.md`.
    pub tile_pair_masks: CudaSlice<u32>,
    pub tile_pair_capacity: u32,
    pub tile_pair_count: u32,
    /// Tile-sorted view of `positions_x`, refreshed at the start of
    /// every `ForceField::step()` by `scatter_positions_to_tile_order`.
    /// Semantics:
    /// `tile_sorted_positions_x[k] = positions_x[sorted_particle_ids[k]]`.
    /// The JIT-composed pair-force kernel reads positions exclusively
    /// from these three buffers (coalesced loads). See
    /// `rqm/forces/tile-based-pair-force.md`.
    pub tile_sorted_positions_x: CudaSlice<Real>,
    pub tile_sorted_positions_y: CudaSlice<Real>,
    pub tile_sorted_positions_z: CudaSlice<Real>,
    /// Sorted-particle-id view used by the tile-based pair-force
    /// kernel. In `CellList` mode this is populated by
    /// `rebuild_tile_data` (copy of the cell-list's sort). In
    /// `Trivial` mode it is the identity permutation, allocated once
    /// at construction.
    pub tile_sorted_particle_ids: CudaSlice<u32>,
}

impl NeighborListState {
    /// Borrow the cell-list-specific data; returns `None` when the state is
    /// in `Trivial` mode.
    pub fn cell_list_data(&self) -> Option<&CellListData> {
        match &self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => Some(cl),
        }
    }

    /// Mutable borrow of the cell-list-specific data; returns `None` when
    /// the state is in `Trivial` mode.
    pub fn cell_list_data_mut(&mut self) -> Option<&mut CellListData> {
        match &mut self.mode {
            NeighborListMode::Trivial => None,
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => Some(cl),
        }
    }

    /// `true` when the state is in CellListOnly mode (bin structure only,
    /// no neighbor list).
    pub fn is_bin_only(&self) -> bool {
        matches!(self.mode, NeighborListMode::CellListOnly(_))
    }

    /// Monotonically-increasing rebuild counter. Bumped once every time
    /// `rebuild` completes successfully. Downstream consumers cache the
    /// observed value to detect when a rebuild has occurred.
    pub fn rebuild_generation(&self) -> u64 {
        self.rebuild_generation
    }

    // rq-14033af1
    pub fn new_cell_list(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        r_cut: Real,
        max_neighbors: u32,
        r_skin: Real,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        debug_assert!(r_cut > 0.0);
        // r_skin == 0 is permitted (degenerate but valid: every
        // step's displacement check triggers a rebuild because
        // max_disp > r_skin * 0.5 = 0 unless the system is fully
        // stationary). Negative is malformed.
        debug_assert!(r_skin >= 0.0);
        debug_assert!(max_neighbors > 0);
        let r_search = r_cut + r_skin;
        let r_search_sq = r_search * r_search;
        let n_cells = compute_cell_layout(sim_box, r_search)?;
        let n_cells_total =
            n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;
        check_n_cells_total(n_cells_total)?;

        let cell_indices = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_counts = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let write_cursors = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let scan_block_totals = alloc_scan_block_totals(&device, n_cells_total)?;
        let sorted_particle_ids = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_offsets = device
            .alloc_zeros::<u32>(n_cells_total + 1)
            .map_err(GpuError::from)?;
        let nl_len = particle_count * max_neighbors as usize;
        let neighbor_list = device
            .alloc_zeros::<u32>(nl_len.max(1))
            .map_err(GpuError::from)?;
        let neighbor_counts = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_x = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_y = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let reference_positions_z = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let disp_sq = device
            .alloc_zeros::<Real>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(1).map_err(GpuError::from)?;

        let tb = alloc_tile_buffers(&device, particle_count)?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::CellList(CellListData {
                n_cells,
                n_cells_total,
                r_cut,
                r_skin,
                r_search_sq,
                cached_generation: sim_box.generation(),
                cell_indices,
                cell_counts,
                write_cursors,
                scan_block_totals,
                sorted_particle_ids,
                cell_offsets,
                reference_positions_x,
                reference_positions_y,
                reference_positions_z,
                disp_sq,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
            n_tiles: tb.n_tiles,
            tile_atom_count: tb.tile_atom_count,
            tile_lane_mask: tb.tile_lane_mask,
            tile_bboxes: tb.tile_bboxes,
            tile_candidate_counts: tb.tile_candidate_counts,
            tile_pair_offsets: tb.tile_pair_offsets,
            tile_pair_list: tb.tile_pair_list,
            tile_pair_masks: tb.tile_pair_masks,
            tile_pair_capacity: tb.tile_pair_capacity,
            tile_pair_count: 0,
            tile_sorted_particle_ids: tb.tile_sorted_particle_ids,
            tile_sorted_positions_x: tb.tile_sorted_positions_x,
            tile_sorted_positions_y: tb.tile_sorted_positions_y,
            tile_sorted_positions_z: tb.tile_sorted_positions_z,
        })
    }

    // rq-9ca00d25 rq-202493a5
    //
    // Build a bin-only cell-list state with explicit grid dimensions.
    // Used by the SPME reciprocal-space slot (see rqm/forces/spme.md).
    // The state produces sorted particle IDs and per-cell offsets but
    // no neighbor list; the neighbor-list-build and displacement-check
    // kernels are never launched.
    // rq-d47caa3d
    pub fn new_cell_list_only(
        gpu: &GpuContext,
        sim_box: &SimulationBox,
        particle_count: usize,
        n_cells_per_direction: [u32; 3],
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();

        let direction_names: [&'static str; 3] = ["a", "b", "c"];
        for d in 0..3 {
            if n_cells_per_direction[d] < 3 {
                return Err(NeighborListError::BoxTooSmallForCells {
                    direction: direction_names[d],
                    width: 0.0,
                    required: 3.0,
                });
            }
        }
        let n_cells = n_cells_per_direction;
        let n_cells_total =
            n_cells[0] as usize * n_cells[1] as usize * n_cells[2] as usize;
        check_n_cells_total(n_cells_total)?;

        // Cell-list scratch + outputs.
        let cell_indices = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_counts = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let write_cursors = device
            .alloc_zeros::<u32>(n_cells_total)
            .map_err(GpuError::from)?;
        let scan_block_totals = alloc_scan_block_totals(&device, n_cells_total)?;
        let sorted_particle_ids = device
            .alloc_zeros::<u32>(particle_count.max(1))
            .map_err(GpuError::from)?;
        let cell_offsets = device
            .alloc_zeros::<u32>(n_cells_total + 1)
            .map_err(GpuError::from)?;

        // Neighbor-list-only buffers are zero-length in bin-only mode.
        let neighbor_list = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;
        let neighbor_counts = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;
        let reference_positions_x = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_y = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let reference_positions_z = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let disp_sq = device.alloc_zeros::<Real>(0).map_err(GpuError::from)?;
        let overflow_flag = device.alloc_zeros::<u32>(0).map_err(GpuError::from)?;

        let tb = alloc_tile_buffers(&device, particle_count)?;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors: 0,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::CellListOnly(CellListData {
                n_cells,
                n_cells_total,
                r_cut: 0.0,
                r_skin: 0.0,
                r_search_sq: 0.0,
                cached_generation: sim_box.generation(),
                cell_indices,
                cell_counts,
                write_cursors,
                scan_block_totals,
                sorted_particle_ids,
                cell_offsets,
                reference_positions_x,
                reference_positions_y,
                reference_positions_z,
                disp_sq,
                overflow_flag,
                needs_rebuild: true,
            }),
            rebuild_generation: 0,
            n_tiles: tb.n_tiles,
            tile_atom_count: tb.tile_atom_count,
            tile_lane_mask: tb.tile_lane_mask,
            tile_bboxes: tb.tile_bboxes,
            tile_candidate_counts: tb.tile_candidate_counts,
            tile_pair_offsets: tb.tile_pair_offsets,
            tile_pair_list: tb.tile_pair_list,
            tile_pair_masks: tb.tile_pair_masks,
            tile_pair_capacity: tb.tile_pair_capacity,
            tile_pair_count: 0,
            tile_sorted_particle_ids: tb.tile_sorted_particle_ids,
            tile_sorted_positions_x: tb.tile_sorted_positions_x,
            tile_sorted_positions_y: tb.tile_sorted_positions_y,
            tile_sorted_positions_z: tb.tile_sorted_positions_z,
        })
    }

    // rq-c96fd9d2
    pub fn new_trivial(
        gpu: &GpuContext,
        _sim_box: &SimulationBox,
        particle_count: usize,
    ) -> Result<Self, NeighborListError> {
        let device = gpu.device.clone();
        let kernels = gpu.kernels.clone();
        let max_neighbors = particle_count as u32;
        let nl_len = particle_count * particle_count;
        let nl_host: Vec<u32> = if nl_len == 0 {
            Vec::new()
        } else {
            let mut v = Vec::with_capacity(nl_len);
            for _ in 0..particle_count {
                for k in 0..particle_count {
                    v.push(k as u32);
                }
            }
            v
        };
        let counts_host: Vec<u32> = vec![particle_count as u32; particle_count];

        let neighbor_list = if nl_host.is_empty() {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&nl_host).map_err(GpuError::from)?
        };
        let neighbor_counts = if counts_host.is_empty() {
            device.alloc_zeros::<u32>(0).map_err(GpuError::from)?
        } else {
            device.htod_sync_copy(&counts_host).map_err(GpuError::from)?
        };

        let mut tb = alloc_tile_buffers(&device, particle_count)?;

        // Trivial mode: populate the tile data as all-pairs. Each
        // tile's neighbour sublist contains every other tile (plus
        // the self-pair). The sorted-id buffer is the identity
        // permutation. Built host-side and uploaded; Trivial mode is
        // for tiny test systems so the upload cost is negligible.
        let n_tiles = tb.n_tiles as usize;
        if particle_count > 0 {
            let identity: Vec<u32> = (0..particle_count as u32).collect();
            device
                .htod_sync_copy_into(
                    &identity,
                    &mut tb.tile_sorted_particle_ids.slice_mut(0..particle_count),
                )
                .map_err(GpuError::from)?;

            // tile_atom_count + lane_mask host-side.
            let tile_counts: Vec<u32> = (0..n_tiles)
                .map(|t| {
                    let start = t * 32;
                    let end = ((t + 1) * 32).min(particle_count);
                    (end - start) as u32
                })
                .collect();
            let tile_masks: Vec<u32> = tile_counts
                .iter()
                .map(|&c| if c == 32 { 0xFFFFFFFFu32 } else { (1u32 << c) - 1 })
                .collect();
            device
                .htod_sync_copy_into(&tile_counts, &mut tb.tile_atom_count)
                .map_err(GpuError::from)?;
            device
                .htod_sync_copy_into(&tile_masks, &mut tb.tile_lane_mask)
                .map_err(GpuError::from)?;

            // All-pairs tile-pair list: tile t's sublist is [0, 1,
            // ..., n_tiles - 1] in ascending order (self-pair appears
            // at position t).
            let total_pairs = n_tiles * n_tiles;
            let pair_offsets: Vec<u32> =
                (0..=n_tiles).map(|t| (t * n_tiles) as u32).collect();
            let pair_list: Vec<u32> = (0..total_pairs)
                .map(|idx| (idx % n_tiles) as u32)
                .collect();
            device
                .htod_sync_copy_into(&pair_offsets, &mut tb.tile_pair_offsets)
                .map_err(GpuError::from)?;
            // Tile-pair list capacity must fit total_pairs; for tiny
            // Trivial-mode workloads the default initial capacity
            // (N_tiles * 256) is plenty for any N_tiles ≤ 256.
            assert!(
                total_pairs <= tb.tile_pair_capacity as usize,
                "Trivial-mode tile-pair list ({total_pairs}) exceeds capacity ({})",
                tb.tile_pair_capacity
            );
            device
                .htod_sync_copy_into(
                    &pair_list,
                    &mut tb.tile_pair_list.slice_mut(0..total_pairs),
                )
                .map_err(GpuError::from)?;
            // Mask for each entry equals the j_tile's lane mask
            // (all real lanes are candidates in Trivial mode).
            let pair_masks: Vec<u32> = pair_list
                .iter()
                .map(|&j_tile| tile_masks[j_tile as usize])
                .collect();
            device
                .htod_sync_copy_into(
                    &pair_masks,
                    &mut tb.tile_pair_masks.slice_mut(0..total_pairs),
                )
                .map_err(GpuError::from)?;
        }
        let tile_pair_count = (n_tiles * n_tiles) as u32;

        Ok(NeighborListState {
            device,
            kernels,
            particle_count,
            max_neighbors,
            neighbor_list,
            neighbor_counts,
            mode: NeighborListMode::Trivial,
            rebuild_generation: 0,
            n_tiles: tb.n_tiles,
            tile_atom_count: tb.tile_atom_count,
            tile_lane_mask: tb.tile_lane_mask,
            tile_bboxes: tb.tile_bboxes,
            tile_candidate_counts: tb.tile_candidate_counts,
            tile_pair_offsets: tb.tile_pair_offsets,
            tile_pair_list: tb.tile_pair_list,
            tile_pair_masks: tb.tile_pair_masks,
            tile_pair_capacity: tb.tile_pair_capacity,
            tile_pair_count,
            tile_sorted_particle_ids: tb.tile_sorted_particle_ids,
            tile_sorted_positions_x: tb.tile_sorted_positions_x,
            tile_sorted_positions_y: tb.tile_sorted_positions_y,
            tile_sorted_positions_z: tb.tile_sorted_positions_z,
        })
    }

    // rq-282af621
    fn refresh_cell_layout_if_box_changed(
        &mut self,
        sim_box: &SimulationBox,
    ) -> Result<bool, NeighborListError> {
        let device = self.device.clone();
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(false),
            // Bin-only mode has a fixed n_cells_per_direction (the FFT
            // grid resolution); the box-generation refresh re-records
            // the generation but does not re-derive n_cells from
            // r_cut/r_skin.
            NeighborListMode::CellListOnly(cl) => {
                if sim_box.generation() != cl.cached_generation {
                    cl.cached_generation = sim_box.generation();
                    cl.needs_rebuild = true;
                    return Ok(true);
                }
                return Ok(false);
            }
            NeighborListMode::CellList(cl) => cl,
        };
        if sim_box.generation() == cl.cached_generation {
            return Ok(false);
        }
        let r_search = cl.r_cut + cl.r_skin;
        let new_n_cells = compute_cell_layout(sim_box, r_search)?;
        let new_n_cells_total =
            new_n_cells[0] as usize * new_n_cells[1] as usize * new_n_cells[2] as usize;
        check_n_cells_total(new_n_cells_total)?;
        let n_cells_total_changed = new_n_cells_total != cl.n_cells_total;
        if n_cells_total_changed {
            cl.cell_offsets = device
                .alloc_zeros::<u32>(new_n_cells_total + 1)
                .map_err(GpuError::from)?;
            cl.cell_counts = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.write_cursors = device
                .alloc_zeros::<u32>(new_n_cells_total)
                .map_err(GpuError::from)?;
            cl.scan_block_totals =
                alloc_scan_block_totals(&device, new_n_cells_total)?;
        }
        cl.n_cells = new_n_cells;
        cl.n_cells_total = new_n_cells_total;
        cl.cached_generation = sim_box.generation();
        if n_cells_total_changed {
            cl.needs_rebuild = true;
        }
        Ok(n_cells_total_changed)
    }

    // rq-c49b2fe6
    pub fn displacement_check(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<Real, NeighborListError> {
        if self.particle_count == 0 {
            return Ok(0.0);
        }
        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(0.0),
            // Bin-only mode has no displacement check; rebuild every step.
            NeighborListMode::CellListOnly(_) => return Ok(0.0),
            NeighborListMode::CellList(cl) => cl,
        };
        timings
            .kernel_start(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;
        neighbor_displacement_squared(
            buffers,
            &cl.reference_positions_x,
            &cl.reference_positions_y,
            &cl.reference_positions_z,
            sim_box,
            &mut cl.disp_sq,
        )?;
        timings
            .kernel_stop(KernelStage::NEIGHBOR_DISPLACEMENT_SQUARED)
            .map_err(map_timings_err)?;

        let host: Vec<Real> = self
            .device
            .dtoh_sync_copy(&cl.disp_sq)
            .map_err(GpuError::from)?;
        let mut max_sq: f64 = 0.0;
        for &v in host[..self.particle_count].iter() {
            let v64 = v as f64;
            if v64 > max_sq {
                max_sq = v64;
            }
        }
        Ok(max_sq.sqrt() as Real)
    }

    // rq-7db97132
    pub fn rebuild(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            match &mut self.mode {
                NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => {
                    cl.needs_rebuild = false;
                }
                NeighborListMode::Trivial => {}
            }
            return Ok(());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(());
        }
        let started = Instant::now();
        let result = self.rebuild_impl(sim_box, buffers, timings);
        timings.record_host(HostStage::NEIGHBOR_LIST_REBUILD, started.elapsed());
        result
    }

    fn rebuild_impl(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        let device = self.device.clone();
        let kernels = self.kernels.clone();
        let max_neighbors = self.max_neighbors;
        let particle_count = self.particle_count;
        let bin_only = matches!(self.mode, NeighborListMode::CellListOnly(_));

        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(()),
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => cl,
        };

        compute_cell_indices_and_histogram(
            buffers,
            sim_box,
            cl.n_cells,
            &mut cl.cell_indices,
            &mut cl.cell_counts,
        )?;

        prefix_scan_cell_counts(
            &kernels,
            &cl.cell_counts,
            &mut cl.cell_offsets,
            &mut cl.scan_block_totals,
            cl.n_cells_total,
            particle_count,
        )?;

        scatter_atoms_into_cells(
            &device,
            &kernels,
            &cl.cell_indices,
            &cl.cell_offsets,
            &mut cl.write_cursors,
            &mut cl.sorted_particle_ids,
            particle_count,
        )?;

        sort_cells_by_particle_id(
            &kernels,
            &cl.cell_offsets,
            &mut cl.sorted_particle_ids,
            cl.n_cells_total,
        )?;

        if bin_only {
            cl.needs_rebuild = false;
            self.rebuild_generation = self.rebuild_generation.wrapping_add(1);
            return Ok(());
        }

        device
            .memset_zeros(&mut cl.overflow_flag)
            .map_err(GpuError::from)?;

        timings
            .kernel_start(KernelStage::NEIGHBOR_LIST_BUILD)
            .map_err(map_timings_err)?;
        neighbor_list_build(
            buffers,
            &cl.sorted_particle_ids,
            &cl.cell_offsets,
            sim_box,
            cl.n_cells,
            cl.r_search_sq,
            max_neighbors,
            &mut self.neighbor_list,
            &mut self.neighbor_counts,
            &mut cl.overflow_flag,
        )?;
        timings
            .kernel_stop(KernelStage::NEIGHBOR_LIST_BUILD)
            .map_err(map_timings_err)?;

        let flag: Vec<u32> = self
            .device
            .dtoh_sync_copy(&cl.overflow_flag)
            .map_err(GpuError::from)?;
        if flag[0] != 0 {
            return Err(NeighborListError::NeighborListOverflow {
                max: max_neighbors,
            });
        }

        timings
            .kernel_start(KernelStage::COPY_POSITIONS_INTO_REFERENCE)
            .map_err(map_timings_err)?;
        copy_positions_into_reference(
            buffers,
            &mut cl.reference_positions_x,
            &mut cl.reference_positions_y,
            &mut cl.reference_positions_z,
        )?;
        timings
            .kernel_stop(KernelStage::COPY_POSITIONS_INTO_REFERENCE)
            .map_err(map_timings_err)?;

        // rq-d8de5e38 — Tile-based pair-force architecture
        // (`rqm/forces/tile-based-pair-force.md`).
        self.rebuild_tile_data(buffers, timings, sim_box)?;

        let cl = match &mut self.mode {
            NeighborListMode::Trivial => return Ok(()),
            NeighborListMode::CellList(cl) | NeighborListMode::CellListOnly(cl) => cl,
        };
        cl.needs_rebuild = false;
        self.rebuild_generation = self.rebuild_generation.wrapping_add(1);
        Ok(())
    }

    /// Populate the tile-based neighbour buffers
    /// (`tile_atom_count`, `tile_lane_mask`, `tile_pair_offsets`,
    /// `tile_pair_list`). Called from `rebuild_impl` after the
    /// cell-list sort has produced `sorted_particle_ids` and
    /// `cell_indices` / `cell_offsets`. The active per-step force
    /// kernel does not yet consume these buffers; the tile-based
    /// composer wires through in a follow-up step.
    pub(crate) fn rebuild_tile_data(
        &mut self,
        buffers: &ParticleBuffers,
        _timings: &mut Timings,
        sim_box: &SimulationBox,
    ) -> Result<(), NeighborListError> {
        use crate::gpu::launch_tile_construction;
        // Tile-based pair-force construction targets the
        // pair-force-bearing CellList mode only.
        let bin_only = !matches!(self.mode, NeighborListMode::CellList(_));
        if bin_only || self.particle_count == 0 {
            self.tile_pair_count = 0;
            return Ok(());
        }
        // Split-borrow: cl borrows self.mode immutably; tile_* are
        // mutable on self. Since `mode` and the tile fields are
        // disjoint fields of self, this is sound via field-level
        // borrowing.
        let NeighborListState {
            mode,
            kernels,
            particle_count,
            n_tiles,
            tile_atom_count,
            tile_lane_mask,
            tile_bboxes,
            tile_candidate_counts,
            tile_pair_offsets,
            tile_pair_list,
            tile_pair_masks,
            tile_pair_capacity,
            tile_pair_count,
            tile_sorted_particle_ids,
            device,
            ..
        } = self;
        let cl = match mode {
            NeighborListMode::CellList(cl) => cl,
            _ => unreachable!(),
        };
        let new_count = launch_tile_construction(
            kernels,
            buffers,
            sim_box,
            &cl.sorted_particle_ids,
            &cl.cell_indices,
            &cl.cell_offsets,
            cl.n_cells,
            cl.r_search_sq,
            *particle_count as u32,
            *n_tiles,
            tile_atom_count,
            tile_lane_mask,
            tile_bboxes,
            tile_candidate_counts,
            tile_pair_offsets,
            tile_pair_list,
            tile_pair_masks,
            *tile_pair_capacity,
        )?;
        *tile_pair_count = new_count;
        // Mirror the cell-list sort into the tile-based view so the
        // dispatcher can read a single sorted-id buffer regardless
        // of mode. Destination is sized for n_tiles * 32 lanes
        // (padded for the partial last tile); copy only the
        // particle_count real entries.
        let n = *particle_count;
        if n > 0 {
            device
                .dtod_copy(
                    &cl.sorted_particle_ids.slice(0..n),
                    &mut tile_sorted_particle_ids.slice_mut(0..n),
                )
                .map_err(GpuError::from)?;
        }
        Ok(())
    }

    // rq-1217c816
    pub fn pre_step(
        &mut self,
        sim_box: &SimulationBox,
        buffers: &ParticleBuffers,
        timings: &mut Timings,
    ) -> Result<(), NeighborListError> {
        if self.particle_count == 0 {
            return Ok(());
        }
        if matches!(self.mode, NeighborListMode::Trivial) {
            return Ok(());
        }

        // CellListOnly mode rebuilds every step unconditionally; the
        // displacement-check + r_skin machinery is bypassed entirely.
        if matches!(self.mode, NeighborListMode::CellListOnly(_)) {
            if let NeighborListMode::CellListOnly(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            return self.rebuild(sim_box, buffers, timings);
        }

        let refreshed = self.refresh_cell_layout_if_box_changed(sim_box)?;

        let (mut rebuild_required, r_skin) = match &self.mode {
            NeighborListMode::Trivial | NeighborListMode::CellListOnly(_) => unreachable!(),
            NeighborListMode::CellList(cl) => (cl.needs_rebuild, cl.r_skin),
        };

        if !refreshed && !rebuild_required {
            let max_disp = self.displacement_check(sim_box, buffers, timings)?;
            if max_disp > r_skin * 0.5 {
                rebuild_required = true;
            }
        }
        if rebuild_required {
            if let NeighborListMode::CellList(cl) = &mut self.mode {
                cl.needs_rebuild = true;
            }
            self.rebuild(sim_box, buffers, timings)?;
        }
        Ok(())
    }
}

// rq-dfad7218
//
// n_cells_d = floor(w_d / (r_cut + r_skin)) per lattice direction
// d ∈ {a, b, c}, where w_d is the box's perpendicular width along that
// direction (see simulation-box.md). Rejects with BoxTooSmallForCells if
// any direction admits fewer than 3 cells.
fn compute_cell_layout(
    sim_box: &SimulationBox,
    r_search: Real,
) -> Result<[u32; 3], NeighborListError> {
    let widths = sim_box.perpendicular_widths();
    let direction_names: [&'static str; 3] = ["a", "b", "c"];
    let mut n_cells = [0u32; 3];
    for d in 0..3 {
        let w = widths[d];
        let nc = (w / r_search).floor() as i64;
        if nc < 3 {
            return Err(NeighborListError::BoxTooSmallForCells {
                direction: direction_names[d],
                width: w,
                required: 3.0 * r_search,
            });
        }
        n_cells[d] = nc as u32;
    }
    Ok(n_cells)
}

// rq-d8e4407a
//
// The device addresses cells with a `u32` cell index (the cell-index
// arithmetic and every scan kernel argument are `u32`), so the cell
// grid may hold at most `u32::MAX` cells. Checked before any device
// buffer is allocated so an over-fine grid fails loud rather than
// overflowing the cell-index arithmetic.
fn check_n_cells_total(n_cells_total: usize) -> Result<(), NeighborListError> {
    let max_supported = u32::MAX as usize;
    if n_cells_total > max_supported {
        return Err(NeighborListError::TooManyCells {
            n_cells_total,
            max_supported,
        });
    }
    Ok(())
}

// rq-a060036e
//
// Lengths of the recursive prefix scan's block-totals stack for a grid
// of `n_cells_total` cells: level 0 holds `ceil(n_cells_total / B)`
// per-block totals, each subsequent level holds `ceil(prev / B)`, and
// the stack ends with the first level of length 1. Every
// `prefix_scan_local_blocks` call needs a block-totals output buffer,
// including the terminal single-block one, so the last length is 1.
/// Same as `scan_stack_lengths`; re-exported under a less
/// neighbour-list-specific name so the tile-pair offset scan in
/// `gpu::kernels` can reuse it without coupling to the cell-list
/// terminology.
pub fn scan_stack_lengths_for(n: usize) -> Vec<usize> {
    scan_stack_lengths(n)
}

pub(crate) fn scan_stack_lengths(n_cells_total: usize) -> Vec<usize> {
    let block = SPATIAL_HASH_SCAN_BLOCK_SIZE as usize;
    let mut lengths = Vec::new();
    let mut len = n_cells_total;
    loop {
        let blocks = len.div_ceil(block);
        lengths.push(blocks);
        if blocks <= 1 {
            break;
        }
        len = blocks;
    }
    lengths
}

pub(crate) fn alloc_scan_block_totals(
    device: &Arc<CudaDevice>,
    n_cells_total: usize,
) -> Result<Vec<CudaSlice<u32>>, NeighborListError> {
    scan_stack_lengths(n_cells_total)
        .into_iter()
        .map(|len| {
            device
                .alloc_zeros::<u32>(len.max(1))
                .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))
        })
        .collect()
}

/// Initial capacity hint for the tile-pair list: `N_tiles *
/// INITIAL_CAPACITY_PER_TILE` entries.
///
/// Without bounding-box pruning (the current skeleton-phase
/// construction kernels enumerate candidate j_tiles purely by
/// cell-sweep), each tile's neighbour-tile sublist is the UNION of
/// the 32 home atoms' 27-cell-sweep candidate tiles. For liquid
/// water at ~100 atoms/nm³ and `r_cut + r_skin ≈ 11 Å` on a
/// near-cubic box this is empirically ~160 candidate j_tiles per
/// tile; 256 leaves a 1.6× headroom before a grow would be
/// required.
///
/// Once bounding-box pruning is implemented (rqm/forces/tile-based-
/// pair-force.md, *Tile-Pair Neighbour List Construction* step 3),
/// this hint can drop substantially.
const TILE_PAIR_INITIAL_CAPACITY_PER_TILE: u32 = 256;

/// Allocate the tile-related GPU buffers. Returns
/// `(n_tiles, tile_atom_count, tile_lane_mask, tile_candidate_counts,
///   tile_pair_offsets, tile_pair_list, tile_pair_capacity)`.
///
/// The tile-pair list is allocated to `N_tiles *
/// TILE_PAIR_INITIAL_CAPACITY_PER_TILE` entries up front. Rebuilds
/// that exceed this capacity grow the buffer; the grow logic lives
/// in `NeighborListState::rebuild`.
pub(crate) struct TileBuffers {
    pub n_tiles: u32,
    pub tile_atom_count: CudaSlice<u32>,
    pub tile_lane_mask: CudaSlice<u32>,
    pub tile_bboxes: CudaSlice<Real>,
    pub tile_candidate_counts: CudaSlice<u32>,
    pub tile_pair_offsets: CudaSlice<u32>,
    pub tile_pair_list: CudaSlice<u32>,
    pub tile_pair_masks: CudaSlice<u32>,
    pub tile_pair_capacity: u32,
    pub tile_sorted_particle_ids: CudaSlice<u32>,
    pub tile_sorted_positions_x: CudaSlice<Real>,
    pub tile_sorted_positions_y: CudaSlice<Real>,
    pub tile_sorted_positions_z: CudaSlice<Real>,
}

pub(crate) fn alloc_tile_buffers(
    device: &Arc<CudaDevice>,
    particle_count: usize,
) -> Result<TileBuffers, NeighborListError> {
    let n_tiles = particle_count.div_ceil(32) as u32;
    let n_tiles_alloc = n_tiles.max(1) as usize;
    let tile_atom_count = device
        .alloc_zeros::<u32>(n_tiles_alloc)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_lane_mask = device
        .alloc_zeros::<u32>(n_tiles_alloc)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_bboxes = device
        .alloc_zeros::<Real>(6 * n_tiles_alloc)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_candidate_counts = device
        .alloc_zeros::<u32>(n_tiles_alloc)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_pair_offsets = device
        .alloc_zeros::<u32>(n_tiles_alloc + 1)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_pair_capacity = n_tiles
        .saturating_mul(TILE_PAIR_INITIAL_CAPACITY_PER_TILE)
        .max(1);
    let tile_pair_list = device
        .alloc_zeros::<u32>(tile_pair_capacity as usize)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_pair_masks = device
        .alloc_zeros::<u32>(tile_pair_capacity as usize)
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    // Allocate sorted_particle_ids large enough to address every
    // tile lane (including the partial last tile's inactive lanes).
    // The kernel never reads inactive lanes' entries; zero-init is
    // sufficient.
    let tile_sorted_particle_ids = device
        .alloc_zeros::<u32>((n_tiles_alloc * 32).max(1))
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    // tile_sorted_positions_* mirror positions in tile-sort order
    // (one entry per atom; no padding for partial last tile because
    // the pair-force kernel reads them only for active lanes via
    // `tile_sorted_positions_*[i_tile * 32 + lane]` where the lane
    // is gated by `tile_lane_mask`).
    let tile_sorted_positions_x = device
        .alloc_zeros::<Real>(particle_count.max(1))
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_sorted_positions_y = device
        .alloc_zeros::<Real>(particle_count.max(1))
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    let tile_sorted_positions_z = device
        .alloc_zeros::<Real>(particle_count.max(1))
        .map_err(|e| NeighborListError::Gpu(GpuError::from(e)))?;
    Ok(TileBuffers {
        n_tiles,
        tile_atom_count,
        tile_lane_mask,
        tile_bboxes,
        tile_candidate_counts,
        tile_pair_offsets,
        tile_pair_list,
        tile_pair_masks,
        tile_pair_capacity,
        tile_sorted_particle_ids,
        tile_sorted_positions_x,
        tile_sorted_positions_y,
        tile_sorted_positions_z,
    })
}

fn map_timings_err(e: crate::timings::TimingsError) -> NeighborListError {
    match e {
        crate::timings::TimingsError::Gpu(g) => NeighborListError::Gpu(g),
    }
}

// rq-2093594f rq-0469400b
#[derive(Debug, Clone)]
pub struct NeighborKernels {
    pub neighbor_displacement_squared: CudaFunction,
    pub neighbor_list_build: CudaFunction,
    pub copy_positions_into_reference: CudaFunction,
    pub compute_cell_indices_and_histogram: CudaFunction,
    pub prefix_scan_local_blocks: CudaFunction,
    pub prefix_scan_apply_block_totals: CudaFunction,
    pub prefix_scan_finalize_offsets: CudaFunction,
    pub scatter_atoms_into_cells: CudaFunction,
    pub sort_cells_by_particle_id: CudaFunction,
    // rq-be571c62 / rq-d8de5e38 — tile-based construction kernels.
    pub compute_tile_metadata: CudaFunction,
    pub compute_tile_bounding_boxes: CudaFunction,
    pub tile_pair_candidate_count: CudaFunction,
    pub tile_pair_finalize_offsets: CudaFunction,
    pub tile_pair_emit: CudaFunction,
    pub compute_tile_pair_masks: CudaFunction,
    pub scatter_positions_to_tile_order: CudaFunction,
}

impl NeighborKernels {
    pub fn load(device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        device.load_ptx(
            Ptx::from_src(kernels::NEIGHBOR),
            "neighbor",
            &[
                "neighbor_displacement_squared",
                "neighbor_list_build",
                "copy_positions_into_reference",
                "compute_cell_indices_and_histogram",
                "prefix_scan_local_blocks",
                "prefix_scan_apply_block_totals",
                "prefix_scan_finalize_offsets",
                "scatter_atoms_into_cells",
                "sort_cells_by_particle_id",
                "compute_tile_metadata",
                "compute_tile_bounding_boxes",
                "tile_pair_candidate_count",
                "tile_pair_finalize_offsets",
                "tile_pair_emit",
                "compute_tile_pair_masks",
                "scatter_positions_to_tile_order",
            ],
        )?;
        Ok(NeighborKernels {
            neighbor_displacement_squared: get_func(
                device,
                "neighbor",
                "neighbor_displacement_squared",
            )?,
            neighbor_list_build: get_func(device, "neighbor", "neighbor_list_build")?,
            copy_positions_into_reference: get_func(
                device,
                "neighbor",
                "copy_positions_into_reference",
            )?,
            compute_cell_indices_and_histogram: get_func(
                device,
                "neighbor",
                "compute_cell_indices_and_histogram",
            )?,
            prefix_scan_local_blocks: get_func(device, "neighbor", "prefix_scan_local_blocks")?,
            prefix_scan_apply_block_totals: get_func(
                device,
                "neighbor",
                "prefix_scan_apply_block_totals",
            )?,
            prefix_scan_finalize_offsets: get_func(
                device,
                "neighbor",
                "prefix_scan_finalize_offsets",
            )?,
            scatter_atoms_into_cells: get_func(device, "neighbor", "scatter_atoms_into_cells")?,
            sort_cells_by_particle_id: get_func(device, "neighbor", "sort_cells_by_particle_id")?,
            compute_tile_metadata: get_func(device, "neighbor", "compute_tile_metadata")?,
            compute_tile_bounding_boxes: get_func(
                device,
                "neighbor",
                "compute_tile_bounding_boxes",
            )?,
            tile_pair_candidate_count: get_func(
                device,
                "neighbor",
                "tile_pair_candidate_count",
            )?,
            tile_pair_finalize_offsets: get_func(
                device,
                "neighbor",
                "tile_pair_finalize_offsets",
            )?,
            tile_pair_emit: get_func(device, "neighbor", "tile_pair_emit")?,
            compute_tile_pair_masks: get_func(
                device,
                "neighbor",
                "compute_tile_pair_masks",
            )?,
            scatter_positions_to_tile_order: get_func(
                device,
                "neighbor",
                "scatter_positions_to_tile_order",
            )?,
        })
    }
}
