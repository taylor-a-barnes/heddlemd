# Feature: Tile-Based Pair-Force Architecture <!-- rq-78f54c5f -->

The fast-class pair-force pipeline organises atoms into tiles of 32
and evaluates pair interactions using a warp-per-tile kernel
structure. The neighbour-list data structure for fast-class slots is
a list of interacting tile-pairs: each entry references one
`(i_tile, j_tile)` pair where at least one atom in `i_tile` is within
`r_cut + r_skin` of at least one atom in `j_tile`. The
JIT-composed pair-force kernel processes one i_tile per warp; each
warp lane owns one of the 32 home atoms in the i_tile and
accumulates pair contributions from every interacting j_tile into a
per-lane register accumulator. After the warp iterates the i_tile's
full neighbour-tile sublist, each lane writes its home atom's
accumulated force, energy, and virial directly to the slot-output
buffer.

This file specifies the data model, the tile-pair neighbour list and
its construction, the pair-force kernel structure, the JIT composer
integration, the determinism invariants, the configuration surface,
and the Feature API. Adjacent files: `rqm/forces/neighbor-list.md`
(cell-list construction is the pre-step for tile assignment),
`rqm/forces/jit-composed-pair-force.md` (the JIT composer's slot
source-fragment contract is unchanged; the outer-loop template is
the only piece that depends on the tile-based structure),
`rqm/forces/framework.md` (the fast-class slot architecture and
slot-output buffer interface are unchanged), `architecture.md`
(determinism invariants are preserved exactly).

The bonded-class and angle-class pipelines, the SPME reciprocal
pipeline, and all integrator / thermostat / barostat slots are
unaffected by this architecture. Per-particle neighbour-list data
structures are not used by any fast-class slot.

## Scope <!-- rq-ce85fbd6 -->

The architecture applies to fast-class pair-force slots:
`lennard_jones`, `coulomb`, `spme_real`, and any user-registered
pair-force slot that participates in `ForceField`'s fast-class
pipeline. The JIT-composed pair-force kernel
(`heddle_jit_composed_pair_force_f` / `heddle_jit_composed_pair_force_fev`)
is the only entry point for these slots' per-step force evaluation.

`max_neighbors` is not a config parameter. The tile-pair list is
sized to the actual interacting-tile count plus a small growth
margin. The per-particle padded neighbour list is not allocated and
not used by any fast-class slot.

## Data Model <!-- rq-63755598 -->

### Tiles <!-- rq-a1b28335 -->

A **tile** is a contiguous group of 32 atoms in the post-cell-sort
atom ordering. There are `N_tiles = ⌈N / 32⌉` tiles. Tile `t`
contains atoms with sorted-index range
`[t × 32, min((t + 1) × 32, N))`. The atom at lane offset `l`
within tile `t` has sorted-index `t × 32 + l`.

If `N` is not divisible by 32, the last tile has `N mod 32` real
atoms and `32 - (N mod 32)` inactive lanes. Inactive lanes contribute
zero force, energy, and virial; the kernel masks them out before
accumulation.

Tile membership is stable across pair-force kernel launches and
changes only when the cell-list sort changes (i.e. at every
neighbour-list rebuild).

### Tile metadata <!-- rq-be571c62 -->

`NeighborListState` carries per-tile metadata in addition to the
cell-list buffers:

- `tile_atom_count: CudaSlice<u32>` of length `N_tiles` — number of
  real atoms in each tile. Equal to 32 for all but possibly the
  last tile.
- `tile_lane_mask: CudaSlice<u32>` of length `N_tiles` — bitmask of
  active lanes for each tile: `(1u << tile_atom_count[t]) - 1`.
  Full tiles carry `0xFFFFFFFF`; the partial last tile (if any)
  carries a mask with zeros in the high bits.

The tile-to-particle mapping uses the existing
`sorted_particle_ids: CudaSlice<u32>` buffer maintained by the cell
list: atom at tile-lane `(t, l)` has particle ID
`sorted_particle_ids[t * 32 + l]`. No separate tile-to-particle
table is allocated.

### Tile-sorted position view <!-- rq-b7601928 -->

The pair-force kernel reads each atom's `(x, y, z)` position by
tile-lane index. To make those reads coalesced, the per-particle
position arrays are mirrored into a parallel **tile-sorted
position view** held on `NeighborListState`:

- `tile_sorted_positions_x: CudaSlice<Real>` of length
  `particle_count`.
- `tile_sorted_positions_y: CudaSlice<Real>` of length
  `particle_count`.
- `tile_sorted_positions_z: CudaSlice<Real>` of length
  `particle_count`.

The semantics are
`tile_sorted_positions_*[k] = positions_*[sorted_particle_ids[k]]`
for every `k` in `[0, particle_count)`. The pair-force kernel
reads these buffers directly via `tile_sorted_positions_*[i_tile * 32 + lane]`
and the cooperative j_tile load `tile_sorted_positions_*[j_tile * 32 + lane]` —
a contiguous 32-element span per j_tile load, coalesced by the
hardware.

Particle ID lookups (`home_atom_id`, `j`) still use
`sorted_particle_ids[k]` — the per-pair functor's `evaluate(i, j, ...)`
call and the slot-output write `slot_force_*[home_atom_id]` both
need the original particle ID. Those are single-u32 loads per
atom; the bandwidth-dominant per-coordinate position loads are
the ones that benefit from coalescing.

The view is refreshed every step by the
`scatter_positions_to_tile_order` kernel; see *Tile-Sorted
Position Scatter* below. The main `ParticleBuffers` stay in
particle-id order — integration, output, post-force per-particle
work, and SPME reciprocal all read from the canonical buffers
without going through this view.

### Tile-pair neighbour list <!-- rq-69063b1d -->

The **tile-pair neighbour list** is a list of `(i_tile, j_tile)`
pairs where at least one atom in `i_tile` is within
`r_cut + r_skin` of at least one atom in `j_tile`. Entries are
grouped by `i_tile`: each tile's neighbour-tile sublist is a
contiguous range, and sublists appear in ascending `i_tile` order.

Per-tile bookkeeping:

- `tile_pair_offsets: CudaSlice<u32>` of length `N_tiles + 1` —
  offsets into `tile_pair_list` and `tile_pair_masks`. Tile `t`'s
  neighbour-tile sublist is
  `tile_pair_list[tile_pair_offsets[t] .. tile_pair_offsets[t + 1]]`.
- `tile_pair_list: CudaSlice<u32>` of length `total_pairs` — the
  j_tile indices for every `(i_tile, j_tile)` pair in the
  neighbour list. The per-tile sublist is sorted by `j_tile`
  ascending.
- `tile_pair_masks: CudaSlice<u32>` of length `total_pairs` — one
  32-bit **j-atom interaction mask** per tile-pair entry. Bit `k`
  of `tile_pair_masks[tile_pair_offsets[t] + i]` is set iff the
  j_atom at tile-lane `k` of `tile_pair_list[tile_pair_offsets[t] + i]`
  is within `r_cut + r_skin` of at least one atom in i_tile `t`,
  considered under minimum image. Inactive j-lanes (lane index ≥
  `tile_atom_count[j_tile]`) always have their bit cleared so the
  pair-force kernel does not have to gate them separately. The
  mask is read once per tile-pair by the pair-force kernel and
  used to short-circuit the per-pair inner loop over j-atoms.

The self-pair `(t, t)` is always present for every tile `t` because
atoms within a tile interact with each other through the tile's
self-pair sublist entry. The self-pair appears as the smallest
j_tile in tile `t`'s sublist.

`total_pairs` is determined at neighbour-list build time from the
actual interacting-tile count plus a 10 % growth margin. If a
rebuild would overflow the allocation, the allocation is grown to
`1.5 × required_capacity` and the rebuild is re-issued. The growth
event is logged at runtime; no host arithmetic is needed beyond the
re-allocation.

## Tile-Pair Neighbour List Construction <!-- rq-d8de5e38 -->

Tile assignment piggybacks on the cell-list sort: after the cell
list emits `sorted_particle_ids` (atoms sorted by cell, ties broken
by particle ID — see `rqm/forces/neighbor-list.md`), atoms
`[t × 32, (t + 1) × 32)` form tile `t` automatically. No additional
sort is performed.

The construction pipeline runs as part of `NeighborListState::rebuild`:

1. **`compute_tile_metadata` kernel.** One thread per tile. For
   tile `t`, writes `tile_atom_count[t]` and `tile_lane_mask[t]`.
   The last tile's count is `N - t * 32`; all others carry `32`.

2. **`compute_tile_bounding_boxes` kernel.** One block per tile,
   32 threads per block (one per home-tile lane). Each lane reads
   its home atom's position; a warp-shuffle butterfly reduction
   produces the per-axis min and max across the 32 atoms. Lanes
   beyond the tile's real atom count contribute neutral identity
   values (`+INFINITY` for min, `-INFINITY` for max) so the
   reduction ignores them. The bounding box is stored as six Real
   per tile in row-major order
   `[min_x, min_y, min_z, max_x, max_y, max_z]` in
   `tile_bboxes: CudaSlice<Real>` of length `6 * N_tiles`.

3. **`tile_pair_candidate_count` kernel.** One block per tile,
   32 threads per block. The block identifies the i_tile's
   **home-cell set** — the set of unique cells that any atom in
   the i_tile occupies — and sweeps the 27-cell neighbourhood of
   each home cell exactly once. Concretely:
   - Each lane reads its home atom's particle ID
     (`sorted_particle_ids[i_tile * 32 + lane]`) and looks up its
     cell (`cell_indices[particle_id]`). Inactive lanes (lane index
     ≥ `tile_atom_count[i_tile]`) carry a sentinel cell value that
     is ignored by the dedup pass.
   - The warp collectively dedups the 32 home-cell values into a
     compact list of unique cells held in shared memory.
     Determinism: cells appear in the unique list in ascending lane
     index order (the lane carrying the first occurrence of each
     cell value contributes it).
   - For each unique home cell `c`, the warp sweeps the 27 cells
     `c + (da, db, dc)` for `(da, db, dc) ∈ {-1, 0, 1}³` using the
     same triclinic wrap logic as the per-particle build. Lanes
     cooperate to iterate the atoms in each swept cell.
   - For each neighbour particle encountered in those cells the
     warp computes `j_tile = sorted_index / 32`. Before marking
     `j_tile` as a candidate the kernel consults the bounding-box
     pruning predicate
     `tile_pair_bbox_prune(bbox_i, bbox_j, lattice, r_search_sq)`:
     the predicate returns `true` (prune) when the two tiles'
     bounding boxes are far enough apart under minimum image that
     no atom-atom pair across them can be within `r_cut + r_skin`.
     A "bit already set" check on the shared-memory bitmask short-
     circuits the bbox-prune call when the candidate has already
     been admitted from an earlier sweep.

   Candidate j_tiles that survive the prune are marked in a
   per-block shared-memory bitmask. Set bits are popcounted into
   `tile_candidate_counts: [u32; N_tiles]`. The self-pair `(t, t)`
   is always counted (no pruning — atoms within the same tile
   always interact through the self-pair sublist entry).

   The home-cell dedup pass guarantees each unique cell in the
   i_tile's home-cell set is swept exactly once. For liquid-density
   workloads where an i_tile's 32 atoms occupy 1–3 cells, this
   eliminates the redundant per-atom cell iteration that would
   otherwise re-walk the same cells up to 32 times per tile.

4. **Prefix scan over `tile_candidate_counts`**, producing
   `tile_pair_offsets`. Uses the same scan primitive as the
   per-particle neighbour-list build.

5. **`tile_pair_emit` kernel.** One block per tile, 32 threads per
   block. Re-enumerates the candidate j_tiles using the same
   home-cell dedup pass and 27-cell sweep as step 3 (identical
   home-cell set, identical sweep order, identical
   `tile_pair_bbox_prune` predicate). The shared-memory bitmask
   records the surviving candidates; a single thread walks the
   bitmask in ascending `j_tile` order and emits each set bit's
   index into `tile_pair_list[tile_pair_offsets[t] + k]` for
   `k = 0, 1, …`. The self-pair appears at its natural sorted
   position within the sublist.

   The two kernels (`tile_pair_candidate_count` and
   `tile_pair_emit`) must produce byte-identical bitmasks for the
   same i_tile across the two passes — the popcount in step 3
   determines the offset into `tile_pair_list` that step 5 writes
   into, so any divergence would either over-emit (overflowing
   the next tile's sublist range) or under-emit (leaving holes in
   the sublist range). Determinism of the home-cell dedup pass and
   the 27-cell sweep order is load-bearing for this equality.

6. **`compute_tile_pair_masks` kernel.** One warp per tile-pair
   entry; grid sized to `⌈total_pairs / WARPS_PER_BLOCK⌉` blocks.
   For each tile-pair `(i_tile, j_tile)` in `tile_pair_list`, the
   warp computes the 32-bit j-atom interaction mask and writes it
   to `tile_pair_masks[entry_index]`. Per-pair execution:
   - Lane `k` (0 ≤ `k` < 32) is responsible for bit `k` of the
     output mask — the bit corresponding to j_atom at tile-lane
     `k` of `j_tile`.
   - If `k ≥ tile_atom_count[j_tile]`, the lane writes a cleared
     bit and exits early.
   - Otherwise the lane reads j_atom `k`'s position
     (`positions_*[sorted_particle_ids[j_tile * 32 + k]]`) and
     iterates the 32 atoms of `i_tile`. For each `(i_atom, j_atom_k)`
     pair the lane computes the minimum-image squared distance and
     compares against `r_search_sq = (r_cut + r_skin)²`. The
     lane's bit is set if any i_atom is within range; the lane
     exits the loop early on the first hit.
   - The 32 lanes combine their per-lane bits into a single u32
     via `__ballot_sync(0xFFFFFFFFu, lane_in_range)`. Lane 0
     writes the result to `tile_pair_masks[entry_index]`.

   The **self-pair** entry `(t, t)` is a special case: the mask is
   hardcoded to `tile_lane_mask[t]` (every real lane is set) and
   the distance computation is skipped. The pair-force kernel's
   self-pair branch separately skips the diagonal `m == lane`
   case, so the diagonal does not contribute even though its bit
   is set.

`total_pairs` is read from `tile_pair_offsets[N_tiles]` after the
scan; if `total_pairs > allocated_capacity`, the allocation is
grown and steps 4, 5, and 6 re-run.

### Determinism <!-- rq-480ea47d -->

The construction order — sublists ascending in `i_tile`, entries
within a sublist ascending in `j_tile`, the self-pair always first
— is deterministic given the deterministic cell-list sort. Two
runs from byte-identical state produce byte-identical tile-pair
lists.

The home-cell dedup pass in steps 3 and 5 admits cells in ascending
lane-index order (the lane carrying the first occurrence of each
cell value is the one that contributes it to the unique list). The
27-cell sweep order is the same as the per-particle build
(`(da, db, dc)` lex outer-to-inner). Within each swept cell atoms
are visited in cell-sorted order. These three deterministic
orderings combine to make the populated bitmask byte-identical
across the two passes and across runs.

The masks produced by step 6 are a pure function of positions and
the deterministic `tile_pair_list` contents. The
`__ballot_sync(0xFFFFFFFFu, ...)` combines the 32 per-lane bits
into a unique u32 regardless of warp execution order. Two runs
from byte-identical state produce byte-identical
`tile_pair_masks`.

The bounding-box pruning is conservative: it may include
tile-pairs where no atom-atom pair is actually within
`r_cut + r_skin`. False positives are harmless — the pair-force
kernel performs the per-atom cutoff check anyway. False negatives
would violate correctness; the bounding-box pruning is implemented
to err on the side of inclusion.

## Tile-Sorted Position Scatter <!-- rq-b1e9e9ae -->

The tile-sorted position view is refreshed at the start of every
per-step pair-force evaluation. A small device kernel —
`scatter_positions_to_tile_order` — populates the three
`tile_sorted_positions_*` buffers from the canonical
`positions_*` arrays via the permutation
`sorted_particle_ids[k]`:

```
for k in 0..particle_count:
    pid = sorted_particle_ids[k]
    tile_sorted_positions_x[k] = positions_x[pid]
    tile_sorted_positions_y[k] = positions_y[pid]
    tile_sorted_positions_z[k] = positions_z[pid]
```

The kernel launches with one thread per atom, block size 256,
grid `ceil(N / 256)`. No shared memory, no synchronisation, no
inter-thread reads. The writes are entirely independent.

The scatter runs as the first device kernel in
`ForceField::step()`, before the JIT-composed pair-force kernel
and any other fast-class slot kernels that read positions
indirectly via tile-sorted indexing. The integration kernel that
precedes `ForceField::step()` has already updated
`positions_*` in place, so the scattered view reflects the
current step's positions.

The scatter is captured into the per-step CUDA graph (when graph
mode is active) and replays once per replay alongside the rest of
the step kernel sequence. Its cost on SPC water 8192 is below
~5 µs per step (single coalesced read + coalesced write per
atom, no compute).

The scatter is deterministic: each thread writes a distinct
`(k, x or y or z)` slot, so the resulting buffers do not depend
on thread scheduling order. Two runs from byte-identical state
produce byte-identical `tile_sorted_positions_*`.

## Pair-Force Kernel <!-- rq-8bed0323 -->

The composed pair-force kernel
(`heddle_jit_composed_pair_force_f` / `heddle_jit_composed_pair_force_fev`)
processes one i_tile per warp. The grid configuration is:

- block size: `WARPS_PER_BLOCK * 32 = 256` threads
- grid x: `⌈N_tiles / WARPS_PER_BLOCK⌉`
- shared memory: 48 bytes per warp (32 × 12 bytes for j_tile
  position cache) plus per-fragment shared-memory allocations
  declared via the JIT composer

### Per-warp execution <!-- rq-71ca093b -->

Each warp owns one i_tile, identified by `i_tile = blockIdx.x *
WARPS_PER_BLOCK + warp_id_in_block`. Lane `l` in the warp owns
home atom at `home_atom_id = sorted_particle_ids[i_tile * 32 + l]`.

The warp processes the i_tile's full neighbour-tile sublist:

1. **Per-warp setup.** Lane `l` reads its home atom position from
   the tile-sorted view:
   `pi_x = tile_sorted_positions_x[i_tile * 32 + l]`,
   `pi_y = tile_sorted_positions_y[i_tile * 32 + l]`,
   `pi_z = tile_sorted_positions_z[i_tile * 32 + l]`. This is a
   contiguous 32-element span across the warp, coalesced by the
   hardware. Inactive lanes (`l ≥ tile_atom_count[i_tile]`) skip
   the read. Each lane initialises its register accumulators
   `(F_x, F_y, F_z, energy, virial)` to zero.

2. **Iterate tile-pair sublist.** For `k = 0 ..
   (tile_pair_offsets[i_tile + 1] - tile_pair_offsets[i_tile])`:
   - Read `j_tile = tile_pair_list[entry_index]` and
     `j_mask = tile_pair_masks[entry_index]` where
     `entry_index = tile_pair_offsets[i_tile] + k`.
   - **Cooperative j_tile load.** Lanes `0..31` cooperatively load
     the 32 j_atoms' positions into shared memory from the
     tile-sorted view: lane `l` reads
     `tile_sorted_positions_*[j_tile * 32 + l]`. The reads are
     coalesced (32 contiguous elements per coordinate). Each lane
     also loads its own particle ID via
     `sorted_particle_ids[j_tile * 32 + l]` for the per-pair
     functor's `evaluate(i, j, ...)` call and for any exclusion-
     table lookups that key by particle ID.
   - **Per-pair inner loop, mask-driven.** Iterate the set bits of
     `j_mask` from low to high (e.g. via
     `while (j_mask != 0) { int m = __ffs(j_mask) - 1;
     j_mask &= j_mask - 1; ... }`):
     - Read j_atom data from shared memory at slot `m`.
     - Compute `r²` between lane `l`'s home atom and j_atom `m`.
     - If `r² ≤ r_cut²`, invoke the composite functor's `evaluate`
       to compute `(factor, energy, virial)`.
     - Look up the exclusion scale via the per-pair exclusion
       table.
     - Accumulate `(factor * dx * scale, factor * dy * scale,
       factor * dz * scale, energy * scale, virial * scale)` into
       lane `l`'s register accumulators.
   - **Self-pair handling.** When `j_tile == i_tile`, the inner
     loop additionally skips the case `m == l` (an atom does not
     interact with itself). The mask for the self-pair includes
     all real j-lanes, so the diagonal skip is the only special
     case.

   Iterating the mask's set bits rather than `0..31` is the
   load-bearing perf property: bits cleared by step 6 correspond
   to j-atoms that step 6 proved cannot have any i-atom within
   `r_cut + r_skin`. The inner loop never visits those j-atoms.
   The `r² ≤ r_cut²` check inside the loop still runs because
   the mask covers the larger `r_search` radius (which includes
   the skin); atoms inside `r_search` but outside `r_cut` fail
   the inner cutoff check and contribute zero, exactly as in the
   skin-less case.

3. **Write to slot output.** After the sublist is exhausted, each
   active lane `l` (where `l < tile_atom_count[i_tile]`) writes
   its accumulated force, energy, and virial directly to the
   slot-output buffer at index `home_atom_id`:
   - `slot_force_x[home_atom_id] = F_x`
   - `slot_force_y[home_atom_id] = F_y`
   - `slot_force_z[home_atom_id] = F_z`
   - `slot_energy[home_atom_id] = energy` (if `WriteEv`)
   - `slot_virial[home_atom_id] = virial` (if `WriteEv`)

   No `atomicAdd`. No warp-tree reduction. Each lane is the sole
   writer of its home atom's slot-output entries across the entire
   kernel launch.

Inactive lanes (lane indices ≥ `tile_atom_count[i_tile]`) do not
read positions, do not contribute to accumulators, and do not write
to the slot output. The kernel structure executes the same control
flow on every lane to avoid divergence; inactive contributions are
masked by the `tile_lane_mask` check before the accumulate step.

### No Newton's third law <!-- rq-033258c1 -->

Each in-cutoff pair `(a, b)` is evaluated twice across the kernel:
once by the warp owning `a`'s tile (computing the contribution to
`a`) and once by the warp owning `b`'s tile (computing the
contribution to `b`). The kernel does not share the symmetric
result between the two warps. This is the source of the
deterministic-reduction property: each atom's force is the sum of
contributions written by exactly one warp lane, in a fixed
evaluation order, with no atomic operations.

The compute count is `2 × N_in_cutoff_pairs` evaluations per
launch, matching the existing warp-per-particle compute count. The
memory access pattern is the architectural improvement: 32 j_atom
positions are loaded cooperatively into shared memory once per
tile-pair and reused for all 32 i_atom evaluations against them.

## JIT Composer Integration <!-- rq-c63f56f3 -->

The JIT composer (see `rqm/forces/jit-composed-pair-force.md`) emits
a single composed kernel per fast-class pipeline by concatenating
fragment source from every active slot. The fragment contract —
`functor_struct_name`, `functor_source`, `entry_point_args`,
`functor_init_source` — is unchanged: each fragment's
`evaluate(r², i, j) -> (factor, energy, virial)` is invoked
identically per-pair regardless of the kernel's outer-loop
structure.

The composer emits a tile-based outer-loop template (the
`OUTER_LOOP_TEMPLATE` constant in `src/forces/jit_composed.rs`)
that iterates one i_tile per warp as described in *Pair-Force
Kernel* above. Entry-point common args include:

- `tile_sorted_positions_x, tile_sorted_positions_y,
  tile_sorted_positions_z` (the tile-sorted position view; see
  *Tile-sorted position view*)
- `sorted_particle_ids` (the cell-sorted index → particle ID map,
  used for `home_atom_id` and per-pair functor calls)
- `tile_pair_offsets, tile_pair_list, tile_pair_masks`
- `tile_atom_count, tile_lane_mask`
- `lattice` (6-float triclinic lattice tuple)
- `slot_force_x, slot_force_y, slot_force_z, slot_energy, slot_virial`
  (per-slot output buffers; one set per active fragment)
- per-fragment args declared by each slot's `entry_point_args`

The canonical `positions_*` arrays are **not** in the pair-force
kernel's common args. Per-particle data accessed by per-fragment
functors (charges, type indices, exclusion tables) still indexes
by particle ID, populated via the slot's `bind_pair_force_args`
and consumed via `sorted_particle_ids` lookups inside the inner
loop.

`max_neighbors` is not in the common args. The per-particle
`neighbor_list` and `neighbor_counts` buffers are not in the common
args. No fast-class slot's `bind_pair_force_args` pushes
`max_neighbors`.

### Source fragment <!-- rq-d2d3ec6b -->

Existing fragment source for the in-tree fast-class slots
(Lennard-Jones, Coulomb, SPME-real) requires no behavioural
changes: the functor's `evaluate` is unchanged. The only
fragment-side update is that `entry_point_args` no longer
declares any `max_neighbors`-related parameters; the composer's
outer-loop template handles tile iteration without per-fragment
involvement.

Fragments may declare additional per-slot shared-memory
allocations (e.g. for charge or type-index caching at the
j_tile level) via the existing
`shared_memory_per_warp_bytes` field on `PairForceFragment`.

## Per-Step Pipeline <!-- rq-36b771fa -->

Per timestep, the fast-class pair-force pipeline runs the
following kernels in order on the default stream:

| Order | Step | Kernel | Operation |
| --- | --- | --- | --- |
| 1 | Displacement check | `neighbor_displacement_squared` | (only on neighbour-list rebuild years) |
| 2 | Cell-list refresh | (per neighbour-list rebuild) | as in `rqm/forces/neighbor-list.md` |
| 3 | Tile metadata | `compute_tile_metadata` | (rebuild only) writes `tile_atom_count`, `tile_lane_mask` |
| 4 | Tile bounding boxes | `compute_tile_bounding_boxes` | (rebuild only) writes `tile_bboxes` |
| 5 | Tile-pair count | `tile_pair_candidate_count` | (rebuild only) writes `tile_candidate_counts`; applies bounding-box pruning |
| 6 | Tile-pair offset scan | prefix scan + `tile_pair_finalize_offsets` | (rebuild only) writes `tile_pair_offsets[0..N_tiles]` and the trailing `tile_pair_count` sentinel |
| 7 | Tile-pair emit | `tile_pair_emit` | (rebuild only) writes `tile_pair_list`; applies the same bounding-box pruning as step 5 |
| 8 | Tile-pair masks | `compute_tile_pair_masks` | (rebuild only) writes `tile_pair_masks`; one 32-bit j-atom interaction mask per tile-pair entry |
| 9 | Tile-sorted position scatter | `scatter_positions_to_tile_order` | every step; refreshes `tile_sorted_positions_x/y/z` from current `positions_*` via `sorted_particle_ids` |
| 10 | Slot accumulator memset | `class_accumulator_memset` | zeros per-slot output buffers |
| 11 | JIT composed pair force | `heddle_jit_composed_pair_force_*` | per-tile pair-force evaluation, one launch per step; reads positions exclusively from the tile-sorted view |
| 12 | Combine class totals | `combine_class_totals` | sums per-class outputs into ParticleBuffers.forces |

Steps 3–8 only run when the neighbour list rebuilds (skin-distance
trigger fires); the cadence is controlled by `r_skin` and matches
the existing rebuild cadence specified in
`rqm/forces/neighbor-list.md`. Step 9 (the position scatter) runs
every step; it depends on the current particle positions which the
integrator updates each step but does not depend on the
neighbour-list rebuild cadence.

## Determinism Invariants <!-- rq-bdf2b3d4 -->

This architecture preserves all three load-bearing invariants from
`architecture.md`:

1. **Deterministic neighbour lists.** The tile-pair list is
   constructed in deterministic order (sublists ascending in
   `i_tile`, entries ascending in `j_tile`, self-pair first). The
   cell-list sort underneath is unchanged and remains
   deterministic.

2. **No atomic float accumulation.** Each atom's force / energy /
   virial is written by exactly one lane in exactly one warp at
   the end of that warp's tile-pair sublist iteration. No
   `atomicAdd`. No cross-warp force sharing.

3. **Fixed-topology in-register accumulation.** Each lane's
   register accumulator sees a fixed-order sequence of
   in-cutoff contributions: tile-pair sublist iteration order is
   fixed; within each tile-pair the j_atom inner loop runs `m =
   0 .. 31` in fixed order; the per-pair functor evaluates
   identically across runs. The accumulation order depends only
   on the tile-pair list contents, not on thread scheduling.

The "two GPU runs are byte-identical" invariant from
`rqm/pipeline-reproducibility.md` holds for tile-based fast-class
slots under the existing reproducibility test suite.

## Configuration <!-- rq-797c961d -->

`[forces].pair_force_kernel = "tile"` is the default and only
supported value. No alternative kernel variants coexist. The
config knob is reserved for future variants but currently has
exactly one accepted value.

`[neighbor_list].max_neighbors` is rejected at config load with
`ConfigError::DeprecatedField` describing that the parameter is no
longer accepted and pointing the user at the tile-based
auto-sizing of `tile_pair_list`.

`[neighbor_list].tile_pair_growth_factor: f32` (optional, default
`1.5`) — multiplier applied when the tile-pair list needs to grow.
Must be `> 1.0` and `≤ 4.0`. Larger values reduce reallocation
frequency at the cost of memory.

`[neighbor_list].tile_pair_initial_capacity_per_tile: u32`
(optional, default `64`) — initial allocation hint: the tile-pair
list is sized to `N_tiles * tile_pair_initial_capacity_per_tile` at
construction, before the first rebuild. Sized generously to avoid
the first rebuild triggering a growth. Tuning this knob is rarely
needed; the auto-grow path handles undersizing.

## Performance Constraints <!-- rq-24ee55f5 -->

The tile-based architecture targets the bandwidth profile of
shared-memory-cached j_tile position reuse. The pair-force kernel
on a 600 GB/s GPU should achieve:

- Direct-space pair force time per step `≤ 0.5 ×` the
  warp-per-particle equivalent on the SPC water 8192 benchmark
  (`r_cut = 10 Å`, density ≈ 100 atoms / nm³).
- Tile-pair list footprint `≤ 100 KB` for any system with
  `N ≤ 100,000` atoms at typical liquid density.
- Neighbour-list rebuild time per rebuild not regressing more than
  10 % vs the per-particle build; the additional tile-pair
  construction kernels offset the simpler in-cell logic.

These targets are validated by the benchmark suite, not by the
spec. Failure to meet a target signals a kernel-tuning issue, not
a spec violation.

## Feature API <!-- rq-0dc1ed40 -->

### Types <!-- rq-6a4fdaa8 -->

- `NeighborListState` — extended with the following fields: <!-- rq-e6577463 -->
  - `tile_atom_count: CudaSlice<u32>` of length `N_tiles`.
  - `tile_lane_mask: CudaSlice<u32>` of length `N_tiles`.
  - `tile_bboxes: CudaSlice<Real>` of length `6 * N_tiles`.
  - `tile_candidate_counts: CudaSlice<u32>` of length `N_tiles`
    (scratch, written by `tile_pair_candidate_count`).
  - `tile_pair_offsets: CudaSlice<u32>` of length `N_tiles + 1`.
  - `tile_pair_list: CudaSlice<u32>` of length `total_pairs_capacity`.
  - `tile_pair_masks: CudaSlice<u32>` of length `total_pairs_capacity`
    (indexed parallel to `tile_pair_list`; one 32-bit j-atom
    interaction mask per tile-pair entry).
  - `tile_pair_count: u32` — current `total_pairs` value (read by
    the pair-force launcher to determine grid size).
  - `tile_sorted_positions_x: CudaSlice<Real>` of length
    `particle_count` — tile-sorted view of `positions_x`,
    refreshed per step by `scatter_positions_to_tile_order`.
  - `tile_sorted_positions_y: CudaSlice<Real>` of length
    `particle_count` — tile-sorted view of `positions_y`.
  - `tile_sorted_positions_z: CudaSlice<Real>` of length
    `particle_count` — tile-sorted view of `positions_z`.

- `NeighborListError` — additional variant: <!-- rq-1376625d -->
  - `TilePairListGrowFailed { requested: u32, available: u32 }` —
    a grow attempt could not allocate the requested capacity. The
    simulation halts; the error reports the requested and
    available sizes.

- `PairForceFragment` — unchanged shape; the <!-- rq-f06621ae -->
  `shared_memory_per_warp_bytes` field is now consulted by the
  composer for j_tile-cache allocation.

### CUDA kernels <!-- rq-64a56174 -->

The following kernels are declared in `kernels/neighbor.cu` and
loaded into the device's neighbour-list PTX module:

- `compute_tile_metadata` <!-- rq-1743db8b -->
- `compute_tile_bounding_boxes` <!-- rq-1a5967a8 -->
- `tile_pair_candidate_count` <!-- rq-8211ff8c -->
- `tile_pair_finalize_offsets` <!-- rq-ec54fd16 -->
- `tile_pair_emit` <!-- rq-917d2f0a -->
- `compute_tile_pair_masks` <!-- rq-f18f5955 -->
- `scatter_positions_to_tile_order` <!-- rq-b78cf2ce -->

The `tile_pair_bbox_prune` device helper (also in
`kernels/neighbor.cu`) implements the bounding-box pruning
predicate consumed by both `tile_pair_candidate_count` and
`tile_pair_emit`. The predicate computes the minimum-image
displacement between the two tiles' centers, subtracts the
per-axis half-extents of each tile, clamps to zero, sums the
squares, and compares against `r_search_sq = (r_cut + r_skin)²`.
It returns `true` when no atom pair across the two tiles can be
within the search radius. The implementation is exact for
orthorhombic boxes and conservative (no false negatives) for
triclinic boxes.

The JIT-composed pair-force kernel
(`heddle_jit_composed_pair_force_f` /
`heddle_jit_composed_pair_force_fev`) is composed at runtime; the
composer source in `src/forces/jit_composed.rs` carries the
tile-based `OUTER_LOOP_TEMPLATE`.

### Composer entry points <!-- rq-bb349328 -->

The composer's tile-based outer loop template (`OUTER_LOOP_TEMPLATE`
in `src/forces/jit_composed.rs`) is the canonical tile-iteration
implementation. It is referenced as a `&'static str` and is
included verbatim in the JIT source.

## Out of Scope <!-- rq-b5c55e6b -->

- **Atomic-based tile-pair force kernel.** A non-deterministic
  variant that uses `atomicAdd` to share Newton's-third-law
  results between warps is not in scope. The deterministic
  invariant from `architecture.md` is load-bearing and forbids
  this optimisation. If a future user-facing fast path is needed,
  it lives in a separate config knob and a separate kernel
  variant.

- **Configurable tile size.** Tile size is fixed at 32, matching
  the CUDA warp size. The per-lane-owns-one-home-atom design
  requires the tile size to equal the warp size.

- **Per-particle padded neighbour list.** No fast-class slot uses a
  per-particle padded neighbour list. The `max_neighbors` config
  parameter is rejected. The `neighbor_list: CudaSlice<u32>` and
  `neighbor_counts: CudaSlice<u32>` buffers are not allocated for
  fast-class slots; if the bonded or angle pipelines retain
  per-particle data structures, those live in their own
  pipelines.

- **Bonded / angle force pipelines.** The tile-based architecture
  applies to fast-class pair forces only. Bonded and angle slots
  continue to use their per-bond / per-angle data structures
  documented in `rqm/forces/jit-composed-intramolecular.md`.

- **Cell-list construction.** The cell-list construction kernels
  (`compute_cell_indices_and_histogram`, prefix scans,
  `scatter_atoms_into_cells`, `sort_cells_by_particle_id`) are
  unchanged; tile assignment piggybacks on the cell-sorted order.

- **SPME reciprocal pipeline.** The SPME reciprocal pipeline (FFT
  spread → R2C FFT → influence multiply → C2R FFT → gather)
  operates on its own per-particle data structures and is
  unaffected by this architecture.

## Gherkin Scenarios <!-- rq-b5961deb -->

```gherkin
Feature: Tile-based pair-force architecture

  Background:
    Given a CUDA-capable GPU available as device 0
    And init_device() has been called
    And the tile size is fixed at 32

  # --- Tile metadata ---

  @rq-54f37d27
  Scenario: Full tile carries lane mask 0xFFFFFFFF
    Given a system with N = 256 atoms (8 full tiles)
    When NeighborListState::rebuild completes
    Then tile_atom_count is [32, 32, 32, 32, 32, 32, 32, 32]
    And tile_lane_mask is [0xFFFFFFFF, 0xFFFFFFFF, …, 0xFFFFFFFF]

  @rq-5e1c8bc6
  Scenario: Partial last tile carries truncated lane mask
    Given a system with N = 100 atoms
    When NeighborListState::rebuild completes
    Then N_tiles equals 4
    And tile_atom_count is [32, 32, 32, 4]
    And tile_lane_mask is [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0000000F]

  # --- Tile-pair list ordering ---

  @rq-640519bc
  Scenario: Per-tile sublists are sorted by ascending i_tile
    Given a system that yields tile-pair list with N_tiles = 4
    When NeighborListState::rebuild completes
    Then tile_pair_offsets is strictly non-decreasing
    And tile_pair_offsets[0] equals 0
    And tile_pair_offsets[N_tiles] equals tile_pair_count

  @rq-2606475c
  Scenario: Sublist entries within a tile are sorted ascending in j_tile
    Given a tile t with at least three neighbour tiles
    When NeighborListState::rebuild completes
    Then tile_pair_list[tile_pair_offsets[t] .. tile_pair_offsets[t+1]] is
      sorted ascending
    And the self-pair (t, t) appears at the smallest position in t's sublist
      whenever any tile has j_tile == t in its sublist

  @rq-0ab46b09
  Scenario: Two GPU runs from identical state produce byte-identical tile-pair lists
    Given a system with byte-identical initial state across two runs
    When NeighborListState::rebuild completes in both runs
    Then tile_atom_count compares byte-identical
    And tile_lane_mask compares byte-identical
    And tile_pair_offsets compares byte-identical
    And tile_pair_list compares byte-identical

  # --- Pair-force kernel determinism ---

  @rq-73c71f43
  Scenario: Two GPU runs produce byte-identical per-particle forces
    Given a system with byte-identical initial state across two runs
    When the JIT-composed pair-force kernel runs one step in both runs
    Then per-particle slot_force_x, slot_force_y, slot_force_z compare byte-identical
    And per-particle slot_energy compares byte-identical (under WriteEv)
    And per-particle slot_virial compares byte-identical (under WriteEv)

  @rq-d425ef81
  Scenario: No atomicAdd appears in the composed pair-force kernel source
    Given a JIT-composed pair-force kernel built from any combination of
      Lennard-Jones, Coulomb, and SPME-real fragments
    When the composer assembles the kernel source
    Then the assembled source contains no atomicAdd call
    And the assembled source contains no atomic_add or __any_sync atomic call

  @rq-d017a73f
  Scenario: Per-particle force accumulator has exactly one writer warp lane
    Given a particle p with home tile t and tile-lane offset l
    When the JIT-composed pair-force kernel runs
    Then exactly one warp processes tile t
    And exactly lane l in that warp writes slot_force_x[p], slot_force_y[p],
      slot_force_z[p], slot_energy[p], slot_virial[p]
    And no other warp lane in the kernel launch writes to those addresses

  # --- Inactive lane handling ---

  @rq-8c9ee1bb
  Scenario: Inactive lane in partial last tile writes nothing
    Given N = 100 atoms (last tile has 4 real atoms)
    When the JIT-composed pair-force kernel runs
    Then slot_force_x[100..128] is not written by the kernel
      (those addresses do not exist; the slot-output buffer is sized to N = 100)
    And lanes 4..31 of the warp owning tile 3 do not contribute to any
      slot-output write

  @rq-9997e2b4
  Scenario: Inactive lane contribution is masked from accumulation
    Given a tile-pair (i_tile, j_tile) where j_tile has 4 real atoms
    When the warp processes the inner loop m = 0..31
    Then per-pair functor evaluations for m >= 4 do not accumulate into
      any lane's register accumulator
    And lanes l where l >= tile_atom_count[i_tile] do not accumulate any
      m's contribution

  # --- Self-pair handling ---

  @rq-12b51aed
  Scenario: Self-pair (t, t) skips the diagonal (m == l)
    Given a tile t with 32 real atoms
    When the warp owning tile t processes the self-pair (t, t)
    Then for every lane l, the inner-loop iteration m == l is skipped
      (an atom does not interact with itself)
    And the remaining 31 pair evaluations per lane accumulate normally

  # --- Tile-pair list growth ---

  @rq-aae9187d
  Scenario: tile_pair_growth_factor = 1.5 triggers growth on overflow
    Given tile_pair_growth_factor = 1.5 and an initial allocation that
      exactly fits the first rebuild's tile-pair count
    When a subsequent rebuild produces 10 % more tile pairs
    Then the tile_pair_list is grown to 1.5 × the new requested capacity
    And the rebuild is re-issued on the larger allocation
    And the simulation continues without halting

  @rq-04d49dee
  Scenario: tile_pair_growth_factor below 1.0 rejected at config load
    When a config sets tile_pair_growth_factor = 0.8
    Then config load returns ConfigError::InvalidValue with field
      "neighbor_list.tile_pair_growth_factor" and reason "value must be > 1.0, got 0.8"

  # --- Configuration ---

  @rq-4248b662
  Scenario: max_neighbors config field is rejected at load
    When a config sets neighbor_list.max_neighbors = 1024
    Then config load returns ConfigError::DeprecatedField with field
      "neighbor_list.max_neighbors" and reason pointing at the tile-based
      auto-sizing

  @rq-af02d035
  Scenario: pair_force_kernel = "warp_per_particle" rejected at load
    When a config sets forces.pair_force_kernel = "warp_per_particle"
    Then config load returns ConfigError::InvalidValue with field
      "forces.pair_force_kernel" and reason "only \"tile\" is supported"

  # --- Integration with the JIT composer ---

  @rq-e78cbbad
  Scenario: Composed kernel emits one launch per step
    Given a phase with active Lennard-Jones, Coulomb, and SPME-real slots
    When the runner runs one timestep
    Then the JIT-composed pair-force kernel is launched exactly once
    And the launch records under KernelStage::JIT_COMPOSED_PAIR_FORCE
    And no per-slot standalone pair-force kernel is launched

  @rq-3901b276
  Scenario: PairForceFragment is unchanged across the migration
    Given an out-of-tree pair-force slot that compiled against the
      pre-migration PairForceFragment shape (functor_struct_name,
      functor_source, entry_point_args, functor_init_source)
    When the slot rebuilds against the migrated codebase
    Then the slot compiles and links without source changes
    And the slot produces the same per-particle forces (within f32 round-off)
      as on the warp-per-particle composer

  @rq-3a9c7455
  Scenario: Composed kernel respects shared_memory_per_warp_bytes per slot
    Given two active fragments, the first requesting 64 bytes of
      per-warp shared memory and the second requesting 128 bytes
    When the composer assembles the kernel
    Then the launch shared-memory configuration reserves at least
      (32 × 12 + 64 + 128) = 576 bytes per warp
    And kernel launch succeeds without out-of-shared-memory error

  # --- Tile-pair list correctness ---

  @rq-9dfcd786
  Scenario: Tile-pair list includes every pair (a, b) with r_ab <= r_cut + r_skin
    Given a system where atom a in tile t_a and atom b in tile t_b
      satisfy r_ab <= r_cut + r_skin
    When NeighborListState::rebuild completes
    Then (t_a, t_b) appears in tile_pair_list (in t_a's sublist)
    And (t_b, t_a) appears in tile_pair_list (in t_b's sublist)
      (i.e. the bidirectional inclusion mirrors the per-particle case)

  @rq-d6e24a47
  Scenario: Tile-pair list excludes obviously distant tiles
    Given two tiles t_a and t_b whose bounding-box centers are >= 3 * (r_cut + r_skin)
      apart and whose bounding boxes do not overlap within (r_cut + r_skin)
    When NeighborListState::rebuild completes
    Then (t_a, t_b) does NOT appear in t_a's tile_pair_list sublist
    And (t_b, t_a) does NOT appear in t_b's tile_pair_list sublist

  @rq-aa870e92
  Scenario: Bounding-box false positives are tolerated by the cutoff check
    Given a tile-pair (t_a, t_b) that the bounding-box pruning includes
      but where no actual atom-atom pair has r <= r_cut
    When the JIT-composed pair-force kernel processes the pair
    Then no force, energy, or virial is accumulated for any (a, b) pair
      across t_a × t_b
    And the kernel completes without error

  # --- Home-cell dedup ---

  @rq-3c94d356
  Scenario: A tile whose 32 atoms occupy 3 cells sweeps exactly 3 home cells
    Given an i_tile whose 32 atoms occupy cells {c_1, c_2, c_3}
      (each cell holds at least one of the tile's atoms; no other cells do)
    When tile_pair_candidate_count runs on the i_tile
    Then the kernel's unique-home-cell list contains exactly {c_1, c_2, c_3}
    And each of c_1, c_2, c_3 is swept exactly once via the 27-cell neighbour
      enumeration

  @rq-4f24333d
  Scenario: A tile whose 32 atoms occupy 1 cell sweeps exactly 1 home cell
    Given an i_tile whose 32 atoms all lie in cell c_1
    When tile_pair_candidate_count runs on the i_tile
    Then the kernel's unique-home-cell list contains exactly {c_1}
    And c_1's 27-cell neighbourhood is swept exactly once

  @rq-5a2609e6
  Scenario: Unique home cells appear in ascending lane-index order
    Given an i_tile whose 32 atoms occupy cells [c_a, c_a, c_b, c_a, c_b, c_a, ...]
      (cells repeated, c_a first appears at lane 0, c_b first appears at lane 2)
    When tile_pair_candidate_count runs on the i_tile
    Then the kernel's unique-home-cell list is [c_a, c_b] in that order
      (lane 0 contributes c_a; lane 2 contributes c_b; later lanes are dedup'd)

  @rq-9008c6f7
  Scenario: Bitmask produced by candidate_count and emit kernels is byte-identical
    Given any i_tile in a non-empty system
    When tile_pair_candidate_count and tile_pair_emit each run on the i_tile
    Then the populated bitmask is byte-identical across the two kernels
    And the number of set bits popcounted by candidate_count equals the
      number of indices walked by emit

  @rq-16fea30b
  Scenario: Inactive lanes contribute no home cells
    Given a partial last tile with k < 32 real atoms
    When tile_pair_candidate_count runs
    Then the unique-home-cell list is derived only from the cells of the k
      real atoms
    And lanes [k..32) contribute the sentinel cell value that is ignored by
      the dedup pass

  # --- Per-tile-pair masks ---

  @rq-c0bd92d5
  Scenario: Self-pair mask equals the j_tile's lane mask
    Given any i_tile t with tile_atom_count[t] = k
    When compute_tile_pair_masks runs
    Then the mask written for the self-pair entry (t, t) is (1u << k) - 1
      (every real lane bit set; inactive lanes cleared)
    And no per-atom distance computation is performed for the self-pair

  @rq-e4f5845f
  Scenario: Inactive j_tile lanes have their mask bits cleared
    Given a tile-pair (i_tile, j_tile) where j_tile has only k < 32 real atoms
    When compute_tile_pair_masks runs on this entry
    Then bits [k..32) of the written mask are zero
    And the pair-force kernel iterating set bits never visits an inactive
      j_atom

  @rq-3ec4f5aa
  Scenario: A mask bit is set iff its j_atom is within r_search of some i_atom
    Given a tile-pair (i_tile, j_tile) and a j_atom at j-lane k
    When the j_atom is within r_cut + r_skin of at least one atom in i_tile
      under minimum image
    Then bit k of tile_pair_masks[entry_index] is set after
      compute_tile_pair_masks runs
    And conversely, when no i_atom is within r_cut + r_skin of the j_atom,
      bit k is cleared

  @rq-a3f54c5f
  Scenario: Two GPU runs produce byte-identical tile_pair_masks
    Given a system with byte-identical initial state across two runs
    When NeighborListState::rebuild completes in both runs
    Then tile_pair_masks compares byte-identical across the two runs
    And the per-particle slot-output forces produced by the pair-force kernel
      compare byte-identical

  @rq-c5e40c8a
  Scenario: Pair-force kernel inner loop visits only set mask bits
    Given a tile-pair (i_tile, j_tile) whose mask has set bits {2, 7, 13}
      and tile_atom_count[j_tile] = 32
    When the JIT-composed pair-force kernel processes the entry
    Then the per-pair inner loop body executes exactly 3 times
      (once for each set bit)
    And j_atoms at j-lanes 0, 1, 3..6, 8..12, 14..31 are never read by the
      inner loop

  @rq-ba20ac7f
  Scenario: Pair-force kernel produces forces consistent with mask-free reference
    Given the same tile_pair_list contents and the masks computed by
      compute_tile_pair_masks
    When run A executes the JIT-composed pair-force kernel with mask-driven
      inner loop
    And run B executes a reference kernel that iterates 0..32 without masks
      (but performs the same r_cut check)
    Then per-particle forces, energies, and virials produced by A and B agree
      bit-for-bit
      (j_atoms outside r_cut + r_skin in B's iteration accumulate zero contribution;
       the mask in A pre-filters those same j_atoms)

  # --- Tile-sorted position scatter ---

  @rq-913c85d3
  Scenario: tile_sorted_positions reflects the post-integration positions
    Given a simulation step that has just completed integration
      (positions_*[pid] are updated in place)
    When scatter_positions_to_tile_order runs at the start of ForceField::step
    Then tile_sorted_positions_x[k] equals positions_x[sorted_particle_ids[k]]
      for every k in [0, particle_count)
    And the same equality holds for y and z

  @rq-a46cbf6b
  Scenario: Two GPU runs produce byte-identical tile_sorted_positions
    Given a system with byte-identical initial state across two runs
    When ForceField::step runs once in each
    Then tile_sorted_positions_x, _y, _z compare byte-identical across
      the two runs

  @rq-537c2c98
  Scenario: Pair-force kernel reads positions exclusively from the tile-sorted view
    Given a JIT-composed pair-force kernel built from any combination of
      Lennard-Jones, Coulomb, and SPME-real fragments
    When the composer assembles the kernel source
    Then the assembled source loads home and j_tile positions via
      tile_sorted_positions_x, _y, _z (indexed by `i_tile * 32 + lane` or
      `j_tile * 32 + lane`)
    And the assembled source does not read positions_x, positions_y, or
      positions_z directly

  @rq-4a5740be
  Scenario: Scatter is launched once per step regardless of rebuild cadence
    Given a simulation that runs 10 steps with neighbour-list rebuild every 5 steps
    When the simulation completes
    Then KernelStage::SCATTER_POSITIONS_TO_TILE_ORDER records exactly 10 launches
    And the launch count is independent of the rebuild cadence

  @rq-84057105
  Scenario: Pair-force results are bit-identical to a non-scattered reference
    Given an i_tile and j_tile pair with their atom positions
    When run A executes the JIT-composed pair-force kernel reading positions
      from tile_sorted_positions_*[k]
    And run B executes a reference kernel reading positions from
      positions_*[sorted_particle_ids[k]] (the gather pattern that the
      scatter eliminates)
    Then per-particle forces, energies, and virials produced by A and B
      agree bit-for-bit (the scatter writes the same bytes the gather
      would have read)

  # --- Per-step pipeline ---

  @rq-4882dc40
  Scenario: Tile metadata kernels run only on neighbour-list rebuild
    Given a simulation that runs 10 steps with neighbour-list rebuild every 5 steps
    When the simulation completes
    Then KernelStage::COMPUTE_TILE_METADATA records exactly 2 launches
    And KernelStage::COMPUTE_TILE_BOUNDING_BOXES records exactly 2 launches
    And KernelStage::TILE_PAIR_CANDIDATE_COUNT records exactly 2 launches
    And KernelStage::TILE_PAIR_EMIT records exactly 2 launches
    And KernelStage::COMPUTE_TILE_PAIR_MASKS records exactly 2 launches
    And KernelStage::JIT_COMPOSED_PAIR_FORCE records exactly 10 launches
    And KernelStage::SCATTER_POSITIONS_TO_TILE_ORDER records exactly 10 launches
```
