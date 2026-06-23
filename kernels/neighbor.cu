// rq-0469400b

#include "precision.cuh"

#include "pbc.cuh"

// Compute the parallelepiped cell index of a Cartesian position. Wraps
// the position into the primary image, transforms to fractional
// coordinates, and bins each fractional component to [0, n_cells_d - 1]
// (clamping handles the +0.5 boundary case).
__device__ static inline void parallelepiped_cell_indices(
    Real x, Real y, Real z,
    const Real *lattice,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int &ca, unsigned int &cb, unsigned int &cc)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  int dummy_a, dummy_b, dummy_c;
  triclinic_wrap_with_image(x, y, z, dummy_a, dummy_b, dummy_c,
                            lx, ly, lz, xy, xz, yz);
  Real s_a, s_b, s_c;
  triclinic_cart_to_frac(x, y, z, lx, ly, lz, xy, xz, yz, s_a, s_b, s_c);
  int ia = (int) Real_floor((s_a + R(0.5)) * (Real) n_cells_a);
  int ib = (int) Real_floor((s_b + R(0.5)) * (Real) n_cells_b);
  int ic = (int) Real_floor((s_c + R(0.5)) * (Real) n_cells_c);
  if (ia < 0) ia = 0;
  if (ia >= (int) n_cells_a) ia = (int) n_cells_a - 1;
  if (ib < 0) ib = 0;
  if (ib >= (int) n_cells_b) ib = (int) n_cells_b - 1;
  if (ic < 0) ic = 0;
  if (ic >= (int) n_cells_c) ic = (int) n_cells_c - 1;
  ca = (unsigned int) ia;
  cb = (unsigned int) ib;
  cc = (unsigned int) ic;
}

// rq-884b5cd6
extern "C" __global__ void neighbor_displacement_squared(
    const Real *positions_x, const Real *positions_y, const Real *positions_z,
    const Real *reference_x, const Real *reference_y, const Real *reference_z,
    const Real *lattice,
    Real *disp_sq,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  Real dx = positions_x[i] - reference_x[i];
  Real dy = positions_y[i] - reference_y[i];
  Real dz = positions_z[i] - reference_z[i];
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
  disp_sq[i] = dx * dx + dy * dy + dz * dz;
}

// rq-a1262872
//
// One block per home cell. The block's threads cooperate so each of the
// 27 neighbour cells' positions is loaded from global memory exactly
// once per block (tiled through dynamic shared memory in chunks of
// blockDim.x candidates) and amortised across all home-cell atoms. Each
// thread owns one home-cell atom's neighbour list and walks the
// shared-memory candidates in cell-sweep order — `(da, db, dc)` lex
// outer-to-inner, particle-ID ascending within each cell. No trailing
// per-atom sort.
//
// Dynamic shared memory layout (in bytes), set at launch:
//   shared_x : Real[blockDim.x]
//   shared_y : Real[blockDim.x]
//   shared_z : Real[blockDim.x]
//   shared_id: unsigned int[blockDim.x]
// Total = 4 * blockDim.x * sizeof(Real).
extern "C" __global__ void neighbor_list_build(
    const Real *positions_x, const Real *positions_y, const Real *positions_z,
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_offsets,
    const Real *lattice,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    Real r_search_sq,
    unsigned int max_neighbors,
    unsigned int *neighbor_list,
    unsigned int *neighbor_counts,
    unsigned int *overflow_flag,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  (void) n;

  extern __shared__ unsigned char smem[];
  Real *shared_x = reinterpret_cast<Real *>(smem);
  Real *shared_y = shared_x + blockDim.x;
  Real *shared_z = shared_y + blockDim.x;
  unsigned int *shared_id =
      reinterpret_cast<unsigned int *>(shared_z + blockDim.x);

  unsigned int home_cell = blockIdx.x;
  unsigned int total_cells = n_cells_a * n_cells_b * n_cells_c;
  if (home_cell >= total_cells) {
    return;
  }

  unsigned int home_start = cell_offsets[home_cell];
  unsigned int home_end = cell_offsets[home_cell + 1u];
  unsigned int home_count = home_end - home_start;
  if (home_count == 0u) {
    return;
  }

  // Decode home cell into (ca, cb, cc) using the same row-major
  // convention as the cell-index/histogram kernel:
  //   home_cell = (ca * n_cells_b + cb) * n_cells_c + cc
  unsigned int ca = home_cell / (n_cells_b * n_cells_c);
  unsigned int rem = home_cell - ca * (n_cells_b * n_cells_c);
  unsigned int cb = rem / n_cells_c;
  unsigned int cc = rem - cb * n_cells_c;

  // Iterate home-cell atoms in chunks of blockDim.x so a single block
  // can service arbitrarily dense cells.
  for (unsigned int home_off = 0u; home_off < home_count;
       home_off += blockDim.x) {
    unsigned int thread_atom = home_off + threadIdx.x;
    bool active = (thread_atom < home_count);

    unsigned int i = 0u;
    Real xi = R(0.0), yi = R(0.0), zi = R(0.0);
    if (active) {
      i = sorted_particle_ids[home_start + thread_atom];
      xi = positions_x[i];
      yi = positions_y[i];
      zi = positions_z[i];
    }
    unsigned int count = 0u;
    unsigned int overflowed = 0u;

    // 27-cell sweep: a outer, b middle, c inner.
    for (int da = -1; da <= 1; ++da) {
      int nca = (int) ca + da;
      while (nca < 0) { nca += (int) n_cells_a; }
      while (nca >= (int) n_cells_a) { nca -= (int) n_cells_a; }
      for (int db = -1; db <= 1; ++db) {
        int ncb = (int) cb + db;
        while (ncb < 0) { ncb += (int) n_cells_b; }
        while (ncb >= (int) n_cells_b) { ncb -= (int) n_cells_b; }
        for (int dc = -1; dc <= 1; ++dc) {
          int ncc = (int) cc + dc;
          while (ncc < 0) { ncc += (int) n_cells_c; }
          while (ncc >= (int) n_cells_c) { ncc -= (int) n_cells_c; }

          unsigned int c_neigh =
              ((unsigned int) nca * n_cells_b + (unsigned int) ncb)
              * n_cells_c + (unsigned int) ncc;
          unsigned int n_start = cell_offsets[c_neigh];
          unsigned int n_end = cell_offsets[c_neigh + 1u];

          // Stream candidates through shared memory in chunks. Each
          // chunk fits exactly one shared-memory tile of blockDim.x
          // candidates; cells with > blockDim.x atoms span multiple
          // chunks. The outer-to-inner cell order and the in-cell
          // sorted_particle_ids order together pin the neighbour append
          // order.
          for (unsigned int chunk_base = n_start; chunk_base < n_end;
               chunk_base += blockDim.x) {
            unsigned int chunk_size = n_end - chunk_base;
            if (chunk_size > blockDim.x) {
              chunk_size = blockDim.x;
            }

            __syncthreads();
            if (threadIdx.x < chunk_size) {
              unsigned int j = sorted_particle_ids[chunk_base + threadIdx.x];
              shared_id[threadIdx.x] = j;
              shared_x[threadIdx.x] = positions_x[j];
              shared_y[threadIdx.x] = positions_y[j];
              shared_z[threadIdx.x] = positions_z[j];
            }
            __syncthreads();

            if (active) {
              for (unsigned int k = 0u; k < chunk_size; ++k) {
                unsigned int j = shared_id[k];
                if (j == i) {
                  continue;
                }
                Real ddx = xi - shared_x[k];
                Real ddy = yi - shared_y[k];
                Real ddz = zi - shared_z[k];
                triclinic_min_image(ddx, ddy, ddz,
                                    lx, ly, lz, xy, xz, yz);
                Real r2 = ddx * ddx + ddy * ddy + ddz * ddz;
                if (r2 <= r_search_sq) {
                  if (count < max_neighbors) {
                    neighbor_list[(size_t) i * (size_t) max_neighbors
                                  + count] = j;
                    count += 1u;
                  } else {
                    overflowed = 1u;
                  }
                }
              }
            }
          }
        }
      }
    }

    if (active) {
      neighbor_counts[i] = count;
      if (overflowed) {
        atomicOr(overflow_flag, 1u);
      }
    }

    // Barrier before the next home_off iteration overwrites shared
    // memory while another wave of threads might still be reading it.
    __syncthreads();
  }
}

// rq-344f7af0
extern "C" __global__ void copy_positions_into_reference(
    const Real *positions_x, const Real *positions_y, const Real *positions_z,
    Real *reference_x, Real *reference_y, Real *reference_z,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  reference_x[i] = positions_x[i];
  reference_y[i] = positions_y[i];
  reference_z[i] = positions_z[i];
}

#define SCAN_BLOCK_SIZE 256u

extern "C" __global__ void compute_cell_indices_and_histogram(
    const Real *positions_x, const Real *positions_y, const Real *positions_z,
    const Real *lattice,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    unsigned int *cell_indices,
    unsigned int *cell_counts,
    unsigned int n)
{
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int ca, cb, cc;
  parallelepiped_cell_indices(positions_x[i], positions_y[i], positions_z[i],
                              lattice,
                              n_cells_a, n_cells_b, n_cells_c,
                              ca, cb, cc);
  unsigned int c = (ca * n_cells_b + cb) * n_cells_c + cc;
  cell_indices[i] = c;
  atomicAdd(&cell_counts[c], 1u);
}

// Per-block exclusive Hillis-Steele scan of input[0..len] into
// output[0..len], writing each block's inclusive total to
// block_totals[blockIdx]. Each thread reads its input element into a
// register before any global write, and blocks write disjoint output
// ranges, so `input` may alias `output` — the recursive scan driver
// relies on this to scan each block-totals level of the stack in place.
extern "C" __global__ void prefix_scan_local_blocks(
    const unsigned int *input,
    unsigned int *output,
    unsigned int *block_totals,
    unsigned int len)
{
  __shared__ unsigned int temp[2u * SCAN_BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + t;
  unsigned int my_input = (gid < len) ? input[gid] : 0u;
  unsigned int pout = 0u;
  unsigned int pin = 1u;
  temp[pout * SCAN_BLOCK_SIZE + t] = my_input;
  __syncthreads();
  for (unsigned int offset = 1u; offset < SCAN_BLOCK_SIZE; offset *= 2u) {
    pout = 1u - pout;
    pin = 1u - pin;
    if (t >= offset) {
      temp[pout * SCAN_BLOCK_SIZE + t] =
          temp[pin * SCAN_BLOCK_SIZE + t]
          + temp[pin * SCAN_BLOCK_SIZE + t - offset];
    } else {
      temp[pout * SCAN_BLOCK_SIZE + t] = temp[pin * SCAN_BLOCK_SIZE + t];
    }
    __syncthreads();
  }
  unsigned int inclusive = temp[pout * SCAN_BLOCK_SIZE + t];
  unsigned int exclusive = inclusive - my_input;
  if (gid < len) {
    output[gid] = exclusive;
  }
  if (t == SCAN_BLOCK_SIZE - 1u) {
    block_totals[blockIdx.x] = inclusive;
  }
}

// Generic add-back: output[gid] += block_offsets[gid / SCAN_BLOCK_SIZE]
// for every gid < len.
extern "C" __global__ void prefix_scan_apply_block_totals(
    const unsigned int *block_offsets,
    unsigned int *output,
    unsigned int len)
{
  unsigned int gid = blockIdx.x * SCAN_BLOCK_SIZE + threadIdx.x;
  if (gid < len) {
    output[gid] += block_offsets[blockIdx.x];
  }
}

// Writes the trailing cell_offsets[n_cells_total] = particle_count
// sentinel slot with a single thread.
extern "C" __global__ void prefix_scan_finalize_offsets(
    unsigned int *cell_offsets,
    unsigned int n_cells_total,
    unsigned int particle_count)
{
  if (blockIdx.x == 0u && threadIdx.x == 0u) {
    cell_offsets[n_cells_total] = particle_count;
  }
}

extern "C" __global__ void scatter_atoms_into_cells(
    const unsigned int *cell_indices,
    const unsigned int *cell_offsets,
    unsigned int *write_cursors,
    unsigned int *sorted_particle_ids,
    unsigned int n)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  unsigned int c = cell_indices[i];
  unsigned int slot = atomicAdd(&write_cursors[c], 1u);
  sorted_particle_ids[cell_offsets[c] + slot] = i;
}

extern "C" __global__ void sort_cells_by_particle_id(
    const unsigned int *cell_offsets,
    unsigned int *sorted_particle_ids,
    unsigned int n_cells_total)
{
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells_total) {
    return;
  }
  unsigned int start = cell_offsets[c];
  unsigned int end = cell_offsets[c + 1u];
  for (unsigned int k = start + 1u; k < end; ++k) {
    unsigned int key = sorted_particle_ids[k];
    int pos = (int) k - 1;
    while (pos >= (int) start && sorted_particle_ids[pos] > key) {
      sorted_particle_ids[pos + 1] = sorted_particle_ids[pos];
      pos -= 1;
    }
    sorted_particle_ids[pos + 1] = key;
  }
}

// ============================================================
// Tile-based architecture kernels
//
// See `rqm/forces/tile-based-pair-force.md` for the architectural
// rationale and the data-model definitions referenced below.
// ============================================================

// rq-d8de5e38 — Per-tile axis-aligned bounding box.
//
// One block per tile; 32 threads per block, one per home-tile
// lane. Each lane reads its home atom's position; a warp reduction
// produces the per-axis min and max across the 32 atoms. The
// bounding box is stored as six Real per tile in row-major order:
// `[min_x, min_y, min_z, max_x, max_y, max_z]`.
//
// Lanes beyond the tile's real atom count contribute neutral
// values (`+INFINITY` for min, `-INFINITY` for max) so the
// reduction ignores them.
extern "C" __global__ void compute_tile_bounding_boxes(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *sorted_particle_ids,
    Real *tile_bboxes,
    unsigned int n,
    unsigned int n_tiles)
{
  unsigned int i_tile = blockIdx.x;
  unsigned int tid = threadIdx.x;
  if (i_tile >= n_tiles || tid >= 32u) {
    return;
  }
  unsigned int home_idx = i_tile * 32u + tid;
  Real x, y, z;
  if (home_idx < n) {
    unsigned int pid = sorted_particle_ids[home_idx];
    x = positions_x[pid];
    y = positions_y[pid];
    z = positions_z[pid];
  } else {
    // Inactive lane: contribute neutral identity values.
    x = INFINITY;
    y = INFINITY;
    z = INFINITY;
  }
  Real min_x = (home_idx < n) ? x : INFINITY;
  Real min_y = (home_idx < n) ? y : INFINITY;
  Real min_z = (home_idx < n) ? z : INFINITY;
  Real max_x = (home_idx < n) ? x : -INFINITY;
  Real max_y = (home_idx < n) ? y : -INFINITY;
  Real max_z = (home_idx < n) ? z : -INFINITY;

  // Warp reductions for min and max along each axis. Use
  // __shfl_xor_sync butterfly so all 32 lanes hold the result;
  // lane 0 writes.
  for (int offset = 16; offset > 0; offset >>= 1) {
    Real ox = __shfl_xor_sync(0xFFFFFFFFu, min_x, offset);
    if (ox < min_x) min_x = ox;
    Real oy = __shfl_xor_sync(0xFFFFFFFFu, min_y, offset);
    if (oy < min_y) min_y = oy;
    Real oz = __shfl_xor_sync(0xFFFFFFFFu, min_z, offset);
    if (oz < min_z) min_z = oz;
    Real bx = __shfl_xor_sync(0xFFFFFFFFu, max_x, offset);
    if (bx > max_x) max_x = bx;
    Real by = __shfl_xor_sync(0xFFFFFFFFu, max_y, offset);
    if (by > max_y) max_y = by;
    Real bz = __shfl_xor_sync(0xFFFFFFFFu, max_z, offset);
    if (bz > max_z) max_z = bz;
  }

  if (tid == 0u) {
    unsigned int base = i_tile * 6u;
    tile_bboxes[base + 0u] = min_x;
    tile_bboxes[base + 1u] = min_y;
    tile_bboxes[base + 2u] = min_z;
    tile_bboxes[base + 3u] = max_x;
    tile_bboxes[base + 4u] = max_y;
    tile_bboxes[base + 5u] = max_z;
  }
}

// rq-d8de5e38 — Bounding-box pruning predicate.
//
// Returns true when the two tile bounding boxes are far enough
// apart under minimum image that no atom-atom pair across them can
// be within `r_search`. The closest atom-atom distance is bounded
// below by the per-axis closest-approach distance after subtracting
// each box's half-extent on that axis.
//
// This implementation handles orthorhombic (`xy=xz=yz=0`) PBC
// exactly and triclinic boxes conservatively. Triclinic
// off-diagonal terms are folded into the per-axis bound by
// shrinking the search distance, which produces some false
// positives (extra tile-pairs admitted) but never false negatives
// (real interactions are never pruned).
__device__ static inline bool tile_pair_bbox_prune(
    const Real *bbox_i,
    const Real *bbox_j,
    Real lx, Real ly, Real lz,
    Real xy, Real xz, Real yz,
    Real r_search_sq)
{
  // Centers and half-extents.
  Real ci_x = R(0.5) * (bbox_i[0] + bbox_i[3]);
  Real ci_y = R(0.5) * (bbox_i[1] + bbox_i[4]);
  Real ci_z = R(0.5) * (bbox_i[2] + bbox_i[5]);
  Real cj_x = R(0.5) * (bbox_j[0] + bbox_j[3]);
  Real cj_y = R(0.5) * (bbox_j[1] + bbox_j[4]);
  Real cj_z = R(0.5) * (bbox_j[2] + bbox_j[5]);
  Real hi_x = R(0.5) * (bbox_i[3] - bbox_i[0]);
  Real hi_y = R(0.5) * (bbox_i[4] - bbox_i[1]);
  Real hi_z = R(0.5) * (bbox_i[5] - bbox_i[2]);
  Real hj_x = R(0.5) * (bbox_j[3] - bbox_j[0]);
  Real hj_y = R(0.5) * (bbox_j[4] - bbox_j[1]);
  Real hj_z = R(0.5) * (bbox_j[5] - bbox_j[2]);

  // Minimum-image center-to-center displacement.
  Real dx = ci_x - cj_x;
  Real dy = ci_y - cj_y;
  Real dz = ci_z - cj_z;
  triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);

  // Per-axis closest-approach distance lower bound.
  Real adx = (dx < R(0.0)) ? -dx : dx;
  Real ady = (dy < R(0.0)) ? -dy : dy;
  Real adz = (dz < R(0.0)) ? -dz : dz;
  Real gap_x = adx - hi_x - hj_x;
  Real gap_y = ady - hi_y - hj_y;
  Real gap_z = adz - hi_z - hj_z;
  if (gap_x < R(0.0)) gap_x = R(0.0);
  if (gap_y < R(0.0)) gap_y = R(0.0);
  if (gap_z < R(0.0)) gap_z = R(0.0);
  Real lower_sq = gap_x * gap_x + gap_y * gap_y + gap_z * gap_z;
  return lower_sq > r_search_sq;
}

// rq-d8de5e38 — Writes the trailing
// `tile_pair_offsets[n_tiles] = tile_pair_offsets[n_tiles - 1] +
// tile_candidate_counts[n_tiles - 1]` sentinel after the exclusive
// prefix scan over `tile_candidate_counts`. Single-thread launch.
extern "C" __global__ void tile_pair_finalize_offsets(
    unsigned int *tile_pair_offsets,
    const unsigned int *tile_candidate_counts,
    unsigned int n_tiles)
{
  if (blockIdx.x != 0u || threadIdx.x != 0u) return;
  if (n_tiles == 0u) {
    tile_pair_offsets[0] = 0u;
    return;
  }
  tile_pair_offsets[n_tiles] =
      tile_pair_offsets[n_tiles - 1u] + tile_candidate_counts[n_tiles - 1u];
}

// rq-be571c62 — Writes per-tile metadata.
//
// `tile_atom_count[t]` is 32 for every tile except the last, which
// has (n - (n_tiles - 1) * 32) real atoms. `tile_lane_mask[t]` is
// `(1u << tile_atom_count[t]) - 1` (with full mask 0xFFFFFFFF for
// the 32-count case to avoid the 1u << 32 undefined-behaviour
// corner). One thread per tile; grid sized to ceil(n_tiles / 256).
extern "C" __global__ void compute_tile_metadata(
    unsigned int *tile_atom_count,
    unsigned int *tile_lane_mask,
    unsigned int n,
    unsigned int n_tiles)
{
  unsigned int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_tiles) {
    return;
  }
  unsigned int start = t * 32u;
  unsigned int end = start + 32u;
  if (end > n) {
    end = n;
  }
  unsigned int count = end - start;
  tile_atom_count[t] = count;
  // Avoid the undefined `1u << 32` for full tiles.
  tile_lane_mask[t] = (count == 32u) ? 0xFFFFFFFFu : ((1u << count) - 1u);
}

// rq-d8de5e38 — Per-tile candidate counting (no bounding-box
// pruning).
//
// One block per i_tile; 32 threads per block, one per home-tile
// lane. Each thread iterates the 27 cells adjacent to its home
// atom's cell and marks every j_tile encountered as a candidate in
// a shared-memory bitmask. The self-pair (t, t) is always
// counted. The kernel writes the popcount of the bitmask to
// `tile_candidate_counts[i_tile]`.
//
// `bitmask_words` must be passed by the host as
// `ceil(n_tiles / 32)`. The kernel uses dynamic shared memory of
// size `bitmask_words * sizeof(unsigned int)` for the bitmask.
//
// The bounding-box pruning step described in
// `rqm/forces/tile-based-pair-force.md` is omitted in this
// implementation; bounding-box false positives are tolerated by
// the spec because the pair-force kernel performs the per-atom
// cutoff check.
extern "C" __global__ void tile_pair_candidate_count(
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_indices,
    const unsigned int *cell_offsets,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    const Real *tile_bboxes,
    const Real *lattice,
    Real r_search_sq,
    unsigned int *tile_candidate_counts,
    unsigned int n,
    unsigned int n_tiles,
    unsigned int bitmask_words)
{
  extern __shared__ unsigned int bitmask[];
  unsigned int tid = threadIdx.x;
  unsigned int i_tile = blockIdx.x;
  if (i_tile >= n_tiles) {
    return;
  }
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  const Real *bbox_i = tile_bboxes + i_tile * 6u;

  // Zero the bitmask cooperatively.
  for (unsigned int w = tid; w < bitmask_words; w += blockDim.x) {
    bitmask[w] = 0u;
  }
  __syncthreads();

  // Always include the self-pair (no pruning — atoms within the
  // same tile always interact through the self-pair sublist entry).
  if (tid == 0u) {
    atomicOr(&bitmask[i_tile / 32u], 1u << (i_tile & 31u));
  }
  __syncthreads();

  // Each lane handles one home atom. Lanes beyond the tile's real
  // atom count are inactive.
  unsigned int home_idx = i_tile * 32u + tid;
  if (home_idx < n && tid < 32u) {
    unsigned int particle_id = sorted_particle_ids[home_idx];
    unsigned int cell_a = cell_indices[particle_id];
    unsigned int ca = cell_a / (n_cells_b * n_cells_c);
    unsigned int rem = cell_a - ca * (n_cells_b * n_cells_c);
    unsigned int cb = rem / n_cells_c;
    unsigned int cc = rem - cb * n_cells_c;

    for (int da = -1; da <= 1; ++da) {
      int nca = (int) ca + da;
      while (nca < 0) { nca += (int) n_cells_a; }
      while (nca >= (int) n_cells_a) { nca -= (int) n_cells_a; }
      for (int db = -1; db <= 1; ++db) {
        int ncb = (int) cb + db;
        while (ncb < 0) { ncb += (int) n_cells_b; }
        while (ncb >= (int) n_cells_b) { ncb -= (int) n_cells_b; }
        for (int dc = -1; dc <= 1; ++dc) {
          int ncc = (int) cc + dc;
          while (ncc < 0) { ncc += (int) n_cells_c; }
          while (ncc >= (int) n_cells_c) { ncc -= (int) n_cells_c; }
          unsigned int c_neigh =
              ((unsigned int) nca * n_cells_b + (unsigned int) ncb)
              * n_cells_c + (unsigned int) ncc;
          unsigned int c_start = cell_offsets[c_neigh];
          unsigned int c_end = cell_offsets[c_neigh + 1u];
          for (unsigned int k = c_start; k < c_end; ++k) {
            // k is the sorted-index of the neighbour particle; its
            // tile is k / 32.
            unsigned int j_tile = k / 32u;
            if (j_tile == i_tile) continue; // self-pair already set
            // Skip if already marked to avoid the bbox check cost.
            unsigned int mask_word = bitmask[j_tile / 32u];
            unsigned int bit = 1u << (j_tile & 31u);
            if (mask_word & bit) continue;
            // Bounding-box pruning: skip if the two tiles are far
            // enough apart under minimum image that no atom-atom
            // pair can be within `r_search`.
            const Real *bbox_j = tile_bboxes + j_tile * 6u;
            if (tile_pair_bbox_prune(
                    bbox_i, bbox_j,
                    lx, ly, lz, xy, xz, yz,
                    r_search_sq)) {
              continue;
            }
            atomicOr(&bitmask[j_tile / 32u], bit);
          }
        }
      }
    }
  }
  __syncthreads();

  // Popcount the bitmask cooperatively.
  unsigned int local_pop = 0u;
  for (unsigned int w = tid; w < bitmask_words; w += blockDim.x) {
    local_pop += __popc(bitmask[w]);
  }
  // Reduce across threads. Use atomicAdd on a single shared scalar.
  __shared__ unsigned int total_pop;
  if (tid == 0u) {
    total_pop = 0u;
  }
  __syncthreads();
  atomicAdd(&total_pop, local_pop);
  __syncthreads();
  if (tid == 0u) {
    tile_candidate_counts[i_tile] = total_pop;
  }
}

// rq-d8de5e38 — Per-tile candidate emission.
//
// One block per i_tile. Recomputes the bitmask in the same way as
// `tile_pair_candidate_count`, then walks the bitmask in ascending
// j_tile order and emits each set bit's index into
// `tile_pair_list[tile_pair_offsets[i_tile] + k]` for k = 0, 1, …
// The self-pair (t, t) is emitted at the position corresponding to
// its sorted j_tile index (smallest among tiles ≥ t, but the
// bitmask walk emits in strict ascending order so the self-pair's
// position is determined by its index relative to other tiles in
// the sublist).
extern "C" __global__ void tile_pair_emit(
    const unsigned int *sorted_particle_ids,
    const unsigned int *cell_indices,
    const unsigned int *cell_offsets,
    unsigned int n_cells_a, unsigned int n_cells_b, unsigned int n_cells_c,
    const Real *tile_bboxes,
    const Real *lattice,
    Real r_search_sq,
    const unsigned int *tile_pair_offsets,
    unsigned int *tile_pair_list,
    unsigned int n,
    unsigned int n_tiles,
    unsigned int bitmask_words)
{
  extern __shared__ unsigned int bitmask[];
  unsigned int tid = threadIdx.x;
  unsigned int i_tile = blockIdx.x;
  if (i_tile >= n_tiles) {
    return;
  }
  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];
  const Real *bbox_i = tile_bboxes + i_tile * 6u;

  // Zero bitmask cooperatively.
  for (unsigned int w = tid; w < bitmask_words; w += blockDim.x) {
    bitmask[w] = 0u;
  }
  __syncthreads();

  // Self-pair.
  if (tid == 0u) {
    atomicOr(&bitmask[i_tile / 32u], 1u << (i_tile & 31u));
  }
  __syncthreads();

  // Same cell-sweep + bbox-prune logic as the count kernel.
  unsigned int home_idx = i_tile * 32u + tid;
  if (home_idx < n && tid < 32u) {
    unsigned int particle_id = sorted_particle_ids[home_idx];
    unsigned int cell_a = cell_indices[particle_id];
    unsigned int ca = cell_a / (n_cells_b * n_cells_c);
    unsigned int rem = cell_a - ca * (n_cells_b * n_cells_c);
    unsigned int cb = rem / n_cells_c;
    unsigned int cc = rem - cb * n_cells_c;

    for (int da = -1; da <= 1; ++da) {
      int nca = (int) ca + da;
      while (nca < 0) { nca += (int) n_cells_a; }
      while (nca >= (int) n_cells_a) { nca -= (int) n_cells_a; }
      for (int db = -1; db <= 1; ++db) {
        int ncb = (int) cb + db;
        while (ncb < 0) { ncb += (int) n_cells_b; }
        while (ncb >= (int) n_cells_b) { ncb -= (int) n_cells_b; }
        for (int dc = -1; dc <= 1; ++dc) {
          int ncc = (int) cc + dc;
          while (ncc < 0) { ncc += (int) n_cells_c; }
          while (ncc >= (int) n_cells_c) { ncc -= (int) n_cells_c; }
          unsigned int c_neigh =
              ((unsigned int) nca * n_cells_b + (unsigned int) ncb)
              * n_cells_c + (unsigned int) ncc;
          unsigned int c_start = cell_offsets[c_neigh];
          unsigned int c_end = cell_offsets[c_neigh + 1u];
          for (unsigned int k = c_start; k < c_end; ++k) {
            unsigned int j_tile = k / 32u;
            if (j_tile == i_tile) continue;
            unsigned int mask_word = bitmask[j_tile / 32u];
            unsigned int bit = 1u << (j_tile & 31u);
            if (mask_word & bit) continue;
            const Real *bbox_j = tile_bboxes + j_tile * 6u;
            if (tile_pair_bbox_prune(
                    bbox_i, bbox_j,
                    lx, ly, lz, xy, xz, yz,
                    r_search_sq)) {
              continue;
            }
            atomicOr(&bitmask[j_tile / 32u], bit);
          }
        }
      }
    }
  }
  __syncthreads();

  // Walk the bitmask in ascending j_tile order and emit indices.
  // Single thread emits to preserve strict ordering; the per-tile
  // sublist is tiny (typically ≤ 60 entries for liquid water) so
  // the serial emission is not on the critical path.
  if (tid == 0u) {
    unsigned int base = tile_pair_offsets[i_tile];
    unsigned int write_offset = 0u;
    for (unsigned int w = 0u; w < bitmask_words; ++w) {
      unsigned int word = bitmask[w];
      while (word != 0u) {
        unsigned int bit = __ffs(word) - 1u; // 0-based bit index
        unsigned int j_tile = w * 32u + bit;
        tile_pair_list[base + write_offset] = j_tile;
        write_offset += 1u;
        word &= ~(1u << bit);
      }
    }
  }
}

// rq-d8de5e38 — Per-tile-pair j-atom interaction mask.
//
// One block per i_tile; MASK_WARPS_PER_BLOCK warps per block (each
// warp processes one entry from i_tile's sublist). Within a warp,
// lane k handles bit k of the output mask: reads j_atom k's
// position, iterates the 32 i_atoms, computes minimum-image
// squared distance, and contributes a per-lane bit. The 32 bits
// combine via __ballot_sync into the output u32.
//
// Inactive j-lanes (k ≥ tile_atom_count[j_tile]) contribute a
// cleared bit. Self-pair (j_tile == i_tile) is short-circuited
// to tile_lane_mask[j_tile] without distance computation.
#define MASK_WARPS_PER_BLOCK 8u
#define MASK_BLOCK_SIZE (MASK_WARPS_PER_BLOCK * 32u)
extern "C" __global__ void compute_tile_pair_masks(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *sorted_particle_ids,
    const unsigned int *tile_pair_offsets,
    const unsigned int *tile_pair_list,
    const unsigned int *tile_atom_count,
    const unsigned int *tile_lane_mask,
    const Real *lattice,
    Real r_search_sq,
    unsigned int *tile_pair_masks,
    unsigned int n_tiles)
{
  unsigned int i_tile = blockIdx.x;
  if (i_tile >= n_tiles) return;
  unsigned int tid = threadIdx.x;
  unsigned int warp_in_block = tid / 32u;
  unsigned int lane = tid & 31u;

  Real lx = lattice[0]; Real ly = lattice[1]; Real lz = lattice[2];
  Real xy = lattice[3]; Real xz = lattice[4]; Real yz = lattice[5];

  unsigned int i_count = tile_atom_count[i_tile];

  // Cache the i_tile's 32 atom positions in shared memory once
  // per block; reused across all sublist entries.
  __shared__ Real i_pos_cache[32][3];
  if (tid < 32u) {
    if (tid < i_count) {
      unsigned int pid =
          sorted_particle_ids[i_tile * 32u + tid];
      i_pos_cache[tid][0] = positions_x[pid];
      i_pos_cache[tid][1] = positions_y[pid];
      i_pos_cache[tid][2] = positions_z[pid];
    } else {
      // Inactive i-lane: place far from everything so distance
      // checks against it always fail.
      i_pos_cache[tid][0] = R(0.0);
      i_pos_cache[tid][1] = R(0.0);
      i_pos_cache[tid][2] = R(0.0);
    }
  }
  __syncthreads();

  unsigned int sublist_start = tile_pair_offsets[i_tile];
  unsigned int sublist_end = tile_pair_offsets[i_tile + 1u];
  unsigned int sublist_len = sublist_end - sublist_start;

  // Each warp handles one entry of the sublist; iterate when the
  // sublist is longer than MASK_WARPS_PER_BLOCK.
  for (unsigned int k = warp_in_block; k < sublist_len;
       k += MASK_WARPS_PER_BLOCK) {
    unsigned int entry_index = sublist_start + k;
    unsigned int j_tile = tile_pair_list[entry_index];

    unsigned int mask = 0u;
    if (j_tile == i_tile) {
      // Self-pair: mask is just the j_tile's lane mask, no
      // distance computation.
      if (lane == 0u) {
        tile_pair_masks[entry_index] = tile_lane_mask[j_tile];
      }
      continue;
    }

    unsigned int j_count = tile_atom_count[j_tile];
    // Determine whether lane k's j_atom is within r_search of any
    // i_atom. Inactive lanes (lane >= j_count) contribute 0.
    bool lane_in_range = false;
    if (lane < j_count) {
      unsigned int j_pid =
          sorted_particle_ids[j_tile * 32u + lane];
      Real jx = positions_x[j_pid];
      Real jy = positions_y[j_pid];
      Real jz = positions_z[j_pid];
      for (unsigned int m = 0u; m < i_count; ++m) {
        Real dx = i_pos_cache[m][0] - jx;
        Real dy = i_pos_cache[m][1] - jy;
        Real dz = i_pos_cache[m][2] - jz;
        triclinic_min_image(dx, dy, dz, lx, ly, lz, xy, xz, yz);
        Real r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= r_search_sq) {
          lane_in_range = true;
          break;
        }
      }
    }

    // Combine 32 per-lane bits into the output u32.
    mask = __ballot_sync(0xFFFFFFFFu, lane_in_range);
    if (lane == 0u) {
      tile_pair_masks[entry_index] = mask;
    }
  }
}

// rq-b7601928 — Tile-sorted position scatter.
//
// One thread per atom: thread `k` reads
// `sorted_particle_ids[k]` to get the particle ID `pid`, then
// reads `positions_*[pid]` and writes the result to
// `tile_sorted_positions_*[k]`. The writes are independent across
// threads, so no synchronisation is required.
//
// The kernel produces the tile-sorted-position view that the
// JIT-composed pair-force kernel reads via
// `tile_sorted_positions_*[i_tile * 32 + lane]` and
// `tile_sorted_positions_*[j_tile * 32 + lane]` — both of which are
// coalesced 32-element spans per warp load.
extern "C" __global__ void scatter_positions_to_tile_order(
    const Real *positions_x,
    const Real *positions_y,
    const Real *positions_z,
    const unsigned int *sorted_particle_ids,
    Real *tile_sorted_positions_x,
    Real *tile_sorted_positions_y,
    Real *tile_sorted_positions_z,
    unsigned int n)
{
  unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) return;
  unsigned int pid = sorted_particle_ids[k];
  tile_sorted_positions_x[k] = positions_x[pid];
  tile_sorted_positions_y[k] = positions_y[pid];
  tile_sorted_positions_z[k] = positions_z[pid];
}
