// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#ifdef _OPENMP
#include <omp.h>
#endif
using namespace Rcpp;

// Helper: Inline WLS unmixing solver
// Returns coefficients for unmixing using weighted least squares
// Uses pre-allocated buffers for efficiency
inline bool solve_wls_fast(arma::vec& coefs,
                           const arma::mat& S,      // P x D (Active spectra)
                           const arma::rowvec& r,   // 1 x D (Raw data)
                           const arma::vec& w,      // D x 1 (Weights)
                           const bool use_weighted,
                           arma::mat& A,            // Buffer: P x P
                           arma::vec& b,            // Buffer: P x 1
                           arma::mat& S_scratch) {  // Buffer: P x D

  if (!use_weighted) {
    // Unweighted OLS: (S S') c = S r'
    A = S * S.t();
    b = S * r.t();
  } else {
    // Weighted OLS: (S W S') c = S W r'
    S_scratch = S;
    for (arma::uword j = 0; j < S.n_cols; ++j) {
      S_scratch.col(j) *= w(j);
    }
    A = S_scratch * S.t();
    b = S_scratch * r.t();
  }

  // Attempt fast solve
  bool ok = arma::solve(coefs, A, b, arma::solve_opts::fast);

  // Fallback to pseudoinverse if singular
  if (!ok) {
    coefs = arma::pinv(A) * b;
  }

  return true;
}

// [[Rcpp::export]]
arma::mat optimize_flowcode_unmix(
    const arma::mat& raw_data,                          // n_cells x n_detectors
    const arma::mat& unmixed,                           // n_cells x n_fluors
    const arma::mat& combined_spectra,                  // n_fluors x n_detectors
    const arma::vec& weights,                           // n_detectors
    const arma::vec& pos_thresholds,                    // n_fluors
    const arma::uvec& af_idx,                           // n_cells (1-indexed from R)
    const arma::mat& af_spectra,                        // n_af_variants x n_detectors
    const arma::uvec& flowcode_ids,                     // n_cells (1-indexed from R, 0 = no flowcode)
    const arma::uvec& has_flowcode,                     // n_cells (logical as 0/1)
    const List& combo_fret,                             // List of matrices (n_fret_variants x n_detectors)
    const List& fret_delta_list,                        // List of delta matrices
    const List& fret_delta_norms,                       // List of norm vectors
    const arma::umat& flowcode_combo_logical,           // n_combos x n_flowcode_fluors (0/1)
    const std::vector<std::string>& flowcode_fluors,    // Fluorophore names for FlowCode
    const std::vector<std::string>& optimize_fluors,    // Fluorophores to optimize
    const List& variants,                               // List of variant matrices per fluorophore
    const List& delta_list,                             // List of delta matrices per fluorophore
    const List& delta_norms,                            // List of delta norms per fluorophore
    const std::vector<std::string>& all_fluor_names,    // All fluorophore names
    const int af_idx_in_spectra,                        // 0-indexed position of AF in spectra
    const int k = 10,                                   // Number of top variants to test
    const bool weighted = false,                        // Use weighted least squares
    const bool cell_weighting = false,                  // Use cell-specific weighting
    const bool cell_weight_regularize = false,          // Regularize cell weights
    const int nthreads = 1) {                           // Number of threads

  const arma::uword n_cells = raw_data.n_rows;
  const arma::uword n_detectors = raw_data.n_cols;
  const arma::uword n_fluors = combined_spectra.n_rows;

  arma::mat result = unmixed;

  // Convert Rcpp Lists to std::vectors of Armadillo objects before the parallel loop
  std::vector<arma::mat> v_combo_fret(combo_fret.size());
  for(int i = 0; i < combo_fret.size(); ++i) v_combo_fret[i] = as<arma::mat>(combo_fret[i]);

  std::vector<arma::mat> v_fret_delta_list(fret_delta_list.size());
  for(int i = 0; i < fret_delta_list.size(); ++i) v_fret_delta_list[i] = as<arma::mat>(fret_delta_list[i]);

  std::vector<arma::vec> v_fret_delta_norms(fret_delta_norms.size());
  for(int i = 0; i < fret_delta_norms.size(); ++i) v_fret_delta_norms[i] = as<arma::vec>(fret_delta_norms[i]);

  std::vector<arma::mat> v_variants(variants.size());
  for(int i = 0; i < variants.size(); ++i) v_variants[i] = as<arma::mat>(variants[i]);

  std::vector<arma::mat> v_delta_list(delta_list.size());
  for(int i = 0; i < delta_list.size(); ++i) v_delta_list[i] = as<arma::mat>(delta_list[i]);

  std::vector<arma::vec> v_delta_norms(delta_norms.size());
  for(int i = 0; i < delta_norms.size(); ++i) v_delta_norms[i] = as<arma::vec>(delta_norms[i]);

  // Build name-to-index maps
  std::map<std::string, arma::uword> fluor_name_to_idx;
  for (arma::uword i = 0; i < all_fluor_names.size(); ++i) {
    fluor_name_to_idx[all_fluor_names[i]] = i;
  }

  std::map<std::string, arma::uword> flowcode_fluor_to_idx;
  for (arma::uword i = 0; i < flowcode_fluors.size(); ++i) {
    flowcode_fluor_to_idx[flowcode_fluors[i]] = i;
  }

  // Identify which fluorophores are not AF (for early exit check)
  std::vector<arma::uword> fluorophore_indices;
  for (arma::uword f = 0; f < n_fluors; ++f) {
    if (f != (arma::uword)af_idx_in_spectra) {
      fluorophore_indices.push_back(f);
    }
  }

#ifdef _OPENMP
  if (nthreads > 0) omp_set_num_threads(nthreads);
  const int nthreads_actual = omp_get_max_threads();
#else
  const int nthreads_actual = 1;
#endif

  // Thread-local buffers
  struct ThreadBuffers {
    arma::rowvec raw_row;
    arma::rowvec cell_raw_modified;  // For FRET subtraction
    arma::vec cell_weights;
    arma::rowvec cell_unmixed_row;
    arma::rowvec fitted;
    arma::rowvec resid;
    arma::vec coefs;
    arma::vec coefs_full;
    arma::mat cell_spectra_curr;
    arma::mat cell_spectra_final;
    arma::mat A;
    arma::vec b;
    arma::mat S_scratch;
    arma::mat A_full;
    arma::vec b_full;
    arma::mat S_scratch_full;
    arma::vec joint_scores;
    std::vector<arma::uword> pos_vec;
    std::vector<bool> pos_flags;
    std::vector<std::pair<double, std::string>> fluor_order;
    std::vector<arma::uword> topK_indices;
  };

  std::vector<ThreadBuffers> thread_buffers(nthreads_actual);

  // Initialize buffers
  for (auto& tb : thread_buffers) {
    tb.raw_row.set_size(n_detectors);
    tb.cell_raw_modified.set_size(n_detectors);
    tb.cell_weights.set_size(n_detectors);
    tb.cell_unmixed_row.set_size(n_fluors);
    tb.fitted.set_size(n_detectors);
    tb.resid.set_size(n_detectors);
    tb.coefs.set_size(n_fluors);
    tb.coefs_full.set_size(n_fluors);
    tb.cell_spectra_curr.set_size(n_fluors, n_detectors);
    tb.cell_spectra_final.set_size(n_fluors, n_detectors);
    tb.A.set_size(n_fluors, n_fluors);
    tb.b.set_size(n_fluors);
    tb.S_scratch.set_size(n_fluors, n_detectors);
    tb.A_full.set_size(n_fluors, n_fluors);
    tb.b_full.set_size(n_fluors);
    tb.S_scratch_full.set_size(n_fluors, n_detectors);
    tb.joint_scores.set_size(1000); // reasonable max
    tb.pos_vec.reserve(n_fluors);
    tb.pos_flags.resize(n_fluors);
    tb.fluor_order.reserve(n_fluors);
    tb.topK_indices.reserve(k);
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (arma::uword cell = 0; cell < n_cells; ++cell) {
#ifdef _OPENMP
    int tid = omp_get_thread_num();
#else
    int tid = 0;
#endif
    ThreadBuffers& tb = thread_buffers[tid];

    // Load cell data
    tb.raw_row = raw_data.row(cell);
    tb.cell_raw_modified = tb.raw_row; // Will be modified for FRET
    tb.cell_unmixed_row = unmixed.row(cell); // Using input unmixed instead of shared result mat for safety

    // Set weights
    tb.cell_weights = weights;
    bool use_weighted = weighted;

    if (cell_weighting) {
      // Cell-specific Poisson-like weighting
      for (arma::uword j = 0; j < n_detectors; ++j) {
        double w = std::abs(tb.raw_row(j));
        if (w < 1e-6) w = 1e-6;
        tb.cell_weights(j) = 1.0 / w;
      }

      if (cell_weight_regularize) {
        tb.cell_weights = (tb.cell_weights + weights) * 0.5;
      }
      use_weighted = true;
    }

    // Get AF index for this cell (convert from 1-indexed R to 0-indexed C++)
    arma::uword cell_af_idx = af_idx(cell) - 1;
    arma::rowvec cell_af = af_spectra.row(cell_af_idx);

    // Initialize final spectra with combined spectra
    tb.cell_spectra_final = combined_spectra;

    // Swap in this cell's pre-selected AF spectrum
    tb.cell_spectra_final.row(af_idx_in_spectra) = cell_af;

    // Check if any fluorophores are present (excluding AF)
    tb.pos_flags.assign(n_fluors, false);
    bool any_pos = false;
    for (arma::uword f : fluorophore_indices) {
      if (tb.cell_unmixed_row(f) >= pos_thresholds(f)) {
        tb.pos_flags[f] = true;
        any_pos = true;
      }
    }

    if (!any_pos) {
      // No fluorophores present, skip optimization
      result.row(cell) = tb.cell_unmixed_row;
      continue;
    }

    arma::uword pos_n;
    double error_final;

    // AF is always present
    tb.pos_flags[af_idx_in_spectra] = true;

    // Build current spectra with positive fluorophores only
    tb.pos_vec.clear();
    for (arma::uword f = 0; f < n_fluors; ++f) {
      if (tb.pos_flags[f]) tb.pos_vec.push_back(f);
    }

    pos_n = tb.pos_vec.size();
    tb.cell_spectra_curr.set_size(pos_n, n_detectors);
    for (arma::uword i = 0; i < pos_n; ++i) {
      tb.cell_spectra_curr.row(i) = tb.cell_spectra_final.row(tb.pos_vec[i]);
    }

    // Initial unmix
    solve_wls_fast(tb.coefs, tb.cell_spectra_curr, tb.cell_raw_modified,
                   tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

    tb.resid = tb.cell_raw_modified - (tb.coefs.t() * tb.cell_spectra_curr);
    error_final = arma::sum(arma::abs(tb.resid));

    // ========================================================================
    // FRET CORRECTION (if applicable)
    // ========================================================================
    if (has_flowcode(cell) == 1) {
      arma::uword fc_id = flowcode_ids(cell) - 1; // Convert to 0-indexed

      // Get FRET variants and deltas for this combo
      const arma::mat& variants_fr = v_combo_fret[fc_id];
      const arma::mat& delta_fr = v_fret_delta_list[fc_id];
      const arma::vec& delta_norm = v_fret_delta_norms[fc_id];

      // Identify disallowed FlowCode fluorophores for this combo
      arma::urowvec combo_logical = flowcode_combo_logical.row(fc_id);
      for (arma::uword fc_idx = 0; fc_idx < flowcode_fluors.size(); ++fc_idx) {
        if (combo_logical(fc_idx) == 0) {
          // This FlowCode fluor is not in the combo
          auto it = fluor_name_to_idx.find(flowcode_fluors[fc_idx]);
          if (it != fluor_name_to_idx.end()) tb.pos_flags[it->second] = false;
        }
      }

      // Rebuild pos_vec after filtering
      tb.pos_vec.clear();
      for (arma::uword f = 0; f < n_fluors; ++f) {
        if (tb.pos_flags[f]) {
          tb.pos_vec.push_back(f);
        }
      }

      pos_n = tb.pos_vec.size();
      arma::mat cell_spectra_fret(pos_n + 1, n_detectors);
      cell_spectra_fret.row(0) = variants_fr.row(0);
      for (arma::uword i = 0; i < pos_n; ++i) cell_spectra_fret.row(i+1) = tb.cell_spectra_final.row(tb.pos_vec[i]);

      arma::vec trial_unmix(pos_n + 1);

      solve_wls_fast(trial_unmix, cell_spectra_fret, tb.cell_raw_modified,
                     tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

      tb.resid = tb.cell_raw_modified - (trial_unmix.t() * cell_spectra_fret);
      error_final = arma::sum(arma::abs(tb.resid));
      arma::rowvec fitted_fret = trial_unmix(0) * cell_spectra_fret.row(0);

      // Score FRET variants
      double fret_coef = trial_unmix(0);
      double r_norm_fret = std::sqrt(arma::dot(tb.resid, tb.resid));
      tb.joint_scores.set_size(delta_fr.n_rows);

      for (arma::uword v = 0; v < delta_fr.n_rows; ++v) {
        // delta_norm is pre-calculated, but check for identity or zero residual
        if (delta_norm(v) < 1e-10 || r_norm_fret < 1e-10 || !std::isfinite(fret_coef)) {
          tb.joint_scores(v) = -1e10;
        } else {
          double score = (arma::dot(delta_fr.row(v), tb.resid) * fret_coef) / (delta_norm(v) * r_norm_fret);
          tb.joint_scores(v) = std::isfinite(score) ? score : -1e10;
        }
      }

      // Select top k variants
      arma::uword k_eff_fret = std::min((arma::uword)k, (arma::uword)delta_fr.n_rows);
      arma::uvec sorted_fret_idx = arma::sort_index(tb.joint_scores.head(delta_fr.n_rows), "descend");

      // Test top k variants
      for (arma::uword i = 0; i < k_eff_fret; ++i) {
        arma::uword var_idx = sorted_fret_idx(i);
        if (tb.joint_scores(var_idx) <= -1e9) break; // Stop if we hit bad variants

        cell_spectra_fret.row(0) = variants_fr.row(var_idx);

        solve_wls_fast(trial_unmix, cell_spectra_fret, tb.cell_raw_modified,
                       tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

        arma::rowvec t_resid = tb.cell_raw_modified - (trial_unmix.t() * cell_spectra_fret);
        double t_err = arma::sum(arma::abs(t_resid));

        if (t_err < error_final) {
          error_final = t_err;
          fitted_fret = trial_unmix(0) * cell_spectra_fret.row(0);
        }
      }

      // Subtract FRET from raw data
      tb.cell_raw_modified -= fitted_fret;

      // Re-unmix after FRET correction
      tb.pos_vec.clear();
      for (arma::uword f = 0; f < n_fluors; ++f) if (tb.pos_flags[f]) tb.pos_vec.push_back(f);
      arma::uword pos_n = tb.pos_vec.size();
      tb.cell_spectra_curr.set_size(pos_n, n_detectors);
      for(arma::uword i = 0; i < pos_n; ++i) tb.cell_spectra_curr.row(i) = tb.cell_spectra_final.row(tb.pos_vec[i]);

      solve_wls_fast(tb.coefs, tb.cell_spectra_curr, tb.cell_raw_modified, tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);
      tb.resid = tb.cell_raw_modified - (tb.coefs.t() * tb.cell_spectra_curr);
      error_final = arma::sum(arma::abs(tb.resid));
    }

    // ========================================================================
    // AF OPTIMIZATION
    // ========================================================================

    // Get current AF coefficient
    arma::uword af_local_idx = 0;
    for (arma::uword i = 0; i < pos_n; ++i) {
      if (tb.pos_vec[i] == (arma::uword)af_idx_in_spectra) {
        af_local_idx = i; break;
      }
    }
    double af_coef = tb.coefs(af_local_idx);
    double r_norm = std::sqrt(arma::dot(tb.resid, tb.resid));

    // Score variants without allocating an 'af_delta' matrix
    tb.joint_scores.set_size(af_spectra.n_rows);
    for (arma::uword v = 0; v < af_spectra.n_rows; ++v) {
      arma::rowvec diff = af_spectra.row(v) - cell_af;
      double d_norm = std::sqrt(arma::dot(diff, diff));

      // SAFEGUARD: Skip if variant is identical to current baseline (d_norm ~ 0)
      // OR if the residual is already essentially zero.
      if (d_norm < 1e-10 || r_norm < 1e-10 || !std::isfinite(af_coef)) {
        tb.joint_scores(v) = -1e10; // Send to the bottom of the sort
      } else {
        double score = (arma::dot(diff, tb.resid) * af_coef) / (d_norm * r_norm);

        // Final check for numerical stability
        if (!std::isfinite(score)) {
          tb.joint_scores(v) = -1e10;
        } else {
          tb.joint_scores(v) = score;
        }
      }
    }

    arma::uword k_eff_af = std::min((arma::uword)k, (arma::uword)af_spectra.n_rows);
    arma::uvec sorted_af_indices = arma::sort_index(tb.joint_scores.head(af_spectra.n_rows), "descend");

    // Test top k AF variants
    for (arma::uword i = 0; i < k_eff_af; ++i) {
      arma::uword var = sorted_af_indices(i);
      if (tb.joint_scores(var) <= 0) break; // Optimization: don't test poor candidates
      tb.cell_spectra_curr.row(af_local_idx) = af_spectra.row(var);

      arma::vec trial_unmix(pos_n);
      solve_wls_fast(trial_unmix, tb.cell_spectra_curr, tb.cell_raw_modified,
                     tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

      arma::rowvec trial_resid = tb.cell_raw_modified - (trial_unmix.t() * tb.cell_spectra_curr);
      double trial_error = arma::sum(arma::abs(trial_resid));

      if (trial_error < error_final) {
        error_final = trial_error;
        tb.cell_spectra_final.row(af_idx_in_spectra) = af_spectra.row(var);
        tb.resid = trial_resid;
        tb.coefs = trial_unmix;
      } else {
        tb.cell_spectra_curr.row(af_local_idx) = tb.cell_spectra_final.row(af_idx_in_spectra);
      }
    }

    // Re-unmix with final AF and reassess positivity
    arma::vec full_unmix(n_fluors);
    solve_wls_fast(full_unmix, tb.cell_spectra_final, tb.cell_raw_modified,
                   tb.cell_weights, use_weighted, tb.A_full, tb.b_full, tb.S_scratch_full);

    tb.pos_flags.assign(n_fluors, false);
    any_pos = false;
    for (arma::uword f : fluorophore_indices) {
      if (full_unmix(f) >= pos_thresholds(f)) {
        tb.pos_flags[f] = true;
        any_pos = true;
      }
    }

    if (!any_pos) {
      // Only AF remains, finalize
      arma::vec final_af_only_unmix(n_fluors);
      solve_wls_fast(final_af_only_unmix, tb.cell_spectra_final, tb.cell_raw_modified,
                     tb.cell_weights, use_weighted, tb.A_full, tb.b_full, tb.S_scratch_full);
      result.row(cell) = final_af_only_unmix.t();
      continue;
    }
    tb.pos_flags[af_idx_in_spectra] = true;

    // Handle FlowCode allowed fluorophores
    if (has_flowcode(cell) == 1) {
      arma::uword fc_id = flowcode_ids(cell) - 1;
      arma::urowvec combo_logical = flowcode_combo_logical.row(fc_id);

      // Reset FlowCode fluors
      for (const auto& fc_name : flowcode_fluors) {
        auto it = fluor_name_to_idx.find(fc_name);
        if (it != fluor_name_to_idx.end()) tb.pos_flags[it->second] = false;
      }

      // Set allowed ones
      for (arma::uword fc_idx = 0; fc_idx < flowcode_fluors.size(); ++fc_idx) {
        if (combo_logical(fc_idx) == 1) {
          auto it = fluor_name_to_idx.find(flowcode_fluors[fc_idx]);
          if (it != fluor_name_to_idx.end()) tb.pos_flags[it->second] = true;
        }
      }
    }

    // Rebuild positive spectra
    tb.pos_vec.clear();
    for (arma::uword f = 0; f < n_fluors; ++f) {
      if (tb.pos_flags[f]) tb.pos_vec.push_back(f);
    }

    pos_n = tb.pos_vec.size();
    tb.cell_spectra_curr.set_size(pos_n, n_detectors);
    for (arma::uword i = 0; i < pos_n; ++i) {
      tb.cell_spectra_curr.row(i) = tb.cell_spectra_final.row(tb.pos_vec[i]);
    }

    tb.coefs.set_size(pos_n);
    solve_wls_fast(tb.coefs, tb.cell_spectra_curr, tb.cell_raw_modified,
                   tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

    tb.resid = tb.cell_raw_modified - (tb.coefs.t() * tb.cell_spectra_curr);
    error_final = arma::sum(arma::abs(tb.resid));

    // ========================================================================
    // FLUOROPHORE OPTIMIZATION
    // ========================================================================

    // Sort fluorophores by abundance
    tb.fluor_order.clear();
    for (arma::uword i = 0; i < pos_n; ++i) {
      arma::uword f_idx = tb.pos_vec[i];
      if (f_idx == (arma::uword)af_idx_in_spectra) continue;
      std::string f_name = all_fluor_names[f_idx];
      for (const auto& opt_name : optimize_fluors) {
        if (opt_name == f_name) {
          tb.fluor_order.emplace_back(tb.coefs(i), f_name);
          break;
        }
      }
    }

    std::sort(tb.fluor_order.begin(), tb.fluor_order.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Optimize each fluorophore
    for (const auto& fl_pair : tb.fluor_order) {
      arma::uword f_idx = fluor_name_to_idx[fl_pair.second];
      arma::uword local_idx = 0;
      for(arma::uword i=0; i<pos_n; ++i) if(tb.pos_vec[i] == f_idx) { local_idx = i; break; }

      // Get the correct list index for this fluorophore
      int opt_idx = -1;
      for(size_t v=0; v < optimize_fluors.size(); ++v) {
        if(optimize_fluors[v] == fl_pair.second) { opt_idx = v; break; }
      }

      if(opt_idx == -1 || v_variants[opt_idx].n_rows == 0) continue;

      const arma::mat& fl_vars = v_variants[opt_idx];
      const arma::mat& fl_deltas = v_delta_list[opt_idx];
      const arma::vec& fl_norms = v_delta_norms[opt_idx];

      double current_coef = tb.coefs(local_idx);
      double r_norm_fl = std::sqrt(arma::dot(tb.resid, tb.resid));
      tb.joint_scores.set_size(fl_deltas.n_rows);

      // score the variants for alignment with the residual
      for(arma::uword v=0; v < fl_deltas.n_rows; ++v) {
        if (fl_norms(v) < 1e-10 || r_norm_fl < 1e-10 || !std::isfinite(current_coef)) {
          tb.joint_scores(v) = -1e10;
        } else {
          double score = (arma::dot(fl_deltas.row(v), tb.resid) * current_coef) / (fl_norms(v) * r_norm_fl);
          tb.joint_scores(v) = std::isfinite(score) ? score : -1e10;
        }
      }

      // pick top k variants
      arma::uword k_eff_fl = std::min((arma::uword)k, (arma::uword)fl_deltas.n_rows);
      arma::uvec s_idx = arma::sort_index(tb.joint_scores.head(fl_deltas.n_rows), "descend");

      // unmix for each variant in top k
      for(arma::uword i=0; i < k_eff_fl; ++i) {
        arma::uword var_idx = s_idx(i);
        if (tb.joint_scores(var_idx) <= -1e9) break;

        tb.cell_spectra_curr.row(local_idx) = fl_vars.row(var_idx);
        arma::vec t_coefs(pos_n);
        solve_wls_fast(t_coefs, tb.cell_spectra_curr, tb.cell_raw_modified, tb.cell_weights, use_weighted, tb.A, tb.b, tb.S_scratch);

        arma::rowvec t_res = tb.cell_raw_modified - (t_coefs.t() * tb.cell_spectra_curr);
        double t_err = arma::sum(arma::abs(t_res));

        if(t_err < error_final) {
          error_final = t_err;
          // update the final spectra with the selected variant
          tb.cell_spectra_final.row(f_idx) = fl_vars.row(var_idx);
          tb.resid = t_res;
          tb.coefs = t_coefs;
        } else {
          // Revert to the best known spectrum for this fluorophore for the next variant test
          tb.cell_spectra_curr.row(local_idx) = tb.cell_spectra_final.row(f_idx);
        }
      }
    }

    // final unmix with optimized spectra
    solve_wls_fast(tb.coefs_full, tb.cell_spectra_final, tb.cell_raw_modified, tb.cell_weights, use_weighted, tb.A_full, tb.b_full, tb.S_scratch_full);
    result.row(cell) = tb.coefs_full.t();
  }

  return result;
}
