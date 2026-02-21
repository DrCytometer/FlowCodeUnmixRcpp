#include <RcppArmadillo.h>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Helper: Fast Top-K selection using partial_sort
std::vector<uword> find_top_k(const vec& scores, int k) {
  int n = (int)scores.n_elem;
  int k_eff = std::min(k, n);
  if (k_eff <= 0) return {};
  std::vector<uword> indices(n);
  std::iota(indices.begin(), indices.end(), 0);
  std::partial_sort(indices.begin(), indices.begin() + k_eff, indices.end(),
                    [&scores](uword i, uword j) { return scores[i] > scores[j]; });
  indices.resize(k_eff);
  return indices;
}

// Helper: Fast median
double quick_median(vec x) {
  if(x.n_elem == 0) return 0;
  std::vector<double> v = conv_to<std::vector<double>>::from(x);
  std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
  return v[v.size()/2];
}

// [[Rcpp::export]]
arma::mat unmix_flowcode_pipeline_cpp(
    arma::mat raw_data_in,
    const arma::mat& spectra,
    const arma::mat& af_spectra,
    const CharacterVector& fluor_names,
    const CharacterVector& flowcode_fluors,
    const CharacterVector& flowcode_tags,
    const NumericVector& flowcode_thresholds,
    const CharacterVector& valid_combos,
    const arma::imat& flowcode_logical,
    const List& fret_spectra_list,
    const arma::vec& pos_thresholds,
    const CharacterVector& optimize_fluors,
    const List& variants,
    const List& delta_list,
    const List& delta_norms,
    int k_opt = 1,
    int n_threads = 1,
    bool optimize = true
) {
  // 1. Memory Layout Optimization (Transpose)
  mat raw_data = raw_data_in.t();
  uword n_cells = raw_data.n_cols;
  uword n_fluors = spectra.n_rows;
  uword n_af = af_spectra.n_rows;
  int n_combos = (int)valid_combos.size();
  
  // --- PRE-CALCULATIONS ---
  mat P = solve(spectra * spectra.t(), spectra);
  mat S_t = spectra.t();
  mat AF_t = af_spectra.t();
  mat v_library_af = P * AF_t;
  mat r_library_af = AF_t - (S_t * v_library_af);
  
  vec r_dots_af(n_af);
  for(uword j = 0; j < n_af; ++j) {
    double d = dot(r_library_af.col(j), r_library_af.col(j));
    r_dots_af[j] = (d == 0) ? 1e-10 : d;
  }
  
  std::vector<std::string> cpp_names = as<std::vector<std::string>>(fluor_names);
  std::map<std::string, int> name_to_idx;
  for(int i = 0; i < (int)cpp_names.size(); ++i) name_to_idx[cpp_names[i]] = i;
  
  // Mapping FlowCode tags
  std::vector<std::string> fc_tag_cpp = as<std::vector<std::string>>(flowcode_tags);
  std::vector<double> fc_thresh_cpp = as<std::vector<double>>(flowcode_thresholds);
  uvec fc_indices((size_t)flowcode_fluors.size());
  for(int i = 0; i < (int)flowcode_fluors.size(); ++i) {
    fc_indices[i] = name_to_idx[as<std::string>(flowcode_fluors[i])];
  }
  
  std::map<std::string, int> combo_map;
  for(int i = 0; i < n_combos; ++i) combo_map[as<std::string>(valid_combos[i])] = i + 1;
  
  // 2. Pre-scaled Variants & Active Opt List
  size_t n_var_total = (size_t)variants.size();
  std::vector<mat> v_mats(n_var_total), D_scaled(n_var_total);
  std::vector<int> var_to_master(n_var_total);
  std::vector<size_t> active_opt_indices;
  
  if (optimize) {
    std::vector<std::string> v_names = as<std::vector<std::string>>(variants.names());
    std::vector<std::string> cpp_opt_names = as<std::vector<std::string>>(optimize_fluors);
    for (size_t f = 0; f < n_var_total; ++f) {
      mat d_mat = as<mat>(delta_list[f]);
      vec d_norm = as<vec>(delta_norms[f]);
      // Reduce Divisions: Pre-scaling
      for(uword r = 0; r < d_mat.n_rows; ++r) d_mat.row(r) /= (d_norm[r] > 1e-12 ? d_norm[r] : 1.0);
      D_scaled[f] = d_mat;
      v_mats[f] = as<mat>(variants[f]);
      var_to_master[f] = name_to_idx.count(v_names[f]) ? name_to_idx[v_names[f]] : -1;
      
      bool is_req = false;
      for (const auto& name : cpp_opt_names) if (name == v_names[f]) { is_req = true; break; }
      if (is_req && !d_norm.is_empty() && (max(d_norm) > 1e-12)) {
        active_opt_indices.push_back(f);
      }
    }
  }
  
  // 3. FRET Pre-calculated Lookup Tables
  std::vector<mat> fret_v_libs(n_combos), fret_r_libs(n_combos), fret_orig_spectra(n_combos);
  for(int i = 0; i < n_combos; ++i) {
    mat f_lib = as<mat>(fret_spectra_list[i]);
    fret_orig_spectra[i] = f_lib;
    fret_v_libs[i] = P * f_lib.t();
    fret_r_libs[i] = f_lib.t() - (S_t * fret_v_libs[i]);
  }
  
  mat unmixed_final(n_cells, n_fluors);
  vec AF_vals(n_cells);
  ivec AF_idx(n_cells), flowcode_ids(n_cells);
  vec FlowCode_Intensity(n_cells, fill::zeros);
  mat fc_data(n_cells, n_combos, fill::zeros);
  
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif
  
  // --- PARALLEL LOOP ---
#pragma omp parallel for schedule(dynamic, 64)
  for(uword i = 0; i < n_cells; ++i) {
    // Aligned Scratchpads
    alignas(64) static thread_local rowvec cell_unmixed, cell_resid_raw, unmixed_curr, resid, t_unmix, t_resid;
    alignas(64) static thread_local mat spectra_curr;
    static thread_local std::vector<std::string> pos_tag_aliases;
    static thread_local std::vector<std::pair<double, std::string>> tag_signals;
    static thread_local std::vector<std::pair<double, size_t>> fluor_order;
    static thread_local std::vector<uword> m_to_c_row, off_idx;
    
    pos_tag_aliases.clear(); tag_signals.clear(); fluor_order.clear(); off_idx.clear();
    if(m_to_c_row.size() != n_var_total) m_to_c_row.assign(n_var_total, 0);
    
    vec cell_raw_vec = raw_data.col(i); 
    rowvec cell_raw = cell_raw_vec.t();
    
    // A. AF Extraction
    vec init_f = (P * cell_raw_vec);
    double min_err_af = datum::inf, b_k_af = 0; uword b_idx_af = 0;
    for(uword j=0; j<n_af; ++j) {
      double k = dot(cell_raw_vec, r_library_af.col(j)) / r_dots_af[j];
      double err = sum(abs(init_f - (k * v_library_af.col(j))));
      if(err < min_err_af) { min_err_af = err; b_k_af = k; b_idx_af = j; }
    }
    AF_vals[i] = b_k_af; AF_idx[i] = (int)b_idx_af + 1;
    cell_resid_raw = cell_raw - (b_k_af * af_spectra.row(b_idx_af));
    cell_unmixed = (P * cell_resid_raw.t()).t();
    
    // B. Debarcoding
    for(uword j = 0; j < (uword)fc_indices.n_elem; ++j) {
      double diff = cell_unmixed[fc_indices[j]] - fc_thresh_cpp[j];
      tag_signals.push_back({diff, fc_tag_cpp[j]});
      if(diff > 0) pos_tag_aliases.push_back(fc_tag_cpp[j]);
    }
    if(pos_tag_aliases.size() > 3) {
      std::sort(tag_signals.begin(), tag_signals.end(), std::greater<>());
      if(tag_signals[2].first > 2 * tag_signals[3].first)
        pos_tag_aliases = {tag_signals[0].second, tag_signals[1].second, tag_signals[2].second};
    }
    int id = 0;
    if(pos_tag_aliases.size() == 3) {
      std::sort(pos_tag_aliases.begin(), pos_tag_aliases.end());
      std::string key = pos_tag_aliases[0] + "_" + pos_tag_aliases[1] + "_" + pos_tag_aliases[2];
      id = combo_map.count(key) ? combo_map[key] : n_combos + 1;
    } else if(pos_tag_aliases.size() > 0) id = n_combos + 1;
    flowcode_ids[i] = id;
    
    // C. FRET Correction (Restored L1-Decider Logic with Pre-calculated Lookup)
    if(id > 0 && id <= n_combos) {
      const mat& v_lib_f = fret_v_libs[id-1];     // Pre-calculated (n_fluors x n_variants)
      const mat& r_lib_f = fret_r_libs[id-1];     // Pre-calculated (n_channels x n_variants)
      const mat& f_orig  = fret_orig_spectra[id-1]; // Original (n_variants x n_channels)
      
      // Identify channels that should NOT have signal (off_idx)
      uvec active_mask = find(flowcode_logical.row(id-1) == 1);
      for(uword f=0; f<n_fluors; ++f) {
        bool active = false;
        for(uword a=0; a<active_mask.n_elem; ++a) {
          if(fc_indices[active_mask[a]] == f) { active=true; break; }
        }
        if(!active) off_idx.push_back(f);
      }
      
      double min_err_fret = datum::inf;
      double b_k_fret = 0;
      uword b_idx_fret = 0;
      
      // Loop through variants as in your original
      for(uword j = 0; j < r_lib_f.n_cols; ++j) {
        // 1. Calculate k using L2 (Dot Product)
        double den = dot(r_lib_f.col(j), r_lib_f.col(j));
        double k = dot(cell_resid_raw, r_lib_f.col(j).t()) / (den == 0 ? 1e-10 : den);
        if(k < 0) k = 0;
        
        // 2. Evaluate fitness using L1 (Absolute Residuals) on off-target fluorophores
        double err = 0;
        for(uword o : off_idx) {
          err += std::abs(cell_unmixed[o] - (k * v_lib_f(o, j)));
        }
        
        // 3. Track the best variant based on L1 error
        if(err < min_err_fret) {
          min_err_fret = err;
          b_k_fret = k;
          b_idx_fret = j;
        }
      }
      
      // Apply best correction
      cell_resid_raw -= b_k_fret * f_orig.row(b_idx_fret);
      cell_unmixed -= (b_k_fret * v_lib_f.col(b_idx_fret)).t();
    }
    
    // D. Optimization (Hardware Tuned)
    if (optimize) {
      mat cell_spectra_final = spectra;
      uvec pos_mask(n_fluors, fill::zeros); bool any_pos = false;
      for (uword f = 0; f < n_fluors; ++f) if (cell_unmixed[f] >= pos_thresholds[f]) { pos_mask[f] = 1; any_pos = true; }
      if (id > 0 && id <= n_combos) {
        for (uword idx : fc_indices) pos_mask[idx] = 0;
        irowvec allowed = flowcode_logical.row(id - 1);
        for (uword j = 0; j < (uword)allowed.n_elem; ++j) if (allowed[j] == 1) { pos_mask[fc_indices[j]] = 1; any_pos = true; }
      }
      if (any_pos) {
        uvec pos_idx = find(pos_mask == 1);
        if (pos_idx.n_elem > 0) {
          spectra_curr = cell_spectra_final.rows(pos_idx);
          unmixed_curr = solve(spectra_curr.t(), cell_resid_raw.t(), solve_opts::fast).t();
          resid = cell_resid_raw - (unmixed_curr * spectra_curr);
          double err_final = sum(abs(resid)), r_n = std::sqrt(dot(resid, resid));
          double inv_rn = 1.0 / (r_n + 1e-12);
          
          for (size_t f_idx : active_opt_indices) {
            int m_idx = var_to_master[f_idx];
            for(uword p = 0; p < pos_idx.n_elem; ++p) if((int)pos_idx[p] == m_idx) { fluor_order.push_back({unmixed_curr[p], f_idx}); m_to_c_row[f_idx] = p; break; }
          }
          if(!fluor_order.empty()) {
            std::sort(fluor_order.begin(), fluor_order.end(), std::greater<>());
            for (auto const& pair : fluor_order) {
              size_t f_idx = pair.second; int m_idx = var_to_master[f_idx]; uword r_curr = m_to_c_row[f_idx];
              if (r_n > 1e-12) {
                vec scores = (D_scaled[f_idx] * resid.t()) * (unmixed_curr[r_curr] * inv_rn);
                std::vector<uword> topK = find_top_k(scores, k_opt);
                for (uword v_idx : topK) {
                  rowvec backup = spectra_curr.row(r_curr);
                  spectra_curr.row(r_curr) = v_mats[f_idx].row(v_idx);
                  t_unmix = solve(spectra_curr.t(), cell_resid_raw.t(), solve_opts::fast).t();
                  t_resid = cell_resid_raw - (t_unmix * spectra_curr);
                  double t_err = sum(abs(t_resid));
                  if (t_err < err_final) {
                    err_final = t_err; unmixed_curr = t_unmix; resid = t_resid;
                    r_n = std::sqrt(dot(resid, resid)); inv_rn = 1.0 / (r_n + 1e-12);
                    cell_spectra_final.row(m_idx) = spectra_curr.row(r_curr);
                  } else { spectra_curr.row(r_curr) = backup; }
                }
              }
            }
            cell_unmixed = solve(cell_spectra_final.t(), cell_resid_raw.t(), solve_opts::fast).t();
          }
        }
      }
    }
    unmixed_final.row(i) = cell_unmixed;
    if(id > 0 && id <= n_combos) {
      uvec active_fc = find(flowcode_logical.row(id-1) == 1);
      vec vals(active_fc.n_elem);
      for(uword j=0; j<active_fc.n_elem; ++j) vals[j] = cell_unmixed[fc_indices[active_fc[j]]];
      double m_val = quick_median(vals); FlowCode_Intensity[i] = m_val; fc_data(i, id-1) = m_val;
    }
  }
  
  // Combine Results
  mat res(n_cells, n_fluors + 3 + n_combos);
  res.cols(0, n_fluors - 1) = unmixed_final;
  res.col(n_fluors) = AF_vals;
  res.col(n_fluors + 1) = conv_to<vec>::from(AF_idx);
  res.col(n_fluors + 2) = FlowCode_Intensity;
  res.cols(n_fluors + 3, res.n_cols - 1) = fc_data;
  return res;
}