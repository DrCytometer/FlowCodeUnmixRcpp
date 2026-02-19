#include <RcppArmadillo.h>
#include <string>
#include <vector>
#include <algorithm>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

// Helper: Fast median
double quick_median(vec x) {
  if(x.n_elem == 0) return 0;
  std::vector<double> v = conv_to<std::vector<double>>::from(x);
  std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
  return v[v.size()/2];
}

// [[Rcpp::export]]
List unmix_flowcode_pipeline_cpp(
    arma::mat raw_data,
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
  uword n_cells = raw_data.n_rows;
  uword n_fluors = spectra.n_rows;
  uword n_af = af_spectra.n_rows;
  int n_combos = valid_combos.size();
  
  // --- 1. SERIAL PRE-CALCULATIONS & DEEP COPIES ---
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
  std::vector<std::string> cpp_opt_names = as<std::vector<std::string>>(optimize_fluors);
  std::vector<std::string> v_names = as<std::vector<std::string>>(variants.names());
  std::vector<std::string> fc_fluor_names = as<std::vector<std::string>>(flowcode_fluors);
  std::vector<std::string> fc_tag_cpp = as<std::vector<std::string>>(flowcode_tags);
  std::vector<double> fc_thresh_cpp = as<std::vector<double>>(flowcode_thresholds);
  
  std::map<std::string, int> name_to_idx;
  for(size_t i = 0; i < cpp_names.size(); ++i) name_to_idx[cpp_names[i]] = (int)i;
  
  uvec fc_indices(fc_fluor_names.size());
  for(size_t i = 0; i < fc_fluor_names.size(); ++i) fc_indices[i] = name_to_idx[fc_fluor_names[i]];
  
  std::map<std::string, int> combo_map;
  for(int i = 0; i < n_combos; ++i) combo_map[as<std::string>(valid_combos[i])] = i + 1;
  
  size_t n_var = variants.size();
  std::vector<mat> v_mats(n_var);
  std::vector<mat> d_mats(n_var);
  std::vector<vec> dn_vecs(n_var);
  std::vector<int> var_to_master(n_var);
  std::vector<bool> should_opt(n_var, false);
  
  for (size_t f = 0; f < n_var; ++f) {
    v_mats[f] = as<mat>(variants[f]);
    d_mats[f] = as<mat>(delta_list[f]);
    dn_vecs[f] = as<vec>(delta_norms[f]);
    bool is_req = false;
    for (const auto& name : cpp_opt_names) if (name == v_names[f]) { is_req = true; break; }
    if (is_req && !dn_vecs[f].is_empty() && (max(dn_vecs[f]) > 1e-12)) should_opt[f] = true;
    var_to_master[f] = name_to_idx.count(v_names[f]) ? name_to_idx[v_names[f]] : -1;
  }
  
  std::vector<mat> fret_libs_cpp(fret_spectra_list.size());
  for(int i = 0; i < (int)fret_spectra_list.size(); ++i) fret_libs_cpp[i] = as<mat>(fret_spectra_list[i]);
  
  // --- 2. OUTPUT CONTAINERS ---
  mat unmixed_final(n_cells, n_fluors);
  vec AF_vals(n_cells);
  ivec AF_idx(n_cells);
  ivec flowcode_ids(n_cells);
  vec FlowCode_Intensity(n_cells, fill::zeros);
  mat fc_data(n_cells, n_combos, fill::zeros);
  
#ifdef _OPENMP
  omp_set_num_threads(n_threads);
#endif
  
  // --- 3. PARALLEL LOOP WITH THREAD-LOCAL STORAGE ---
#pragma omp parallel for schedule(dynamic, 64)
  for(uword i = 0; i < n_cells; ++i) {
    // TLS Scratchpads (persistent across loop iterations for each thread)
    static thread_local rowvec cell_unmixed, cell_resid_raw, unmixed_curr, resid, t_unmix, t_resid;
    static thread_local mat spectra_curr;
    static thread_local std::vector<std::string> pos_tag_aliases;
    static thread_local std::vector<std::pair<double, std::string>> tag_signals;
    static thread_local std::vector<std::pair<double, size_t>> fluor_order;
    static thread_local std::vector<uword> m_to_c_row, off_idx;
    
    // Reset Scratchpads
    pos_tag_aliases.clear(); tag_signals.clear(); fluor_order.clear(); off_idx.clear();
    if(m_to_c_row.size() != n_var) m_to_c_row.assign(n_var, 0);
    
    rowvec cell_raw = raw_data.row(i);
    
    // A. AF Extraction
    vec init_f = (P * cell_raw.t());
    double min_err_af = datum::inf; double b_k_af = 0; uword b_idx_af = 0;
    for(uword j=0; j<n_af; ++j) {
      double k = dot(cell_raw, r_library_af.col(j)) / r_dots_af[j];
      if(k < 0) k = 0;
      double err = sum(abs(init_f - (k * v_library_af.col(j))));
      if(err < min_err_af) { min_err_af = err; b_k_af = k; b_idx_af = j; }
    }
    AF_vals[i] = b_k_af; AF_idx[i] = (int)b_idx_af + 1;
    cell_resid_raw = cell_raw - (b_k_af * af_spectra.row(b_idx_af));
    cell_unmixed = (P * cell_resid_raw.t()).t();
    
    // B. Debarcoding
    for(uword j = 0; j < fc_indices.n_elem; ++j) {
      double signal = cell_unmixed[fc_indices[j]];
      double diff = signal - fc_thresh_cpp[j];
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
    
    // C. FRET Correction
    if(id > 0 && id <= n_combos) {
      const mat& f_lib = fret_libs_cpp[id-1];
      mat v_lib_f = P * f_lib.t(), r_lib_f = f_lib.t() - (S_t * v_lib_f);
      uvec active_mask = find(flowcode_logical.row(id-1) == 1);
      for(uword f=0; f<n_fluors; ++f) {
        bool active = false;
        for(uword a=0; a<active_mask.n_elem; ++a) if(fc_indices[active_mask[a]] == f) active=true;
        if(!active) off_idx.push_back(f);
      }
      double min_err_fret = datum::inf, b_k_fret = 0; uword b_idx_fret = 0;
      for(uword j=0; j<f_lib.n_rows; ++j) {
        double den = dot(r_lib_f.col(j), r_lib_f.col(j));
        double k = dot(cell_resid_raw, r_lib_f.col(j)) / (den == 0 ? 1e-10 : den);
        if(k < 0) k = 0;
        double err = 0;
        for(uword o : off_idx) err += std::abs(cell_unmixed[o] - (k * v_lib_f(o, j)));
        if(err < min_err_fret) { min_err_fret = err; b_k_fret = k; b_idx_fret = j; }
      }
      cell_resid_raw -= b_k_fret * f_lib.row(b_idx_fret);
      cell_unmixed -= (b_k_fret * v_lib_f.col(b_idx_fret)).t();
    }
    
    // D. Optimization
    mat cell_spectra_final = spectra;
    if (optimize) {
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
          for (size_t f = 0; f < n_var; ++f) {
            if (!should_opt[f]) continue;
            int m_idx = var_to_master[f];
            for(uword p = 0; p < pos_idx.n_elem; ++p) if((int)pos_idx[p] == m_idx) { fluor_order.push_back({unmixed_curr[p], f}); m_to_c_row[f] = p; break; }
          }
          std::sort(fluor_order.begin(), fluor_order.end(), std::greater<>());
          for (auto const& pair : fluor_order) {
            size_t f_idx = pair.second; int m_idx = var_to_master[f_idx]; uword r_curr = m_to_c_row[f_idx]; 
            if (r_n > 1e-12) {
              vec sc = (d_mats[f_idx] * resid.t()) * unmixed_curr[r_curr] / (dn_vecs[f_idx] * r_n);
              uvec topK = sort_index(sc, "descend");
              for (uword v_i = 0; v_i < std::min((uword)k_opt, (uword)topK.n_elem); ++v_i) {
                rowvec backup = spectra_curr.row(r_curr);
                spectra_curr.row(r_curr) = v_mats[f_idx].row(topK[v_i]);
                t_unmix = solve(spectra_curr.t(), cell_resid_raw.t(), solve_opts::fast).t();
                t_resid = cell_resid_raw - (t_unmix * spectra_curr);
                double t_err = sum(abs(t_resid));
                if (t_err < err_final) { err_final = t_err; unmixed_curr = t_unmix; resid = t_resid; r_n = std::sqrt(dot(resid, resid)); cell_spectra_final.row(m_idx) = spectra_curr.row(r_curr); }
                else { spectra_curr.row(r_curr) = backup; }
              }
            }
          }
          cell_unmixed = solve(cell_spectra_final.t(), cell_resid_raw.t(), solve_opts::fast).t();
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
  return List::create(_["unmixed"] = unmixed_final, _["AF"] = AF_vals, _["af.idx"] = AF_idx, _["flowcode.ids"] = flowcode_ids, _["FlowCode"] = FlowCode_Intensity, _["fc.data"] = fc_data);
}