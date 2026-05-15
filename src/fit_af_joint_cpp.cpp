#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

//' @title Fit Autofluorescence Joint — C++ implementation
//'
//' @description
//' C++ equivalent of \code{fit.af.joint()}. Selects the best-fitting
//' autofluorescence spectrum per cell using a combined proportional score that
//' balances two complementary error signals:
//'
//' \itemize{
//'   \item \strong{Fluorophore spillover} (\code{p_fluor}): L1 norm of unmixed
//'     fluorophore values after subtracting the AF shadow, expressed as a
//'     proportion of the baseline spillover with no AF correction at all.
//'   \item \strong{Raw-space residual} (\code{p_resid}): L2 norm of the
//'     detector-space residual after subtracting the fitted AF signal,
//'     expressed as a proportion of the baseline residual with no AF
//'     correction.
//' }
//'
//' The combined score for variant \eqn{j} is
//' \deqn{\text{score}_j = \alpha \cdot p_{\text{resid},j}
//'       + (1-\alpha) \cdot p_{\text{fluor},j}}
//' and the variant that minimises this score is selected.
//'
//' The unmixing matrix \eqn{P = (SS^T)^{-1}S} is computed internally from
//' \code{spectra}. Any \code{"AF"} row in \code{spectra} is silently stripped
//' before use, matching the behaviour of the R implementation.
//'
//' @param raw_data       Numeric matrix, cells × detectors.
//' @param unmixed        Numeric matrix, cells × fluorophores. OLS unmixed
//'   values computed without any AF correction (fluorophore spectra only).
//' @param spectra        Numeric matrix, fluorophores × detectors. Row names
//'   must be set; any row named \code{"AF"} is dropped automatically.
//' @param af_spectra     Numeric matrix, AF variants × detectors.
//' @param alpha          Numeric scalar in `[0, 1]` (default 0.5). Weight for
//'   the raw-detector residual term relative to the fluorophore-spillover
//'   term.
//' @param n_threads      Integer. Number of OpenMP threads (default 1).
//'
//' @return A named \code{List} with four elements:
//' \describe{
//'   \item{\code{unmixed}}{Matrix (cells × fluorophores) of AF-corrected
//'     unmixed values.}
//'   \item{\code{AF}}{Numeric vector (length \code{nrow(raw_data)}) of
//'     per-cell AF intensities (the fitted scalar \eqn{k}).}
//'   \item{\code{af.idx}}{Integer vector of the AF variant index selected per
//'     cell (1-based).}
//'   \item{\code{fitted.af}}{Matrix (cells × detectors) of the AF signal
//'     projected back into raw detector space.}
//' }
//'
//' @export
// [[Rcpp::export]]
 List fit_af_joint_cpp(
     const arma::mat& raw_data,
     const arma::mat& unmixed,
     const arma::mat& spectra,
     const arma::mat& af_spectra,
     double            alpha     = 0.5,
     int               n_threads = 1
 ) {

   // ---- Dimensions -----------------------------------------------------------
   uword n_cells   = raw_data.n_rows;
   uword n_det     = raw_data.n_cols;
   uword n_fluors  = spectra.n_rows;
   uword n_af      = af_spectra.n_rows;

   // ---- Pre-compute AF projection libraries (shared across cells) ------------
   //
   //   P            : OLS unmixing matrix  (n_fluors × n_det)
   //   S_t          : spectra transposed   (n_det    × n_fluors)
   //   v_library    : how much AF variant j leaks into each fluorophore
   //                  (n_fluors × n_af)
   //   r_library    : AF variant j orthogonal to fluorophore space
   //                  (n_det    × n_af)
   //   r_dots       : r_j · r_j  denominators for the k OLS solve
   //                  (n_af)
   //
   // These mirror exactly the pre-computations in the R implementation:
   //   P          <- solve(spectra %*% t(spectra)) %*% spectra
   //   v.library  <- P %*% t(af.spectra)
   //   r.library  <- t(af.spectra) - S %*% v.library

   mat S_t        = spectra.t();                          // n_det    × n_fluors
   mat AF_t       = af_spectra.t();                       // n_det    × n_af
   mat P          = solve(spectra * S_t, spectra);        // n_fluors × n_det
   mat v_library  = P   * AF_t;                          // n_fluors × n_af
   mat r_library  = AF_t - (S_t * v_library);            // n_det    × n_af

   vec r_dots(n_af);
   for (uword j = 0; j < n_af; ++j) {
     double d   = dot(r_library.col(j), r_library.col(j));
     r_dots[j]  = (d == 0.0) ? 1e-10 : d;
   }

   // ---- Transpose raw_data for cache-friendly column access ------------------
   //
   // raw_t.col(i) gives detector readings for cell i as a contiguous vector,
   // which is faster in the inner loop than raw_data.row(i).
   mat raw_t = raw_data.t();     // n_det × n_cells
   mat unm_t = unmixed.t();      // n_fluors × n_cells

   // ---- Output storage -------------------------------------------------------
   mat  unmixed_out(n_fluors, n_cells);   // filled as columns, transposed at end
   vec  AF_vals(n_cells,     fill::zeros);
   ivec af_idx(n_cells,      fill::zeros);
   mat  fitted_af_out(n_det, n_cells, fill::zeros);  // transposed at end

#ifdef _OPENMP
   omp_set_num_threads(n_threads);
#endif

   // ---- Main per-cell loop ---------------------------------------------------
#pragma omp parallel for schedule(dynamic, 64)
   for (uword i = 0; i < n_cells; ++i) {

     vec cell_raw  = raw_t.col(i);    // n_det    × 1
     vec cell_unm  = unm_t.col(i);   // n_fluors × 1

     // ---- Baseline errors (denominators for proportional scores) -------------
     //
     // base_e_fluor : L1 norm of unmixed values (no AF correction at all).
     //                Matches R: rowSums(abs(unmixed)) + 1e-6
     //
     // base_e_resid : L2 norm of detector-space residual when using the
     //                non-negative-clamped unmixed fit back into raw space.
     //                Matches R: raw - (clamp(unmixed, 0) %*% spectra)
     double base_e_fluor = sum(abs(cell_unm)) + 1e-6;

     vec cell_unm_nn = clamp(cell_unm, 0.0, datum::inf);   // n_fluors × 1
     vec resid_0     = cell_raw - (S_t * cell_unm_nn);     // n_det    × 1
     double base_e_resid = norm(resid_0, 2) + 1e-6;

     // ---- Score every AF variant and track the best --------------------------
     double best_score = datum::inf;
     double best_k     = 0.0;
     uword  best_j     = 0;

     for (uword j = 0; j < n_af; ++j) {

       // Optimal AF scalar k for this variant (non-negative OLS):
       //   k = (raw · r_j) / (r_j · r_j),  clamped to >= 0
       double k = dot(cell_raw, r_library.col(j)) / r_dots[j];
       if (k < 0.0) k = 0.0;

       // Fluorophore-spillover after AF correction  (L1, unclamped)
       //   adj_f = cell_unm - k * v_j
       double e_fluor = sum(abs(cell_unm - k * v_library.col(j)));

       // Raw-detector residual after subtracting fitted AF  (L2)
       //   adj_resid = resid_0 - k * r_j
       double e_resid = norm(resid_0 - k * r_library.col(j), 2);

       // Combined proportional score — lower is better
       double score = alpha       * (e_resid / base_e_resid)
         + (1.0-alpha) * (e_fluor / base_e_fluor);

       if (score < best_score) {
         best_score = score;
         best_k     = k;
         best_j     = j;
       }
     }

     // ---- Final reconstruction for the winning AF variant -------------------
     //
     // unmixed_corrected = cell_unm - best_k * v_{best_j}
     // fitted_af         = best_k  * af_spectra_{best_j}   (in detector space)
     unmixed_out.col(i) = cell_unm - best_k * v_library.col(best_j);
     fitted_af_out.col(i) = best_k * AF_t.col(best_j);

     AF_vals[i] = best_k;
     af_idx[i]  = (int)best_j + 1;   // 1-based, matching R
   }

   // ---- Transpose back to cells-in-rows layout expected by callers -----------
   mat unmixed_final  = unmixed_out.t();   // n_cells × n_fluors
   mat fitted_af_final = fitted_af_out.t(); // n_cells × n_det

   return List::create(
     Named("unmixed")   = unmixed_final,
     Named("AF")        = AF_vals,
     Named("af.idx")    = af_idx,
     Named("fitted.af") = fitted_af_final
   );
 }
