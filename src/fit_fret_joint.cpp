#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

using namespace Rcpp;
using namespace arma;

//' @title Fit FRET Signals Joint — C++ implementation
//'
//' @description
//' C++ equivalent of \code{fit.fret.joint()}.  Selects the best-fitting FRET
//' correction spectrum per cell using a combined proportional score that
//' balances two complementary error signals:
//'
//' \itemize{
//'   \item \strong{Off-channel fluorophore spillover} (\code{p_fluor}): L1 norm
//'     of unmixed values in the fluorophores that are \emph{absent} from the
//'     cell's FlowCode barcode (the "OFF channels"), after subtracting the FRET
//'     shadow.  Expressed as a proportion of the baseline OFF-channel spillover
//'     with no FRET correction.
//'   \item \strong{Raw-space residual} (\code{p_resid}): L2 norm of the
//'     detector-space residual after subtracting the fitted FRET signal,
//'     expressed as a proportion of the baseline residual with no FRET
//'     correction.
//' }
//'
//' The combined score for variant \eqn{j} is
//' \deqn{\text{score}_j = \alpha \cdot p_{\text{resid},j}
//'       + (1-\alpha) \cdot p_{\text{fluor},j}}
//' and the variant that minimises this score is selected.
//'
//' Because FRET spectra differ per barcode, the function receives a
//' \code{List} of per-barcode FRET matrices and an integer vector
//' (\code{cell_barcode_ids}) that maps each cell to its barcode.  The OLS
//' unmixing matrix \eqn{P = (SS^T)^{-1}S} is computed once internally from
//' \code{spectra}.  Cells that belong to a placeholder barcode (a single
//' all-zero FRET row) are returned uncorrected.
//'
//' @param raw_data         Numeric matrix, cells × detectors.
//' @param unmixed          Numeric matrix, cells × fluorophores.  OLS unmixed
//'   values computed without any FRET correction.
//' @param spectra          Numeric matrix, fluorophores × detectors.
//' @param fret_spectra_list A \code{List} of numeric matrices, one per
//'   barcode ID.  Each matrix is FRET-variants × detectors.  The list is
//'   1-indexed; element \code{i} corresponds to barcode ID \code{i}.
//' @param cell_barcode_ids Integer vector (length \code{nrow(raw_data)}).
//'   Barcode ID for each cell (1-based, matching \code{fret_spectra_list}).
//' @param off_indices_list A \code{List} of integer vectors (one per barcode
//'   ID), each giving the 1-based column indices into \code{unmixed} that are
//'   the OFF-channel fluorophores for that barcode.
//' @param alpha            Numeric scalar in `[0, 1]` (default 0.5).  Weight for
//'   the raw-detector residual term relative to the off-channel fluorophore
//'   spillover term.
//' @param n_threads        Integer.  Number of OpenMP threads (default 1).
//'
//' @return A named \code{List} with two elements:
//' \describe{
//'   \item{\code{unmixed}}{Matrix (cells × fluorophores) of FRET-corrected
//'     unmixed values.}
//'   \item{\code{fitted.fret}}{Matrix (cells × detectors) of the FRET signal
//'     projected back into raw detector space.}
//' }
//'
//' @export
// [[Rcpp::export]]
 List fit_fret_joint_cpp(
     const arma::mat& raw_data,
     const arma::mat& unmixed,
     const arma::mat& spectra,
     const Rcpp::List& fret_spectra_list,
     const arma::ivec& cell_barcode_ids,
     const Rcpp::List& off_indices_list,
     double alpha     = 0.5,
     int    n_threads = 1
 ) {

   // ---- Dimensions -----------------------------------------------------------
   uword n_cells   = raw_data.n_rows;
   uword n_det     = raw_data.n_cols;
   uword n_fluors  = spectra.n_rows;
   uword n_barcodes = (uword)fret_spectra_list.size();

   // ---- Global unmixing matrix P = (S S^T)^{-1} S ---------------------------
   mat S_t = spectra.t();                             // n_det    × n_fluors
   mat P   = solve( spectra * S_t, spectra );         // n_fluors × n_det

   // ---- Pre-compute per-barcode FRET projection libraries -------------------
   //
   //   v_libraries[b]  : n_fluors × fret_n  — spillover of each variant into fluors
   //   r_libraries[b]  : n_det    × fret_n  — detector-space orthogonal complement
   //   r_dots[b]       : fret_n             — r_j · r_j denominators for k solve
   //
   // These mirror the R pre-computations:
   //   v.library <- unmixing.matrix %*% t(fret_library)
   //   r.library <- t(fret_library) - (S %*% v.library)

   std::vector< arma::mat > v_libraries( n_barcodes );
   std::vector< arma::mat > r_libraries( n_barcodes );
   std::vector< arma::vec > r_dots_list( n_barcodes );
   std::vector< arma::mat > fret_mats( n_barcodes );
   std::vector< arma::uvec > off_idx_vecs( n_barcodes );  // 0-based

   for ( uword b = 0; b < n_barcodes; ++b ) {

     // FRET spectra matrix for barcode b  (fret_n × n_det)
     arma::mat fret_b = Rcpp::as< arma::mat >( fret_spectra_list[ b ] );
     fret_mats[ b ] = fret_b;

     uword fret_n = fret_b.n_rows;

     arma::mat AF_t = fret_b.t();                      // n_det × fret_n
     arma::mat v_b  = P * AF_t;                        // n_fluors × fret_n
     arma::mat r_b  = AF_t - ( S_t * v_b );            // n_det    × fret_n

     arma::vec dots( fret_n );
     for ( uword j = 0; j < fret_n; ++j ) {
       double d  = dot( r_b.col(j), r_b.col(j) );
       dots[ j ] = ( d == 0.0 ) ? 1e-10 : d;
     }

     v_libraries[ b ] = v_b;
     r_libraries[ b ] = r_b;
     r_dots_list[ b ] = dots;

     // OFF-channel indices — convert from R 1-based to C++ 0-based
     arma::ivec off_r = Rcpp::as< arma::ivec >( off_indices_list[ b ] );
     arma::uvec off_0( off_r.n_elem );
     for ( uword k = 0; k < off_r.n_elem; ++k )
       off_0[ k ] = (uword)( off_r[ k ] - 1 );
     off_idx_vecs[ b ] = off_0;
   }

   // ---- Transpose inputs for cache-friendly column access --------------------
   mat raw_t = raw_data.t();     // n_det    × n_cells
   mat unm_t = unmixed.t();      // n_fluors × n_cells

   // ---- Output storage -------------------------------------------------------
   mat unmixed_out  ( n_fluors, n_cells );
   mat fitted_fret_t( n_det,    n_cells, fill::zeros );

   // Initialise unmixed_out with the uncorrected values (cells with placeholder
   // barcodes will be returned unchanged)
   unmixed_out = unm_t;

#ifdef _OPENMP
   omp_set_num_threads( n_threads );
#endif

   // ---- Main per-cell loop ---------------------------------------------------
#pragma omp parallel for schedule(dynamic, 64)
   for ( uword i = 0; i < n_cells; ++i ) {

     int bid_r = cell_barcode_ids[ (arma::uword)i ];  // 1-based barcode id
     if ( bid_r < 1 || (uword)bid_r > n_barcodes ) continue;
     uword b = (uword)( bid_r - 1 );                  // 0-based index

     const arma::mat& v_b    = v_libraries[ b ];
     const arma::mat& r_b    = r_libraries[ b ];
     const arma::vec& dots   = r_dots_list [ b ];
     const arma::mat& fret_b = fret_mats   [ b ];
     const arma::uvec& off_0 = off_idx_vecs[ b ];

     uword fret_n = fret_b.n_rows;

     // Skip placeholder combos (single all-zero FRET row)
     if ( fret_n == 1 ) {
       bool all_zero = true;
       for ( uword d = 0; d < n_det; ++d ) {
         if ( std::abs( fret_b(0, d) ) > 1e-15 ) { all_zero = false; break; }
       }
       if ( all_zero ) continue;
     }

     arma::vec cell_raw = raw_t.col( i );   // n_det    × 1
     arma::vec cell_unm = unm_t.col( i );   // n_fluors × 1

     // ---- Baseline errors (denominators for proportional scores) -------------

     // OFF-channel baseline (L1, unclamped)
     double base_e_fluor = 1e-6;
     for ( uword oi : off_0 )
       base_e_fluor += std::abs( cell_unm[ oi ] );

     // Raw-space baseline: residual using non-negative-clamped unmixed fit
     arma::vec cell_unm_nn = clamp( cell_unm, 0.0, datum::inf );
     arma::vec resid_0     = cell_raw - ( S_t * cell_unm_nn );
     double base_e_resid   = norm( resid_0, 2 ) + 1e-6;

     // ---- Score every FRET variant and track the best -----------------------
     double best_score = datum::inf;
     double best_k     = 0.0;
     uword  best_j     = 0;

     for ( uword j = 0; j < fret_n; ++j ) {

       // Optimal FRET scalar k (non-negative OLS):
       double k = dot( cell_raw, r_b.col(j) ) / dots[ j ];
       if ( k < 0.0 ) k = 0.0;

       // OFF-channel L1 after FRET correction
       double e_fluor = 0.0;
       for ( uword oi : off_0 )
         e_fluor += std::abs( cell_unm[ oi ] - k * v_b( oi, j ) );

       // Raw-space L2 residual after subtracting fitted FRET
       double e_resid = norm( resid_0 - k * r_b.col(j), 2 );

       // Combined proportional score — lower is better
       double score = alpha       * ( e_resid / base_e_resid )
         + ( 1.0 - alpha ) * ( e_fluor / base_e_fluor );

       if ( score < best_score ) {
         best_score = score;
         best_k     = k;
         best_j     = j;
       }
     }

     // ---- Final reconstruction for the winning FRET variant -----------------
     //
     // unmixed_corrected = cell_unm - best_k * v_{best_j}
     // fitted_fret       = best_k  * fret_spectrum_{best_j}  (detector space)
     unmixed_out.col( i )   = cell_unm - best_k * v_b.col( best_j );
     fitted_fret_t.col( i ) = best_k * fret_b.row( best_j ).t();
   }

   // ---- Transpose back to cells-in-rows layout expected by callers -----------
   mat unmixed_final    = unmixed_out.t();   // n_cells × n_fluors
   mat fitted_fret_final = fitted_fret_t.t(); // n_cells × n_det

   return List::create(
     Named("unmixed")     = unmixed_final,
     Named("fitted.fret") = fitted_fret_final
   );
 }
