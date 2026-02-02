#' Optimize FlowCode spectral unmixing
#'
#' High-performance C++ backend for FlowCode-based spectral optimization.
#'
#' @param raw_data Numeric matrix (n_cells x n_detectors)
#' @param unmixed Numeric matrix (n_cells x n_fluors)
#' @param combined_spectra Numeric matrix (n_fluors x n_detectors)
#' @param weights Numeric vector (n_detectors)
#' @param pos_thresholds Numeric vector (n_fluors)
#' @param af_idx Integer vector (n_cells), 1-based AF indices
#' @param af_spectra Numeric matrix (n_af_variants x n_detectors)
#' @param flowcode_ids Integer vector (n_cells), 0 = no flowcode
#' @param has_flowcode Integer/logical vector (n_cells)
#' @param combo_fret List of FRET spectra matrices
#' @param fret_delta_list List of delta matrices
#' @param fret_delta_norms List of norm vectors
#' @param flowcode_combo_logical Integer matrix (n_combos x n_flowcode_fluors)
#' @param flowcode_fluors Character vector
#' @param optimize_fluors Character vector
#' @param variants List of variant matrices per fluorophore
#' @param delta_list List of delta matrices per fluorophore
#' @param delta_norms List of delta norms per fluorophore
#' @param all_fluor_names Character vector
#' @param af_idx_in_spectra Integer, 0-based index of AF in spectra
#' @param k Integer, number of variants to test
#' @param weighted Logical, use weighted least squares
#' @param cell_weighting Logical, use cell-specific weights
#' @param cell_weight_regularize Logical, regularize cell weights
#' @param nthreads Integer, number of threads
#'
#' @return Numeric matrix (n_cells x n_fluors)
#'
#' @export
optimize_flowcode_unmix <- function(
    raw_data,
    unmixed,
    combined_spectra,
    weights,
    pos_thresholds,
    af_idx,
    af_spectra,
    flowcode_ids,
    has_flowcode,
    combo_fret,
    fret_delta_list,
    fret_delta_norms,
    flowcode_combo_logical,
    flowcode_fluors,
    optimize_fluors,
    variants,
    delta_list,
    delta_norms,
    all_fluor_names,
    af_idx_in_spectra,
    k = 10L,
    weighted = FALSE,
    cell_weighting = FALSE,
    cell_weight_regularize = FALSE,
    nthreads = 1L
) {
  .optimize_flowcode_unmix(
    raw_data,
    unmixed,
    combined_spectra,
    weights,
    pos_thresholds,
    af_idx,
    af_spectra,
    flowcode_ids,
    has_flowcode,
    combo_fret,
    fret_delta_list,
    fret_delta_norms,
    flowcode_combo_logical,
    flowcode_fluors,
    optimize_fluors,
    variants,
    delta_list,
    delta_norms,
    all_fluor_names,
    af_idx_in_spectra,
    as.integer(k),
    weighted,
    cell_weighting,
    cell_weight_regularize,
    as.integer(nthreads)
  )
}
