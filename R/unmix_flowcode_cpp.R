# unmix_flowcode.r

#' @title Unmix FlowCode
#'
#' @description
#' Unmix FlowCode samples, correcting FRET errors and debarcoding the data.
#'
#' @importFrom parallelly availableCores
#'
#' @param raw.data Expression data from raw FCS files. Cells in rows and
#' detectors in columns. Columns should be fluorescent data only and must
#' match the columns in spectra.
#' @param spectra Spectral signatures of fluorophores, normalized between 0
#' and 1, with fluorophores in rows and detectors in columns.
#' @param af.spectra Spectral signatures of autofluorescences, normalized
#' between 0 and 1, with fluorophores in rows and detectors in columns. Prepare
#' using `get.af.spectra`.
#' @param spectra.variants Named list (names are fluorophores) carrying matrices
#' of spectral signature variations for each fluorophore. Prepare using
#' `get.spectral.variants()`. Default is `NULL`.
#' @param flowcode.spectra Structured output from `get.flowcode.spectra()`, which
#' details the combination-level spectral unmixing errors due to FRET-like
#' artefacts.
#' @param asp The AutoSpectral parameter list. Prepare using
#' `get.autospectral.param`.
#' @param thresholds Optional named numeric vector of positivity thresholds for
#' the FlowCode fluorophores. Overrides the thresholds provided by `flowcode.spectra`.
#' Default is `NULL`, which is unused.
#' @param k Numeric, controls the number of variants tested for each fluorophore,
#' autofluorescence and FRET spectrum. Default is `10`, which will be good, `1`
#' is fastest. Values up to `10` provide additional benefit in unmixing quality.
#' @param parallel Logical, whether to use parallel processing for the per-cell
#' unmixing. Default is `FALSE`.
#' @param threads Numeric. Number of threads to use for parallel processing.
#' Defaults to `1` for sequential processing, or `0` (all cores) if `parallel=TRUE`.
#' @param verbose Logical, whether to send messages to the console.
#' Default is `TRUE`.
#' @param optimize Logical, whether to perform per-cell spectral optimization.
#' Faster without this, usually better with it.
#'
#' @return Unmixed data with cells in rows and fluorophores in columns.
#'
#' @export

unmix.flowcode.cpp <- function(
    raw.data,
    spectra,
    af.spectra,
    spectra.variants,
    flowcode.spectra,
    asp,
    thresholds = NULL,
    k = 1,
    parallel = TRUE,
    threads = if ( parallel ) 0 else 1,
    verbose = TRUE,
    optimize = TRUE
  ) {

  ### Setup-----------
  # check for AF in spectra, remove if present
  if ( "AF" %in% rownames( spectra ) )
    spectra <- spectra[ rownames( spectra ) != "AF", , drop = FALSE ]

  fluorophores <- rownames( spectra )

  # check for fatal errors
  if ( nrow( af.spectra ) < 2 )
    stop( "Multiple AF spectra must be provided." )

  # set positivity thresholds vector
  pos.thresholds <- rep( Inf, nrow( spectra ) )
  names( pos.thresholds ) <- fluorophores
  # fill with data
  pos.thresholds[ names( spectra.variants$thresholds ) ] <- spectra.variants$thresholds

  # unpack spectral variants
  variants <- spectra.variants$variants
  delta.list <- spectra.variants$delta.list
  delta.norms <- spectra.variants$delta.norms

  if ( is.null( pos.thresholds ) )
    stop( "Check that spectral variants have been calculated using get.spectra.variants" )
  if ( !( length( variants ) > 1 ) )
    stop( "Multiple fluorophore spectral variants must be provided." )

  optimize.fluors <- fluorophores[ fluorophores %in% names( variants ) ]
  if ( !( length( optimize.fluors ) > 0 ) )
    stop( "No matching fluorophores between supplied spectra and spectral variants.
             No spectral optimization performed." )


  # unpack FlowCode FRET and thresholds
  flowcode.thresholds <- flowcode.spectra$Thresholds
  fret.spectra <- flowcode.spectra$FRET
  flowcode.fluors <- flowcode.spectra$Flowcode.fluors
  combo.df <- flowcode.spectra$Combos
  flowcode.combo.logical <- flowcode.spectra$Logical.combo

  # overwrite thresholds if user has provided new ones
  if ( !is.null( thresholds ) )
    flowcode.thresholds <- thresholds

  if ( is.null( flowcode.thresholds ) )
    stop( "Check that FlowCode thresholds have been supplied correctly (use get.flowcode.spectra)." )
  if ( ! all( names( flowcode.thresholds ) %in% fluorophores ) )
    stop( "FlowCode thresholds don't match the fluorophore names supplied." )
  if ( !( length( fret.spectra ) > 1 ) )
    stop( "Multiple FRET options for FlowCode spectra must be provided." )
  ## more checks to be implemented here
  # check for match between flowcode fluors and `fluorophores`

  # if delta.list and delta.norms are not provided by AutoSpectral (<v1.0.0), calculate
  if ( is.null( delta.list ) ) {
    # calculate deltas for each fluorophore's variants
    delta.list <- lapply( optimize.fluors, function( fl ) {
      variants[[ fl ]] - matrix(
        spectra[ fl, ],
        nrow = nrow( variants[[ fl ]] ),
        ncol = ncol( spectra ),
        byrow = TRUE
      )
    } )
    names( delta.list ) <- optimize.fluors

    # precompute delta norms
    delta.norms <- lapply( delta.list, function( d ) {
      sqrt( rowSums( d^2 ) )
    } )
  }

  # set number of threads to use
  if ( parallel ) {
    if ( is.null( threads ) ) threads <- asp$worker.process.n
    if ( threads == 0 ) threads <- parallelly::availableCores()
  } else {
    threads <- 1
  }

  # Pre-pasting combos for debarcoding
  valid.combos <- apply(
    combo.df[ , -1 ], 1, function( x ) paste( sort( toupper( x ) ), collapse = "_" ) )
  fc.tags <- toupper( names( flowcode.fluors ) )
  
  # call C++ pipeline
  results <- unmix_flowcode_pipeline_cpp(
    raw_data = as.matrix( raw.data ),
    spectra = as.matrix( spectra ),
    af_spectra = as.matrix( af.spectra ),
    fluor_names = rownames( spectra ),
    flowcode_fluors = as.character( flowcode.fluors ),
    flowcode_tags = fc.tags,
    flowcode_thresholds = as.numeric( flowcode.thresholds ),
    valid_combos = valid.combos,
    flowcode_logical = as.matrix( flowcode.combo.logical ),
    fret_spectra_list = fret.spectra,
    pos_thresholds = as.numeric( pos.thresholds ),
    optimize_fluors = optimize.fluors,
    variants = variants,
    delta_list = delta.list,
    delta_norms = delta.norms,
    k_opt = k,
    n_threads = threads,
    optimize = optimize
  )
  
  # Final assembly of the return matrix
  unmixed <- cbind(
    results$unmixed, 
    AF = results$AF, 
    `AF Index` = results$af.idx, 
    FlowCode = results$FlowCode, 
    results$fc.data
  )
  
  colnames( unmixed ) <- c(
    fluorophores, "AF", "AF Index", "FlowCode", paste( "Tag:", combo.df$Id ) )

  return( unmixed )
}
