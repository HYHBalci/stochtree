% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{createPreprocessorFromJsonString}
\alias{createPreprocessorFromJsonString}
\title{Reload a covariate preprocessor object from a JSON string containing a serialized preprocessor}
\usage{
createPreprocessorFromJsonString(json_string)
}
\arguments{
\item{json_string}{in-memory JSON string containing covariate preprocessor metadata}
}
\value{
Preprocessor object that can be used with the \code{preprocessPredictionData} function
}
\description{
Reload a covariate preprocessor object from a JSON string containing a serialized preprocessor
}
\examples{
cov_mat <- matrix(1:12, ncol = 3)
preprocess_list <- preprocessTrainData(cov_mat)
preprocessor_json_string <- savePreprocessorToJsonString(preprocess_list$metadata)
preprocessor_roundtrip <- createPreprocessorFromJsonString(preprocessor_json_string)
}
