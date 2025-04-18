% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/data.R
\name{createOutcome}
\alias{createOutcome}
\title{Create an outcome object}
\usage{
createOutcome(outcome)
}
\arguments{
\item{outcome}{Vector of outcome values}
}
\value{
\code{Outcome} object
}
\description{
Create an outcome object
}
\examples{
X <- matrix(runif(10*100), ncol = 10)
y <- -5 + 10*(X[,1] > 0.5) + rnorm(100)
outcome <- createOutcome(y)
}
