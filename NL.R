# This code is licensed under the GNU General Public License v3.0
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# *********************************************
#
# Implemenation of "Natural Learning"
#
# website: www.natural-learning.cc
# Author: Hadi Fanaee-T
# Associate Professor of Machine Learning
# School of Information Technology
# Halmstad University, Sweden
# Email: hadi.fanaee@hh.se
#
#
# Please cite the following paper if you use the code
#
# *********************************************
# Hadi Fanaee-T, "Natural Learning", arXiv:2404.05903
# https://arxiv.org/abs/2404.05903
#
# *********************************************
# BibTeX
# *********************************************
#
# @article{fanaee2024natural,
#   title={Natural Learning},
#   author={Fanaee-T, Hadi},
#   journal={arXiv preprint arXiv:2404.05903},
#   year={2024}
#}
#

# LSH Hash
LSH <- function(X, num_hash_functions = 10, hash_size = 100) {
  n <- nrow(X)
  m <- ncol(X)
  set.seed(42)
  hash_functions <- matrix(rnorm(m * num_hash_functions), nrow = m, ncol = num_hash_functions)
  hash_codes <- sign(X %*% hash_functions)
  hash_tables <- vector("list", length = hash_size)
  
  for (i in 1:n) {
    hash_code <- hash_codes[i, ]
    hash_index <- sum(hash_code) %% hash_size + 1
    
    if (is.null(hash_tables[[hash_index]])) {
      hash_tables[[hash_index]] <- i
    } else {
      hash_tables[[hash_index]] <- c(hash_tables[[hash_index]], i)
    }
  }
  
  list(hash_functions = hash_functions, hash_tables = hash_tables)
}

# LSH Query
query <- function(query_point, X, hash_functions, hash_tables, hash_size) {
  query_hash_code <- sign(query_point %*% hash_functions)
  query_hash_index <- sum(query_hash_code) %% hash_size + 1
  candidate_neighbors <- hash_tables[[query_hash_index]]
  best_distance <- Inf
  best_neighbor <- -1
  
  for (i in candidate_neighbors) {
    candidate_point <- X[i, ]
    distance <- sqrt(sum((candidate_point - query_point)^2))
    
    if (distance < best_distance && distance != 0) {
      best_distance <- distance
      best_neighbor <- i
    }
  }
  
  return(best_neighbor)
}

# NL Training
NL <- function(X, y) {
  n <- nrow(X)
  p <- ncol(X)
  M <- seq_len(p)
  ids0 <- which(y == 0)
  ids1 <- which(y == 1)
  L <- 0
  e_best <- Inf
  
  while (TRUE) {
    L <- L + 1
    Mdl0 <- LSH(X[ids0, M])
    Mdl1 <- LSH(X[ids1, M])
    
    for (i in 1:n) {
      s <- query(X[i, M], X[ids0, M], Mdl0$hash_functions, Mdl0$hash_tables, length(Mdl0$hash_tables))
      
      if (s != -1) {
        s <- ids0[s]
        o <- query(X[i, M], X[ids1, M], Mdl1$hash_functions, Mdl1$hash_tables, length(Mdl1$hash_tables))
        
        if (o != -1) {
          o <- ids1[o]
          
          if (y[i] == 1) {
            tmp_o <- o
            o <- s
            s <- tmp_o
          }
          
          vs <- abs(X[s, M] - X[i, M])
          vo <- abs(X[o, M] - X[i, M])
          v <- vo - vs
          C <- M[v > 0]
          
          if (length(C) > 1) {
            
            yhat <- NULL
            
            for (r in 1:n) {
              distances <- sqrt(sum((X[r, C] - X[c(s), C])^2))
              distanceo <- sqrt(sum((X[r, C] - X[c(o), C])^2))
              if (distances<=distanceo){
                yhat[c(r)]<-y[c(s)]
              }else{
                yhat[c(r)]<-y[c(o)]
              }
            }  
            
            e <- sum(yhat != y)
            
            if (!is.na(e) && e < e_best) {
              e_best <- e
              C_best <- C
              s_best <- s
              o_best <- o
              L_best <- L
              cat(sprintf("Layer=%d Sample=%d/%d Prototype=[%d,%d], NumFeatures=%d/%d, Error=%.4f\n", 
                          L, i, n, s_best, o_best, length(C_best), length(M), e_best/n))
            }
          }
        }
      }
    }
    
    if (length(C_best) == length(M)) break
    else {
      M <- C_best
      e_best <- Inf
    }
  }
  
  bp <- paste("[", s_best, " (class ", y[s_best], "), ", o_best, " (class ", y[o_best], ")]", sep = "")
  features <- paste(M, collapse = " ")
  cat("Best Prototypes=", bp, ", BestError=", e_best / n, ", Prototype Features=[", features, "]\n", sep = "")
  
  list(
    PrototypeSampleIDs = c(s_best, o_best),
    PrototypeFeatureIDs = M,
    Error = e_best / n,
    NLayers = L_best,
    MX = X[c(s_best, o_best), M],
    My = y[c(s_best, o_best)]
  )
}

# Predict function
predict <- function(Mdl,X_test) {
  y_pred <- NULL
  n <- nrow(X_test)
  for (r in 1:n) {
    distance_s <- sqrt(sum((X_test[r, Mdl$PrototypeFeatureIDs] - Mdl$MX[1,])^2))
    distance_o <- sqrt(sum((X_test[r, Mdl$PrototypeFeatureIDs] - Mdl$MX[2,])^2))
    if (distance_s<=distance_o){
      y_pred[c(r)]<-Mdl$My[1]
    }else{
      y_pred[c(r)]<-Mdl$My[2]
    }
  }  
  return(y_pred)
}

#setwd("CURDIR")

# Read data
X_train <- as.matrix(read.csv('X_train.csv', header = FALSE))
X_test <- as.matrix(read.csv('X_test.csv', header = FALSE))
y_train <- as.integer(unlist(read.csv('y_train.csv', header = FALSE)))
y_test <- as.integer(unlist(read.csv('y_test.csv', header = FALSE)))

# Train NL model
Mdl <- NL(X_train, y_train)
y_pred <- predict(Mdl,X_test)
err_test <- sum(y_pred != y_test)/nrow(X_test)
cat("Test Error =", as.character(err_test), "\n")

