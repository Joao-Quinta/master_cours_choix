library(plgp)

# kernel for 2D problem -> given in class
kernel_SE_2 <- function(XX,X){
  TMP <- matrix(nrow=ncol(XX), ncol=ncol(X))
  for (r in 1:nrow(TMP))
    for (c in 1:ncol(TMP)){
      TMP[r,c] <- exp(-(XX[1,r]-X[1,c])^2 - (XX[2,r]-X[2,c])^2)}
  TMP
}

# training data -> known (locations,depth)
n = 5
X <- matrix(nrow=2,ncol=5)
X[1,1:n] <- c(1,6,11,16,21)# x axis locations
X[2,1:n] <- c(1,6,11,16,21)# y axis locations
Y <- c(1, 10, 5, 7, -1)

#lets consider the x* locations, this is from A to B
# A = (0,0)
# B = (22,22)
m <- 23
X_e <- matrix(nrow=2, ncol=m*m-1)
for (i in 1:m-1){
  for (j in 1:m-1){
    cc <- i*m+j
    X_e[1,cc] <- i
    X_e[2,cc] <- j
  }
}

Y_e <- matrix(nrow=1, ncol=m*m-1)

# Gaussian processes applied to this problem
sigma_yy <- kernel_SE_2(X,X)
sigma_yy_inv <- solve(sigma_yy)
sigma_yey <- kernel_SE_2(X_e,X)


sigma_yeye <- kernel_SE_2(X_e, X_e)

# compute A and b
A <- sigma_yeye - sigma_yey %*% sigma_yy_inv %*% t(sigma_yey)
A <- 0.5 * (A+ t(A))
b <- sigma_yey %*% sigma_yy_inv %*% Y

sample <- 20
Y_e <- rmvnorm(sample,b,A) # sample 100 functions

avg_Y_e <- matrix(nrow = 1, ncol = ncol(Y_e))
for (i in 1:ncol(Y_e)){
  sum <- 0
  for (j in 1:sample){
    sum <- sum + Y_e[j,i]
  }
  sum <- sum/sample
  avg_Y_e[i] <- sum
}





