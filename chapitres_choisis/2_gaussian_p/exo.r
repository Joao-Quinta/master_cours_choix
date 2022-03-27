library(plgp)


kernel_SE_2 <- function(XX,X){
  TMP <- matrix(nrow=ncol(XX), ncol=ncol(X))
  for (r in 1:ncol(TMP))
    for (c in 1:ncol(TMP)){
      TMP[r,c] <- exp(-(XX[1,r]-X[1,c])^2 - (XX[2,r]-X[2,c])^2)}
  TMP
}
# training data -> known points f(x)=y
n <- 8
# x are n evenly spaced values from 0 to 2*pi
x <- matrix(seq(0, 2*pi, length=n), ncol=1)
# y are the images for points x, f is sin
y <- sin(x)
# y ← c(1,3,2,5,5,7,3,4) # another possibility
# D is the euclidean distance between xi and xj
D <- distance(x)
# covariance matrix can be computed from D -> covariance
e <- 0.001
sigma_yy <- exp(-D) + diag(e,ncol(D))


# m is the number of points in x*, that we want to predict an y*
m <- 8
# we compute all points, they range from -0,5 to 2*pi+0.5
# we can see that the range for x* is larger than x
x_e <- matrix(seq(0, 2*pi, length=m), ncol=1)

# we can compute the distance matrix for x*
D_ee <- distance(x_e)
# as well as the corresponding covariance matrix
sigma_yeye <- exp(-D_ee) + diag(e,ncol(D_ee))

# finally we need to compute the covariance matrix between y* and y
# we do this by first computing the distances between x and x*
D_e <- distance(x_e, x)
# and then we simply compute the covariance matrix as before from D*
sigma_yey <- exp(-D_e)

# the inverse of a matrix is found by using function solve()
sigma_yy_inv <- solve(sigma_yy) # = inv(cov(D(x)))

# with inverse of the covariance matrix of x -> inv(cov(D(x)))
# we can compute A and b as seen in class
A <- sigma_yeye - sigma_yey %*% sigma_yy_inv %*% t(sigma_yey)
A <- 0.5 * (A+ t(A))
b <- sigma_yey %*% sigma_yy_inv %*% y

# finally we just need to sample, here we want to sample 100 functions
sample <- 100
y_e <- rmvnorm(sample,b,A) # sample 100 functions


matplot(x_e, t(y_e), type = "l", col = "gray", lty=1, xlab = "x", ylab = "y")
points(x,y,pch=20,cex=2)
lines(x_e, b, lwd=2)
lines(x_e, sin(x_e), col="blue")
q1 <- b + qnorm(0.05,0,sqrt(diag(A)))
q2 <- b + qnorm(0.95,0,sqrt(diag(A)))
lines(x_e, q1, lwd=2, lty=2, col="red")
lines(x_e, q2, lwd=2, lty=2, col="red")

