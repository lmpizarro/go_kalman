package main
//
// https://en.wikipedia.org/wiki/Kalman_filter
//
// https://medium.com/wireless-registry-engineering/gonum-tutorial-linear-algebra-in-go-21ef136fc2d7
//
// https://github.com/gonum/matrix/blob/master/mat64/matrix_test.go
//

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	)

func matPrint(X mat.Matrix) {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// eye returns a new identity matrix of size n×n.
func eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

/*
	arguments

		P -> P0.0 the previous updated covariance

		A -> System Matrix
	
		Q -> covariance process noise


	return P1.0 the predicted covariance
*/
func predicted_cov_p(A, P, Q * mat.Dense) * mat.Dense {

	P.Product(A, P, A.T())
	P.Add(Q,P)
	return P
}

func predicted_estate(A, B, X, U * mat.Dense) * mat.Dense {
	X.Product(A, X)
	U.Product(B,U)
	X.Add(X,U)
	return X
}

func innovation_residual(M, H, X *mat.Dense) * mat.Dense {
	Z := mat.NewDense(1, 1, []float64{0})
	Z.Product(H,X)
	M.Sub(M, Z)
	return M
}

func innovation_cov(H,P,R *mat.Dense) *mat.Dense{
	Z := mat.NewDense(1, 1, []float64{0})
	Z.Product(H, P, H.T())
	R.Add(R,Z)
	return R
}

func kalman_gain(P, H, S * mat.Dense) * mat.Dense {
	S.Inverse(S)
	_, c := H.Dims()
	p := make([]float64, c)
	K := mat.NewDense(c, 1, p)
	K.Product(P, H.T(), S)
	return K
}

func update_estate(X, K, y_tilde *mat.Dense) *mat.Dense {
	rk,_ := K.Dims()
	_, cy := y_tilde.Dims()
	p := mat.NewDense(rk, cy, nil)
	p.Product(K, y_tilde)
	X.Add(X, p)
	return X
}

func update_cov_p (P, K, H *mat.Dense) *mat.Dense{
 
	rk,_ := K.Dims()
	_, cy := H.Dims()

	p := mat.NewDense(rk, cy, nil)
	p.Product(K, H)
	r, _ := P.Dims()
	identity := eye(r)

	identity.Sub(identity, p)
	P.Product(identity, P)

	return P
}
func main(){
	p := make([]float64, 4)
	p[0] = 1.0
	p[1] = 2.0
	p[2] = 2.0
	p[3] = 1.0
	
	P := mat.NewDense(2, 2, p)

	A := mat.NewDense(2, 2, []float64{1,0,0,1})
	Q := mat.NewDense(2, 2, []float64{1,0,0,1})
	B := mat.NewDense(2, 2, []float64{1,0,0,1})
	X := mat.NewDense(2, 1, []float64{1,0,})
	U := mat.NewDense(2, 1, []float64{1,0})
	H := mat.NewDense(1, 2, []float64{1,0})
	M := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{1})
	

	S := innovation_cov(H,P,R)
	K := kalman_gain(P, H, S)
	matPrint(K)
	y_tilde := innovation_residual(M, H, X)
	matPrint(y_tilde)
	update_estate(X,K,y_tilde)
	update_cov_p(P, K, H)
	matPrint(predicted_cov_p(P,A,Q))
	matPrint(predicted_estate(A,B,X,U))	
}