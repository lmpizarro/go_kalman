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

func zero(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i +=  1 {
		d[i] = 0
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

	r, c := P.Dims()
	p := mat.NewDense(r, c, nil)

	p.Product(A, P, A.T())
	p.Add(Q, p)
	return p
}

func predicted_estate(A, B, X, U * mat.Dense) * mat.Dense {
	rx, cx := X.Dims()
	ru, cu := U.Dims()


	x := mat.NewDense(rx, cx, nil)
	u := mat.NewDense(ru, cu, nil)

	x.Product(A, X)
	u.Product(B,U)

	x.Add(X,u)

	return x
}

func innovation_residual(M, H, X *mat.Dense) * mat.Dense {
	r, _ := H.Dims()
	_, c := X.Dims()

	z := mat.NewDense(r, c, []float64{0})
	z.Product(H,X)
	matPrint(z)
	z.Scale(-1.0, z)
	// a <- meas b <- HX
	z.Add(z, M)
	return z
}

func innovation_cov(H, P10, R *mat.Dense) *mat.Dense{
	r, _ := H.Dims()
	Z := mat.NewDense(r, r, []float64{0})

	Z.Product(H, P10, H.T())
	Z.Add(R,Z)
	return Z
}

func kalman_gain(P10, H, S * mat.Dense) * mat.Dense {
	S.Inverse(S)
	_, c := H.Dims()
	p := make([]float64, c)
	K := mat.NewDense(c, 1, p)
	K.Product(P10, H.T(), S)
	return K
}

func update_estate(X, K, y_tilde *mat.Dense) *mat.Dense {
	rk,_ := K.Dims()
	_, cy := y_tilde.Dims()
	p := mat.NewDense(rk, cy, nil)
	p.Product(K, y_tilde)
	p.Add(X, p)
	return p
}

func update_cov_p(P, K, H *mat.Dense) *mat.Dense{
 
	rk,_ := K.Dims()
	_, cy := H.Dims()

	p := mat.NewDense(rk, cy, nil)
	p.Product(K, H)
	r, _ := P.Dims()
	p.Scale(-1.0, p)
	identity := eye(r)

	identity.Add(p, identity)
	identity.Product(identity, P)

	return identity
}

func main(){

	Dt := 1.0
	Dt1 := 1.0
	p1 := 5.0
	p2 := 5.0
	p3 := 45.0
	p4 := 10.0
	//  p5 := 1.0

	A := mat.NewDense(2, 2, []float64{1,Dt,0,1})
	B := mat.NewDense(2, 2, []float64{0,0,0,0})
	H := mat.NewDense(1, 2, []float64{1,0})
	matPrint(A)
	matPrint(B)
	matPrint(H)

	U := mat.NewDense(2, 1, []float64{.5*Dt1*Dt1,Dt1})
	Y := mat.NewDense(1, 1, []float64{2})
	matPrint(U)
	matPrint(Y)
	
	r_sys, _ := A.Dims()
	X00 := mat.NewDense(r_sys, 1, []float64{0,0})

	P00 := mat.NewDense(r_sys, r_sys, []float64{p4,0,0,p4})
	Q := mat.NewDense(r_sys, r_sys, []float64{p1*p1,p1*p2,p2*p1,p2*p2})
	R :=  mat.NewDense(1, 1, []float64{p3})
	matPrint(P00)
	matPrint(Q)
	matPrint(R)

	for i := 1; i < 200; i++ {
		X10 := predicted_estate(A, B, X00, U)	
		P10 := predicted_cov_p(P00, A, Q)

		y_tilde := innovation_residual(Y, H, X10)
		S1 := innovation_cov(H, P10, R)
		K1 := kalman_gain(P10, H, S1)
		X11 := update_estate(X10, K1, y_tilde)
		P11 := update_cov_p(P10, K1, H)
		// matPrint(K1)
		// y_t_p = innovation_residual(Y,H, X11)
		// matPrint(y_tilde)

		X00.Scale(1, X11)
		P00.Scale(1,P11)
	}
}