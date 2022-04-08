package go_kalman

//
// https://en.wikipedia.org/wiki/Kalman_filter
//
// https://medium.com/wireless-registry-engineering/gonum-tutorial-linear-algebra-in-go-21ef136fc2d7
//
// https://github.com/gonum/matrix/blob/master/mat64/matrix_test.go
//

import (
	"gonum.org/v1/gonum/mat"
)

/*
	arguments

		P -> P0.0 the previous updated covariance

		A -> System Matrix
	
		Q -> covariance process noise


	return P1.0 the predicted covariance
*/
func PredictedCovP(A, P, Q * mat.Dense) * mat.Dense {

	r, c := P.Dims()
	p := mat.NewDense(r, c, nil)

	p.Product(A, P, A.T())
	p.Add(Q, p)
	return p
}

func PredictedEstate(A, B, X, U * mat.Dense) * mat.Dense {
	rx, cx := X.Dims()
	ru, cu := U.Dims()


	x := mat.NewDense(rx, cx, nil)
	u := mat.NewDense(ru, cu, nil)

	x.Product(A, X)
	u.Product(B,U)

	x.Add(X,u)

	return x
}

func InnovationResidual(M, H, X *mat.Dense) (* mat.Dense, * mat.Dense) {
	r, _ := H.Dims()
	_, c := X.Dims()

	z := mat.NewDense(r, c, []float64{0})
	m := mat.NewDense(r, c, []float64{0})

	m.Product(H,X)
	// matPrint(z)
	z.Scale(-1.0, m)
	// a <- meas b <- HX
	z.Add(z, M)
	return z, m
}

func InnovationCov(H, P10, R *mat.Dense) *mat.Dense{
	r, _ := H.Dims()
	Z := mat.NewDense(r, r, []float64{0})

	Z.Product(H, P10, H.T())
	Z.Add(R,Z)
	return Z
}

func KalmanGain(P10, H, S * mat.Dense) * mat.Dense {
	S.Inverse(S)
	_, c := H.Dims()
	p := make([]float64, c)
	K := mat.NewDense(c, 1, p)
	K.Product(P10, H.T(), S)
	return K
}

func UpdateState(X, K, y_tilde *mat.Dense) *mat.Dense {
	rk,_ := K.Dims()
	_, cy := y_tilde.Dims()
	p := mat.NewDense(rk, cy, nil)
	p.Product(K, y_tilde)
	p.Add(X, p)
	return p
}

func UpdateCovP(P, K, H *mat.Dense) *mat.Dense{
 
	rk,_ := K.Dims()
	_, cy := H.Dims()

	p := mat.NewDense(rk, cy, nil)
	p.Product(K, H)
	r, _ := P.Dims()
	p.Scale(-1.0, p)
	identity := Eye(r)

	identity.Add(p, identity)
	identity.Product(identity, P)

	return identity
}