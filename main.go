package main

//
// https://en.wikipedia.org/wiki/Kalman_filter
//
// https://medium.com/wireless-registry-engineering/gonum-tutorial-linear-algebra-in-go-21ef136fc2d7
//
// https://github.com/gonum/matrix/blob/master/mat64/matrix_test.go
//

import (
	"os"
	"fmt"
	"bufio"
	"gonum.org/v1/gonum/mat"
	"math/rand"
)

func matPrint(info string, X mat.Matrix) {
	fmt.Println(info)
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

func random_walk(r_val float64) float64{
	r := rand.Float64()
	
	if r < .5 {
		return -r_val
	}
	
	return r_val
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

func innovation_residual(M, H, X *mat.Dense) (* mat.Dense, * mat.Dense) {
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

func Wrt(vals, filter_out []float64, filename string) error {

	f, err := os.Create(filename)
    if err != nil {
        return err
    }
    // remember to close the file
    defer f.Close()
	    // create new buffer
    buffer := bufio.NewWriter(f)

	buffer.WriteString("vals,out\n")
	for i, line := range filter_out {
		d := fmt.Sprintf("%f,%f\n", vals[i], line)
        _, err := buffer.WriteString(d)
        if err != nil {
            return err
        }
    }

    // flush buffered data to the file
    if err := buffer.Flush(); err != nil {
         return err
    }

	return nil
}

func main(){

	size := 128
	vals := make([]float64, size)
	filter := make([]float64, size)
	
	measure := 0.0
	Dt := 1.0
	Dt1 := 0.0
	q1 := .3
	q2 := .3
	r11 := 1.2
	p11 := 1.0
	p12 := 1.0

	A := mat.NewDense(2, 2, []float64{1,Dt,0,1})
	B := mat.NewDense(2, 2, []float64{0,0,0,0})
	H := mat.NewDense(1, 2, []float64{1,0})
	matPrint("A ", A)
	matPrint("B ", B)
	matPrint("H ", H)

	U := mat.NewDense(2, 1, []float64{.5*Dt1*Dt1,Dt1})
	Y := mat.NewDense(1, 1, []float64{2})
	matPrint("U ", U)
	matPrint("Y ", Y)
	
	r_sys, _ := A.Dims()
	X00 := mat.NewDense(r_sys, 1, []float64{0,0})

	P00 := mat.NewDense(r_sys, r_sys, []float64{p11,0,0,p12})
	Q := mat.NewDense(r_sys, r_sys, []float64{q1*q1,q1*q2,q2*q1,q2*q2})
	R :=  mat.NewDense(1, 1, []float64{r11})
	matPrint("POO ", P00)
	matPrint("Q ", Q)
	matPrint("R ", R)

	for i := 1; i < size; i++ {
		measure += random_walk(rand.Float64())
		if measure <= 0.0 {
			measure = -measure
		}
		vals[i] = measure
		Y.Set(0,0, measure)
		X10 := predicted_estate(A, B, X00, U)	
		P10 := predicted_cov_p(P00, A, Q)

		y_pre, m_pre := innovation_residual(Y, H, X10)
		
		S1 := innovation_cov(H, P10, R)
		K1 := kalman_gain(P10, H, S1)
		X11 := update_estate(X10, K1, y_pre)
		P11 := update_cov_p(P10, K1, H)
		// matPrint(K1)
		_, m_post := innovation_residual(Y,H, X11)
		
		filter_out := m_post.At(0,0)
		filter[i] = filter_out 
		Wrt(vals, filter, "/home/lmpizarro/devel/project/GOLANG/indicators/pkg/go_kalman/filter.csv")
		fmt.Printf("measure %e m_pre %e m_post %e\n", 
		            measure, m_pre.At(0,0), filter_out)

		X00.Scale(1, X11)
		P00.Scale(1,P11)
	}
}