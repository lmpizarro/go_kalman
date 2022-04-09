package go_kalman_tests

import (
        "gonum.org/v1/gonum/mat" 
        "testing"
		gk "github.com/lmpizarro/go_kalman"
		"math/rand"
		"fmt"
    )

func random_walk(r_val float64) float64{
	r := rand.Float64()
	
	if r < .5 {
		return -r_val
	}
		
	return r_val
}
func TestKalman(t *testing.T) {

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

	gk.MatPrint("A ", A)
	gk.MatPrint("B ", B)
	gk.MatPrint("H ", H)

	U := mat.NewDense(2, 1, []float64{.5*Dt1*Dt1,Dt1})
	Y := mat.NewDense(1, 1, []float64{2})
	gk.MatPrint("U ", U)
	gk.MatPrint("Y ", Y)

	
	r_sys, _ := A.Dims()
	X00 := mat.NewDense(r_sys, 1, []float64{0,0})

	P00 := mat.NewDense(r_sys, r_sys, []float64{p11,0,0,p12})
	Q := mat.NewDense(r_sys, r_sys, []float64{q1*q1,q1*q2,q2*q1,q2*q2})
	R :=  mat.NewDense(1, 1, []float64{r11})

	gk.MatPrint("POO ", P00)
	gk.MatPrint("Q ", Q)
	gk.MatPrint("R ", R)

	for i := 1; i < size; i++ {

		measure += random_walk(1.0)
		if measure < 0.0 {
			measure = 1.0
		}
		
		vals[i] = measure
		Y.Set(0,0, measure)

		X10 := gk.PredictedEstate(A, B, X00, U)	
		P10 := gk.PredictedCovP(P00, A, Q)
		y_pre, m_pre := gk.InnovationResidual(Y, H, X10)
		
		S1 := gk.InnovationCov(H, P10, R)
		K1 := gk.KalmanGain(P10, H, S1)
		X11 := gk.UpdateState(X10, K1, y_pre)
		P11 := gk.UpdateCovP(P10, K1, H)
		// matPrint(K1)
		_, m_post := gk.InnovationResidual(Y,H, X11)
		
		filter_out := m_post.At(0,0)
		filter[i] = filter_out

		fmt.Printf("measure %e m_pre %e m_post %e\n", 
		            measure, m_pre.At(0,0), filter_out)

		X00.Scale(1, X11)
		P00.Scale(1,P11)
	}
	filename := "kalman_simple"
	name := fmt.Sprintf("data/%s.csv",filename)
	gk.Wrt(vals, filter, name)
}