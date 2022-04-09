package go_kalman

import (
	"gonum.org/v1/gonum/mat" 
	"errors"
)

var A * mat.Dense
var B * mat.Dense
var C * mat.Dense
var Q * mat.Dense
var P00 * mat.Dense
var R * mat.Dense
var X00 * mat.Dense
var Y * mat.Dense
var U * mat.Dense

func SetSystem(Aa, Bb, Hh * mat.Dense) error {

	A = Aa
	B = Bb
	C = Hh

	ra, _ := A.Dims()
	rb, _ := Bb.Dims() 

	if ra != rb {
		return errors.New("bad system")
	}

	return nil 
}

func SetCovariance(Qq, Pp, Rr * mat.Dense) error {

	Q = Qq
	P00 = Pp
	R = Rr

	ra, _ := Qq.Dims()
	rb, _ := Pp.Dims() 

	if ra != rb {
		return errors.New("bad system")
	}

	return nil
}

func SetInitialCondition(Xx, Uu, Yy * mat.Dense) error {

	X00 = Xx
	U = Uu
	Y = Yy

	rx, _ := Xx.Dims()
	ru, _ := Uu.Dims()

	if rx != ru {
		return errors.New("bad system")
	}

	return nil
}

func KPrint(){
	MatPrint("A", A)
	MatPrint("B", B)
	MatPrint("H", C)
	MatPrint("P", P00)
	MatPrint("Q", Q)
	MatPrint("R", R)
}

func Update(Yy *mat.Dense) *mat.Dense{

	X10 := PredictedEstate(A, B, X00, U)	
	P10 := PredictedCovP(P00, A, Q)
	y_pre, _ := InnovationResidual(Yy, C, X10)
	
	S1 := InnovationCov(C, P10, R)
	K1 := KalmanGain(P10, C, S1)
	X11 := UpdateState(X10, K1, y_pre)
	P11 := UpdateCovP(P10, K1, C)
	_, m_post := InnovationResidual(Y,C, X11)

	X00.Scale(1, X11)
	P00.Scale(1,P11)

	return m_post
}

func KalmanDefault2x2() * mat.Dense {
	Dt := 1.0
	q1 := .2
	q2 := .2
	r11 := 1.0
	p11 := 1.0
	p12 := 1.0

	A := mat.NewDense(2, 2, []float64{1,Dt,0,1})
	B := mat.NewDense(2, 2, []float64{0,0,0,0})
	H := mat.NewDense(1, 2, []float64{1,0})
	r_sys, _ := A.Dims()
	r_out, _ := H.Dims()
	P := mat.NewDense(r_sys, r_sys, []float64{p11,0,0,p12})
	Q := mat.NewDense(r_sys, r_sys, []float64{q1*q1,q1*q2,q2*q1,q2*q2})
	R :=  mat.NewDense(r_out, r_out, []float64{r11})

	err := SetSystem(A, B, H)
	if err != nil {panic("error")}

	err = SetCovariance(Q, P, R)
	if err != nil {panic("error")}

 	U := mat.NewDense(2, 1, []float64{.5*Dt*Dt,Dt})
	Y := mat.NewDense(1, 1, []float64{2})
	X00 := mat.NewDense(r_sys, 1, []float64{0,0})

	SetInitialCondition(X00, U, Y)

	return Y
}