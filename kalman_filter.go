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

	err := errors.New("Bad System")

	err = nil
	return err
}

func SetCovariance(Qq, Pp, Rr * mat.Dense) error {

	Q = Qq
	P00 = Pp
	R = Rr

	err := errors.New("Bad System")

	err = nil
	return err
}


func SetInitialCondition(Xx, Uu, Yy * mat.Dense) error {

	X00 = Xx
	U = Uu
	Y = Yy

	err := errors.New("Bad System")

	err = nil
	return err
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

