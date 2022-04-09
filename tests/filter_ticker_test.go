package go_kalman_tests

import (
	"github.com/piquette/finance-go/datetime"
	"testing"
	"gonum.org/v1/gonum/mat" 
	gk "github.com/lmpizarro/go_kalman"
	"fmt"
)


func TestFilterTicker(t *testing.T) {

	start := datetime.Datetime{Month: 2, Day: 10, Year: 2021}
	end := datetime.Datetime{Month: 3, Day: 25, Year: 2022}

	ticker := "AMZN"
	values, _ := gk.Historical(ticker, start, end)

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

	err := gk.SetSystem(A, B, H)
	if err != nil {panic("error")}

	err = gk.SetCovariance(Q, P, R)
	if err != nil {panic("error")}

 	U := mat.NewDense(2, 1, []float64{.5*Dt*Dt,Dt})
	Y := mat.NewDense(1, 1, []float64{2})
	X00 := mat.NewDense(r_sys, 1, []float64{0,0})

	gk.SetInitialCondition(X00, U, Y)

	out_vals := make([]float64, len(values))

	gk.RandMeasure = 0.0
	for i := 1; i < len(values); i++ {
		Y.Set(0,0, values[i])
		ypred := gk.Update(Y)
		out_vals[i] = ypred.At(0,0)
	}

	filename := fmt.Sprintf("data/%s.csv",ticker)
	gk.Wrt(values, out_vals, filename)


}