package go_kalman_tests

import (
	"github.com/piquette/finance-go/datetime"
	"testing"
	gk "github.com/lmpizarro/go_kalman"
	"fmt"
)


func TestFilterTicker(t *testing.T) {

	start := datetime.Datetime{Month: 2, Day: 10, Year: 2016}
	end := datetime.Datetime{Month: 4, Day: 7, Year: 2022}

	ticker := "AMZN"
	values, _ := gk.Historical(ticker, start, end)

	Y := gk.KalmanDefault2x2()

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