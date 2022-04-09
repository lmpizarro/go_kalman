package go_kalman

import (
	"github.com/piquette/finance-go/chart"
	"github.com/piquette/finance-go/datetime"
	"github.com/shopspring/decimal"

	"errors"
)

func Historical(symbol string, start, end datetime.Datetime) ([]float64, error) {

	p := &chart.Params{
		Symbol:   symbol,
		Start:    &start,
		End:      &end,
		Interval: datetime.OneDay,
	}

	iter := chart.Get(p)

	var values []float64

	// Iterate over results. Will exit upon any error.
	for iter.Next() {
		b := iter.Bar()
		avg := decimal.Avg(b.Low, b.Close, b.Open, b.High)
		val, _ := avg.Float64()
		values = append(values, val)
	}
	// Catch an error, if there was one.
	if iter.Err() != nil {
		// Uh-oh!
		return values, errors.New("statistics error")
	}
	
	return values, nil
}