package go_kalman_tests

import (
	"github.com/piquette/finance-go/datetime"

	gk "github.com/lmpizarro/go_kalman"
	"testing"
	"fmt"
)


func TestTicker(t *testing.T) {

	start := datetime.Datetime{Month: 2, Day: 10, Year: 2021}
	end := datetime.Datetime{Month: 3, Day: 25, Year: 2022}

	ticker := "AMZN"
	values, _ := gk.Historical(ticker, start, end)

	filename := fmt.Sprintf("data/%s.csv",ticker)
	gk.Wrt(values, values, filename)
}