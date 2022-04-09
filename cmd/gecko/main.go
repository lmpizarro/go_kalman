package main

import (
	"fmt"
	"log"
	"time"

	gk "github.com/lmpizarro/go_kalman"
	gecko "github.com/superoo7/go-gecko/v3"
)

func main() {
	cg := gecko.NewClient(nil)
	p := make([]float64, 10)

	for i := 0; i < 10; i++ {
		p[i] = 1.0
	}

	Y, _ := gk.KalmanModel2x2x10(p)
	for {
	    time.Sleep(5 * time.Second)
		price, err := cg.SimpleSinglePrice("bitcoin", "usd")

		if err != nil {
			log.Fatal(err)
		}
		Y.Set(0,0, float64(price.MarketPrice))
		ypred := gk.Update(Y)
		y_kalman := ypred.At(0,0)
		fmt.Println(price.MarketPrice, y_kalman)

	}
}