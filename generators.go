package go_kalman
import (

	"math/rand"
)

var RandMeasure float32

func RandomWalk(r_val float64) float32{
	r := rand.Float64()
	
	if r < .5 {
		RandMeasure += -float32(r_val)
	} else {
		RandMeasure += float32(r_val)
	}

	if RandMeasure < 0 {
		RandMeasure *= -1.0
	}
	
	return RandMeasure
}

