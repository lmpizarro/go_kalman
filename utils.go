package go_kalman

import (
	"fmt"
	"gonum.org/v1/gonum/mat"

	"os"
	"bufio"
)

func MatPrint(info string, X mat.Matrix) {
	fmt.Println(info)
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	fmt.Printf("%v\n", fa)
}

// Eye returns a new identity matrix of size n×n.
func Eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return mat.NewDense(n, n, d)
}

func Zero(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := 0; i < n*n; i +=  1 {
		d[i] = 0
	}
	return mat.NewDense(n, n, d)
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
