package main

import (
	"errors"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

const (
	size          = 2500
	goroutinesNum = 20
	step          = size / goroutinesNum
)

func createMatrix(size int, parallel bool) [][]int {
	matrix := make([][]int, size)
	for i := range matrix {
		matrix[i] = make([]int, size)
		for j := range matrix {
			x := rand.Intn(10)
			if parallel {
				x = 0
			}

			matrix[i][j] = x
		}
	}
	return matrix
}

func multiplyMatrix(A, B [][]int) [][]int {
	C := make([][]int, len(A))
	for i := range C {
		C[i] = make([]int, len(C))
		for j := range C {
			for k := range C {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
	return C
}

func multiplyMatrixColumns(A, B [][]int) [][]int {
	C := make([][]int, len(A))
	for i := range C {
		for j := range C {
			if len(C[j]) == 0 {
				C[j] = make([]int, len(C))
			}
			for k := range C {
				C[j][i] += A[j][k] * B[k][i]
			}
		}
	}
	return C
}

func multiplyMatrixParallelCol(wg *sync.WaitGroup, A, B, C [][]int, start, end int) {
	defer wg.Done()
	for i := range C {
		for j := start; j < end; j++ {
			for k := range C {
				C[j][i] += A[j][k] * B[k][i]
			}
		}
	}
}

func multiplyMatrixParallel(wg *sync.WaitGroup, A, B, C [][]int, start, end int) {
	defer wg.Done()
	for i := start; i < end; i++ {
		for j := range C {
			for k := range C {
				C[i][j] += A[i][k] * B[k][j]
			}
		}
	}
}

func compareMatrix(A, B [][]int) error {
	for i := range A {
		for j := range A {
			if A[i][j] != B[i][j] {
				return errors.New(fmt.Sprintln("Matrix have difference"))
			}
		}
	}
	return nil
}

func main() {
	A := createMatrix(size, false)
	B := createMatrix(size, false)
	fmt.Println("Starting default counting...")
	t := time.Now()
	C := multiplyMatrixColumns(A, B)
	fmt.Println("multiplyMatrix counting time:", time.Since(t))

	wg := &sync.WaitGroup{}
	wg.Add(goroutinesNum)
	CParallel := createMatrix(size, true)

	fmt.Println("Starting parallel counting...")
	fmt.Println("Count of parts:", goroutinesNum)
	t = time.Now()

	for i := 0; i < goroutinesNum; i++ {
		end := (i + 1) * step
		if i == goroutinesNum-1 {
			end = size
		}
		go multiplyMatrixParallelCol(wg, A, B, CParallel, i*step, end)
	}

	wg.Wait()
	fmt.Println("multiplyMatrixParallel counting time:", time.Since(t))
	fmt.Println()

	if err := compareMatrix(C, CParallel); err != nil {
		panic(err)
	} else {
		fmt.Println("Parallel counting equal default counting")
	}
}
