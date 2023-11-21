package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

func mean(data []float64) float64 {
	sum := 0.0
	for _, last_X := range data {
		sum += last_X
	}
	return sum / float64(len(data))
}

func variance(data []float64) float64 {
	mean := mean(data)
	v := 0.0
	for i := 0; i < len(data); i++ {
		diff := data[i] - mean
		v += diff * diff
	}
	return math.Sqrt(v / float64(len(data)-1))
}

func main() {
	path := os.Args[1]
	T_mid, _ := strconv.ParseFloat(os.Args[2], 64)
	step, _ := strconv.ParseFloat(os.Args[3], 64)
	now := os.Args[4]

	graph := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))

	T_start := math.Max(2.0/3.0*T_mid, 2.0)
	T_end := 2 * T_mid

	var X_avg float64
	var X_std float64

	X_max := 0.0
	T_max := 0.0

	X_data := []float64{}
	T_data := []float64{}

	for T := T_start; T < T_end; T += step {
		stdout, err := exec.Command("./calc_suscept", path, fmt.Sprint(T), now).Output()
		if err != nil {
			panic(err)
		}

		line := strings.TrimSuffix(string(stdout), "\n")
		X, _ := strconv.ParseFloat(strings.Split(line, " ")[1], 64)

		X_data = append(X_data, X)
		T_data = append(T_data, T)

		if len(X_data) > 5 {
			X_avg = mean(X_data[len(X_data)-5:])
			X_std = variance(X_data[len(X_data)-5:])

			if math.Abs(X-X_avg) > 4*X_std {
				X_data = X_data[:len(X_data)-1]
				T_data = T_data[:len(T_data)-1]
				continue
			}
		}

		if X_avg-X_max > 0 && X_avg-X_max < 4*X_std {
			X_max = X_avg
			T_max = T
		}

		fmt.Fprintf(os.Stderr, "[debug] %s: X_avg = %f, X_var = %f, T = %f\n", graph, X_avg, X_std, T)
	}

	fmt.Printf("%s_%s %f\n", graph, now, T_max)
}
