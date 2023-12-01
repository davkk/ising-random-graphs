package main

import (
	"fmt"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
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

	T_start := math.Max(0.5*T_mid, 2.0)
	// T_end := 1.5 * T_mid

	var X_avg float64
	var X_std float64

	X_max := 0.0
	T_max := 0.0

	X_data := []float64{}
	T_data := []float64{}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		// Wait for a signal
		<-sigs
		fmt.Printf("%s_%s %f\n", graph, now, T_max)
		os.Exit(0)
	}()

	for T := T_start; ; T += step {
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
		}

		if X_avg > X_max {
			X_max = X_avg
			T_max = T
		}

		fmt.Fprintf(os.Stderr, "[debug] %s: X_avg = %f, X_var = %f, T = %f\n", graph, X_avg, X_std, T)
	}
}
