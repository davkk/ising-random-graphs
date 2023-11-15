package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

func main() {
	graphs := os.Args[1:]

	queue := make(chan float64, 2)

	for idx, graph := range graphs {
		temp := 1.0
		X_max := 0.0
		T_max := 0.0

		count := 0
		for count < 10 {
			cmd := exec.Command("./run_parallel", graph, fmt.Sprint(temp))
			stdout, _ := cmd.Output()

			results := strings.Split(string(stdout), "\n")
			take := float64(len(results) - 1)

			X := 0.0
			for line := 0; line < len(results)-1; line++ {
				parsed, _ := strconv.ParseFloat(results[line], 64)
				X += parsed
			}
			X /= take

			queue <- X

			if X > X_max {
				X_max = X
				T_max = temp
				fmt.Fprintln(os.Stderr, X_max, T_max)
				count = 0
			}

			if len(queue) > 1 {
				last_X := <-queue

				if last_X-X > X/10 {
					count += 1
				}
			}

			temp += 0.02
		}

		fmt.Println(idx+1, T_max)
	}
}
