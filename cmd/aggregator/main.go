package main

import (
	"flag"
	"fmt"
	"github.com/darwayne/knn-playground/internal/knn"
	"log"
	"math/rand"
	"strings"
	"time"
)

func main() {
	endpointsFlag := flag.String("endpoints", "localhost:50051", "comma separated list of endpoints to connect to")
	port := flag.Int("port", 50051, "port to listen on")

	flag.Parse()
	endpoints := strings.Split(strings.TrimSpace(*endpointsFlag), ",")
	agg, err := knn.NewAggregator(endpoints...)
	if err != nil {
		log.Fatalf("error creating aggregator: %v", err)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("Aggregator Listening on", *port)
	fmt.Println("aggregator error", agg.Serve(*port))
}
