package main

import (
	"flag"
	"fmt"
	"github.com/darwayne/knn-playground/internal/knn"
	"log"
	"math/rand"
	"time"
)

func main() {
	port := flag.Int("port", 50051, "port to listen on")
	flag.Parse()

	srv, err := knn.NewServer(knn.NewService(), *port)
	if err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}

	rand.Seed(time.Now().UnixNano())

	fmt.Println("server listening on", *port)
	fmt.Println("Exited server with", srv.Serve())
}
