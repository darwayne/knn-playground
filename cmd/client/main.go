package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/darwayne/knn-playground/internal/knn"
	"github.com/google/uuid"

	"github.com/darwayne/knn-playground/internal/knn/knnpb"
)

func main() {
	url := flag.String("url", "localhost:50051", "url to connect to")
	flag.Parse()
	fmt.Println("Hello from client")

	c, err := knn.NewClient(*url)
	if err != nil {
		log.Fatalf("could not connect: %v", err)
	}
	defer c.Close()

	rand.Seed(time.Now().UnixNano())

	fmt.Printf("Created client %f\n", c)

	ctx := context.Background()

	//streamRequest(ctx, c)

	searchRequest(ctx, c)
	//singleRequest(ctx, c)

	time.Sleep(time.Minute)
}

func searchRequest(ctx context.Context, cli knnpb.KNNServiceClient) {
	req := &knnpb.QueryRequest{
		Query: &knnpb.Query{
			Points: knn.GenPoints(1000, 128)[rand.Intn(999)],
		},
	}

	ctx2, _ := context.WithTimeout(ctx, 2*time.Minute)

	s := time.Now()
	resp, err := cli.Search(ctx2, req)
	e := time.Since(s)

	if err != nil {
		log.Fatalf("Error getting search result: %v", err)
	}

	log.Printf("Search Result(%v) is %q", e, resp)
}

func streamRequest(ctx context.Context, cli knnpb.KNNServiceClient) {
	stream, err := cli.CreateStreamVector(ctx)
	if err != nil {
		log.Fatalf("Error receiving stream: %v", err)
	}

	batchSize := 10
	arr := make([]*knnpb.Vector, 0, batchSize)
	for i := 0; i < 40000; i++ {

		vec := &knnpb.Vector{
			Id:     uuid.New().String(),
			Points: knn.GenPoints(1, 128)[0],
		}
		arr = append(arr, vec)

		if len(arr) == batchSize {
			req := &knnpb.CreateStreamVectorRequest{
				Vectors: arr,
			}

			s := time.Now()
			err := stream.Send(req)
			e := time.Since(s)

			if err != nil {
				log.Fatalf("error sending stream: %v", err)
			}

			log.Println("Send completed in", e)

			arr = make([]*knnpb.Vector, 0, batchSize)
		}

	}

	msg, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("Error recieving: %v", err)
	}
	log.Println("\n\nstreamed response is", msg, "\n=======")
}

func singleRequest(ctx context.Context, cli knnpb.KNNServiceClient) {
	for i := 0; i < 100000; i++ {
		req := &knnpb.CreateSingleVectorRequest{
			Vector: &knnpb.Vector{
				Id:     uuid.New().String(),
				Points: knn.GenPoints(1, 128)[0],
			}}

		s := time.Now()
		res, err := cli.CreateSingleVector(ctx, req)
		e := time.Since(s)

		if err != nil {
			log.Fatalf("error getting response: %v", err)
		}

		log.Println("Response is", res, "and completed in", e)
	}
}
