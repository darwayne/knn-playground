package knn

import (
	"context"
	"fmt"
	"github.com/darwayne/knn-playground/internal/knn/knnpb"
	"github.com/hashicorp/go-multierror"
	"github.com/pkg/errors"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"io"
	"math"
	"math/rand"
	"net"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type Aggregator struct {
	clientsMu *sync.RWMutex
	clients   []*Client
}

func NewAggregator(endpoints ...string) (*Aggregator, error) {
	clients := make([]*Client, 0, len(endpoints))
	for _, endpoint := range endpoints {
		client, err := NewClient(endpoint)
		if err != nil {
			return nil, errors.Wrapf(err, "error creating client: %s", endpoint)
		}

		clients = append(clients, client)
	}

	return &Aggregator{
		clientsMu: &sync.RWMutex{},
		clients:   clients,
	}, nil
}

func (s *Aggregator) Serve(port int) error {
	lis, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", port))
	if err != nil {
		return err
	}

	server := grpc.NewServer()
	knnpb.RegisterKNNServiceServer(server, s)
	knnpb.RegisterAggregatorServiceServer(server, s)

	if err := server.Serve(lis); err != nil {
		return err
	}

	return nil
}

func (s *Aggregator) AddEndpoint(ctx context.Context, req *knnpb.AddEndpointRequest) (*knnpb.AddEndpointResponse, error) {
	if req == nil || req.Endpoint == "" {
		return nil, status.Error(codes.InvalidArgument, "invalid endpoint")
	}

	client, err := NewClient(req.Endpoint)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "error creating client: %s", req.Endpoint)
	}

	s.clientsMu.Lock()
	s.clients = append(s.clients, client)
	s.clientsMu.Unlock()

	return nil, nil
}

func (s *Aggregator) RemoveEndpoint(ctx context.Context, req *knnpb.RemoveEndpointRequest) (*knnpb.RemoveEndpointResponse, error) {
	if req == nil || req.Endpoint == "" {
		return nil, status.Error(codes.InvalidArgument, "invalid endpoint")
	}

	indexToRemove := -1

	newClients := make([]*Client, 0, s.TotalClients())
	s.clientsMu.RLock()
	for idx, client := range s.clients {
		var removed bool
		if client.url == req.Endpoint {
			indexToRemove = idx
			removed = true
		}

		if !removed {
			newClients = append(newClients, client)
		}
	}
	s.clientsMu.RUnlock()

	if indexToRemove == -1 {
		return nil, status.Errorf(codes.NotFound, "endpoint not found: %s", req.Endpoint)
	}

	s.clientsMu.Lock()
	s.clients = newClients
	s.clientsMu.Unlock()

	return nil, nil
}

func (s *Aggregator) CreateSingleVector(ctx context.Context, req *knnpb.CreateSingleVectorRequest) (*knnpb.CreateSingleVectorResponse, error) {
	client := s.RandomClient()

	return client.CreateSingleVector(ctx, req)
}

func (s *Aggregator) CreateStreamVector(vectorServer knnpb.KNNService_CreateStreamVectorServer) error {
	ctx := vectorServer.Context()
	for {
		msg, err := vectorServer.Recv()
		if err != nil {
			if err == io.EOF {
				vectorServer.SendAndClose(&knnpb.CreateStreamVectorResponse{
					Result: uint32(s.Size()),
				})
				return nil
			}

			return err
		}

		_, err = s.CreateMultipleVector(ctx, &knnpb.CreateMultipleVectorRequest{
			Vectors: msg.Vectors,
		})

		if err != nil {
			return err
		}
	}
	return nil
}

func (s *Aggregator) Search(ctx context.Context, req *knnpb.QueryRequest) (*knnpb.QueryResponse, error) {
	// TODO: update req so total neighbors can be reduced since multiple endpoints will be hit

	ctx2, cancel1 := context.WithTimeout(ctx, time.Minute)
	defer cancel1()

	clientCtx, cancel := context.WithCancel(ctx2)
	defer cancel()

	totalWorkers := len(s.clients) - 1

	resultChan := make(chan []*knnpb.QueryResult, totalWorkers)
	errChan := make(chan error, totalWorkers)
	wg := &sync.WaitGroup{}

	for _, c := range s.clients {
		wg.Add(1)
		go func(client *Client) {
			defer wg.Done()
			resp, err := client.Search(clientCtx, req)
			if err != nil {
				select {
				case errChan <- err:
				case <-clientCtx.Done():

				}
				return
			}

			if resp != nil {
				select {
				case resultChan <- resp.Results:
				case <-clientCtx.Done():

				}
			}
		}(c)
	}

	wg.Wait()
	close(errChan)
	close(resultChan)

	var err error
	for e := range errChan {
		err = multierror.Append(err, e)
	}

	response := knnpb.QueryResponse{}
	for results := range resultChan {
		for _, r := range results {
			response.Results = append(response.Results, r)
		}
	}

	if len(response.Results) > 0 {
		sort.Slice(response.Results, func(i, j int) bool {
			return response.Results[i].Distance < response.Results[i].Distance
		})

		// TODO: base results on actual neighbors request
		response.Results = response.Results[:1]
	}

	return &response, err
}

func (s *Aggregator) CreateMultipleVector(ctx context.Context, req *knnpb.CreateMultipleVectorRequest) (*knnpb.CreateMultipleVectorResponse, error) {
	var err error
	wg := &sync.WaitGroup{}

	var size uint32
	totalClients := s.TotalClients()

	errChan := make(chan error, totalClients)
	chunks := chunkVectors(totalClients, req.Vectors...)

	for idx, c := range s.RandomClients() {
		if idx >= len(chunks) {
			break
		}

		wg.Add(1)
		go func(client *Client, chunk []*knnpb.Vector) {
			defer wg.Done()
			resp, err := client.CreateMultipleVector(ctx, &knnpb.CreateMultipleVectorRequest{
				Vectors: chunk,
			})
			if err != nil {
				errChan <- err
				return
			}

			if resp != nil {
				atomic.AddUint32(&size, resp.Result)
			}
		}(c, chunks[idx])
	}
	wg.Wait()
	close(errChan)

	for e := range errChan {
		err = multierror.Append(err, e)
	}

	return &knnpb.CreateMultipleVectorResponse{
		Result: size,
	}, err
}

func (s *Aggregator) GetSize(context.Context, *knnpb.SizeRequest) (*knnpb.SizeResponse, error) {
	return &knnpb.SizeResponse{
		Size: uint32(s.Size()),
	}, nil
}

func (s *Aggregator) TotalClients() int {
	s.clientsMu.RLock()
	result := len(s.clients)
	s.clientsMu.RUnlock()

	return result
}

func (s *Aggregator) RandomClient() *Client {
	idx := rand.Intn(s.TotalClients())
	s.clientsMu.RLock()
	result := s.clients[idx]
	s.clientsMu.RUnlock()

	return result
}

func (s *Aggregator) RandomClients() []*Client {
	total := s.TotalClients()
	clients := make([]*Client, 0, total)
	s.clientsMu.RLock()
	clients = append([]*Client{}, s.clients...)
	s.clientsMu.RUnlock()
	result := make([]*Client, 0, total)
	indexes := rand.Perm(total)

	for idx := range indexes {
		result = append(result, clients[idx])
	}

	return result
}

func (s *Aggregator) Size() int {
	var result uint64

	s.clientsMu.RLock()
	wg := &sync.WaitGroup{}
	for _, c := range s.clients {
		wg.Add(1)
		go func(client *Client) {
			defer wg.Done()
			resp, err := client.GetSize(context.Background(), &knnpb.SizeRequest{})
			if err != nil {
				return
			}

			if resp != nil {
				atomic.AddUint64(&result, uint64(resp.Size))
			}
		}(c)
	}
	s.clientsMu.RUnlock()

	wg.Wait()

	return int(result)
}

func chunkVectors(size int, vectors ...*knnpb.Vector) [][]*knnpb.Vector {
	itemSize := len(vectors)
	eachArrSize := int(math.Ceil(float64(itemSize) / float64(size)))
	results := make([][]*knnpb.Vector, 0, size)

	for i := 0; i < itemSize; i += eachArrSize {
		end := i + eachArrSize

		if end > itemSize {
			end = itemSize
		}

		results = append(results, vectors[i:end])
	}

	return results
}
