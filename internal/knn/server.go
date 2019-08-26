package knn

import (
	"context"
	"fmt"
	"github.com/darwayne/knn-playground/internal/knn/knnpb"
	"google.golang.org/grpc"
	"io"
	"log"
	"net"
)

type Server struct {
	svc *Service
	lis net.Listener
}

func NewServer(svc *Service, port int) (*Server, error) {
	lis, err := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", port))
	if err != nil {
		return nil, err
	}

	srver := &Server{
		svc: svc,
		lis: lis,
	}
	return srver, nil
}

func (s *Server) Serve() error {
	server := grpc.NewServer()
	knnpb.RegisterKNNServiceServer(server, s)

	if err := server.Serve(s.lis); err != nil {
		return err
	}

	return nil
}

func (s *Server) CreateSingleVector(ctx context.Context, req *knnpb.CreateSingleVectorRequest) (*knnpb.CreateSingleVectorResponse, error) {
	log.Println("CreateSingleVector called", s.svc.Size())
	vector := req.GetVector()
	id := vector.GetId()
	points := vector.GetPoints()
	namespace := vector.GetNamespace()

	return &knnpb.CreateSingleVectorResponse{
		Result: uint32(s.svc.CreateSingleVector(id, namespace, points)),
	}, nil
}

type Vectors []*knnpb.Vector

func (vs Vectors) ToSingleVectors() []SingleVector {
	vectors := make([]SingleVector, 0, len(vs))

	for _, v := range vs {
		vectors = append(vectors, SingleVector{
			ID:        v.Id,
			Namespace: v.Namespace,
			Points:    v.Points,
		})
	}

	return vectors
}

func (s *Server) CreateMultipleVector(ctx context.Context, req *knnpb.CreateMultipleVectorRequest) (*knnpb.CreateMultipleVectorResponse, error) {
	log.Println("CreateMultipleVector called", s.svc.Size())
	vectors := Vectors(req.GetVectors()).ToSingleVectors()

	return &knnpb.CreateMultipleVectorResponse{
		Result: uint32(s.svc.CreateMultipleVectors(vectors...)),
	}, nil
}

func (s *Server) CreateStreamVector(vectorServer knnpb.KNNService_CreateStreamVectorServer) error {

	for {
		msg, err := vectorServer.Recv()
		if err != nil {
			if err == io.EOF {
				vectorServer.SendAndClose(&knnpb.CreateStreamVectorResponse{
					Result: uint32(s.svc.Size()),
				})
				return nil
			}

			return err
		}

		s.svc.CreateMultipleVectors(Vectors(msg.Vectors).ToSingleVectors()...)
	}

	return nil
}

func (s *Server) Search(ctx context.Context, req *knnpb.QueryRequest) (*knnpb.QueryResponse, error) {
	query := req.GetQuery()
	searchResults := s.svc.Search(query.Namespace, &EuclideanPoint{
		Vector: query.Points,
	})

	results := make([]*knnpb.QueryResult, 0, len(searchResults))

	for _, r := range searchResults {
		results = append(results, &knnpb.QueryResult{
			Id:       r.ID,
			Distance: r.Distance,
		})
	}

	return &knnpb.QueryResponse{
		Results: results,
	}, nil
}

func (s *Server) GetSize(context.Context, *knnpb.SizeRequest) (*knnpb.SizeResponse, error) {
	return &knnpb.SizeResponse{
		Size: uint32(s.svc.Size()),
	}, nil
}
