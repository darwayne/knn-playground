package knn

import (
	"github.com/darwayne/knn-playground/internal/knn/knnpb"
	"google.golang.org/grpc"
	"time"
)

type Client struct {
	url  string
	conn *grpc.ClientConn
	knnpb.KNNServiceClient
}

func NewClient(url string) (*Client, error) {
	conn, err := grpc.Dial(url,
		grpc.WithInsecure(),
		grpc.WithBackoffConfig(grpc.BackoffConfig{
			MaxDelay: time.Minute,
		}),
	)
	if err != nil {
		return nil, err
	}

	svcCli := knnpb.NewKNNServiceClient(conn)

	return &Client{
		url:              url,
		conn:             conn,
		KNNServiceClient: svcCli,
	}, nil
}

func (c *Client) Close() error {
	return c.conn.Close()
}
