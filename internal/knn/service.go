package knn

import "sync"

type Service struct {
	vectorMu *sync.RWMutex
	vectors  map[string]map[string]Distancer

	nodesMu *sync.RWMutex
	nodes   map[string][]Distancer
}

func NewService() *Service {
	return &Service{
		nodesMu: &sync.RWMutex{},
		nodes:   make(map[string][]Distancer, 0),

		vectorMu: &sync.RWMutex{},
		vectors:  make(map[string]map[string]Distancer),
	}
}

type SingleVector struct {
	ID        string
	Namespace string
	Points    []float32
}

func (s *Service) CreateSingleVector(id, namespace string, points []float32) int {
	node := &EuclideanPoint{
		ID:     id,
		Vector: points,
	}
	useDefaultIfNotSet(&namespace)

	s.CreateNameSpace(namespace)

	s.nodesMu.Lock()
	s.nodes[namespace] = append(s.nodes[namespace], node)
	s.nodesMu.Unlock()

	s.vectorMu.Lock()
	s.vectors[namespace][id] = node
	s.vectorMu.Unlock()

	return s.Size()
}

func (s *Service) CreateMultipleVectors(vectors ...SingleVector) int {
	if len(vectors) == 0 {
		return s.Size()
	}

	nodes := make([]Distancer, 0, len(vectors))
	namespace := vectors[0].Namespace
	useDefaultIfNotSet(&namespace)

	s.CreateNameSpace(namespace)

	for _, v := range vectors {
		nodes = append(nodes, &EuclideanPoint{
			ID:     v.ID,
			Vector: v.Points,
		})
	}

	s.nodesMu.Lock()
	s.nodes[namespace] = append(s.nodes[namespace], nodes...)
	s.nodesMu.Unlock()

	s.vectorMu.Lock()
	for idx, n := range nodes {
		s.vectors[namespace][vectors[idx].ID] = n
	}
	s.vectorMu.Unlock()

	return s.Size()
}

func (s *Service) Size() int {
	s.nodesMu.RLock()
	length := len(s.nodes)
	s.nodesMu.RUnlock()

	return length
}

func (s *Service) CreateNameSpace(namespace string) bool {
	s.vectorMu.RLock()
	val := s.vectors[namespace]
	s.vectorMu.RUnlock()

	if val == nil {
		s.vectorMu.Lock()
		s.nodesMu.Lock()
		s.vectors[namespace] = make(map[string]Distancer)
		s.nodes[namespace] = make([]Distancer, 0)
		s.nodesMu.Unlock()
		s.vectorMu.Unlock()
	}

	return val == nil
}

type QueryResult struct {
	ID       string
	Distance float32
}

func (s *Service) Search(namespace string, query Distancer) []QueryResult {
	useDefaultIfNotSet(&namespace)
	s.nodesMu.RLock()
	result := KNN(query, s.nodes[namespace]...)
	s.nodesMu.RUnlock()

	if result == nil {
		return nil
	}

	euc := result.(*EuclideanPoint)
	return []QueryResult{
		{
			ID:       euc.ID,
			Distance: query.Distance(euc),
		},
	}
}

func useDefaultIfNotSet(namespace *string) {
	if namespace != nil && *namespace == "" {
		*namespace = "___default"
	}
}
