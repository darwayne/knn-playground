package knn

import (
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/google/uuid"
	"github.com/monochromegane/go-avx"
)

type Distancer interface {
	Distance(d interface{}) float32
}

func KNN(query Distancer, points ...Distancer) Distancer {
	if len(points) == 0 {
		return nil
	}

	var result Distancer
	var bestDistance float32
	for _, p := range points {
		dist := query.Distance(p)
		if result == nil || dist < bestDistance {
			bestDistance = dist
			result = p
		}
	}

	return result
}

func ParallelKNN(query Distancer, points ...Distancer) Distancer {
	if len(points) < 100 {
		return KNN(query, points...)
	}

	totalWorkers := runtime.NumCPU() - 1
	resultChan := make(chan Distancer, totalWorkers)
	wg := &sync.WaitGroup{}

	chunks := chunkDistancers(totalWorkers, points...)
	for _, chunk := range chunks {
		wg.Add(1)
		go func(work []Distancer) {
			defer wg.Done()
			resultChan <- KNN(query, work...)
		}(chunk)
	}
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	closestPoints := make([]Distancer, 0, totalWorkers)
	for result := range resultChan {
		closestPoints = append(closestPoints, result)
	}

	return KNN(query, closestPoints...)
}

func chunkDistancers(size int, items ...Distancer) [][]Distancer {
	itemSize := len(items)
	eachArrSize := int(math.Ceil(float64(itemSize) / float64(size)))
	results := make([][]Distancer, 0, size)

	for i := 0; i < itemSize; i += eachArrSize {
		end := i + eachArrSize

		if end > itemSize {
			end = itemSize
		}

		results = append(results, items[i:end])
	}

	return results
}

func NewEuclideanPoints(arrs ...[]float32) EuclideanPoints {
	result := make(EuclideanPoints, len(arrs))
	for idx := range arrs {
		result[idx] = &EuclideanPoint{
			ID:     uuid.New().String(),
			Vector: arrs[idx],
		}
	}

	return result
}

type EuclideanPoint struct {
	ID     string
	Vector []float32
}

type EuclideanPoints []*EuclideanPoint

func (e EuclideanPoints) ToDistancers() []Distancer {
	ps := make([]Distancer, len(e))
	for idx, p := range e {
		ps[idx] = p
	}

	return ps
}

func (e *EuclideanPoint) Distance(d interface{}) float32 {
	point, ok := d.(*EuclideanPoint)
	if !ok {
		return 0
	}

	return avx.EuclideanDistance(len(e.Vector), e.Vector, point.Vector)
}

func GenPoints(total, dim int) [][]float32 {
	result := make([][]float32, total)
	for idx := range result {
		r := make([]float32, dim)
		for i := 0; i < dim; i++ {
			r[i] = rand.Float32()
		}

		result[idx] = r
	}

	return result
}
