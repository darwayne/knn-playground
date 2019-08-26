package knn

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"testing"
	"time"
)

func TestKNN(t *testing.T) {
	as := NewEuclideanPoints([]float32{3, 4, 5, 6})[0]
	points := NewEuclideanPoints(
		[]float32{1, 2, 3, 4},
		[]float32{19, 15, 16, 17},
		[]float32{29, 25, 26, 27},
	)

	ps := make([]Distancer, len(points))
	for idx, p := range points {
		ps[idx] = p
	}

	fmt.Println(KNN(as, ps...))
}

func TestParallelKNN(t *testing.T) {
	as := NewEuclideanPoints([]float32{3, 4, 5, 6})[0]
	points := NewEuclideanPoints(
		[]float32{1, 2, 3, 4},
		[]float32{19, 15, 16, 17},
		[]float32{29, 25, 26, 27},
	)

	fmt.Println(ParallelKNN(as, points.ToDistancers()...))
}

func BenchmarkChunkDistancers(b *testing.B) {
	points := NewEuclideanPoints(genPoints(100000, 128)...).
		ToDistancers()
	num := runtime.NumCPU() - 1

	for i := 0; i < b.N; i++ {
		_ = chunkDistancers(num, points...)
	}
}

// Without AVX:
// 100: 427,839  1k: 4,644,175 10k: 45,078,098 100k: 544,539,846
// With AVX:
// 100: 8,858  1k: 87,441 10k: 944,566 100k: 12,721,693
func BenchmarkKNN(b *testing.B) {
	as := NewEuclideanPoints(genPoints(1, 128)...)[0]
	points := NewEuclideanPoints(genPoints(100, 128)...).ToDistancers()

	for i := 0; i < b.N; i++ {
		_ = KNN(as, points...)
	}
}

// Without AVX:
// 100: 152,205 1k: 934,473 10k: 6,954,788 100k: 81,367,415
// With AVX:
// 100: 14,237 1k: 41,677 10k: 303,512 100k: 2,895,545
func BenchmarkParallelKNN(b *testing.B) {
	as := NewEuclideanPoints(genPoints(1, 128)...)[0]
	points := NewEuclideanPoints(genPoints(100, 128)...).ToDistancers()

	for i := 0; i < b.N; i++ {
		_ = ParallelKNN(as, points...)
	}
}

func genPoints(total, dim int) [][]float32 {
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

func TestKNN2(t *testing.T) {
	fmt.Println(math.Ceil(float64(1) / float64(2)))
	points := NewEuclideanPoints(genPoints(5, 128)...)
	fmt.Println(len(chunkDistancers(6, points.ToDistancers()...)))

	fmt.Println(runtime.GOMAXPROCS(0))
}

func TestMemoryUsage(t *testing.T) {
	s := time.Now()
	points := NewEuclideanPoints(genPoints(8000000, 128)...).ToDistancers()
	fmt.Println("it took", time.Since(s), "to generate 8m random points")
	s = time.Now()
	ParallelKNN(points[0], points...)
	fmt.Println("search completed in", time.Since(s))
	time.Sleep(time.Minute)
}

func TestEuclidean(t *testing.T) {
	a := []float32{1, 2, 3, 4, 5, 6, 7}
	b := []float32{8, 9, 10}

	full := []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	query := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	aQuery := []float32{1, 1, 1, 1, 1, 1, 1}
	bQuery := []float32{1, 1, 1, 1}

	fmt.Println("full", avx.EuclideanDistance(10, query, full))
	fmt.Println("a", avx.EuclideanDistance(7, a, aQuery))
	fmt.Println("b", avx.EuclideanDistance(3, b, bQuery))
}
