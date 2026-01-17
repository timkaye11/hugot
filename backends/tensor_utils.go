package backends

// Flatten2DInt64 flattens a 2D int64 slice into a 1D slice.
// Each row in the input is placed sequentially in the output.
func Flatten2DInt64(data [][]int64, rows, cols int) []int64 {
	result := make([]int64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			if i < len(data) && j < len(data[i]) {
				result[idx] = data[i][j]
			}
		}
	}
	return result
}

// Flatten2DInt64Pair flattens two 2D int64 slices (e.g., token IDs and attention mask).
// This is more efficient than calling Flatten2DInt64 twice.
func Flatten2DInt64Pair(data1, data2 [][]int64, rows, cols int) ([]int64, []int64) {
	result1 := make([]int64, rows*cols)
	result2 := make([]int64, rows*cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			idx := i*cols + j
			if i < len(data1) && j < len(data1[i]) {
				result1[idx] = data1[i][j]
			}
			if i < len(data2) && j < len(data2[i]) {
				result2[idx] = data2[i][j]
			}
		}
	}
	return result1, result2
}
