package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"

	"github.com/gonum/stat"
	"gonum.org/v1/gonum/mat"
)

type SVM struct {
	degree float64
	C      float64
	alpha  []float64
	b      float64
	X      *mat.Dense
	Y      []float64
	gamma  float64
	coef0  float64
}

func (svm SVM) RBFKernel(x1, x2 []float64) float64 {
	gamma := svm.gamma
	norm := 0.0
	for i := 0; i < len(x1); i++ {
		norm += math.Pow(x1[i]-x2[i], 2)
	}
	return math.Exp(-gamma * norm)
}

func (svm SVM) polyKernel(x1, x2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(x1); i++ {
		dot += x1[i] * x2[i]
	}
	value := svm.C + dot
	return math.Pow(value, float64(svm.degree))
}

func (svm SVM) sigmoidKernel(x1, x2 []float64) float64 {
	dot := 0.0
	for i := 0; i < len(x1); i++ {
		dot += x1[i] * x2[i]
	}
	value := svm.gamma*dot + svm.coef0
	return math.Tanh(value)
}

func (svm *SVM) train(x [][]float64, y []float64) {
	//rand.Seed(time.Now().UnixNano())

	svm.X = mat.NewDense(len(x), len(x[0]), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[0]); j++ {
			svm.X.Set(i, j, x[i][j])
		}
	}

	svm.Y = y

	alpha := make([]float64, len(x))
	for i := 0; i < len(alpha); i++ {
		alpha[i] = 0
	}

	xMat := mat.NewDense(len(x), len(x[0]), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x[0]); j++ {
			xMat.Set(i, j, x[i][j])
		}
	}

	gramMatrix := mat.NewDense(len(x), len(x), nil)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(x); j++ {
			gramMatrix.Set(i, j, svm.polyKernel(x[i], x[j]))
			//gramMatrix.Set(i, j, svm.RBFKernel(x[i], x[j]))
			//gramMatrix.Set(i, j, svm.sigmoidKernel(x[i], x[j]))
		}
	}

	var supportVectors []int
	for epoch := 0; epoch < 100; epoch++ {
		for i := 0; i < len(x); i++ {
			f := mat.Dot(mat.NewVecDense(len(x), alpha), mat.NewVecDense(len(x), svm.Y))
			if svm.Y[i]*f < 1 {
				alpha[i] += svm.C - svm.Y[i]*f
			}
		}

		if epoch%10 == 0 {
			for i := 0; i < len(x); i++ {
				if alpha[i] > 0 {
					supportVectors = append(supportVectors, i)
				}
			}
		}
	}

	for i := 0; i < len(x); i++ {
		alpha[i] *= svm.Y[i]
	}
	svm.alpha = alpha

	// Compute the value of svm.b
	var bSum float64
	numSupportVectors := float64(len(supportVectors))
	for _, i := range supportVectors {
		bSum += svm.Y[i] - mat.Dot(mat.NewVecDense(len(x), svm.alpha), gramMatrix.RowView(i))
	}
	svm.b = bSum / numSupportVectors
}

func (svm SVM) predict(testdata []float64) float64 {
	result := 0.0

	for i := 0; i < len(svm.alpha); i++ {
		dotProduct := svm.polyKernel(svm.X.RawRowView(i), testdata)
		//dotProduct := svm.RBFKernel(svm.X.RawRowView(i), testdata)
		//dotProduct := svm.sigmoidKernel(svm.X.RawRowView(i), testdata)
		result += svm.alpha[i] * svm.Y[i] * dotProduct
	}
	result += svm.b
	//return result
	if result >= 1 {
		return 1.0
	} else {
		return 0.0
	}

}

//scaling functions

func scaleFeatures(x [][]float64) [][]float64 {
	scaled := make([][]float64, len(x))
	for i := range scaled {
		scaled[i] = make([]float64, len(x[i]))
	}
	for j := range x[0] {
		col := make([]float64, len(x))
		for i := range x {
			col[i] = x[i][j]
		}
		mean, std := stat.MeanStdDev(col, nil)
		for i := range x {
			scaled[i][j] = (x[i][j] - mean) / std
		}
	}
	return scaled
}

func scaleLabels(y []float64) []float64 {
	scaled := make([]float64, len(y))
	mean, std := stat.MeanStdDev(y, nil)
	for i := range y {
		scaled[i] = (y[i] - mean) / std
	}
	return scaled
}

// Spliting Data for checking accuracy
func split(x [][]float64, y []float64, ratio float64, seed int64) (train_x, test_x [][]float64, train_y, test_y []float64) {
	// Set the seed for the random number generator
	rand.Seed(seed)

	// Calculate the number of training examples
	nTrain := int(math.Round(float64(len(x)) * (1.0 - ratio)))

	// Shuffle the indices of the examples
	indices := rand.Perm(len(x))

	// Split the indices into training and test indices
	trainIndices := indices[:nTrain]
	testIndices := indices[nTrain:]

	// Initialize the training and test sets
	train_x = make([][]float64, nTrain)
	train_y = make([]float64, nTrain)
	test_x = make([][]float64, len(x)-nTrain)
	test_y = make([]float64, len(x)-nTrain)

	// Fill in the training and test sets
	for i, idx := range trainIndices {
		train_x[i] = x[idx]
		train_y[i] = y[idx]
	}
	for i, idx := range testIndices {
		test_x[i] = x[idx]
		test_y[i] = y[idx]
	}

	return train_x, test_x, train_y, test_y
}

func accuracy(predicted []float64, actual []float64) float64 {
	n := len(predicted)
	nCorrect := 0
	for i := 0; i < n; i++ {
		if predicted[i] == actual[i] {
			nCorrect++
		}
	}
	return (float64(nCorrect) / float64(n)) * 100
}

func main() {

	// Read data from CSV file
	file, err := os.Open("C:/Users/rbadr/Downloads/diabetes.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	// Exclude first row which contains variable names
	if _, err := reader.Read(); err != nil {
		log.Fatal(err)
	}

	records, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// Extract features and target variable
	var x [][]float64
	var y []float64

	for _, record := range records {
		xRow := make([]float64, 0, len(record)-1)
		for i, value := range record {
			if i != len(record)-1 {
				val, err := strconv.ParseFloat(value, 64)
				if err != nil {
					log.Fatal(err)
				}
				xRow = append(xRow, val)
			} else {
				val, err := strconv.ParseFloat(value, 64)
				if err != nil {
					log.Fatal(err)
				}
				y = append(y, val)
			}
		}
		x = append(x, xRow)
	}

	scaledX := scaleFeatures(x)
	//scaledY := scaleLabels(y)

	train_x, test_x, train_y, test_y := split(scaledX, y, 0.2, 43)

	//train the SVM model
	svm := SVM{degree: 3, C: 1, gamma: 0.01, coef0: 0}

	svm.train(train_x, train_y)

	// testdata := []float64{11, 155, 76, 28, 150, 33.3, 1.353, 51}
	// scaledTestData := scaleLabels(testdata)
	// predicted := svm.predict(scaledTestData)
	//fmt.Println("Scaled Test data", scaledTestData)
	//fmt.Println("Scaled Test data", scaledX)
	//fmt.Printf("Predicted value is: %v\n", predicted)

	predicted := make([]float64, len(test_x))
	for i, x := range test_x {
		predicted[i] = svm.predict(x)
	}

	acc := accuracy(predicted, test_y)
	fmt.Printf("Accuracy is: %v\n", acc)
	//fmt.Println(predicted)
	//fmt.Println(test_y)
}
