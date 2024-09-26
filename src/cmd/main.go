package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"strings"
	"time"

	"concurrente/internal/randomforest"
)

const (
	labelIndex = 14 // columna exporta
)

var (
	numTrees    int     = 10
	subsetRatio float64 = 0.8
	trainRatio  float64 = 0.8
	datasetSize int     = 100000
)

func main() {
	for {
		printMenu()
		choice := getUserChoice()

		switch choice {
		case 1:
			setAlgorithmParameters()
		case 2:
			setSimulationParameters()
		case 3:
			runSimulation()
		case 4:
			compareRuntimes()
		case 5:
			predictExporta()
		case 6:
			fmt.Println("Saliendo del programa. ¡Hasta luego!")
			return
		default:
			fmt.Println("Opción no válida. Por favor, intente de nuevo.")
		}
	}
}

func printMenu() {
	fmt.Println("\n--- Menú de Random Forest Paralelo ---")
	fmt.Println("1. Establecer Parámetros del Algoritmo")
	fmt.Println("2. Establecer Parámetros de Simulación")
	fmt.Println("3. Ejecutar Simulación")
	fmt.Println("4. Comparar Tiempos de Ejecución para Diferentes Tamaños de Datos")
	fmt.Println("5. Predecir 'exporta' para un Registro Individual")
	fmt.Println("6. Salir")
	fmt.Print("Ingrese su elección: ")
}

func getUserChoice() int {
	var choice int
	_, err := fmt.Scan(&choice)
	if err != nil {
		fmt.Println("Error al leer la entrada:", err)
		return 0
	}
	return choice
}

func setAlgorithmParameters() {
	fmt.Printf("\nNúmero actual de árboles: %d\n", numTrees)
	fmt.Print("Ingrese nuevo número de árboles (o presione Enter para mantener el actual): ")
	input := readLine()
	if input != "" {
		if val, err := strconv.Atoi(input); err == nil {
			numTrees = val
		}
	}

	fmt.Printf("\nRatio de subconjunto actual: %.2f\n", subsetRatio)
	fmt.Print(
		"Ingrese nuevo ratio de subconjunto (0-1) (o presione Enter para mantener el actual): ",
	)
	input = readLine()
	if input != "" {
		if val, err := strconv.ParseFloat(input, 64); err == nil && val > 0 && val <= 1 {
			subsetRatio = val
		}
	}

	fmt.Printf(
		"\nParámetros del algoritmo actualizados: Árboles = %d, Ratio de Subconjunto = %.2f\n",
		numTrees,
		subsetRatio,
	)
}

func setSimulationParameters() {
	fmt.Printf("\nRatio actual de división entrenamiento/prueba: %.2f\n", trainRatio)
	fmt.Print(
		"Ingrese nuevo ratio de división entrenamiento/prueba (0-1) (o presione Enter para mantener el actual): ",
	)
	input := readLine()
	if input != "" {
		if val, err := strconv.ParseFloat(input, 64); err == nil && val > 0 && val < 1 {
			trainRatio = val
		}
	}

	fmt.Printf("\nTamaño actual del conjunto de datos: %d\n", datasetSize)
	fmt.Print(
		"Ingrese nuevo tamaño del conjunto de datos (o presione Enter para mantener el actual): ",
	)
	input = readLine()
	if input != "" {
		if val, err := strconv.Atoi(input); err == nil && val > 0 {
			datasetSize = val
		}
	}

	fmt.Printf(
		"\nParámetros de simulación actualizados: Ratio de Entrenamiento = %.2f, Tamaño del Conjunto de Datos = %d\n",
		trainRatio,
		datasetSize,
	)
}

func runSimulation() {
	fmt.Printf("\n--- Ejecutando Simulación ---\n")
	fmt.Printf(
		"Parámetros del Algoritmo: Árboles = %d, Ratio de Subconjunto = %.2f\n",
		numTrees,
		subsetRatio,
	)
	fmt.Printf(
		"Parámetros de Simulación: Ratio de Entrenamiento = %.2f, Tamaño del Conjunto de Datos = %d\n",
		trainRatio,
		datasetSize,
	)

	allData := readAndPrepareData(datasetSize)
	trainData, testData := splitData(allData, trainRatio)

	rf := randomforest.NewParallelRandomForest(numTrees, subsetRatio)
	trainTime, evalTime, accuracy := testRandomForestParallel(rf, trainData, testData)

	fmt.Printf("\nResultados:\n")
	fmt.Printf("Tiempo de Entrenamiento: %v\n", trainTime)
	fmt.Printf("Tiempo de Evaluación: %v\n", evalTime)
	fmt.Printf("Tiempo Total: %v\n", trainTime+evalTime)
	fmt.Printf("Precisión: %.2f%%\n", accuracy*100)
}

func compareRuntimes() {
	rowSizes := []int{1000, 10000, 100000, 1000000}

	for _, size := range rowSizes {
		fmt.Printf("\n--- Probando con %d filas ---\n", size)
		allData := readAndPrepareData(size)
		trainData, testData := splitData(allData, trainRatio)

		rf := randomforest.NewParallelRandomForest(numTrees, subsetRatio)
		trainTime, evalTime, accuracy := testRandomForestParallel(rf, trainData, testData)

		fmt.Printf("Tiempo de Entrenamiento: %v\n", trainTime)
		fmt.Printf("Tiempo de Evaluación: %v\n", evalTime)
		fmt.Printf("Tiempo Total: %v\n", trainTime+evalTime)
		fmt.Printf("Precisión: %.2f%%\n", accuracy*100)
	}
}

func predictExporta() {
	fmt.Println("\nIngrese los valores para cada columna (separados por '|'):")
	input := readLine()

	record := strings.Split(input, "|")
	if len(record) != 18 {
		fmt.Println("Error: Número inválido de columnas. Se esperaban 18 columnas.")
		return
	}

	allData := readAndPrepareData(datasetSize)
	trainData, _ := splitData(allData, 1.0) // Usar todos los datos para entrenamiento

	rf := randomforest.NewParallelRandomForest(numTrees, subsetRatio)
	rf.Train(trainData)

	prediction := rf.Predict(record[:len(record)-1]) // Excluir la última columna (fec_creacion)

	fmt.Printf("Predicción para 'exporta': ")
	if prediction >= 0.5 {
		fmt.Println("SI")
	} else {
		fmt.Println("NO")
	}
}

func testRandomForestParallel(
	rf *randomforest.ParallelRandomForest,
	trainData, testData [][]string,
) (time.Duration, time.Duration, float64) {
	trainTimeStart := time.Now()
	rf.Train(trainData)
	trainTime := time.Since(trainTimeStart)

	evalTimeStart := time.Now()
	accuracy := evaluateModel(rf, testData)
	evalTime := time.Since(evalTimeStart)

	return trainTime, evalTime, accuracy
}

func evaluateModel(rf *randomforest.ParallelRandomForest, testData [][]string) float64 {
	correct := 0
	for _, sample := range testData {
		features := sample[:len(sample)-1]
		label := sample[labelIndex]
		prediction := rf.Predict(features)
		if (prediction >= 0.5 && label == "SI") || (prediction < 0.5 && label == "NO") {
			correct++
		}
	}
	return float64(correct) / float64(len(testData))
}

func readAndPrepareData(limit int) [][]string {
	allData, err := readCSV("datasets/bd_mujeres_2023.csv", '|', limit)
	if err != nil {
		fmt.Println("Error al leer el CSV:", err)
		os.Exit(1)
	}
	fmt.Printf("Se leyeron %d registros del conjunto de datos\n", len(allData))
	return allData
}

func readCSV(filename string, separator rune, limit int) ([][]string, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.Comma = separator

	var data [][]string
	for i := 0; ; i++ {
		record, err := reader.Read()
		if err != nil {
			break
		}
		if i == 0 {
			continue // saltar encabezado
		}
		if limit > 0 && i > limit {
			break
		}
		data = append(data, record)
	}
	return data, nil
}

func splitData(data [][]string, trainRatio float64) ([][]string, [][]string) {
	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	splitIndex := int(float64(len(data)) * trainRatio)
	return data[:splitIndex], data[splitIndex:]
}

func readLine() string {
	reader := bufio.NewReader(os.Stdin)
	input, _ := reader.ReadString('\n')
	return strings.TrimSpace(input)
}

