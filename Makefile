xgbParallel:xgbParallel.cpp
	g++ -pthread -std=c++17 -Wall -Wextra  -o xgbParallel xgbParallel.cpp -lm

#"Format de l'executable: <<nom fichier>> <<nombre-thread>> <<fichier-dataset>> <<fichier-labels>> <<train-test-percent>> <<subsample_cols>> <<min_child_weight>> <<depth>> <<min_leaf>> <<learning_rate>> <<trees>> <<lambda>> <<gamma>> <<epsilon>> <<NanOuPas>>\n"
#"Datasets/diabetesData.csv" "Datasets/labels.csv"

run:xgbParallel
	./xgbParallel 4 "Datasets/diabetesData.csv" "Datasets/labels.csv" 0.2 1 1 8 1 0.2 100 0 0.1 0.2 non
	
clean:
	rm xgbParallel

xgboostTest:xgboostTest.cpp
	g++ -pthread -std=c++17 -o xgboostTest xgboostTest.cpp -lm

run_xgboostTest:xgboostTest
	./xgboostTest 1

clean_xgboostTest:
	rm xgboostTest
