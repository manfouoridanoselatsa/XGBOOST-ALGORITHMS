xgboost_exec:xgboost_exec.cpp
	g++ -pthread -std=c++17 -o xgboost_exec xgboost_exec.cpp -lm

run:xgboost_exec
	./xgboost_exec

clean:
	rm xgboost_exec
