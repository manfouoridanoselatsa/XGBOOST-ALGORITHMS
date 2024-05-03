#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <memory>
//#include "eigen-3.3.9/Eigen/Dense"
//#include "MyXGBClassifier.h"
//#include <bits/stdc++.h>

/*class MyXGBClassifierHello {
    private:
		int x;
	public:
	    MyXGBClassifierHello(int y){
			x = y;
			std::cout << "The value of x is: "<< x << "\n";
	    }
};*/

struct TreeNode {
    double leafValue;
    int featureIndex;
    double split_point;
    double gain;
    TreeNode* leftChild;
    TreeNode* rightChild;
    TreeNode() : leafValue(0.0), featureIndex(-1), split_point(0.0),gain(0.0), leftChild(nullptr), rightChild(nullptr) {}
};

class MyXGBClassificationTree {
	private:
	    int max_depth;
	    double reg_lambda;
	    double prune_gamma;
	    TreeNode* estimator1;
	    TreeNode* estimator2;
	    std::vector<std::vector<double>> feature;
	    std::vector<double> residual;
	    std::vector<double> prev_yhat;
	public:
		
	//constructeur de ma classe
	MyXGBClassificationTree(int max_depth, double reg_lambda, double prune_gamma, std::vector<std::vector<double>> feature,std::vector<double> residual,std::vector<double> prev_yhat) {
		this->max_depth = max_depth;
		this->reg_lambda = reg_lambda;
		this->prune_gamma = prune_gamma;
		//this->estimator1;
		//this->estimator2;
		this->feature = feature;
		this->residual=residual;
		this->prev_yhat=prev_yhat;
		
	}
	
	//constructeur de ma classe
	MyXGBClassificationTree(int max_depth, double reg_lambda, double prune_gamma) {
		this->max_depth = max_depth;
		this->reg_lambda = reg_lambda;
		this->prune_gamma = prune_gamma;
		//this->estimator1;
		//this->estimator2;
		//this->feature;
		//this->residual;
		//this->prev_yhat;
	}

/*   ici on cree la fonction node_split qui vas ce charger de diviser un noeud d'un type dictionnaire en d'autre noeud*/
    	std::unordered_map<std::string,std::vector<double>> node_split(std::vector<int> did) {
        
        int b_fid = 0;
        double b_point = 0;
        double r = this->reg_lambda;
        double max_gain = -std::numeric_limits<double>::infinity();
        int d = this->feature[0].size(); // nombre de feature
        
        double G =  std::accumulate(this->residual.begin() + did[0], this->residual.begin() + did[did.size()-1]+1, 0.0);
        double H =  inner_product(this->prev_yhat.begin() + did[0], this->prev_yhat.begin() + did[did.size()-1] + 1, this->prev_yhat.begin() + did[0], 0.0, std::plus<double>(), [](double a, double b) { return a * (1.0 - b); });
        double p_score = (G * G) / (H + r);
        
        for (int k = 0; k < d; k++) {
            
		    double GL = 0.0, HL = 0.0;
		    std::vector<double> x_feat(did.size());
		    for (int i = 0; i < did.size(); i++) {
			x_feat[i] = this->feature[did[i]][k];
		    }
		    
		    
		    
		    std::vector<double>x_uniq(std::begin(x_feat), std::end(x_feat));
		    std::sort(std::begin(x_uniq), std::end(x_uniq));
		    
		    
		    std::vector<double> s_point(x_uniq.size() - 1);
		    for (int i = 1; i < x_uniq.size(); i++) {
			s_point[i - 1] = (x_uniq[i - 1] + x_uniq[i]) / 2;
		    }
		    
		    
		    
		    double l_bound = -std::numeric_limits<double>::infinity();
		    //std::cout<<"\nhello "<<k<<"\n";
		    for (double j : s_point) {
			std::vector<int> left;
			std::vector<int> right;
			
			for (int i : did) {
			    if (this->feature[i][k] > l_bound && this->feature[i][k] <= j) {
				left.push_back(i);
			    } else if (this->feature[i][k] > j) {
				right.push_back(i);
			    }
			    
			
			}
			
			for(int i=0;i<left.size();i++){
			    GL += this->residual[i];
			    HL += this->prev_yhat[i]*(1.0 - this->prev_yhat[i]);
			}
			
			//GL += std::accumulate(residualLeft.begin() + left[0],residualLeft.begin() + left[left.size()-1]+1, 0.0);
			/*GL += std::accumulate(this->residual.begin(),
			this->residual.end(), 0.0);*/
			
			//HL += 0.000063;//inner_product(prev_yhatLeft.begin() + left[0], prev_yhatLeft.begin() + left[left.size()-1] + 1, prev_yhatLeft.begin() + left[0], 0.0, std::plus<double>(), [](double a, double b) { return a * (1.0 - b); });
			
			double GR = G - GL;
			double HR = H - HL;
			
			double gain = (GL * GL) / (HL + r) + (GR * GR) / (HR + r) - p_score;
			

			if (gain >= max_gain) {
			    max_gain = gain;
			    b_fid = k;
			    b_point = j;
			}
			l_bound = j;
			
		   }
		   
                  
            };
            
            
            
            std::vector<double> b_left, b_right;
            if (max_gain >= this->prune_gamma) {
                
                for (int i = 0; i < did.size(); i++) {
                    if (this->feature[did[i]][b_fid] <= b_point) {
                        b_left.push_back((double)did[i]);
                    } else {
                        b_right.push_back((double)did[i]);
                    }
                }
               
                
                std::vector<double> vec_b_fid;
                vec_b_fid.push_back(b_fid);
                
                std::vector<double> vec_split_point;
                vec_split_point.push_back(b_point);
                
                std::vector<double> vec_gain;
                vec_gain.push_back(max_gain);
                
                
                
                
                
                
              return {{"fid", vec_b_fid}, {"split_point", vec_split_point}, {"gain", vec_gain}, {"left", b_left}, {"right", b_right}};
                //return return_map; 
            };
       // };
        return {};
    };
    
    
        
    TreeNode* recursive_split(std::vector<int> did, int curr_depth) {
        
        if (curr_depth >= this->max_depth){
        	TreeNode* leafNode = new TreeNode();
        	leafNode->leafValue = output_value(did);
        	return leafNode;
        };
        
        if (did.size() <=1){
        	TreeNode* leafNode = new TreeNode();
        	leafNode->leafValue = output_value(did);
        	return leafNode;
        };
        
        
        std::unordered_map<std::string,std::vector<double>> nodeRep = node_split(did);
        
        TreeNode* node = new TreeNode();
	 node->featureIndex = nodeRep["fid"][0];
	 node->split_point = nodeRep["split_point"][0];
	 node->gain = nodeRep["gain"][0];
	 
	 int leftSize = nodeRep["left"].size();
	 int rightSize= nodeRep["right"].size();
	 //std::cout<<"\n\nhello "<<rightSize<<" "<<leftSize<<"\n";
	 std::vector<int> left ;
	 std::vector<int> right;
	 
	 for (int i = 0; i < leftSize; i++) {
            left.push_back((int)nodeRep["left"][i]);
            //std::cout<<"hello "<<(int)nodeRep["left"][i] <<"\n";
         }
         //std::cout<<"\n\n";
         for (int i = 0; i < rightSize; i++) {
            right.push_back((int)nodeRep["right"][i]);
            //std::cout<<"hello "<<(int)nodeRep["right"][i] <<"\n";
         }
	 
	 node->leftChild = recursive_split(left,curr_depth+1);
	 node->rightChild = recursive_split(right,curr_depth+1);
        
        return node;
    }


    double output_value(std::vector<int> did) {
        double r = std::accumulate(this->residual.begin() + did[0], this->residual.begin() + did[did.size()-1]+1, 0.0);
        double H = inner_product(this->prev_yhat.begin() + did[0], this->prev_yhat.begin() + did[did.size()-1] + 1, this->prev_yhat.begin() + did[0], 0.0, std::plus<double>(), [](double a, double b) { return a * (1.0 - b); });
        return r / (H + this->reg_lambda);
    }
    
    
    TreeNode*  fit(std::vector<std::vector<double>> x, std::vector<double> y, std::vector<double> prev_yhat) {
        this->feature = x;
        this->residual = y;
        this->prev_yhat = prev_yhat;
	
        std::vector<int> root_did(x.size());
        std::iota(root_did.begin(), root_did.end(), 0);
        
        this->estimator2 = this->recursive_split(root_did, 1);
        return this->estimator2;
    }
    
    double predict_leaf_value(std::vector<double> features){
    	    TreeNode* treeTemp = new TreeNode();
    	    treeTemp = this->estimator2;
	    while (treeTemp->leftChild != nullptr && treeTemp->rightChild != nullptr) {
		if (features[treeTemp->featureIndex] <= treeTemp->split_point) {
		    treeTemp = treeTemp->leftChild;
		} else {
		    treeTemp = treeTemp->rightChild;
		}
	    }

	    return treeTemp->leafValue;
    }
    
    std::vector<double>  predict_leaf_multi_entry(std::vector<std::vector<double>> x_test){
	
	std::vector<double> y_pred(x_test.size());
        for (int i = 0; i < x_test.size(); i++) {
            y_pred[i] = predict_leaf_value(x_test[i]);
        }
        return y_pred;
    }
	~MyXGBClassificationTree(){
		//std::cout<<" Hello destroy:"<<std::endl;
	};	
	
};

// Mon Classifieur
class MyXGBClassifier {
	private:
	    int n_estimators;
	    int max_depth;
	    double eta;
	    double prune_gamma;
	    double reg_lambda;
	    double base_score;
	    std::vector<MyXGBClassificationTree> models;
	    std::vector<double> loss;
	public:
	    MyXGBClassifier(int n_estimators = 10, int max_depth = 3, double learning_rate = 0.3, double prune_gamma = 0.0, double reg_lambda = 0.0, double base_score = 0.5){

			this->n_estimators = n_estimators;
			this->max_depth = max_depth;
			this->eta=learning_rate;
			this->prune_gamma  = prune_gamma;
			this->reg_lambda = reg_lambda;
			this->base_score = base_score;
			
	   }
	   
	   double F2P(double x) {
			return std::exp(x) / (1.0 + std::exp(x));
	   }

	   std::vector<double> fit(std::vector<std::vector<double>> x, std::vector<int> y) {
			double F0 = std::log(this->base_score / (1.0 - this->base_score));
			int number_user = x.size();
			std::vector<double> Fm(number_user, F0);
			std::vector<double> y_hat(number_user);
			
			for(int i=0;i<number_user;i++){
				y_hat[i] = this->F2P(Fm[i]);
			}
			
			this->models.clear();
			this->loss.clear();

			for (int m = 0; m < this->n_estimators; m++) {
				
				std::vector<double> residual(y.size());
				
				for (int i = 0; i < y.size(); i++) {
				    residual[i] = y[i] - y_hat[i];
				}
				
				//transform(y.begin(), y.end(), y_hat.begin(), residual.begin(), [](double a, double b) { return a - b; });
				
				MyXGBClassificationTree model(this->max_depth, this->reg_lambda, this->prune_gamma);
				;
				model.fit(x, residual, y_hat);
				

				std::vector<double>  gamma =model.predict_leaf_multi_entry(x);
				
				for (int i = 0; i < Fm.size(); i++) {
				    Fm[i] += eta * gamma[i];
				}

				transform(Fm.begin(), Fm.end(), y_hat.begin(), [&](double x) { return this->F2P(x); });
				/*for (int i = 0; i < Fm.size(); i++) {
				    y_hat[i] = this->F2P(Fm[i]);
				}*/


				models.push_back(model);

				double loss_value = 0.0;
				for (int i = 0; i < y.size(); i++) {
				    loss_value -= y[i] * log(y_hat[i]) + (1.0 - y[i]) * log(1.0 - y_hat[i]);
				}
				
				loss.push_back(loss_value);
				
			}

			return this->loss;
		}
		
		double predict_simple_entry(std::vector<double> x_test, bool proba = false) {
		    double Fm= this->base_score;
		    for (auto& model : this->models) {
			double gamma = model.predict_leaf_value(x_test);
			Fm = Fm + this->eta*gamma;
		    }

		    if (proba) {
			return this->F2P(Fm);
		    } else {
			return this->F2P(Fm) > 0.5 ? 1.0 : 0.0;
		    }

	    	}	
			
		std::vector<double> predict_multi_entry(std::vector<std::vector<double>> x_test, bool proba = false) {
		    std::vector<double> Fm(x_test.size(), this->base_score);
		    for (auto& model : this->models) {
			std::vector<double> gamma = model.predict_leaf_multi_entry(x_test);
			transform(Fm.begin(), Fm.end(), gamma.begin(), Fm.begin(), [&](double a, double b) { return a + this->eta * b; });
		    }

		    std::vector<double> y_pred(x_test.size());
		    if (proba) {
			transform(Fm.begin(), Fm.end(), y_pred.begin(), [&](double x) { return this->F2P(x); });
		    } else {
			transform(Fm.begin(), Fm.end(), y_pred.begin(), [&](double x) { return this->F2P(x) > 0.5 ? 1.0 : 0.0; });
		    }

		    return y_pred;
	    	}

	   

	
};



void printCSV(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        std::vector<std::string> row;
        while (std::getline(iss, value, ',')) {
            row.push_back(value);
        }

    }

    file.close();
}

std::vector<std::vector<double>> extractCSVDataset(const std::string& filename) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        exit(0);
    }

    std::string line;
    std::vector<std::vector<double>> row;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string value;
        
        std::vector<double> rowLine;
        while (std::getline(iss, value, ',')) {
            rowLine.push_back(std::stod(value));
        }
        row.push_back(rowLine);

    }

    file.close();
    
    return row;
}    

std::vector<int> calculate_confusion_matrix(std::vector<int> true_labels, std::vector<double> predicted_labels, int num_samples, std::vector<int> confusion_matrix) {
    for (int i = 0; i < num_samples; i++) {
        int true_label = true_labels[i];
        int predicted_label = (int)predicted_labels[i];
        
        // Increment the corresponding cell in the confusion matrix
        confusion_matrix[true_label * 2 + predicted_label]++;
    }
    
    return  confusion_matrix;
}

double calculate_accuracy(std::vector<int> confusion_matrix) {
    int true_positive = confusion_matrix[3];
    int true_negative = confusion_matrix[0];
    int total_samples = true_positive + true_negative+confusion_matrix[1]+confusion_matrix[2];
    printf("\npoints %d\n", total_samples);
    double accuracy = (double)(true_positive + true_negative) / total_samples;
    return accuracy;
}


void print_confusion_matrix(std::vector<int> confusion_matrix) {
    printf("Confusion Matrix:\n");
    printf("          Predicted\n");
    printf("          0     1\n");
    printf("Actual 0  %d     %d\n", confusion_matrix[0], confusion_matrix[1]);
    printf("       1  %d     %d\n", confusion_matrix[2], confusion_matrix[3]);
}
    

int main() {
    double y_init = 0.5; // Assuming y is a vector of doubles
    int n_depth =8;
    int n_tree = 100;
    double eta = 0.1;
    double reg_lambda = 0.1;
    double prune_gamma = 0.6;
    
    if (__cplusplus == 202101L) std::cout << "C++23";
    else if (__cplusplus == 202002L) std::cout << "C++20";
    else if (__cplusplus == 201703L) std::cout << "C++17";
    else if (__cplusplus == 201402L) std::cout << "C++14";
    else if (__cplusplus == 201103L) std::cout << "C++11";
    else if (__cplusplus == 199711L) std::cout << "C++98";
    else std::cout << "pre-standard C++." << __cplusplus;
    std::cout << "\n\n\n";
    /*std::vector<std::vector<double>>x{{ 10.3, 20.2, 30.5, 30.5, 32.5,10.3, 20.2, 30.5, 30.5, 32.5},
                                        { 12.8, 22.0, 33.2, 0.5, 31.5,12.8, 22.0, 33.2, 0.5, 31.5},
                                        { 3.6, 18.1, 28.3 , 6.5, 33.5,3.6, 18.1, 28.3 , 6.5, 33.5},
                                        { 3.8, 12.1, 20.3 , 8.5, 22.5,3.8, 12.1, 20.3 , 8.5, 22.5},
                                        { 7.2, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5},
                                        { 8.6, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5},
                                        { 14.2, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5}
                                        };
    std::vector<int> y{ 0,1,0,1,0,0,0 };
    std::vector<double> prev_yhat{0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    std::vector<double> residus{-0.5,0.5,-0.5,0.5,-0.5,-0.5,-0.5};*/

    //MyXGBClassifierHello my_model(6);
    //std::vector<int> indice{0,1,2,3,4,5,6};
    //MyXGBClassificationTree my_model(n_depth,reg_lambda, prune_gamma,x,residus,prev_yhat);
    //my_model.node_split(indice);
    //my_model.recursive_split(indice,1);
    /*MyXGBClassificationTree my_model(n_depth,reg_lambda, prune_gamma,x,residus,prev_yhat);
    std::vector<int> indice{0,1,2,3,4};
    //my_model.node_split(indice);
    my_model.recursive_split(indice,1);
    
    */
    // donnee d'entrainnement
    std::string filename = "diabetesData.csv";
    std::vector<std::vector<double>> x = extractCSVDataset(filename);
     
    filename = "diabetesLabel.csv";
    std::vector<std::vector<double>> yTemp = extractCSVDataset(filename);
    
    std::vector<int> y;
     
     for(int i=0; i<yTemp.size();i++){
        y.push_back(((int)yTemp[i][0]));
        
     }
     // initialisation du classifieur
     MyXGBClassifier tryIt(n_tree,n_depth,eta,prune_gamma,reg_lambda,y_init);
     // entrainnement
     std::vector<double> loss = tryIt.fit(x,y);
     
     // valeur de perte par arbre
     for(int i=0; i<n_tree;i++){
        std::cout<<" "<<loss[i];
     }
     std::cout<<"\n\n\n";
     
     //donnee de test
     filename = "diabetesDataTest.csv";
     std::vector<std::vector<double>> xTest = extractCSVDataset(filename);
     
     filename = "diabetesLabelTest.csv";
     std::vector<std::vector<double>> yTempTest = extractCSVDataset(filename);
     
     std::vector<int> yTest;
    
     for(int i=0; i<yTempTest.size();i++){
        yTest.push_back(((int)yTempTest[i][0]));
     }
     //prediction des donnees de test
     std::vector<double>predicted_labels= tryIt.predict_multi_entry(xTest);
      
     int confusionSize = 4 ;
     std::vector<int> confusion_matrix(confusionSize,0);
     confusion_matrix = calculate_confusion_matrix(yTest, predicted_labels, yTest.size(), confusion_matrix);
     
     
      print_confusion_matrix(confusion_matrix);
      
      double accuracy = calculate_accuracy(confusion_matrix);
      printf("\n\nAccuracy: %.2f%%\n", accuracy * 100);
    
      
     // prediction d'une entree
     std::vector<double> testData =  { 6,154,74,32,193,29.3,0.839,39  };
     std::cout<<"\n"<<tryIt.predict_simple_entry(testData)<<"\n";
     

    return 0;
}
