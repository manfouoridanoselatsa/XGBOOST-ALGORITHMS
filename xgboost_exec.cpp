#include <iostream>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <memory>
#include <stack>
#include <random>
#include <pthread.h>
#include <sys/time.h>


static struct timeval _t1, _t2;
static struct timezone _tz;

#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

 long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec );
}

class Node {
	
	//private:
	    
	    
	public:
	    
	      //constructeur de ma classe
	      Node(std::vector<int>  idxs,int depth = 0,double val=-10000.0,double score =-10000.0,int var_idx =-1,double split=0.0,int left=-1,int right = -1) {
		
			this->idxs = idxs;
			this->depth = depth;
			this->val = val;
			this->score = score;
			this->var_idx = var_idx;
			this->split = split;
			this->left = left;
			this->right = right;
			
		}
		/*
		double get_val(){
			return this->val;
		}
		
		void set_val(double val){
			this->val = val;
		}
		
		double get_score(){
			return this->score;
		}
		void set_score(double score){
			this->score = score;
		}
		
		double get_split(){
			return this->score;
		}
		void set_split(double split){
			this->split = split;
		}
		
		double get_left(){
			return this->left;
		}
		void set_left(int left){
			this->left = left;
		}
		
		double get_right(){
			return this->right;
		}
		void set_right(int right){
			this->right = right;
		}*/
		
		std::vector<int>  idxs;
        int depth;
        double val;
        double score;
        int var_idx ;
        double split;
        int left;
        int right;
    
    	~Node() {};
};

class TreeGenerator{

	//private:
		
    	
	public:
		// pour creer notre arbre
		std::vector<Node> tree;
		std::stack<int> node_stack;
		//variables
		std::vector<int> column_subsample;
		double lambda;
		double gamma;
		int row_count;
		int col_count; 
		double subsample_cols;
		std::vector<double>  gradient;
	    	std::vector<double> hessian;
	    	int min_leaf;
	    	int depth;
	    	std::vector<int> idxs;
	    	int min_child_weight;
	    	std::vector<std::vector<double>> x;	
		
		
		TreeGenerator(std::vector<std::vector<double>> x, std::vector<double>  gradient,std::vector<double> hessian,std::vector<int>  idxs, double subsample_cols = 0.8 , int min_leaf = 1,int min_child_weight = 1 ,int depth = 4,double lambda = 1,double gamma = 1) {
			
			this->x = x;
			
			this->gradient = gradient;
			this->hessian = hessian;
			this->idxs = idxs;
			this->depth = depth;
			this->min_leaf = min_leaf;
			this->lambda = lambda;
			this->gamma  = gamma;
			this->min_child_weight = min_child_weight;
			this->row_count = idxs.size();
			this->col_count = x[1].size(); 
			this->subsample_cols =subsample_cols;
			
			
			this->column_subsample = col_subsample(false);
			this->find_varsplit();
			//this->printNode();
			
		}
		
		std::vector<int> col_subsample(bool random_col){
			if(random_col == true){
				// ici on veut faire la selection aleatoire des features a utiliser
				std::vector<int> permutation(this->col_count);
				std::vector<int> selected;

				// On initialise notre vecteur avec les valeurs quelcquonque
				for (int i = 0; i < this->col_count; i++) {
					permutation[i] = i;
				}
				

				// Melangeons de facon aleatoire le contenu de nos vecteurs
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(permutation.begin(), permutation.end(), g);

				// Faire le stockage des valeurs
				selected.reserve(round(this->subsample_cols * this->col_count));
				for(int i = 0; i < round(this->subsample_cols * this->col_count); i++) {
					selected.emplace_back(permutation[i]);
				}
				return selected;
			}else{
				std::vector<int> root_did(this->col_count);
			    	std::iota(root_did.begin(), root_did.end(), 0);
			    	return root_did;
			}
		}
		
		double compute_gamma( std::vector<double> gradient, std::vector<double>  hessian){
			double G=0.0;
			double H=0.0;
			
			int gradient_size = gradient.size();
			for (int i = 0; i < gradient_size; i++) {
				G += gradient[i];
				H += hessian[i];
			}
        
        		return (-G/(H + this->lambda));
		}
		
		std::vector<Node> get_tree(){
			return this->tree;
		}
		
		std::vector<int> get_idxs(){
			return this->idxs;
		}
		
		
		
		void find_varsplit() {
		
		    tree.emplace_back(Node(this->idxs, 0, 0.0,-10000.0,-1, 0.0, -1,-1));
		    node_stack.push(0);
		    
		    while(!node_stack.empty()){
		    	// ici on continu l'entrainnement
		    	int node_idx = node_stack.top();
		    	node_stack.pop();
		    	//std::cout<<node_idx<<std::endl;
		    	find_greedy_split(node_idx);
		    }
		    //std::cout<<std::endl;
		    
		}
		
		void find_greedy_split(int node_idx){
			
			std::vector<double> gradientS;
			std::vector<double>  hessianS;
			
			std::vector<int> lhsF;
			std::vector<int>  rhsF;
			
			for(int i : tree[node_idx].idxs){
				gradientS.emplace_back(this->gradient[i]);
				hessianS.emplace_back(this->hessian[i]);
				//std::cout<<i<<" ";		
			}
			//std::cout<<std::endl;
			
			
			int node_tree_size = tree[node_idx].idxs.size();
			
			if( node_tree_size <= 1 || tree[node_idx].depth >= this->depth-1){
				tree[node_idx].val = this->compute_gamma(gradientS,hessianS);
				return;
			}
			
			
			for (int c : this->column_subsample){
				std::vector<double> xsplit;
				
				// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
				for (int idx : tree[node_idx].idxs){
					xsplit.emplace_back(this->x[idx][c]);
				}
				int xsplit_size = xsplit.size();
				
				for (int r = 0; r < xsplit_size; r++){
					std::vector<bool> lhs(xsplit_size, false);
					std::vector<bool> rhs(xsplit_size, false);
					int lhs_sum=0;
					int rhs_sum=0;
					std::vector<int> lhs_indices, rhs_indices;
					double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
					
					for (int i = 0; i < xsplit_size; i++){
						if (xsplit[i] <= xsplit[r]) {
							lhs[i] = true;
							lhs_sum++;
							lhs_indices.emplace_back(tree[node_idx].idxs[i]);
							lhs_hessian_sum += hessianS[i];
						} else{
							rhs[i] = true;
							rhs_sum++;
							rhs_indices.emplace_back(tree[node_idx].idxs[i]);
							rhs_hessian_sum += hessianS[i];
						}
						
					}
					
					
					
					if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum < this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
						continue;
					}
					
					
					double  curr_score = this->gain(lhs,rhs,node_idx);
					
					if (curr_score > tree[node_idx].score){
						tree[node_idx].var_idx = c;
						tree[node_idx].score  = curr_score;
						tree[node_idx].split = xsplit[r];
						lhsF.clear();
						rhsF.clear();
						lhsF = lhs_indices;
						rhsF = rhs_indices;
					}
					lhs_indices.clear();
					rhs_indices.clear();
					lhs.clear();
					rhs.clear();
				}
			
				xsplit.clear();
			}
			//ajout des noeuds gauche et droit dans notre arbre
			int left_child_idx = tree.size();
			tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, -10000.0,-1, 0.0, -1,-1));
			int right_child_idx = tree.size();
			tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, -10000.0,-1, 0.0, -1,-1));
			
			//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
			
			tree[node_idx].left = left_child_idx;
			tree[node_idx].right = right_child_idx;
		
			node_stack.push(left_child_idx);
			node_stack.push(right_child_idx);
			
		}
		
		double gain(std::vector<bool> lhs,std::vector<bool> rhs,int node_idx){
			std::vector<double> hessian,gradient;
			for (int idx : tree[node_idx].idxs) {
				gradient.emplace_back(this->gradient[idx]);
				hessian.emplace_back(this->hessian[idx]);
			}
			
			double lhs_hessian,lhs_gradient,rhs_gradient,rhs_hessian;
			int lhs_size = lhs.size();
			for (int i = 0; i < lhs_size; i++) {
				if(lhs[i] == true){
					lhs_gradient += gradient[i];
					lhs_hessian += hessian[i];
				}
			}
			int rhs_size = rhs.size();
			for (int i = 0; i < rhs_size; i++) {
				if(rhs[i] == true){
					rhs_gradient += gradient[i];
					rhs_hessian += hessian[i];
				}
			}
			
			double gain = 0.5*(((lhs_gradient*lhs_gradient)/(lhs_hessian + this->lambda)) + ((rhs_gradient*rhs_gradient)/(rhs_hessian + this->lambda)) - (((lhs_gradient + rhs_gradient)*(lhs_gradient + rhs_gradient))/(lhs_hessian + rhs_hessian + this->lambda))) - this->gamma;
			gradient.clear();
			hessian.clear();
			
			return gain;
			
		}
		
		std::vector<double> predict(std::vector<std::vector<double>> x) {
		    int x_size = x.size();
		    std::vector<double> y_pred(x_size);
		    
		    for (int i = 0; i < x_size; i++) {
		    	y_pred[i] = this->predict_row(x[i]);
		    }

		    return y_pred;
	    }
	    
	    double predict_row(std::vector<double> xi){
	    	int node_idx = 0;
	    	Node node = tree[node_idx];
	    	int tree_size = tree.size();
	    	for(int i=0;i<tree_size;i++){
	    		
	    		if(tree[node_idx].var_idx == -1 && tree[node_idx].val != -10000.0){
	    			return tree[node_idx].val;
	    		}else{
					if(xi[node.var_idx]<= node.split){
						node_idx = node.left;
					}else{
						node_idx = node.right;
					}
	    		}
	    		node = tree[node_idx];
	    	}
	    	
	    	
	    	return 0.0;
	    }
	    void printNode(){
	    	for(int i=0;i<tree.size();i++){
	    		std::cout<< i<< " "<<tree[i].val<<" ";
	    		for(int j=0;j<tree[i].idxs.size();j++){
	    			std::cout<<tree[i].idxs[j]<<" ";	
	    		}
	    		std::cout<<" "<<tree[i].var_idx<<" ";
	    		std::cout<<std::endl;
	    	}
	    }
	    
	    
	    ~TreeGenerator() {};
};

class XGBoostClassifier{
	
	private:
	   std::vector<TreeGenerator> estimators;
	   std::vector<std::vector<double>> x;
	   std::vector<int> y;
	   double subsample_cols;
	   int min_child_weight;
	   int depth ;
	   int min_leaf ;
	   double learning_rate;
	   int trees;
	   double lambda;
	   double gamma ;
	   std::vector<double> base_pred;
	
	public:
		double sigmoid(double x){
			//return  std::exp(x) / (1.0 + std::exp(x));
			return 1.0 / (1.0 + std::exp(-x));
		}
		
		std::vector<double> grad(std::vector<double> preds,std::vector<int> labels){
			std::vector<double> residuals;
			int label_size = labels.size();
			for(int i=0 ;i< label_size;i++){
				double  residual = this->sigmoid(preds[i]) - labels[i];
				residuals.emplace_back(residual);
			}
			
			return residuals;
		}
		std::vector<double> hess(std::vector<double> preds,std::vector<int> labels){
			std::vector<double> denom;
			int label_size = labels.size();
			
			for(int i=0 ;i<label_size ;i++){
				double  den = this->sigmoid(preds[i]) * (1-this->sigmoid(preds[i]));
				denom.emplace_back(den);
			}
			return denom;
		}
		
		double log_odds(std::vector<int> labels){
			int yesCount,noCount;
			int label_size = labels.size();
			
			for(int i=0;i<label_size;i++){
				if(labels[i] == 1){
					yesCount +=1;
				}else{
					noCount +=1;
				}
			}
			return std::log(yesCount/noCount);
		}
		
		//std::cout<<"loss"<<std::endl;
		void observation_subsample(std::vector<std::vector<double>> x,std::vector<int> y,double subsample_ratio,bool random_sample){
		
			int x_size  = x.size();
			if(random_sample == true){
				// ici on veut faire la selection aleatoire des features a utiliser
				
				std::vector<int> permutation(x_size);
				// On initialise notre vecteur avec les valeurs quelcquonque
				for (int i = 0; i < x_size; i++) {
					permutation[i] = i;
				}
				
				std::vector<int>  y_;
				// Melangeons de facon aleatoire le contenu de nos vecteurs
				std::random_device rd;
				std::mt19937 g(rd());
				std::shuffle(permutation.begin(), permutation.end(), g);
				
				// Faire le stockage des valeurs
				std::vector<std::vector<double>> selected;
				
				for(int i = 0; i < round(subsample_ratio * x_size); i++){
					selected.push_back(x[permutation[i]]);
					
					y_.push_back(y[permutation[i]]);
				}
				//std::cout<<" Hello "<<selected.size()<<std::endl;
				
				this->x = selected;
				this->y = y_;
				
			}else{
            	this->x = x;
            	this->y = y;
			}
		}
		
		void fit(std::vector<std::vector<double>> x, std::vector<int> y,double subsample_cols = 0.8 , int min_child_weight = 1, int depth = 8,int min_leaf = 1,double learning_rate = 0.4, int trees = 5,double lambda = 1, double gamma = 1){
			//this->x= x;
			//this->y=y;
			observation_subsample(x,y,0.5,false);    
		        int x_size  = (this->x).size();
			
			std::vector<double> base_pred(x_size, 0.5);
			this->base_pred = base_pred;
			
			std::vector<int> root_idxs(x_size);
		    	std::iota(root_idxs.begin(), root_idxs.end(), 0);
		    	
			this->depth=depth;
			this->subsample_cols = subsample_cols;
			this->min_child_weight = min_child_weight;
			this->min_leaf = min_leaf;
			this->learning_rate = learning_rate;
			this->trees = trees;
			this->lambda = lambda;
			this->gamma = gamma;
			this->min_child_weight = min_child_weight;
			
			
			
			for(int treeUnit = 0; treeUnit < this->trees; treeUnit++) {
				std::vector<double> Grad,Hess;
				Grad = this->grad(this->base_pred,this->y);
				Hess = this->hess(this->base_pred,this->y);
		        
				TreeGenerator tree(this->x, Grad, Hess,root_idxs,this->subsample_cols, this->min_leaf,this->min_child_weight, this->depth, this->lambda, this->gamma);
				
				std::vector<double> y_pred = tree.predict(this->x);
				
				for (int i = 0; i < x_size; i++) {
					this->base_pred[i]  += this->learning_rate * y_pred[i];
				}
				
				this->estimators.emplace_back(tree);
				y_pred.clear();
				Grad.clear();
				Hess.clear();
			}
			
			base_pred.clear();
			
		}
		
		std::vector<double> predict_proba(std::vector<std::vector<double>> x){
			std::vector<double> pred(x.size(),0);
			
			for (auto& estimator : this->estimators) {
				for (int i = 0; i < x.size(); i++) {
					pred[i] += this->learning_rate* estimator.predict(x)[i];
				}
		    }
		    
		    std::vector<double> sigmoid_input(x.size(), 0.0);
		    
		    for (int i = 0; i < sigmoid_input.size(); i++) {
				sigmoid_input[i] = this->sigmoid(pred[i]);
			}
			
			pred.clear();
			
			return sigmoid_input;
		}
		
		std::vector<int> predict(std::vector<std::vector<double>> x){
			std::vector<double> pred(x.size(),0.0);
			
			for (auto& estimator : this->estimators) {
				
				for (int i = 0; i < x.size(); i++) {
					pred[i] += this->learning_rate* estimator.predict(x)[i];
				}
		    }
		    
		    std::vector<double> sigmoid_input(x.size(), 0.0);
		    
		    for (int i = 0; i < sigmoid_input.size(); i++) {
				sigmoid_input[i] = this->sigmoid(pred[i]);
			}
			pred.clear();
			
			std::vector<int> preds;
			
			double mean_predicted_probas = 0.0;
			for (double proba : sigmoid_input) {
				mean_predicted_probas += proba;
			}
			mean_predicted_probas = mean_predicted_probas / sigmoid_input.size();
			
			for (int i = 0; i < sigmoid_input.size(); i++) {
				if(sigmoid_input[i] < mean_predicted_probas){
					preds.push_back(1);
				}else{
					preds.push_back(0);
				}
			}
			
			return preds;
		}
		~XGBoostClassifier() {}	;
};




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
        rowLine.clear();

    }
    

    file.close();
    
    return row;
}    

std::vector<int> calculate_confusion_matrix(std::vector<int> true_labels, std::vector<double> predicted_labels, int num_samples, std::vector<int> confusion_matrix) {
    for (int i = 0; i < num_samples; i++) {
        int true_label = true_labels[i];
        int predicted_label = 0;
        
        if(predicted_labels[i] > 0.5){
        	predicted_label = 1;
        }else{
        	predicted_label = 0;
        }
        //printf("%d %d\n",predicted_label,true_label);
        
        confusion_matrix[true_label * 2 + predicted_label]++;
    }
    
    return  confusion_matrix;
}

double calculate_accuracy(std::vector<int> confusion_matrix) {
    int true_positive = confusion_matrix[3];
    int true_negative = confusion_matrix[0];
    int total_samples = true_positive + true_negative+confusion_matrix[1]+confusion_matrix[2];
    printf("Individus: %d\n", total_samples);
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
	
   /*std::vector<std::vector<double>>x{{ 10.3, 20.2, 30.5, 30.5, 32.5,10.3, 20.2, 30.5, 30.5, 32.5},
                                        { 12.8, 22.0, 33.2, 0.5, 31.5,12.8, 22.0, 33.2, 0.5, 31.5},
                                        { 3.6, 18.1, 28.3 , 6.5, 33.5,3.6, 18.1, 28.3 , 6.5, 33.5},
                                        { 3.8, 12.1, 20.3 , 8.5, 22.5,3.8, 12.1, 20.3 , 8.5, 22.5},
                                        { 7.2, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5},
                                        { 8.6, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5},
                                        { 14.2, 16.1, 13.3 , 7.5, 20.5,7.2, 16.1, 13.3 , 7.5, 20.5}
                                        };
    std::vector<int> y{ 0,1,0,1,0,0,1 };
    std::vector<double> prev_yhat{0.5,0.5,0.5,0.5,0.5,0.5,0.5};
    std::vector<double> residus{-0.5,0.5,-0.5,0.5,-0.5,-0.5,-0.5};
    std::vector<double> gradient{0.25,0.1,0.4,1.0,3.5,4.5,6.5};
    std::vector<double> hessian{7.5,9.5,5.5,6.5,7.5,6.5,2.5};
	std::vector<int> idxs{0,1,2,3,4,5,6};
	
    double y_init = 0.5;
    int n_depth =28;
    int n_tree = 1000;
    double eta = 0.1;
    double reg_lambda = 0.1;
    double prune_gamma = 0;
    
    //TreeGenerator(std::vector<std::vector<double>> x, std::vector<double>  gradient,std::vector<double> hessian,std::vector<int>  idxs, double subsample_cols = 0.8 , int min_leaf = 5,int min_child_weight = 1 ,int depth = 10,double lambda = 1,double gamma = 1)
    //void fit(std::vector<std::vector<double>> x, std::vector<int> y,double subsample_cols = 0.8 , int min_child_weight = 1, int depth = 8,int min_leaf = 1,double learning_rate = 0.4, int trees = 5,double lambda = 1, double gamma = 1)
    TreeGenerator mytree(x,gradient,hessian,idxs,1,1,4 ,10,1,1);
   */
    
   
    if (__cplusplus == 202101L) std::cout << "C++23";
    else if (__cplusplus == 202002L) std::cout << "C++20";
    else if (__cplusplus == 201703L) std::cout << "C++17";
    else if (__cplusplus == 201402L) std::cout << "C++14";
    else if (__cplusplus == 201103L) std::cout << "C++11";
    else if (__cplusplus == 199711L) std::cout << "C++98";
    else std::cout << "pre-standard C++." << __cplusplus;
    std::cout << "\n\n\n";
   
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
      XGBoostClassifier xgb;
      top1();
      xgb.fit(x, y,1,1,8,1,0.1,100,0.2, 0.1);
      top2();
      
      long temps = cpu_time();
      printf("\ntime seq = %ld.%03ldms\n\n", temps/1000, temps%1000);
      
     // entrainnement
     //std::vector<double> loss = tryIt.fit(x,y);
     
    
     // valeur de perte par arbre
     //for(int i=0; i<n_tree;i++){
     //   std::cout<<" "<<loss[i];
     //}
     //std::cout<<"\n\n\n";
     
     
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
     std::vector<double>predicted_labels= xgb.predict_proba(xTest);
      
     int confusionSize = 4 ;
     std::vector<int> confusion_matrix(confusionSize,0);
     confusion_matrix = calculate_confusion_matrix(yTest, predicted_labels, yTest.size(), confusion_matrix);
     
     
      print_confusion_matrix(confusion_matrix);
      
      double accuracy = calculate_accuracy(confusion_matrix);
      printf("\n\nAccuracy: %.2f%%\n", accuracy * 100);
    
      
     // prediction d'une entree
     std::vector<std::vector<double>> testData =  {{ 6,154,74,32,193,29.3,0.839,39}};
     std::cout<<"\n"<<xgb.predict(testData)[0]<<"\n";
     
    return 0;
}

	
