#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <memory>
#include <stack>
#include <random>
#include <thread>
#include <mutex>
#include <sys/time.h>
#include "xgb.h"
std::mutex mtx;
std::mutex mtx1;

static struct timeval _t1, _t2;
static struct timeval _t3, _t4;
static struct timezone _tz;
static struct timezone _tz1;

#define top1() gettimeofday(&_t1, &_tz)
#define top2() gettimeofday(&_t2, &_tz)

#define top3() gettimeofday(&_t3, &_tz1)
#define top4() gettimeofday(&_t4, &_tz1)

static unsigned long _temps_residuel = 0;
static unsigned long _temps_residuel2 = 0;
void init_cpu_time(void)
{
   top1(); top2();
   _temps_residuel = 1000000L * _t2.tv_sec + _t2.tv_usec -
                     (1000000L * _t1.tv_sec + _t1.tv_usec );
}

void init_cpu_time_2(void)
{
   top3(); top4();
   _temps_residuel2 = 1000000L * _t4.tv_sec + _t4.tv_usec -
                     (1000000L * _t3.tv_sec + _t3.tv_usec );
}


 long cpu_time(void) /* retourne des microsecondes */
{
   return 1000000L * _t2.tv_sec + _t2.tv_usec -
           (1000000L * _t1.tv_sec + _t1.tv_usec- _temps_residuel );
}
 long cpu_time_2(void) /* retourne des microsecondes */
{
   return 1000000L * _t4.tv_sec + _t4.tv_usec -
           (1000000L * _t3.tv_sec + _t3.tv_usec - _temps_residuel2 );
}

// structure pour parallelisation
typedef struct{
	int node_idx;
	int thread_id;
	int num_threads;
	int xsplit_size;
	int xfeat_val_size;
	int feature;
	std::vector<double> hessianS;
	std::vector<double> xsplit;
	std::vector<double> xfeat_val;
	TreeGenerator* treeGen;

}split_data;


//constructeur de ma classe
Node::Node(std::vector<int>&  idxs,int depth = 0,double val=0.0,double score =0.0,int var_idx =-1,double split=0.0,int left=-1,int right = -1) {

	this->idxs = idxs;
	this->depth = depth;
	this->val = val;
	this->score = score;
	this->var_idx = var_idx;
	this->split = split;
	this->left = left;
	this->right = right;

}

TreeGenerator::TreeGenerator(std::vector<std::vector<double>>& x, std::vector<double>&  gradient,std::vector<double>& hessian,std::vector<int>& idxs, double subsample_cols = 0.8 , int min_leaf = 1,int min_child_weight = 1 ,int depth = 4,double lambda = 1,double gamma = 1,double eps=0.5,int num_threads=4,int choice=0) {
	
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
	this->num_threads = num_threads;
	this->choice = choice;
	
	if(this->subsample_cols < 1.0){
		//std::cout<<"Hello"<<std::endl;
		this->column_subsample = col_subsample(true);		
	}else{
		this->column_subsample = col_subsample(false);
	}
	
	this->eps=eps;
	
	/*if(this->choice == 0){
	   this->find_varsplit_seq();
	}else if(this->choice == 1){
	   this->find_varsplit_par();
	}else if(this->choice == 2){
	   this->find_varsplit_par_feat();
	}else{
		this->find_varsplit_par_node();
	}*/

	switch (choice) {
        case 0: this->find_varsplit_seq(); break;
        case 1: this->find_varsplit_par(); break;
        case 2: this->find_varsplit_par_feat(); break;
		case 3: this->find_varsplit_par_feat_plus_SplitPoint(); break;
		case 4: this->find_varsplit_seq_approx_quantile_sketch(); break;
		case 5: this->find_varsplit_greedy_approximation(); break;
		case 6: this->find_varsplit_seq_with_na(); break;
        default: this->find_varsplit_par_node(); break;
    }

	//this->printNode();
	
}

std::vector<int> TreeGenerator::col_subsample(bool random_col){
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
		//std::cout<<selected.size()<<std::endl;
		return selected;
	}else{
		std::vector<int> root_did(this->col_count);
	    std::iota(root_did.begin(), root_did.end(), 0);
	    return root_did;
	}
}

double TreeGenerator::compute_node_output( std::vector<double>& gradient, std::vector<double>& hessian){
	double G=0.0;
	double H=0.0;
	
	int gradient_size = gradient.size();
	for (int i = 0; i < gradient_size; i++) {
		G += gradient[i];
		H += hessian[i];
	}

	return (-G/(H + this->lambda));
}

void TreeGenerator::find_greedy_split_par(int node_idx) {
    std::vector<double> gradientS;
    std::vector<double> hessianS;

    std::vector<int> lhsF;
    std::vector<int> rhsF;

    // Collect gradients and hessians for the current node
    for (int i : tree[node_idx].idxs) {
        gradientS.emplace_back(this->gradient[i]);
        hessianS.emplace_back(this->hessian[i]);
    }

    int node_tree_size = tree[node_idx].idxs.size();
    tree[node_idx].val = this->compute_node_output(gradientS, hessianS);

    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        return;
    }

   int num_threads =this->num_threads;  // Get the number of available cores
    std::vector<std::thread> threads(num_threads);
    
    // Lambda function to process a subset of split points
    auto process_splitPoints = [&](int thread_id) {
        for (int c : this->column_subsample) {
            std::vector<double> xsplit_1;
			std::vector<double> xsplit_u_2_2;

            for (int idx : tree[node_idx].idxs) {
                xsplit_1.emplace_back(this->x[idx][c]);
            }
            int xsplit_size = xsplit_1.size();

            std::vector<double> xsplit;
            xsplit.emplace_back(xsplit_1[0]);

            for (int i = 1; i < xsplit_size; i++) {
                if (std::find(xsplit.begin(), xsplit.end(), xsplit_1[i]) == xsplit.end()) {
                    xsplit.emplace_back(xsplit_1[i]);
                }
            }

            std::sort(xsplit.begin(), xsplit.end());
			for(int i=1;i<(int)xsplit.size();i++){
				xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
			}
			xsplit = xsplit_u_2_2;

            for (int r = thread_id; r < (int)xsplit.size(); r += num_threads) {
                std::vector<bool> lhs(xsplit_size, false);
                std::vector<bool> rhs(xsplit_size, false);
                int lhs_sum = 0;
                int rhs_sum = 0;
                std::vector<int> lhs_indices, rhs_indices;
                double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;

                for (int i = 0; i < xsplit_size; i++) {
                    if (xsplit_1[i] <= xsplit[r]) {
                        lhs[i] = true;
                        lhs_sum++;
                        lhs_indices.emplace_back(tree[node_idx].idxs[i]);
                        lhs_hessian_sum += hessianS[i];
                    } else {
                        rhs[i] = true;
                        rhs_sum++;
                        rhs_indices.emplace_back(tree[node_idx].idxs[i]);
                        rhs_hessian_sum += hessianS[i];
                    }
                }

                double curr_score = this->gain(lhs, rhs, node_idx);

                std::lock_guard<std::mutex> lock(mtx);
                if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0) {
                    tree[node_idx].var_idx = c;
                    tree[node_idx].score = curr_score;
                    tree[node_idx].split = xsplit[r];
                    lhsF = lhs_indices;
                    rhsF = rhs_indices;
                }
            }
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_splitPoints, i);
    }

    // Join threads
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

	//threads.clear();

    // Add left and right children if the best split is found
    if (!lhsF.empty() && !rhsF.empty()) {
        int left_child_idx = tree.size();
        tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        int right_child_idx = tree.size();
        tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;

        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    }
}

void TreeGenerator::find_varsplit_par(){

    tree.emplace_back(Node(this->idxs, 0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    do{
    	// ici on continu l'entrainnement
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_par(node_idx);
    	
    }while(!node_stack.empty());
}




void TreeGenerator::find_greedy_split_par_feat(int node_idx) {    
    std::vector<double> gradientS;
    std::vector<double> hessianS;

    std::vector<int> lhsF;
    std::vector<int> rhsF;

    // Collect gradients and hessians for the current node
    for (int i : tree[node_idx].idxs) {
        gradientS.emplace_back(this->gradient[i]);
        hessianS.emplace_back(this->hessian[i]);
    }

    int node_tree_size = tree[node_idx].idxs.size();
    tree[node_idx].val = this->compute_node_output(gradientS, hessianS);

    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        return;
    }

	int num_threads = this->num_threads;
    std::vector<std::thread> threads(num_threads);

    auto process_column = [&](int thread_id) {
		 for (int i = thread_id; i < (int)this->column_subsample.size(); i += num_threads) {
			int c = this->column_subsample[i];
			std::vector<double> xsplit_1;
			std::vector<double> xsplit_u_2_2;

			for (int idx : tree[node_idx].idxs) {
				xsplit_1.emplace_back(this->x[idx][c]);
			}
			int xsplit_size = xsplit_1.size();

			std::vector<double> xsplit;
			xsplit.emplace_back(xsplit_1[0]);

			for (int i = 1; i < xsplit_size; i++) {
                if (std::find(xsplit.begin(), xsplit.end(), xsplit_1[i]) == xsplit.end()) {
                    xsplit.emplace_back(xsplit_1[i]);
                }
            }

			std::sort(xsplit.begin(), xsplit.end());
			for(int i=1;i<(int)xsplit.size();i++){
				xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
			}
			xsplit = xsplit_u_2_2;

			for (double split_val : xsplit) {
				std::vector<bool> lhs(xsplit_size, false);
				std::vector<bool> rhs(xsplit_size, false);
				int lhs_sum = 0;
				int rhs_sum = 0;
				std::vector<int> lhs_indices, rhs_indices;
				double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;

				for (int i = 0; i < xsplit_size; i++) {
					if (xsplit_1[i] <= split_val) {
						lhs[i] = true;
						lhs_sum++;
						lhs_indices.emplace_back(tree[node_idx].idxs[i]);
						lhs_hessian_sum += hessianS[i];
					} else {
						rhs[i] = true;
						rhs_sum++;
						rhs_indices.emplace_back(tree[node_idx].idxs[i]);
						rhs_hessian_sum += hessianS[i];
					}
				}

				double curr_score = this->gain(lhs, rhs, node_idx);
				// Protect shared variables with mutex
				std::lock_guard<std::mutex> lock(mtx);
				if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0) {
					tree[node_idx].var_idx = c;
					tree[node_idx].score = curr_score;
					tree[node_idx].split = split_val;
					lhsF = lhs_indices;
					rhsF = rhs_indices;
				}
			}
		 }
    };

    // Launch threads using modulo parallelization
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_column, i);
    }

    // Join threads
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    // Add left and right children if the best split is found
    if (!lhsF.empty() && !rhsF.empty()) {
        int left_child_idx = tree.size();
        tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        int right_child_idx = tree.size();
        tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;

        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    }
}

void TreeGenerator::find_varsplit_par_feat(){

    tree.emplace_back(Node(this->idxs,  0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    
    do{
    	// ici on continu l'entrainnement
    	
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_par_feat(node_idx);
    	
    }while(!node_stack.empty());
}


// Function to find the best split for a node, parallelizing over both features and split points simultaneously
void TreeGenerator::find_greedy_split_par_feat_plus_SplitPoint(int node_idx) {
    std::vector<double> gradientS;
    std::vector<double> hessianS;

    std::vector<int> lhsF;
    std::vector<int> rhsF;

	std::mutex inner_mtx;
    // Collect gradients and hessians for the current node
    for (int i : tree[node_idx].idxs) {
        gradientS.emplace_back(this->gradient[i]);
        hessianS.emplace_back(this->hessian[i]);
    }

    int node_tree_size = tree[node_idx].idxs.size();
    tree[node_idx].val = this->compute_node_output(gradientS, hessianS);

    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        return;
    }

    int num_threads = this->num_threads;
    std::vector<std::thread> threads(num_threads);

    auto process_column_and_split = [&](int thread_id) {
        for (int i = thread_id; i < (int)this->column_subsample.size(); i += num_threads) {
            int c = this->column_subsample[i];
            std::vector<double> xsplit_1;

            for (int idx : tree[node_idx].idxs) {
                xsplit_1.emplace_back(this->x[idx][c]);
            }
            int xsplit_size = xsplit_1.size();

            std::vector<double> xsplit;
            xsplit.emplace_back(xsplit_1[0]);

            for (int i = 1; i < xsplit_size; i++) {
                if (std::find(xsplit.begin(), xsplit.end(), xsplit_1[i]) == xsplit.end()) {
                    xsplit.emplace_back(xsplit_1[i]);
                }
            }

            std::sort(xsplit.begin(), xsplit.end());

            std::vector<double> xsplit_u_2_2;
            for(int i = 1; i < (int)xsplit.size(); i++) {
                xsplit_u_2_2.emplace_back((xsplit[i-1] + xsplit[i]) / 2.0);
            }
            xsplit = xsplit_u_2_2;

            // Inner parallelization over split points
            auto process_split_point = [&](int split_thread_id){
                for (int r = split_thread_id; r < (int)xsplit.size(); r += num_threads) {
                    std::vector<bool> lhs(xsplit_size, false);
                    std::vector<bool> rhs(xsplit_size, false);
                    std::vector<int> lhs_indices, rhs_indices;
                    double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;

                    for (int i = 0; i < xsplit_size; i++) {
                        if (xsplit_1[i] <= xsplit[r]) {
                            lhs[i] = true;
                            lhs_indices.emplace_back(tree[node_idx].idxs[i]);
                            lhs_hessian_sum += hessianS[i];
                        } else {
                            rhs[i] = true;
                            rhs_indices.emplace_back(tree[node_idx].idxs[i]);
                            rhs_hessian_sum += hessianS[i];
                        }
                    }

                    double curr_score = this->gain(lhs, rhs, node_idx);

                    // Protect shared variables with mutex
                    std::lock_guard<std::mutex> lock(inner_mtx);
                    if (curr_score > tree[node_idx].score && !lhs_indices.empty() && !rhs_indices.empty()) {
                        tree[node_idx].var_idx = c;
                        tree[node_idx].score = curr_score;
                        tree[node_idx].split = xsplit[r];
                        lhsF = lhs_indices;
                        rhsF = rhs_indices;
                    }
                }
            };
			
            // Launch inner threads for split points
			std::vector<std::thread> inner_threads(num_threads);
            for (int j = 0; j < num_threads; ++j) {
                inner_threads[j] = std::thread(process_split_point, j);
            }

            // Join inner threads
            for (auto &t : inner_threads) {
                if (t.joinable()) {
                    t.join();
                }
            }
        }
    };

    // Launch outer threads for feature processing
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_column_and_split, i);
    }

    // Join outer threads
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Add left and right children if the best split is found
    if (!lhsF.empty() && !rhsF.empty()) {
        int left_child_idx = tree.size();
        tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        int right_child_idx = tree.size();
        tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;

        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    }
}

// Function to build the tree using parallelized feature and split point search
void TreeGenerator::find_varsplit_par_feat_plus_SplitPoint(){
    tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
    node_stack.push(0);
    
    do{
        int node_idx = node_stack.top();
        node_stack.pop();
        this->find_greedy_split_par_feat_plus_SplitPoint(node_idx);
    }while(!node_stack.empty());
}



void TreeGenerator::find_greedy_split_seq_with_na(int node_idx){
	std::vector<double> gradientS;
    std::vector<double> hessianS;

    std::vector<int> lhsF;
    std::vector<int> rhsF;

    // Collect gradients and hessians for the current node
    for (int i : tree[node_idx].idxs) {
        gradientS.emplace_back(this->gradient[i]);
        hessianS.emplace_back(this->hessian[i]);
    }

    int node_tree_size = tree[node_idx].idxs.size();
	tree[node_idx].val = this->compute_node_output(gradientS, hessianS);
	
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        return;
    }

	for (int c : this->column_subsample){
		std::vector<double> xsplit_1;
		
		// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
		for (int idx : tree[node_idx].idxs){
			xsplit_1.emplace_back(this->x[idx][c]);
		}
		int xsplit_size = xsplit_1.size();
		if (xsplit_size == 0) continue;

		std::vector<double> xsplit;
		std::vector<double> xsplit_u_2_2;
		xsplit.emplace_back(xsplit_1[0]);

		for(int i=1;i<xsplit_size;i++){
			bool myBool = false;
			for(int j=0;j<(int)xsplit.size();j++){
				if(xsplit[j] == xsplit_1[i]){
					myBool = true;
					break;
				}
			}
			// l'on evite de récupérer les valeurs Nan dans le vecteur qui contiendra les points de subdivision
			if(myBool == false && xsplit_1[i] != -0.00000000000000001){
				xsplit.emplace_back(xsplit_1[i]);
			}	
		}

		std::sort(xsplit.begin(), xsplit.end());

		for(int i=1;i<(int)xsplit.size();i++){
			xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
		}
		xsplit = xsplit_u_2_2;
		
		for(int r = 0; r < (int)xsplit.size(); r++){
			std::vector<bool> lhs(xsplit_size, false);
			std::vector<bool> rhs(xsplit_size, false);
			int lhs_sum=0;
			int rhs_sum=0;
			std::vector<int> lhs_indices, rhs_indices;
			double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
			
			// L'on met les differents individus a valeur Nan au niveau du fils gauche  pour en tirer les meilleurs valeur
			for (int i = 0; i < xsplit_size; i++){
				if (xsplit_1[i] <= xsplit[r] || xsplit_1[i] == -0.00000000000000001) {
					lhs[i] = true;
					lhs_sum++;
					lhs_indices.emplace_back(tree[node_idx].idxs[i]);
					lhs_hessian_sum += hessianS[i];
				}else{
					rhs[i] = true;
					rhs_sum++;
					rhs_indices.emplace_back(tree[node_idx].idxs[i]);
					rhs_hessian_sum += hessianS[i];
				}
				
			}		
			
			/*if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum <= this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
				continue;
			}*/
			
			
			double  curr_score = this->gain(lhs,rhs,node_idx);
			
			if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0){
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

			// L'on met les differents individus à valeur Nan au niveau du fils droit pour en tirer les meilleurs valeur de subdivision
			lhs_hessian_sum = 0.0;
			rhs_hessian_sum = 0.0;
			lhs_sum=0;
			rhs_sum=0;

			for (int i = 0; i < xsplit_size; i++){
				if (xsplit_1[i] > xsplit[r] || xsplit_1[i] == -0.00000000000000001) {
					rhs[i] = true;
					rhs_sum++;
					rhs_indices.emplace_back(tree[node_idx].idxs[i]);
					rhs_hessian_sum += hessianS[i];
				}else{
					lhs[i] = true;
					lhs_sum++;
					lhs_indices.emplace_back(tree[node_idx].idxs[i]);
					lhs_hessian_sum += hessianS[i];
				}
				
			}
			
			
			
			/*if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum <= this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
				continue;
			}*/
			
			
			curr_score = this->gain(lhs,rhs,node_idx);
			
			if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0){
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
	
	if(lhsF.size() != 0 && rhsF.size() != 0){
		//ajout des noeuds gauche et droit dans notre arbre
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		
		//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
		
		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	}

}

void TreeGenerator::find_varsplit_seq_with_na(){

    tree.emplace_back(Node(this->idxs, 0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    do{
    	// ici on continu l'entrainnement
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_seq_with_na(node_idx);
    	
    }while(!node_stack.empty());
}


std::mutex tree_mtx;  // Protects tree
std::mutex stack_mtx;  // Protects node_stack


void TreeGenerator::find_greedy_split_par_node(int node_idx){    
	std::vector<double> gradientS;
    std::vector<double> hessianS;

    std::vector<int> lhsF;
    std::vector<int> rhsF;

    // Collect gradients and hessians for the current node
    for (int i : tree[node_idx].idxs) {
        gradientS.emplace_back(this->gradient[i]);
        hessianS.emplace_back(this->hessian[i]);
    }

    int node_tree_size = tree[node_idx].idxs.size();
	tree[node_idx].val = this->compute_node_output(gradientS, hessianS);
	
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        return;
    }

    for (int c : this->column_subsample) {
        std::vector<double> xsplit_1;
		std::vector<double> xsplit_u_2_2;
		
        for (int idx : tree[node_idx].idxs) {
        	//if (idx < 0 || idx >= this->x.size()) continue;
            xsplit_1.emplace_back(this->x[idx][c]);
        }
        int xsplit_size = xsplit_1.size();
		if (xsplit_size == 0) continue;
		
        std::vector<double> xsplit;
        xsplit.emplace_back(xsplit_1[0]);

        for (int i = 1; i < xsplit_size; i++) {
            if(std::find(xsplit.begin(), xsplit.end(), xsplit_1[i]) == xsplit.end()) {
                xsplit.emplace_back(xsplit_1[i]);
            }
        }

        std::sort(xsplit.begin(), xsplit.end());
		for(int i=1;i<(int)xsplit.size();i++){
			xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
		}
		xsplit = xsplit_u_2_2;

        for (double split_val : xsplit) {
            std::vector<bool> lhs(xsplit_size, false);
            std::vector<bool> rhs(xsplit_size, false);
            int lhs_sum = 0;
            int rhs_sum = 0;
            std::vector<int> lhs_indices, rhs_indices;
            double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;

            for (int i = 0; i < xsplit_size; i++){
                if (xsplit_1[i] <= split_val){
                    lhs[i] = true;
                    lhs_sum++;
                    lhs_indices.emplace_back(tree[node_idx].idxs[i]);
                    lhs_hessian_sum += hessianS[i];
                } else {
                    rhs[i] = true;
                    rhs_sum++;
                    rhs_indices.emplace_back(tree[node_idx].idxs[i]);
                    rhs_hessian_sum += hessianS[i];
                }
            }

            double curr_score = this->gain(lhs, rhs, node_idx);

            // Protect shared variables with mutex
            std::lock_guard<std::mutex> tree_lock(tree_mtx);
            if (curr_score > tree[node_idx].score && !lhs_indices.empty() && !rhs_indices.empty()) {
                tree[node_idx].var_idx = c;
                tree[node_idx].score = curr_score;
                tree[node_idx].split = split_val;
                lhsF = lhs_indices;
                rhsF = rhs_indices;
            }
        }
    }

    // Add left and right children if the best split is found
    
    //std::lock_guard<std::mutex> lock(mtx);
	if (!lhsF.empty() && !rhsF.empty()) {
		std::lock_guard<std::mutex> tree_lock(tree_mtx);	
		int left_child_idx = tree.size();
        tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        
		int right_child_idx = tree.size();
        tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
 		
		
		tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;
		
       
        std::lock_guard<std::mutex> stack_lock(stack_mtx);
		node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    }
    
}


void TreeGenerator::find_varsplit_par_node(){
    tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
    node_stack.push(0);

    //int num_threads = this->num_threads;
	std::cout<<"num thread av "<<num_threads<<std::endl;

    int k=0;
   	while(!node_stack.empty()){
		std::vector<std::thread> threads(num_threads);
    	std::vector<int> tabPresentNode;
    	int m=0;
    	
    	{
            std::lock_guard<std::mutex> stack_lock(stack_mtx);
            while (!node_stack.empty()) {
                int node_idx = node_stack.top();
                node_stack.pop();
                if (tree[node_idx].depth == k) {
                    tabPresentNode.push_back(node_idx);
                    std::cout << tabPresentNode[m] << " ";
                    m++;
                }
            }
        }
        
        auto worker = [&](int thread_id){
        	for(int i= thread_id;i<(int)tabPresentNode.size();i+= num_threads){
        		//std::cout<<"num thread av "<<num_threads<<" thread id "<<thread_id<<" i "<<i<<std::endl;
				this->find_greedy_split_par_node(tabPresentNode[i]);	
        	}
        };
        
        for (int r = 0; r < num_threads; r++){
		    threads[r] = std::thread(worker, r);
		}

		for(auto &t : threads){
		    if (t.joinable()){
		        t.join();
		    }
		}
		
		std::cout<<"\n"<<tabPresentNode.size()<<"\n\n";
		k = k+1;
		
		if(node_stack.empty() || threads.empty() || tabPresentNode.empty()){
        	break;
        }
        tabPresentNode.clear();
    };
}

// Split Sequentiel
void TreeGenerator::find_greedy_split_seq(int node_idx){	
	std::vector<double> gradientS;
	std::vector<double>  hessianS;
	
	std::vector<int> lhsF;
	std::vector<int>  rhsF;
	
	for(int i : tree[node_idx].idxs){
		gradientS.emplace_back(this->gradient[i]);
		hessianS.emplace_back(this->hessian[i]);
	}
	
	
	int node_tree_size = tree[node_idx].idxs.size();
	tree[node_idx].val = this->compute_node_output(gradientS,hessianS);

	if(node_tree_size <= 1 || tree[node_idx].depth >= this->depth){
		return;
	}

	for (int c : this->column_subsample){
		std::vector<double> xsplit_1;
		
		// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
		for (int idx : tree[node_idx].idxs){
			xsplit_1.emplace_back(this->x[idx][c]);
		}
		int xsplit_size = xsplit_1.size();
		//if (xsplit_size == 0) continue;

		std::vector<double> xsplit;
		std::vector<double> xsplit_u_2_2;
		xsplit.emplace_back(xsplit_1[0]);

		for(int i=1;i<xsplit_size;i++){
			bool myBool = false;
			for(int j=0;j<(int)xsplit.size();j++){
				if(xsplit[j] == xsplit_1[i]){
					myBool = true;
					break;
				}
			}
			if(myBool == false){
				xsplit.emplace_back(xsplit_1[i]);
			}	
		}

		std::sort(xsplit.begin(), xsplit.end());

		for(int i=1;i<(int)xsplit.size();i++){
			xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
		}
		xsplit = xsplit_u_2_2;
		
		for(int r = 0; r < (int)xsplit.size(); r++){
			std::vector<bool> lhs(xsplit_size, false);
			std::vector<bool> rhs(xsplit_size, false);
			int lhs_sum=0;
			int rhs_sum=0;
			std::vector<int> lhs_indices, rhs_indices;
			double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
			
			for (int i = 0; i < xsplit_size; i++){
				if (xsplit_1[i] <= xsplit[r]){
					lhs[i] = true;
					lhs_sum++;
					lhs_indices.emplace_back(tree[node_idx].idxs[i]);
					lhs_hessian_sum += hessianS[i];
				}else{
					rhs[i] = true;
					rhs_sum++;
					rhs_indices.emplace_back(tree[node_idx].idxs[i]);
					rhs_hessian_sum += hessianS[i];
				}
				
			}
			
			
			
			/*if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum <= this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
				continue;
			}*/
			
			
			double  curr_score = this->gain(lhs,rhs,node_idx);
			
			if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0){
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
	
	if(lhsF.size() != 0 && rhsF.size() != 0){
		//ajout des noeuds gauche et droit dans notre arbre
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		
		//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
		
		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	}
}

void TreeGenerator::find_varsplit_seq(){

    tree.emplace_back(Node(this->idxs, 0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    do{
    	// ici on continu l'entrainnement
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_seq(node_idx);
    	
    }while(!node_stack.empty());
}



// Split Approximatif
void TreeGenerator::find_greedy_split_greedy_approximation(int node_idx){	
	std::vector<double> gradientS;
	std::vector<double>  hessianS;
	
	std::vector<int> lhsF;
	std::vector<int>  rhsF;
	
	for(int i : tree[node_idx].idxs){
		gradientS.emplace_back(this->gradient[i]);
		hessianS.emplace_back(this->hessian[i]);
	}
	
	
	int node_tree_size = tree[node_idx].idxs.size();
	tree[node_idx].val = this->compute_node_output(gradientS,hessianS);

	if(node_tree_size <= 1 || tree[node_idx].depth >= this->depth){
		return;
	}
	
	for (int c : this->column_subsample){
		std::vector<double> xsplit_1;
		
		// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
		for (int idx : tree[node_idx].idxs){
			xsplit_1.emplace_back(this->x[idx][c]);
		}
		int xsplit_size = xsplit_1.size();
		std::sort(xsplit_1.begin(), xsplit_1.end());
		int num_blocks = std::round(1/(this->eps));
		std::vector<std::vector<double>> xsplit_blocks(num_blocks);
		
		for(int t=0;t<num_blocks;t++){
			std::vector<double> xsplit_temp;
			for(int p=t*std::round((xsplit_size)/num_blocks);p<std::round((xsplit_size)/num_blocks)*(t+1);p++){
				xsplit_temp.emplace_back(xsplit_1[p]);
				std::cout<<xsplit_1[p]<<" ";
			}
			if (xsplit_temp.size() == 0) continue;
			std::cout<<std::endl;
			xsplit_blocks[t] = xsplit_temp;
			std::cout<<xsplit_blocks[t].size()<<std::endl;
			xsplit_temp.clear();
		}
		
		if (xsplit_blocks[0].size() == 0) continue;

		for(int t=0;t<num_blocks;t++){
			std::vector<double> xsplit;
			std::vector<double> xsplit_u_2_2;
			xsplit.emplace_back(xsplit_blocks[t][0]);
			
			for(int i=1;i<(int)xsplit_blocks[t].size();i++){
				bool myBool = false;
				
				for(int j=0;j<(int)xsplit.size();j++){
					if(xsplit[j] == xsplit_blocks[t][i]){
						myBool = true;
						break;
					}
				}
				if(myBool == false){
					xsplit.emplace_back(xsplit_blocks[t][i]);
				}	
			}
			
			for(int i=1;i<(int)xsplit.size();i++){
				xsplit_u_2_2.emplace_back((xsplit[i-1]+xsplit[i])/2.0);
			}
		    /*xsplit_u_2_2;*/
			if (xsplit_u_2_2.size() == 0) continue;
			for(int r = 0; r < (int)xsplit_u_2_2.size(); r++){
				std::vector<bool> lhs(xsplit_size, false);
				std::vector<bool> rhs(xsplit_size, false);
				int lhs_sum=0;
				int rhs_sum=0;
				std::vector<int> lhs_indices, rhs_indices;
				double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
				
				for (int i = 0; i < (int)xsplit_blocks[t].size(); i++){
					if(xsplit_blocks[t][i] < xsplit_u_2_2[r]) {
						lhs[i] = true;
						lhs_sum++;
						lhs_indices.emplace_back(tree[node_idx].idxs[i]);
						lhs_hessian_sum += hessianS[i];
					}else{
						rhs[i] = true;
						rhs_sum++;
						rhs_indices.emplace_back(tree[node_idx].idxs[i]);
						rhs_hessian_sum += hessianS[i];
					}
					
				}
				
				/*if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum <= this->min_child_weight || lhs_hessian_sum < this->min_child_weight) {
					continue;
				}*/
				
				double  curr_score = this->gain(lhs,rhs,node_idx);
				
				if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0){
					tree[node_idx].var_idx = c;
					tree[node_idx].score  = curr_score;
					tree[node_idx].split = xsplit_u_2_2[r];
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
			xsplit_u_2_2.clear();
		}	
	}
	
	if(lhsF.size() != 0 && rhsF.size() != 0){
		//ajout des noeuds gauche et droit dans notre arbre
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		
		//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
		
		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	}
}

void TreeGenerator::find_varsplit_greedy_approximation(){
    tree.emplace_back(Node(this->idxs, 0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    do{
    	// ici on continu l'entrainnement
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_greedy_approximation(node_idx);
    	
    }while(!node_stack.empty());
}


// split Weighted quantile sketch
void TreeGenerator::find_greedy_split_seq_approx_quantile_sketch(int node_idx){	
	std::vector<double> gradientS;
	std::vector<double>  hessianS;
	
	std::vector<int> lhsF;
	std::vector<int>  rhsF;
	
	for(int i : tree[node_idx].idxs){
		gradientS.emplace_back(this->gradient[i]);
		hessianS.emplace_back(this->hessian[i]);
	}
	
	double hessian_sum_all = 0.0;

	for(int i=0;i<(int)hessianS.size();i++){
		hessian_sum_all += hessianS[i];
	}

	int node_tree_size = tree[node_idx].idxs.size();
	tree[node_idx].val = this->compute_node_output(gradientS,hessianS);

	if(node_tree_size <= 1 || tree[node_idx].depth >= this->depth){
		return;
	}

	for(int c : this->column_subsample){
		std::vector<double> xsplit_1;
		
		// on selectionne les valeurs d'une caracteristiques donnee suivant un certain nombre d'indice
		for (int idx : tree[node_idx].idxs){
			xsplit_1.emplace_back(this->x[idx][c]);
		}
		size_t xsplit1_size = xsplit_1.size();
		//std::vector<double> xsplit;
		std::sort(xsplit_1.begin(), xsplit_1.end());
		std::vector<double> ranks(xsplit1_size);
		
		for(size_t i = 0; i < xsplit1_size; ++i) {
			double rank_sum = 0.0;
			for(size_t j = 0; j < i; ++j) {
				if (xsplit_1[j] <= xsplit_1[i]) {
					rank_sum += hessianS[j];
				}
			}
			ranks[i] = rank_sum / hessian_sum_all;
		}

		for (size_t row = 0; row < xsplit1_size-1; ++row) {
			double rk_sk_j = ranks[row];
			double rk_sk_j_1 = ranks[row + 1];
			double diff = std::fabs(rk_sk_j - rk_sk_j_1);

			if (diff > this->eps) {
				continue;
			}

			double split_value = (ranks[row + 1] + ranks[row]) / 2.0;
			
			std::vector<bool> lhs(xsplit1_size, false);
			std::vector<bool> rhs(xsplit1_size, false);
			int lhs_sum=0;
			int rhs_sum=0;
			std::vector<int> lhs_indices, rhs_indices;
			double lhs_hessian_sum = 0.0, rhs_hessian_sum = 0.0;
			
			for (size_t i = 0; i < xsplit1_size; i++){
				if (xsplit_1[i] <= split_value) {
					lhs[i] = true;
					lhs_sum++;
					lhs_indices.emplace_back(tree[node_idx].idxs[i]);
					lhs_hessian_sum += hessianS[i];
				}else{
					rhs[i] = true;
					rhs_sum++;
					rhs_indices.emplace_back(tree[node_idx].idxs[i]);
					rhs_hessian_sum += hessianS[i];
				}
				
			}
			
			/*if (rhs_sum < this->min_leaf || lhs_sum < this->min_leaf || rhs_hessian_sum <= this->min_child_weight || lhs_hessian_sum < this->min_child_weight ) {
				continue;
			}*/
			
			
			double  curr_score = this->gain(lhs,rhs,node_idx);
			
			if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0){
				tree[node_idx].var_idx = c;
				tree[node_idx].score  = curr_score;
				tree[node_idx].split = split_value;
				lhsF.clear();
				rhsF.clear();
				lhsF = lhs_indices;
				rhsF = rhs_indices;	
			}
		}
	
		//xsplit_1.clear();
	}
	//gradientS.clear();
	//hessianS.clear();
	
	if(lhsF.size() != 0 && rhsF.size() != 0){
		//ajout des noeuds gauche et droit dans notre arbre
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF,tree[node_idx].depth+1,0.0, 0.0,-1, 0.0, -1,-1));
		
		//std::cout<<node_idx<<" "<<node_tree_size<<" "<<tree[node_idx].idxs[0]<<" "<<tree[node_idx].idxs[1]<<" "<<tree.size()<<std::endl;
		
		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	}
}

void TreeGenerator::find_varsplit_seq_approx_quantile_sketch(){

    tree.emplace_back(Node(this->idxs, 0, 0.0,0.0,-1, 0.0, -1,-1));
    node_stack.push(0);
    do{
    	// ici on continu l'entrainnement
    	int node_idx = node_stack.top();
    	node_stack.pop();
    	this->find_greedy_split_seq_approx_quantile_sketch(node_idx);
    	
    }while(!node_stack.empty());
}



double TreeGenerator::gain(std::vector<bool>& lhs,std::vector<bool>& rhs,int node_idx){
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
	//std::cout<< rhs_gradient <<" "<<lhs_gradient<<std::endl;
	
	double gain = 0.5*(((lhs_gradient*lhs_gradient)/(lhs_hessian + this->lambda)) + ((rhs_gradient*rhs_gradient)/(rhs_hessian + this->lambda)) - (((lhs_gradient + rhs_gradient)*(lhs_gradient + rhs_gradient))/(lhs_hessian + rhs_hessian + this->lambda))) - this->gamma;
	gradient.clear();
	hessian.clear();
	
	return gain;
	
}

std::vector<double> TreeGenerator::predict(std::vector<std::vector<double>>& x) {
    int x_size = x.size();
    std::vector<double> y_pred(x_size);
	std::vector<std::thread> threads(num_threads);

    auto worker = [&](int thread_id){
		for (int i = thread_id; i < x_size; i += num_threads) {
			y_pred[i] = this->predict_row(x[i]);
		}
	};
	
	for (int r = 0; r < num_threads; r++){
		threads[r] = std::thread(worker, r);
	}

	for(auto &t : threads){
		if (t.joinable()){
			t.join();
		}
	}

    return y_pred;
}

double TreeGenerator::predict_row(std::vector<double>& xi){
	int node_idx = 0;
	Node node = tree[node_idx];
	int tree_size = tree.size();
	for(int i=0;i<tree_size;i++){
		
		if(tree[node_idx].var_idx == -1 /*&& tree[node_idx].val >= 0.0*/){
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

void TreeGenerator::printNode(){
	for(int i=0;i<(int)tree.size();i++){
		std::cout<< i<< " "<<tree[i].val<<" ";
		for(int j=0;j<(int)tree[i].idxs.size();j++){
			std::cout<<tree[i].idxs[j]<<" ";	
		}
		std::cout<<" "<<tree[i].var_idx<<" "<<tree[i].score;
		std::cout<<std::endl;
	}
}

double XGBoostClassifier::sigmoid(double x){
	//return  std::exp(x) / (1.0 + std::exp(x));
	return 1.0 / (1.0 + std::exp(-x));
}
		
void XGBoostClassifier::grad_hess(std::vector<double>& preds,std::vector<int>& labels,std::vector<double> & grads,std::vector<double> & hess){
	int label_size = labels.size();
	std::vector<std::thread> threads(num_threads);
	auto worker = [&](int thread_id){
		for(int i = thread_id;i< label_size;i+= num_threads){
			double  residual =(this->sigmoid(preds[i]) - labels[i]);
			double  den = this->sigmoid(preds[i]) * (1.0-this->sigmoid(preds[i]));
			
			grads[i] = (residual);
			hess[i] = (den);
			
		}
	};
	
	for (int r = 0; r < num_threads; r++){
		threads[r] = std::thread(worker, r);
	}

	for(auto &t : threads){
		if (t.joinable()){
			t.join();
		}
	}

}
		
double XGBoostClassifier::log_odds(std::vector<int> labels){
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
void XGBoostClassifier::observation_subsample(std::vector<std::vector<double>>& x,std::vector<int>& y,double subsample_ratio,bool random_sample){

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
		
std::vector<double>  XGBoostClassifier::fit(std::vector<std::vector<double>>& x, std::vector<int>& y,double subsample_cols = 0.8 , int min_child_weight = 1, int depth = 8,int min_leaf = 1,double learning_rate = 0.4, int trees = 5,double lambda = 1, double gamma = 1,double eps=0.5,int num_threads=4,int choice=0){
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
	this->num_threads = num_threads;
	this->min_child_weight = min_child_weight;
	
	std::string outputFileName= "outputTreePerTime"+std::to_string(choice)+".csv";
	std::ofstream csvFile(outputFileName, std::ios::out);

	// Check if the file was opened successfully
	if (!csvFile.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
	}
	// Write the header row
	csvFile << "NumArbre,ExecutionTime" << std::endl;
	double temps_double_data = 0;
	std::vector<double> loss;
	
	for(int treeUnit = 0; treeUnit < this->trees; treeUnit++) {
			
		std::vector<double> Grad(x_size,0.0),Hess(x_size,0.0);
		grad_hess(this->base_pred,this->y,Grad,Hess);
		init_cpu_time();
		top1();
		TreeGenerator tree(this->x, Grad, Hess,root_idxs,this->subsample_cols, this->min_leaf,this->min_child_weight, this->depth, this->lambda, this->gamma,eps,num_threads,choice);
		top2();
		long temps = cpu_time();
		temps_double_data += temps/1000.0;

		csvFile <<treeUnit<<","<<temps_double_data<< std::endl;
		std::vector<double> y_pred = tree.predict(this->x);
		
		for (int i = 0; i < x_size; i++) {
			this->base_pred[i]  += this->learning_rate * y_pred[i];
		}
		
		this->estimators.emplace_back(tree);
		
		double loss_value = 0.0;
		
		double sum_squared_error = 0.0;
		for (int i = 0; i < x_size; i++) {
			double error = y[i] - y_pred[i];
			sum_squared_error += error * error;
		}

		double mean_squared_error = sum_squared_error / x_size;
		loss_value=sqrt(mean_squared_error);
		
		/*for (int i = 0; i < x_size; i++) {
			if (this->base_pred[i] <= 0.0) {
			    loss_value += -log(1e-15);
			} else if (this->base_pred[i] >= 1.0) {
			    loss_value += -log(1.0 - 1e-15);
			} else {
			    loss_value += -y[i] * log(this->base_pred[i]) - (1.0 - y[i]) * log(1.0 - this->base_pred[i]);
			}
		}*/
		
		
		
		//std::cout<<"\n"<<loss_value<<"\n";
		loss.push_back(loss_value);
		y_pred.clear();
		Grad.clear();
		Hess.clear();
	}
	
	// Fermer le fichier
	csvFile.close();
	base_pred.clear();
	
	return loss;
}
		
std::vector<double> XGBoostClassifier::predict_proba(std::vector<std::vector<double>> x){
	int x_size = x.size();
	std::vector<double> pred(x_size,0);
	std::vector<double> sigmoid_input(x_size, 0.0);
	
	std::vector<std::thread> threads(num_threads);
	for (auto& estimator : this->estimators) {
		std::vector<double> estimator_pred =estimator.predict(x);
		auto worker = [&](int thread_id){
			for (int i = thread_id; i < x_size; i += num_threads) {
				pred[i] += this->learning_rate*estimator_pred[i];
				sigmoid_input[i] = this->sigmoid(pred[i]);
			}
		};
		
		for (int r = 0; r < num_threads; r++){
			threads[r] = std::thread(worker, r);
		}

		for(auto &t : threads){
			if (t.joinable()){
				t.join();
			}
		}

    }
    
    
	
	pred.clear();
	
	return sigmoid_input;
}
		
std::vector<int> XGBoostClassifier::predict(std::vector<std::vector<double>> x){
	std::vector<double> pred(x.size(),0.0);
	
	for (auto& estimator : this->estimators) {
		
		for (int i = 0; i < (int)x.size(); i++){
			pred[i] += this->learning_rate* estimator.predict(x)[i];
		}
    }
    
    std::vector<double> sigmoid_input(x.size(), 0.0);
    
    for (int i = 0; i < (int)sigmoid_input.size(); i++) {
		sigmoid_input[i] = this->sigmoid(pred[i]);
	}
	pred.clear();
	
	std::vector<int> preds;
	
	double mean_predicted_probas = 0.0;
	for (double proba : sigmoid_input) {
		mean_predicted_probas += proba;
	}
	mean_predicted_probas = mean_predicted_probas / sigmoid_input.size();
	
	for (int i = 0; i < (int)sigmoid_input.size(); i++) {
		if(sigmoid_input[i] < mean_predicted_probas){
			preds.push_back(1);
		}else{
			preds.push_back(0);
		}
	}
	
	return preds;
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
        rowLine.clear();

    }
    

    file.close();
    
    return row;
}    

std::vector<std::vector<double>> extractCSVDatasetWithNa(const std::string& filename) {

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
        std::vector<std::string> fields;
        std::vector<double> rowLine;
        while (std::getline(iss, value, ',')){
			if(value.empty() || value == "NaN" || value== "NA" || value == "null"){
				rowLine.push_back(-0.00000000000000001);
			}else{
            	rowLine.push_back(std::stod(value));
			}
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
        
        if(predicted_labels[i] >= 0.5){
        	predicted_label = 1;
        }else{
        	predicted_label = 0;
        }
        
        confusion_matrix[true_label * 2 + predicted_label]++;
    }
    
    return  confusion_matrix;
}




void print_confusion_matrix(std::vector<int> confusion_matrix,std::ofstream &logFileResult) {
    printf("Confusion Matrix:\n");
	logFileResult<<"Confusion Matrix:"<<std::endl;
    printf("          Predicted\n");
	logFileResult<<"          Predicted"<<std::endl;
    printf("          0     1\n");
	logFileResult<<"          0     1"<<std::endl;
    printf("Actual 0  %d     %d\n", confusion_matrix[0], confusion_matrix[1]);
	logFileResult<<"          "<<confusion_matrix[0]<<"     "<<confusion_matrix[1]<<std::endl;
    printf("       1  %d     %d\n", confusion_matrix[2], confusion_matrix[3]);
	logFileResult<<"          "<<confusion_matrix[2]<<"     "<<confusion_matrix[3]<<std::endl;
}

void calculate_metrics(std::vector<int> true_labels, std::vector<double> predicted_labels, int num_samples){
	// Initialize counters for precision and recall calculations
    int true_positive = 0;
    int false_positive = 0;
    int false_negative = 0;
    int true_negative = 0;

    // Calculate accuracy, precision, and recall
    for (int i = 0; i < num_samples; ++i) {
        // Round the prediction to get binary output (0 or 1)
        int predicted_label = round(predicted_labels[i]);
        int actual_label = (int)true_labels[i];

        if (predicted_label == 1 && actual_label == 1) {
            true_positive++;
        } else if (predicted_label == 1 && actual_label == 0) {
            false_positive++;
        } else if (predicted_label == 0 && actual_label == 1) {
            false_negative++;
        } else if (predicted_label == 0 && actual_label == 0) {
            true_negative++;
        }
    }

    // Calculate precision, recall, and accuracy
    float precision = true_positive / (float)(true_positive + false_positive);
    float recall = true_positive / (float)(true_positive + false_negative);
    float accuracy = (true_positive + true_negative) / (float)(true_positive + false_positive+true_negative+false_negative);

    // Print the results
    printf("Accuracy: %.2f%%\n", accuracy * 100.0);
    printf("Precision: %.2f%%\n", precision * 100.0);
    printf("Recall: %.2f%%\n", recall * 100.0);
}

double calculate_accuracy(std::vector<int> confusion_matrix) {
    int true_positive = confusion_matrix[3];
    int true_negative = confusion_matrix[0];
    int total_samples = true_positive + true_negative+confusion_matrix[1]+confusion_matrix[2];
    printf("Individus: %d\n", total_samples);
    double accuracy = (double)(true_positive + true_negative) / total_samples;
    return accuracy;
}

double calculate_recall(std::vector<int> confusion_matrix) {
    int true_positive = confusion_matrix[3];
    //int true_negative = confusion_matrix[0];
    int false_negative = confusion_matrix[1];
    double recall = (double)(true_positive)/(true_positive+false_negative);
    return recall;
}

double calculate_precision(std::vector<int> confusion_matrix){
    int true_positive = confusion_matrix[3];
    //int true_negative = confusion_matrix[0];
    int false_positive = confusion_matrix[2];
    double precision = (double)(true_positive)/(true_positive+false_positive);
    return precision;
}


std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<int>, std::vector<int>> train_test_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y, double test_size = 0.2, unsigned seed = 0) {

    // Check if sizes of X and y match
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same number of elements.");
    }

    // Create a vector of indices and shuffle it
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));

    // Determine the splitting point
    int test_count = static_cast<int>(X.size() * test_size);
    int train_count = X.size() - test_count;

    // Create the train and test sets
    std::vector<std::vector<double>> X_train, X_test;
	std::vector<int>  y_train, y_test;
    for (int i = 0; i < train_count; ++i) {
        X_train.push_back(X[indices[i]]);
        y_train.push_back(y[indices[i]]);
    }
    for (int i = train_count; i < (int)X.size(); ++i) {
        X_test.push_back(X[indices[i]]);
        y_test.push_back(y[indices[i]]);
    }

    return make_tuple(X_train, X_test, y_train, y_test);
}


int main(int argc, char * argv[]) {
	init_cpu_time_2();
   
   	if(argc != 15){
		printf("Nombre d'argument incorrect\n");
		printf("Format de l'executable: <<nom fichier>> <<nombre-thread>> <<fichier-dataset>> <<fichier-labels>> <<train-test-percent>> <<subsample_cols>> <<min_child_weight>> <<depth>> <<min_leaf>> <<learning_rate>> <<trees>> <<lambda>> <<gamma>> <<epsilon>>\n");
		exit(-1);
	}
	
	// Declatration des parametres utilisateurs
	int num_threads = atoi(argv[1]);
	std::string filename_dataset = argv[2];
	std::string filename_labels = argv[3];
    double train_test_percent= std::stod(argv[4]);
	double subsample_cols_percent= std::stod(argv[5]);
	int  min_child_weight_arg =  atoi(argv[6]);
	int depth_arg = atoi(argv[7]);
	int min_leaf_arg = atoi(argv[8]);
	double learning_rate_arg = std::stod(argv[9]);
	int num_trees = atoi(argv[10]);
	double lambda_arg = std::stod(argv[11]);
	double gamma_arg = std::stod(argv[12]);
	double eps = std::stod(argv[13]);
	std::string nanOrNot = argv[14];
	// donnee d'entrainnement
	// ./xgbParallel 4 "Datasets/diabetesData.csv" "Datasets/labels.csv" 0.2 1 1 8 1 0.2 100 0.1 0.2 0.5
	//"Datasets/cleanDiabetes_data_WiDS2021_5k.csv" "Datasets/cleanDiabetes_labels_WiDS2021_5k.csv"
	//"Datasets/cleanDiabetes_data_WiDS2021.csv" "Datasets/cleanDiabetes_labels_WiDS2021.csv"
	//std::string filename = "Datasets/diabetesData.csv";
	std::vector<std::vector<double>> x;
	std::vector<std::vector<double>> yTemp;
	if(nanOrNot != "oui"){
		x = extractCSVDataset(filename_dataset);
		yTemp = extractCSVDataset(filename_labels);
	}else{
		x = extractCSVDatasetWithNa(filename_dataset);
		yTemp = extractCSVDatasetWithNa(filename_labels);
	}
	

	std::vector<int> y;

	for(int i=0; i<(int)yTemp.size();i++){
		y.push_back(((int)yTemp[i][0]));
	}

	auto [x_train, xTest, y_train, yTest] = train_test_split(x, y, train_test_percent);

	// execution sequentielle
	std::cout<<"seq execution in progress..."<<std::endl;
	XGBoostClassifier xgb;
	top3();
	std::vector<double> loss = xgb.fit(x_train, y_train,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,num_threads,1);
	top4();
	
	long temps = cpu_time_2();
	printf("\ntime seq = %ld.%03ldms\n\n", temps/1000, temps%1000);
	printf("\n");
	
	std::string outputFileNameLoss= "outputPerLossPerTree.csv";
	std::ofstream csvFileLoss(outputFileNameLoss, std::ios::out);
	// Check if the file was opened successfully
	if (!csvFileLoss.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	
	int loss_size = loss.size();
	for(int i=0;i<loss_size;i++){
		csvFileLoss<<i<<","<<loss[i]<< std::endl;
	}
	csvFileLoss.close();
	/*
	std::string outputFileName= "outputSPeedUpPerThread.csv";
	std::ofstream csvFileSPeedUpPerThread(outputFileName, std::ios::out);

	// Check if the file was opened successfully
	if (!csvFileSPeedUpPerThread.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	
	std::ofstream csvFileLossPar("outputCsvFileLossPar.csv", std::ios::out);

	// Check if the file was opened successfully
	if (!csvFileLossPar.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	
	// Write the header row
	csvFileSPeedUpPerThread << "NumThread,ParallelPerSplitPoint,ParallelPerFeature" << std::endl;
	csvFileLossPar<<"NumTree,LossParFeat,LossParSplitPoint"<<std::endl;
	
	for(int i=2;i<=num_threads;i++){
		std::cout<<"parallel execution using "<<i<<" threads"<<std::endl;
		// execution parallel par split point
		std::cout<<"parallel execution per split point in progress..."<<std::endl;
		XGBoostClassifier xgb1;
		top3();
		std::vector<double> loss1 =xgb1.fit(x_train, y_train,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,i,1);
		top4();

		long temps1 = cpu_time_2();
		printf("\ntime par split point = %ld.%03ldms\n\n", temps1/1000, temps1%1000);
		
		// execution parallel par feature
		std::cout<<"parallel execution per feature in progress..."<<std::endl;
		XGBoostClassifier xgb2;
		top3();
		std::vector<double> loss2 =xgb2.fit(x_train, y_train,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,i,2);
		top4();

		long temps2 = cpu_time_2();
		printf("\ntime par feat = %ld.%03ldms\n\n", temps2/1000, temps2%1000);

		// // execution parallel par feature plus split point
		// std::cout<<"parallel execution per feature plus split point in progress..."<<std::endl;
		// XGBoostClassifier xgb3;
		// top3();
		// std::vector<double> loss3 =xgb3.fit(x, y,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,i,3);
		// top4();

		// long temps3 = cpu_time_2();
		// printf("\ntime par split point plus feature= %ld.%03ldms\n\n", temps3/1000, temps3%1000); <<","<<loss3[i]

		
		if(i == num_threads){
			int loss_size1 = loss1.size();
			for(int i=0;i<loss_size1;i++){
				csvFileLossPar<<i<<","<<loss1[i]<<","<<loss2[i]<< std::endl;
			}
		}

		
		
		// calcul des speedUp
		float speedUpSplitPoint =  (temps/1000.0 )/(temps1/1000.0 );
		printf("\nspeedUp split point  %.8f\n\n",speedUpSplitPoint);

		float speedUpFeat =  (temps/1000.0)/(temps2/1000.0);
		printf("\nspeedUp feature %.8f\n\n",speedUpFeat);

		// float speedUpFeatPlusSplitPoint =  (temps/1000.0)/(temps3/1000.0 );
		// printf("\nspeedUpFeatPlusSplitPoint %.8f\n\n",speedUpFeatPlusSplitPoint); <<speedUpFeatPlusSplitPoint

		csvFileSPeedUpPerThread<<i<<","<< speedUpSplitPoint <<","<<speedUpFeat<<std::endl;
		//loss1.clear();
		//loss2.clear();
	}
	csvFileLossPar.close();
	csvFileSPeedUpPerThread.close();
	*/

	//prediction des donnees de test
	std::vector<double>predicted_labels= xgb.predict_proba(xTest);

	int confusionSize = 4 ;

	std::ofstream logFileResult("resultLog.txt", std::ios::app);
	// Check if the file was opened successfully
	if (!logFileResult.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	std::vector<int> confusion_matrix(confusionSize,0);
	confusion_matrix = calculate_confusion_matrix(yTest, predicted_labels, yTest.size(), confusion_matrix);

	logFileResult<<argv[0]<<" "<<num_threads<<" "<<filename_dataset<<" "<<filename_labels<<" "<<train_test_percent<<" "<<subsample_cols_percent<<" "<<min_child_weight_arg<<" "<<depth_arg<<" "<<min_leaf_arg<<" "<<learning_rate_arg<<" "<<num_trees<<" "<<lambda_arg<<" "<<gamma_arg<<" "<<eps<<" "<<nanOrNot<< std::endl;

	print_confusion_matrix(confusion_matrix,logFileResult);
	double accuracy = calculate_accuracy(confusion_matrix);
	printf("\n\nAccuracy: %.2f%%\n", accuracy*100);
	logFileResult<<"Accuracy:"<<accuracy*100<< std::endl;
	double precision = calculate_precision(confusion_matrix);
	printf("Precision: %.2f%%\n", precision*100);
	logFileResult<<"Precision:"<<precision*100<< std::endl;
	double recall = calculate_recall(confusion_matrix);
	printf("Recall: %.2f%%\n", recall*100);
	logFileResult<<"Recall:"<<recall*100<< std::endl;
	logFileResult<<"-------------------------------------------------------------------"<< std::endl;
	logFileResult.close();

	// prediction d'une entree
	std::vector<std::vector<double>> testData =  {{ 6,154,74,32,193,29.3,0.839,39}};
	std::cout<<"\n"<<xgb.predict(testData)[0]<<"\n";

    return 0;
}
