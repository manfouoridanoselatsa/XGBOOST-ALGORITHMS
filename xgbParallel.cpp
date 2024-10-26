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
#include <unordered_set>
#include <mutex>
#include <sys/time.h>
#include "xgb.h"
#include "bitoniq.h"
#include <set>
std::mutex mtx;
std::mutex mtx1;

/* ici on defini les differentes outils qui nous permettrons de calculer le temps d'execution d'un algorithme donne*/
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


/*Ici je defini le constructeur de ma classe permettant de creer un noeud d'un arbre*/
Node::Node(std::vector<int>&  idxs,int depth = 0,double val=0.0,double score =0.0,int var_idx =-1,double split=0.0,int left=-1,int right = -1,bool IsLeaf = false) {

	this->idxs = idxs;
	this->depth = depth;
	this->val = val;
	this->score = score;
	this->var_idx = var_idx;
	this->split = split;
	this->left = left;
	this->right = right;
	this->IsLeaf = IsLeaf;
}


/* Ici Je definis le constructeur de la classe me permettant de creer un arbre */
TreeGenerator::TreeGenerator(std::vector<std::vector<double>>& x, std::vector<double>&  gradient,std::vector<double>& hessian,std::vector<int>& idxs, double subsample_cols = 0.8 , int min_leaf = 1,int min_child_weight = 1 ,int depth = 4,double lambda = 1,double gamma = 1,double eps=0.5,int num_threads=4,int choice=0,int num_bins=3) {
	
	// top3();
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
	this->num_bins = num_bins;
	
	if(this->subsample_cols < 1.0){
		//std::cout<<"Hello"<<std::endl;
		this->column_subsample = col_subsample(true);		
	}else{
		this->column_subsample = col_subsample(false);
	}
	// top4();
	
	// long temps = cpu_time_2();
	// printf("\nexecution time = %ld.%03ldms\n\n", temps/1000, temps%1000);
	// printf("\n");
	
	this->eps=eps;

	/* Cette Section nous permet de choisir quelle algorihtme l'on veut utiliser dans notre main suivant un chiffre donné */
	switch (choice) {
        case 0: this->find_varsplit_seq(); break;
        case 1: this->find_varsplit_par(); break;
        case 2: this->find_varsplit_par_feat(); break;
        case 3: this->find_varsplit_par_plus_feat(); break;
    }

	//this->printNode();
	
}

// valeur de sortie d'un noeud donné
double TreeGenerator::compute_node_output(int &node_idx){
	
	double gradient_sum = 0.0;
	double hessian_sum = 0.0;
	for(int i : tree[node_idx].idxs){
		gradient_sum += this->gradient[i];
		hessian_sum += this->hessian[i];
	}

	return (-gradient_sum/(hessian_sum + lambda));
}

double TreeGenerator::gain(double &s_Gg,double &s_Hg,double &s_Gr,double &s_Hr){
	double gain = 0.5*(((s_Gg*s_Gg)/(s_Hg + this->lambda)) + ((s_Gr*s_Gr)/(s_Hr + this->lambda)) - (((s_Gg + s_Gr)*(s_Gg + s_Gr))/(s_Hg + s_Hr + this->lambda)))- this->gamma;
	return gain;
}
/* Nous permet d'extraire les données de notre jeux de données*/
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



/* Nous permet d'extraire les données de notre jeux de données avec prise en compte des valeurs NA*/
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


/* Calcul de la matrice de confusion */
std::vector<int> calculate_confusion_matrix(std::vector<int>& true_labels, std::vector<double>& predicted_labels, int num_samples, std::vector<int>& confusion_matrix) {
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




void print_confusion_matrix(std::vector<int>& confusion_matrix,std::ofstream &logFileResult) {
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

void calculate_metrics(std::vector<int>& true_labels, std::vector<double>& predicted_labels, int num_samples){
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

double calculate_accuracy(std::vector<int>& confusion_matrix) {
    int true_positive = confusion_matrix[3];
    int true_negative = confusion_matrix[0];
    int total_samples = true_positive + true_negative+confusion_matrix[1]+confusion_matrix[2];
    printf("Individus: %d\n", total_samples);
    double accuracy = (double)(true_positive + true_negative) / total_samples;
    return accuracy;
}

double calculate_recall(std::vector<int>& confusion_matrix) {
    int true_positive = confusion_matrix[3];
    //int true_negative = confusion_matrix[0];
    int false_negative = confusion_matrix[1];
    double recall = (double)(true_positive)/(true_positive+false_negative);
    return recall;
}

double calculate_precision(std::vector<int>& confusion_matrix){
    int true_positive = confusion_matrix[3];
    //int true_negative = confusion_matrix[0];
    int false_positive = confusion_matrix[2];
    double precision = (double)(true_positive)/(true_positive+false_positive);
    return precision;
}


/* Fonction de train test split*/
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

/* Cette fonction est un utilitaire que nous allons utiliser
pour comparer les éléments de type feature_val*/
bool compareByVal(const feature_val& a, const feature_val& b) {
    if (a.value == b.value) {
        return a.idx > b.idx; // Si les âges sont égaux, trier par nom
    }
    return a.value <= b.value; // Trier par âge
}




/* Ici je definis la fonction me permettant de choisir aleatoirement un nombre de colonne donne pour creer un arbre*/
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


/* Cette fonction est la version séquentielle de l'algorithme Find greedy split en utilisant les feature.
Il ce charge de  réaliser la subdivision pour un noeud donné node_idx en noeud fils gauche et droit*/

/*void TreeGenerator::find_greedy_split_seq(int node_idx) {	
	
	if (tree[node_idx].IsLeaf) {
		return;
	}

	std::vector<int> lhsF, rhsF;
	int node_tree_size = tree[node_idx].idxs.size();

	if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
		tree[node_idx].val = this->compute_node_output(node_idx);
		tree[node_idx].IsLeaf = true;
		return;
	}

	// top3();
	for (int c : this->column_subsample) {
		std::unordered_set<double> unique_values;
		std::vector<double> xsplit_1(node_tree_size);

		for (int i = 0; i < node_tree_size; ++i) {
			xsplit_1[i] = this->x[tree[node_idx].idxs[i]][c];
			unique_values.insert(xsplit_1[i]);
		}

		std::vector<double> xsplit(unique_values.begin(), unique_values.end());
		std::sort(xsplit.begin(), xsplit.end());

		int xsplit_size = xsplit.size();
		std::vector<double> midpoints(xsplit_size - 1);
		for (int i = 1; i < xsplit_size; ++i) {
			midpoints[i - 1] = (xsplit[i - 1] + xsplit[i]) / 2.0;
		}

		std::vector<int> lhs_indices, rhs_indices;
		for (double midpoint : midpoints) {
			int lhs_sum = 0, rhs_sum = 0;
			double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;
			lhs_indices.clear();
			rhs_indices.clear();

			for (int i = 0; i < node_tree_size; ++i) {
				if (xsplit_1[i] <= midpoint) {
					lhs_sum++;
					lhs_indices.push_back(tree[node_idx].idxs[i]);
					s_Gg += this->gradient[tree[node_idx].idxs[i]];
					s_Hg += this->hessian[tree[node_idx].idxs[i]];
				} else {
					rhs_sum++;
					rhs_indices.push_back(tree[node_idx].idxs[i]);
					s_Gr += this->gradient[tree[node_idx].idxs[i]];
					s_Hr += this->hessian[tree[node_idx].idxs[i]];
				}
			}
			
			// if (lhs_sum < this->min_leaf || rhs_sum < this->min_leaf || s_Hr <= this->min_child_weight || s_Hg <= this->min_child_weight) {
			// 	 continue;
			// }

			double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);

			if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0) {
				tree[node_idx].var_idx = c;
				tree[node_idx].score = curr_score;
				tree[node_idx].split = midpoint;
				lhsF = lhs_indices;
				rhsF = rhs_indices;
			}
		}
		xsplit_1.clear();
	}

	// top4();
	
	// long temps = cpu_time_2();
	// printf("\nexecution time = %ld.%03ldms\n\n", temps/1000, temps%1000);
	

	if (!lhsF.empty() && !rhsF.empty()) {
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	} else {
		tree[node_idx].val = this->compute_node_output(node_idx);
		tree[node_idx].IsLeaf = true;
	}
	
}*/

void TreeGenerator::find_greedy_split_seq(int node_idx) {    
    if (tree[node_idx].IsLeaf) {
        return;
    }

    int node_tree_size = tree[node_idx].idxs.size();
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
        return;
    }

    // Structure to hold split information
    struct SplitInfo {
        double score;
        int var_idx;
        double split_point;
        std::vector<int> left_indices;
        std::vector<int> right_indices;

        SplitInfo() : score(std::numeric_limits<double>::lowest()), 
                     var_idx(-1), split_point(0.0) {}

        bool operator<(const SplitInfo& other) const {
            if (score != other.score) return score < other.score;
            if (var_idx != other.var_idx) return var_idx < other.var_idx;
            return split_point < other.split_point;
        }
    };

    // Keep track of best split
    SplitInfo best_split;

    // First, collect all feature values and split points
    struct FeatureSplits {
        int feature_idx;
        std::vector<double> feature_values;
        std::vector<double> split_points;
    };
    std::vector<FeatureSplits> all_feature_splits;

    // Gather all feature values and split points first
    for (int c : this->column_subsample) {
        FeatureSplits feature_split;
        feature_split.feature_idx = c;
        feature_split.feature_values.resize(node_tree_size);
        
        std::set<double> unique_values;  // Using set for deterministic ordering
        for (int i = 0; i < node_tree_size; ++i) {
            feature_split.feature_values[i] = this->x[tree[node_idx].idxs[i]][c];
            unique_values.insert(feature_split.feature_values[i]);
        }

        // Calculate split points
        auto it = unique_values.begin();
        if (it != unique_values.end()) {
            auto prev_val = *it;
            ++it;
            for (; it != unique_values.end(); ++it) {
                feature_split.split_points.push_back((prev_val + *it) / 2.0);
                prev_val = *it;
            }
        }
        
        if (!feature_split.split_points.empty()) {
            all_feature_splits.push_back(std::move(feature_split));
        }
    }

    // Process all splits sequentially
    for (const auto& feature_split : all_feature_splits) {
        const int c = feature_split.feature_idx;
        const auto& feature_values = feature_split.feature_values;
        const auto& split_points = feature_split.split_points;

        for (const double split_val : split_points) {
            std::vector<int> lhs_indices, rhs_indices;
            double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;

            // Deterministic splitting
            for (int i = 0; i < node_tree_size; ++i) {
                int idx = tree[node_idx].idxs[i];
                if (feature_values[i] <= split_val) {
                    lhs_indices.push_back(idx);
                    s_Gg += this->gradient[idx];
                    s_Hg += this->hessian[idx];
                } else {
                    rhs_indices.push_back(idx);
                    s_Gr += this->gradient[idx];
                    s_Hr += this->hessian[idx];
                }
            }

            if (!lhs_indices.empty() && !rhs_indices.empty()) {
                double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);
                
                // Update best split if better
                if (curr_score > best_split.score || 
                    (curr_score == best_split.score && 
                     (c < best_split.var_idx || 
                      (c == best_split.var_idx && split_val < best_split.split_point)))) {
                    best_split.score = curr_score;
                    best_split.var_idx = c;
                    best_split.split_point = split_val;
                    best_split.left_indices = lhs_indices;
                    best_split.right_indices = rhs_indices;
                }
            }
        }
    }
	all_feature_splits.clear();
    // Create child nodes if a valid split was found
    if (!best_split.left_indices.empty() && !best_split.right_indices.empty()) {
        tree[node_idx].var_idx = best_split.var_idx;
        tree[node_idx].score = best_split.score;
        tree[node_idx].split = best_split.split_point;

        int left_child_idx = tree.size();
        tree.emplace_back(Node(best_split.left_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        
        int right_child_idx = tree.size();
        tree.emplace_back(Node(best_split.right_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;
        
        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    } else {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
    }
}

void TreeGenerator::find_varsplit_seq() {
	tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
	node_stack.push(0);
	
	while (!node_stack.empty()) {
		int node_idx = node_stack.top();
		node_stack.pop();
		this->find_greedy_split_seq(node_idx);
	}
}


/*void TreeGenerator::find_greedy_split_par_feat(int node_idx) {    
    if (tree[node_idx].IsLeaf) {
        return;
    }

    int node_tree_size = tree[node_idx].idxs.size();
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
        return;
    }

    // Structure to hold split information
    struct SplitInfo {
        double score;
        int var_idx;
        double split_point;
        std::vector<int> left_indices;
        std::vector<int> right_indices;

        SplitInfo() : score(std::numeric_limits<double>::lowest()), 
                     var_idx(-1), split_point(0.0) {}

        bool operator<(const SplitInfo& other) const {
            if (score != other.score) return score < other.score;
            if (var_idx != other.var_idx) return var_idx < other.var_idx;
            return split_point < other.split_point;
        }
    };

    // Vector to store best split for each thread
    std::vector<SplitInfo> thread_best_splits(num_threads);
    std::vector<std::thread> threads(num_threads);
    
    auto process_column = [&](int thread_id) {
        // Thread-local best split
        SplitInfo& local_best = thread_best_splits[thread_id];
        
        int numCol = (int)this->column_subsample.size();
        for (int tt = thread_id; tt < numCol; tt += num_threads) {
            int c = this->column_subsample[tt];
            
            // Gather unique values
            std::vector<double> feature_values(node_tree_size);
            std::set<double> unique_values; // Using set for deterministic ordering
            
            for (int i = 0; i < node_tree_size; ++i) {
                feature_values[i] = this->x[tree[node_idx].idxs[i]][c];
                unique_values.insert(feature_values[i]);
            }
            
            std::vector<double> split_points;
            {
                auto it = unique_values.begin();
                auto prev_val = *it;
                ++it;
                for (; it != unique_values.end(); ++it) {
                    split_points.push_back((prev_val + *it) / 2.0);
                    prev_val = *it;
                }
            }

            // Process each split point
            for (double split_val : split_points) {
                std::vector<int> lhs_indices, rhs_indices;
                double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;
                
                // Deterministic splitting
                for (int i = 0; i < node_tree_size; ++i) {
                    int idx = tree[node_idx].idxs[i];
                    if (feature_values[i] <= split_val) {
                        lhs_indices.push_back(idx);
                        s_Gg += this->gradient[idx];
                        s_Hg += this->hessian[idx];
                    } else {
                        rhs_indices.push_back(idx);
                        s_Gr += this->gradient[idx];
                        s_Hr += this->hessian[idx];
                    }
                }

                if (!lhs_indices.empty() && !rhs_indices.empty()) {
                    double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);
                    
                    // Update thread-local best if better
                    if (curr_score > local_best.score || 
                        (curr_score == local_best.score && 
                         (c < local_best.var_idx || 
                          (c == local_best.var_idx && split_val < local_best.split_point)))) {
                        local_best.score = curr_score;
                        local_best.var_idx = c;
                        local_best.split_point = split_val;
                        local_best.left_indices = lhs_indices;
                        local_best.right_indices = rhs_indices;
                    }
                }
            }
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_column, i);
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Find the best split among all threads deterministically
    SplitInfo best_split;
    for (const auto& thread_split : thread_best_splits) {
        if (thread_split.score > best_split.score || 
            (thread_split.score == best_split.score && 
             (thread_split.var_idx < best_split.var_idx || 
              (thread_split.var_idx == best_split.var_idx && 
               thread_split.split_point < best_split.split_point)))) {
            best_split = thread_split;
        }
    }
	thread_best_splits.clear();
    // Create child nodes if a valid split was found
    if (!best_split.left_indices.empty() && !best_split.right_indices.empty()) {
        tree[node_idx].var_idx = best_split.var_idx;
        tree[node_idx].score = best_split.score;
        tree[node_idx].split = best_split.split_point;

        int left_child_idx = tree.size();
        tree.emplace_back(Node(best_split.left_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        
        int right_child_idx = tree.size();
        tree.emplace_back(Node(best_split.right_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;
        
        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    } else {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
    }
}*/

void TreeGenerator::find_greedy_split_par_feat(int node_idx){    
    if (tree[node_idx].IsLeaf) {
        return;
    }

    int node_tree_size = tree[node_idx].idxs.size();
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
        return;
    }

    // Structure to hold split information
    struct SplitInfo {
        double score;
        int var_idx;
        double split_point;
        std::vector<int> left_indices;
        std::vector<int> right_indices;

        SplitInfo() : score(std::numeric_limits<double>::lowest()), 
                     var_idx(-1), split_point(0.0) {}

        bool operator<(const SplitInfo& other) const {
            if (score != other.score) return score < other.score;
            if (var_idx != other.var_idx) return var_idx < other.var_idx;
            return split_point < other.split_point;
        }
    };

    // Vector to store best split for each thread/feature group
    std::vector<SplitInfo> feature_group_best_splits(num_threads);
    std::vector<std::thread> threads(num_threads);

    // Divide features among threads
    std::vector<std::vector<int>> feature_groups(num_threads);
    for (size_t i = 0; i < column_subsample.size(); ++i) {
        feature_groups[i % num_threads].push_back(column_subsample[i]);
    }

    auto process_features = [&](int thread_id) {
        // Thread-local best split
        SplitInfo& local_best = feature_group_best_splits[thread_id];
        const auto& features = feature_groups[thread_id];
        
        // Process all features assigned to this thread
        for (int feature_idx : features) {
            // Collect feature values for current feature
            std::vector<double> feature_values(node_tree_size);
            std::set<double> unique_values;  // Using set for deterministic ordering
            
            for (int i = 0; i < node_tree_size; ++i) {
                feature_values[i] = this->x[tree[node_idx].idxs[i]][feature_idx];
                unique_values.insert(feature_values[i]);
            }

            // Calculate split points for current feature
            std::vector<double> split_points;
            auto it = unique_values.begin();
            if (it != unique_values.end()) {
                auto prev_val = *it;
                ++it;
                for (; it != unique_values.end(); ++it) {
                    split_points.push_back((prev_val + *it) / 2.0);
                    prev_val = *it;
                }
            }

            // Evaluate all split points for current feature
            for (double split_val : split_points) {
                std::vector<int> lhs_indices, rhs_indices;
                double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;

                // Deterministic splitting
                for (int i = 0; i < node_tree_size; ++i) {
                    int idx = tree[node_idx].idxs[i];
                    if (feature_values[i] <= split_val) {
                        lhs_indices.push_back(idx);
                        s_Gg += this->gradient[idx];
                        s_Hg += this->hessian[idx];
                    } else {
                        rhs_indices.push_back(idx);
                        s_Gr += this->gradient[idx];
                        s_Hr += this->hessian[idx];
                    }
                }

                if (!lhs_indices.empty() && !rhs_indices.empty()) {
                    double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);
                    
                    // Update thread-local best if better
                    if (curr_score > local_best.score || 
                        (curr_score == local_best.score && 
                         (feature_idx < local_best.var_idx || 
                          (feature_idx == local_best.var_idx && split_val < local_best.split_point)))) {
                        local_best.score = curr_score;
                        local_best.var_idx = feature_idx;
                        local_best.split_point = split_val;
                        local_best.left_indices = lhs_indices;
                        local_best.right_indices = rhs_indices;
                    }
                }
            }
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_features, i);
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Find best split across all threads
    SplitInfo best_split = feature_group_best_splits[0];
    for (int i = 1; i < num_threads; ++i) {
        const auto& thread_split = feature_group_best_splits[i];
        if (thread_split.score > best_split.score || 
            (thread_split.score == best_split.score && 
             (thread_split.var_idx < best_split.var_idx || 
              (thread_split.var_idx == best_split.var_idx && 
               thread_split.split_point < best_split.split_point)))) {
            best_split = thread_split;
        }
    }
	feature_group_best_splits.clear();
	feature_groups.clear();
    // Apply the best split
    /*if (best_split.score > std::numeric_limits<double>::lowest()) {
        // Create child nodes and update current node
        tree[node_idx].var_idx = best_split.var_idx;
        tree[node_idx].split = best_split.split_point;
        
        int left_child_idx = tree.size();
        int right_child_idx = tree.size() + 1;
        
        tree[node_idx].left_child = left_child_idx;
        tree[node_idx].right_child = right_child_idx;
        
        // Create left child
        TreeNode left_child;
        left_child.depth = tree[node_idx].depth + 1;
        left_child.idxs = std::move(best_split.left_indices);
        tree.push_back(left_child);
        
        // Create right child
        TreeNode right_child;
        right_child.depth = tree[node_idx].depth + 1;
        right_child.idxs = std::move(best_split.right_indices);
        tree.push_back(right_child);
    } else {
        // No valid split found, make this a leaf node
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
    }*/
    
    if (!best_split.left_indices.empty() && !best_split.right_indices.empty()) {
        tree[node_idx].var_idx = best_split.var_idx;
        tree[node_idx].score = best_split.score;
        tree[node_idx].split = best_split.split_point;

        int left_child_idx = tree.size();
        tree.emplace_back(Node(best_split.left_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        
        int right_child_idx = tree.size();
        tree.emplace_back(Node(best_split.right_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;
        
        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    } else {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
    }
    
}

void TreeGenerator::find_varsplit_par_feat() {
	tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
	node_stack.push(0);
	
	while (!node_stack.empty()) {
		int node_idx = node_stack.top();
		node_stack.pop();
		this->find_greedy_split_par_feat(node_idx);
	}
}



/* Cette fonction est la version parallel de l'algorithme Find greedy split en utilisant les points de subdivision.
Il ce charge de  réaliser la subdivision pour un noeud donné node_idx en noeud fils gauche et droit*/

void TreeGenerator::find_greedy_split_par(int node_idx) {    
    if (tree[node_idx].IsLeaf) {
        return;
    }

    int node_tree_size = tree[node_idx].idxs.size();
    if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
        return;
    }

    // Structure to hold split information
    struct SplitInfo {
        double score;
        int var_idx;
        double split_point;
        std::vector<int> left_indices;
        std::vector<int> right_indices;

        SplitInfo() : score(std::numeric_limits<double>::lowest()), 
                     var_idx(-1), split_point(0.0) {}

        bool operator<(const SplitInfo& other) const {
            if (score != other.score) return score < other.score;
            if (var_idx != other.var_idx) return var_idx < other.var_idx;
            return split_point < other.split_point;
        }
    };

    // Vector to store best split for each thread
    std::vector<SplitInfo> thread_best_splits(num_threads);
    std::vector<std::thread> threads(num_threads);

    // First, collect all feature values and split points
    struct FeatureSplits {
        int feature_idx;
        std::vector<double> feature_values;
        std::vector<double> split_points;
    };
    std::vector<FeatureSplits> all_feature_splits;

    // Gather all feature values and split points first
    for (int c : this->column_subsample) {
        FeatureSplits feature_split;
        feature_split.feature_idx = c;
        feature_split.feature_values.resize(node_tree_size);
        
        std::set<double> unique_values;  // Using set for deterministic ordering
        for (int i = 0; i < node_tree_size; ++i) {
            feature_split.feature_values[i] = this->x[tree[node_idx].idxs[i]][c];
            unique_values.insert(feature_split.feature_values[i]);
        }

        // Calculate split points
        auto it = unique_values.begin();
        if (it != unique_values.end()) {
            auto prev_val = *it;
            ++it;
            for (; it != unique_values.end(); ++it) {
                feature_split.split_points.push_back((prev_val + *it) / 2.0);
                prev_val = *it;
            }
        }
        
        if (!feature_split.split_points.empty()) {
            all_feature_splits.push_back(std::move(feature_split));
        }
    }

    auto process_splits = [&](int thread_id) {
        // Thread-local best split
        SplitInfo& local_best = thread_best_splits[thread_id];
        
        // Process splits in round-robin fashion across threads
        for (size_t feat_idx = 0; feat_idx < all_feature_splits.size(); ++feat_idx) {
            const auto& feature_split = all_feature_splits[feat_idx];
            const int c = feature_split.feature_idx;
            const auto& feature_values = feature_split.feature_values;
            const auto& split_points = feature_split.split_points;

            for (size_t sp = thread_id; sp < split_points.size(); sp += num_threads) {
                double split_val = split_points[sp];
                std::vector<int> lhs_indices, rhs_indices;
                double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;

                // Deterministic splitting
                for (int i = 0; i < node_tree_size; ++i) {
                    int idx = tree[node_idx].idxs[i];
                    if (feature_values[i] <= split_val) {
                        lhs_indices.push_back(idx);
                        s_Gg += this->gradient[idx];
                        s_Hg += this->hessian[idx];
                    } else {
                        rhs_indices.push_back(idx);
                        s_Gr += this->gradient[idx];
                        s_Hr += this->hessian[idx];
                    }
                }

                if (!lhs_indices.empty() && !rhs_indices.empty()) {
                    double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);
                    
                    // Update thread-local best if better
                    if (curr_score > local_best.score || 
                        (curr_score == local_best.score && 
                         (c < local_best.var_idx || 
                          (c == local_best.var_idx && split_val < local_best.split_point)))) {
                        local_best.score = curr_score;
                        local_best.var_idx = c;
                        local_best.split_point = split_val;
                        local_best.left_indices = lhs_indices;
                        local_best.right_indices = rhs_indices;
                    }
                }
            }
        }
    };

    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        threads[i] = std::thread(process_splits, i);
    }

    // Join threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Find the best split among all threads deterministically
    SplitInfo best_split;
    for (const auto& thread_split : thread_best_splits) {
        if (thread_split.score > best_split.score || 
            (thread_split.score == best_split.score && 
             (thread_split.var_idx < best_split.var_idx || 
              (thread_split.var_idx == best_split.var_idx && 
               thread_split.split_point < best_split.split_point)))) {
            best_split = thread_split;
        }
    }
	all_feature_splits.clear();
	thread_best_splits.clear();
    // Create child nodes if a valid split was found
    if (!best_split.left_indices.empty() && !best_split.right_indices.empty()) {
        tree[node_idx].var_idx = best_split.var_idx;
        tree[node_idx].score = best_split.score;
        tree[node_idx].split = best_split.split_point;

        int left_child_idx = tree.size();
        tree.emplace_back(Node(best_split.left_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
        
        int right_child_idx = tree.size();
        tree.emplace_back(Node(best_split.right_indices, 
                              tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

        tree[node_idx].left = left_child_idx;
        tree[node_idx].right = right_child_idx;
        
        node_stack.push(left_child_idx);
        node_stack.push(right_child_idx);
    } else {
        tree[node_idx].val = this->compute_node_output(node_idx);
        tree[node_idx].IsLeaf = true;
    }
}


void TreeGenerator::find_varsplit_par() {
	tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
	node_stack.push(0);
	
	while (!node_stack.empty()) {
		int node_idx = node_stack.top();
		node_stack.pop();
		this->find_greedy_split_par(node_idx);
	}
}



/* On fusionne les 2 stratégies de parallélisation*/
void TreeGenerator::find_greedy_split_par_plus_feat(int node_idx){	
	if (tree[node_idx].IsLeaf) {
		return;
	}

	std::vector<int> lhsF, rhsF;
	int node_tree_size = tree[node_idx].idxs.size();

	if (node_tree_size <= 1 || tree[node_idx].depth >= this->depth) {
		tree[node_idx].val = this->compute_node_output(node_idx);
		tree[node_idx].IsLeaf = true;
		return;
	}
	
	std::vector<std::thread> extern_threads(num_threads);

	auto process_column = [&](int thread_id){
		int numCol = (int)this->column_subsample.size();
		for (int tt = thread_id; tt < numCol; tt += num_threads){
		   int c = this->column_subsample[tt];
		// for (int c : this->column_subsample){
			
			std::unordered_set<double> unique_values;
			std::vector<double> xsplit_1(node_tree_size);

			for (int i = 0; i < node_tree_size; ++i) {
				xsplit_1[i] = this->x[tree[node_idx].idxs[i]][c];
				unique_values.insert(xsplit_1[i]);
			}

			std::vector<double> xsplit(unique_values.begin(), unique_values.end());
			std::sort(xsplit.begin(), xsplit.end());

			int xsplit_size = xsplit.size();
			std::vector<double> midpoints(xsplit_size - 1);
			for (int i = 1; i < xsplit_size; ++i) {
				midpoints[i - 1] = (xsplit[i - 1] + xsplit[i]) / 2.0;
			}

			
			
			int midpoints_size = midpoints.size();

			std::vector<std::thread> threads(num_threads);

			auto process_splitPoints = [&](int thread_id){
				for(int midpoint_id = thread_id; midpoint_id < midpoints_size; midpoint_id += num_threads){
					double midpoint = midpoints[midpoint_id];
					int lhs_sum = 0, rhs_sum = 0;
					std::vector<int> lhs_indices, rhs_indices;
					double s_Gg = 0.0, s_Hg = 0.0, s_Gr = 0.0, s_Hr = 0.0;
			
					for (int i = 0; i < node_tree_size; ++i) {
						if (xsplit_1[i] <= midpoint) {
							lhs_sum++;
							lhs_indices.push_back(tree[node_idx].idxs[i]);
							s_Gg += this->gradient[tree[node_idx].idxs[i]];
							s_Hg += this->hessian[tree[node_idx].idxs[i]];
						} else {
							rhs_sum++;
							rhs_indices.push_back(tree[node_idx].idxs[i]);
							s_Gr += this->gradient[tree[node_idx].idxs[i]];
							s_Hr += this->hessian[tree[node_idx].idxs[i]];
						}
					}
						
					// if (lhs_sum < this->min_leaf || rhs_sum < this->min_leaf || s_Hr <= this->min_child_weight || s_Hg <= this->min_child_weight) {
					// 	 continue;
					// }

					
					double curr_score = this->gain(s_Gg, s_Hg, s_Gr, s_Hr);
					
					{
	                    std::lock_guard<std::mutex> lock(mtx1);
						if (curr_score > tree[node_idx].score && lhs_indices.size() != 0 && rhs_indices.size() != 0) {
							tree[node_idx].var_idx = c;
							tree[node_idx].score = curr_score;
							tree[node_idx].split = midpoint;
							lhsF = lhs_indices;
							rhsF = rhs_indices;
						}
					}

					lhs_indices.clear();
					rhs_indices.clear();
				}
				
			};
			for (int i = 0; i < num_threads; ++i) {
		        threads[i] = std::thread(process_splitPoints, i);
		    }

		    for (auto &t : threads) {
		        if (t.joinable()) {
		            t.join();
		        }
		    }
		}

	};
	for (int i = 0; i < num_threads; ++i) {
        extern_threads[i] = std::thread(process_column, i);
    }
	
	for (auto &t : extern_threads) {
        if (t.joinable()) {
            t.join();
        }
    }

	if (lhsF.size() != 0 && rhsF.size() != 0) {
		int left_child_idx = tree.size();
		tree.emplace_back(Node(lhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));
		int right_child_idx = tree.size();
		tree.emplace_back(Node(rhsF, tree[node_idx].depth + 1, 0.0, 0.0, -1, 0.0, -1, -1));

		tree[node_idx].left = left_child_idx;
		tree[node_idx].right = right_child_idx;

		node_stack.push(left_child_idx);
		node_stack.push(right_child_idx);
	}else {
		tree[node_idx].val = this->compute_node_output(node_idx);
		tree[node_idx].IsLeaf = true;
	}
}

void TreeGenerator::find_varsplit_par_plus_feat() {
	tree.emplace_back(Node(this->idxs, 0, 0.0, 0.0, -1, 0.0, -1, -1));
	node_stack.push(0);
	
	while (!node_stack.empty()) {
		int node_idx = node_stack.top();
		node_stack.pop();
		this->find_greedy_split_par_plus_feat(node_idx);
	}
}

/* Cette fonction permet de realiser la prediction pour un arbre donné*/
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


/* Cette fonction est un utilitaire utilisé par la fonction predict ci haut pour recuperer la valeur  du noeud feuille 
à laquelle correspond un individus*/
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


/* Permet d'afficher notre arbre de décision créé*/
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


/* Nous permet de calculer la fonction sigmoid*/
double XGBoostClassifier::sigmoid(double x){
	//return  std::exp(x) / (1.0 + std::exp(x));
	return 1.0 / (1.0 + std::exp(-x));
}


/* Nous permet de calculer les gradients et les hessiens de chaque individus*/	
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


/* Nous permet au besoin de calculer notre modele initial f_0*/
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
		

/* Nous permet de tirer aleatoirement un pourcentage d'individus
 pour  creer un arbre donné*/
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


/*Cette fontion ce charge de creer notre modele
 XGOOST suivant l'algorithme choisit */
std::vector<double>  XGBoostClassifier::fit(std::vector<std::vector<double>>& x, std::vector<int>& y,double subsample_cols = 0.8 , int min_child_weight = 1, int depth = 8,int min_leaf = 1,double learning_rate = 0.4, int trees = 5,double lambda = 1, double gamma = 1,double eps=0.5,int num_threads=4,int choice=0,int num_bins=3){
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
	this->num_bins = num_bins;
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
	
	for(int treeUnit = 0; treeUnit < this->trees; treeUnit++){
		std::cout<<"Tree Number:"<<treeUnit<<std::endl;
		std::vector<double> Grad(x_size,0.0),Hess(x_size,0.0);
		grad_hess(this->base_pred,this->y,Grad,Hess);
		init_cpu_time();
		top1();
		TreeGenerator tree(this->x, Grad, Hess,root_idxs,this->subsample_cols, this->min_leaf,this->min_child_weight, this->depth, this->lambda, this->gamma,eps,num_threads,choice,this->num_bins);
		top2();
		long temps = cpu_time();
		temps_double_data += temps/1000.0 + (0.0001*(temps%1000));

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
		
		//std::cout<<"\n"<<loss_value<<"\n";
		loss.push_back(loss_value);
		y_pred.clear();
		Grad.clear();
		Hess.clear();
	}
	
	// Fermer le fichier
	csvFile.close();
	base_pred.clear();
	root_idxs.clear();

	
	return loss;
}


/* Nous permet de calculer les predictions de 
notre jeux de test en retournant les valeurs sous forme de probabilité*/
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

int main(int argc, char * argv[]) {
	init_cpu_time_2();
   
   	if(argc != 17){
		printf("Nombre d'argument incorrect\n");
		printf("Format de l'executable: <<nom fichier>> <<nombre-thread>> <<fichier-dataset>> <<fichier-labels>> <<train-test-percent>> <<subsample_cols>> <<min_child_weight>> <<depth>> <<min_leaf>> <<learning_rate>> <<trees>> <<lambda>> <<gamma>> <<epsilon>> <<ValeurNAOuPas>> <<choixAlgo>> <<choix algo>> <<numbins>>\n");
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
	int choice = atoi(argv[15]);
	int num_bins =  atoi(argv[16]);

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
	/*std::vector<std::vector<double>> x_train = extractCSVDataset("Datasets/xtrain.csv");
	std::vector<std::vector<double>> xTest = extractCSVDataset("Datasets/xtest.csv");

	std::vector<std::vector<double>> y_trainTemp = extractCSVDataset("Datasets/ytrain.csv");
	std::vector<std::vector<double>> y_testTemp = extractCSVDataset("Datasets/ytest.csv");

	std::vector<int> y_train;
	for(int i=0; i<(int)y_trainTemp.size();i++){
	 	y_train.push_back(((int)y_trainTemp[i][0]));
	}

	std::vector<int> yTest;
	for(int i=0; i<(int)y_testTemp.size();i++){
		yTest.push_back(((int)y_testTemp[i][0]));
	}*/
	// execution sequentielle
	std::cout<<"execution in progress..."<<std::endl;
	XGBoostClassifier xgb;
	top3();
	std::vector<double> loss = xgb.fit(x_train,y_train,subsample_cols_percent,min_child_weight_arg,
	depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,num_threads,0,num_bins);
	top4();
	
	long temps = cpu_time_2();
	printf("\nexecution time seq = %ld.%03ldms\n\n", temps/1000, temps%1000);
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
	
	
	std::string outputFileName= "outputSPeedUpPerThread.csv";
	std::ofstream csvFileSPeedUpPerThread(outputFileName, std::ios::out);

	// Check if the file was opened successfully
	if (!csvFileSPeedUpPerThread.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	
	std::ofstream csvFileLossPar("outputCsvFileLossPar.csv", std::ios::out);

	// Check if the file was opened successfully
	if (!csvFileLossPar.is_open()){
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	
	// Write the header row
	csvFileSPeedUpPerThread << "NumThread,ParallelPerSplitPoint,ParallelPerFeature" << std::endl;
	csvFileLossPar<<"NumTree,LossParFeat,LossParSplitPoint"<<std::endl;
	
	for(int i=2;i<=num_threads;i+=2){
		std::cout<<"parallel execution using "<<i<<" threads"<<std::endl;
		// execution parallel par split point
		std::cout<<"parallel execution per split point in progress..."<<std::endl;
		XGBoostClassifier xgb1;
		top3();
		std::vector<double> loss1 =xgb1.fit(x_train, y_train,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,i,1,num_bins);
		top4();

		long temps1 = cpu_time_2();
		printf("\ntime par split point = %ld.%03ldms\n\n", temps1/1000, temps1%1000);
		
		// execution parallel par feature
		std::cout<<"parallel execution per feature in progress..."<<std::endl;
		XGBoostClassifier xgb2;
		top3();
		std::vector<double> loss2 =xgb2.fit(x_train, y_train,subsample_cols_percent,min_child_weight_arg,depth_arg,min_leaf_arg,learning_rate_arg,num_trees,lambda_arg, gamma_arg,eps,i,2,num_bins);
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
		loss1.clear();
		loss2.clear();
	}
	csvFileLossPar.close();
	csvFileSPeedUpPerThread.close();
	
	//prediction des donnees de test
	std::vector<double>predicted_labels= xgb.predict_proba(xTest);
	int confusionSize = 4;

	std::ofstream logFileResult("resultLog.txt", std::ios::app);
	// Check if the file was opened successfully
	if(!logFileResult.is_open()) {
		std::cerr << "Error: Unable to open the CSV file." << std::endl;
		return 1;
	}
	std::vector<int> confusion_matrix(confusionSize,0);
	confusion_matrix = calculate_confusion_matrix(yTest, predicted_labels, yTest.size(), confusion_matrix);

	logFileResult<<argv[0]<<" "<<num_threads<<" "<<filename_dataset<<" "<<filename_labels<<" "<<train_test_percent<<" "<<subsample_cols_percent<<" "<<min_child_weight_arg<<" "<<depth_arg<<" "<<min_leaf_arg<<" "<<learning_rate_arg<<" "<<num_trees<<" "<<lambda_arg<<" "<<gamma_arg<<" "<<eps<<" "<<nanOrNot<<" "<<choice<<" "<<num_bins<<std::endl;

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
	logFileResult<<"execution time:"<< temps/1000<<"."<<temps%1000<<"ms"<<std::endl;
	logFileResult<<"-----------------------------------------------------------------------------------------"<< std::endl;
	logFileResult.close();

    return 0;
}
