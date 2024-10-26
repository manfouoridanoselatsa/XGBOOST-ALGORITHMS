#ifndef MYCLASS_H
#define MYCLASS_H

typedef struct{
    double value;
    int idx;
}feature_val;

typedef struct{
	int bin_id;
	std::vector<int> list_idxs_bin;
	double gradient_sum_bin;
	double gradient_sum_left;
	double gradient_sum_right;
    double hessian_sum_bin;
	double hessian_sum_left;
	double hessian_sum_right;
	//double score_left;
    //double score_right;
    double gain;
    double bin_val_start;
    double bin_val_ending;

}Histogram;

class Node{
	public: 
	      //constructeur de ma classe
	    Node(std::vector<int>&  idxs,int depth,double val,double score ,int var_idx,double split,int left,int right, bool IsLeaf);
		
	std::vector<int>  idxs;
        int depth;
        double val;
        double score;
        int var_idx ;
        double split;
        int left;
        int right;
    	bool IsLeaf;
    	~Node() {};
};



//void find_greedy_split_par(split_data *data);

class TreeGenerator{

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
		int num_threads;
		int num_bins;
		int depth;
		int choice;
		std::vector<int> idxs;
		int min_child_weight;
		std::vector<int> lhsF;
		std::vector<int> rhsF;
	    double eps;
		std::vector<std::vector<double>> x;
		TreeGenerator(std::vector<std::vector<double>>& x, std::vector<double>&  gradient,std::vector<double>& hessian,std::vector<int>&  idxs, double subsample_cols, int min_leaf,int min_child_weight ,int depth,double lambda,double gamma,double eps,int num_thread,int choice,int num_bins);
		void find_greedy_split_par(int node_idx);
		void find_greedy_split_seq(int node_idx);
		void find_greedy_split_par_feat(int node_idx);
		void find_greedy_split_par_node(int node_idx);
		void find_greedy_split_par_plus_feat(int node_idx);
		void Histogram_Finding_Algorithm(int node_idx,int num_bins);

		std::vector<int> col_subsample(bool random_col);
		inline double compute_node_output(int &node_idx);
		void find_varsplit_par();
		void find_varsplit_seq();
		void find_varsplit_par_feat();
		void find_varsplit_par_node();
		void find_varsplit_par_plus_feat();
		void find_varsplit_Histogram();
		double gain(double &s_Gg, double &s_Hg, double &s_Gr, double &s_Hr);
		std::vector<double> predict(std::vector<std::vector<double>>& x);
		double predict_row(std::vector<double>& xi);
		void printNode();
		//TreeGenerator &operator=(const TreeGenerator &p);	
	
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
	   int min_leaf;
	   int num_threads;
	   int num_bins;
	   double learning_rate;
	   int trees;
	   double lambda;
	   double gamma ;
	   std::vector<double> base_pred;
	
	public:
		double sigmoid(double x);
		void grad_hess(std::vector<double>& preds,std::vector<int>& labels,std::vector<double> & grads,std::vector<double> & hess);
		double log_odds(std::vector<int> labels);
		void observation_subsample(std::vector<std::vector<double>>& x,std::vector<int>& y,double subsample_ratio,bool random_sample);
		std::vector<double> fit(std::vector<std::vector<double>>& x, std::vector<int>& y,double subsample_cols, int min_child_weight, int depth,int min_leaf,double learning_rate, int trees,double lambda, double gamma,double eps,int num_threads,int choice,int num_bins);
		std::vector<double> predict_proba(std::vector<std::vector<double>> x);
		~XGBoostClassifier() {};
};
#endif // MYCLASS_H
