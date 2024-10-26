#ifndef BITONIQ_H
#define BITONIQ_H

typedef std::vector<double> vector_t;

vector_t allocate_vector(int n);
void generate_vector_values(vector_t& vec, int n);
void print_vec_portion(vector_t vec, int begin, int end);
void print_vec(vector_t vec, int n);
void compare_and_swap (vector_t& vec, int i, int j, int dir);
void sort_a_bitonique_suite(vector_t& vec, int begin, int n, int dir);
void sort_arbitrary_suite(vector_t& vec, int begin, int n, int dir);
void tri_fusion(vector_t& arr, int l, int r);
void fusion(vector_t& arr, int l, int m, int r);
int plus_grande_puissance_de_deux(int n);
void tri_bitonique_fusion(vector_t& tab_init, int taille_tab_init);

#endif // BITONIQ_H
