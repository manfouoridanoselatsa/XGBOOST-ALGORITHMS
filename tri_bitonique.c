#include<pthread.h>
#include<stdlib.h>
#include<stdio.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include "bitoniq.h"
#include <math.h>
#define NUM_THREADS 2

#define MAX_VAL 50000
#define MIN_VAL 1
#define ASC_ORDER 1
#define DES_ORDER 2


//typedef double* vector_t;
//typedef std::vector<double> vector_t;

vector_t allocate_vector(int n)
{
	vector_t vec(n);
	
	return vec;
}

void generate_vector_values(vector_t& vec, int n)
{
	int i;
	
	//Random seed
	srandom(time(0)+clock()+random());
	#pragma omp parallel for
	for(i=0; i<n; i++)
	{
		vec[i] = rand() % MAX_VAL + MIN_VAL;
	}
}

void print_vec_portion(vector_t vec, int begin, int end)
{
	int i;
	printf("Printing vector [%d, %d]: \n", begin, end);
	for(i=begin; i<=end; i++)
	{	
		printf("%f ",vec[i]);
	}
		printf("\n");
}

void print_vec(vector_t vec, int n)
{
	int i;
	printf("Printing vector elements\n");
	for(i=0; i<n; i++)
	{	
		printf("%f ",vec[i]);
	}
		printf("\n");
}

void compare_and_swap (vector_t& vec, int i, int j, int dir){

	double temp;
	if (((vec[i]<vec[j]) && (dir == DES_ORDER)) || ((vec[i]>vec[j]) && (dir == ASC_ORDER)))
	{
		temp = vec[i];
		vec[i] = vec[j];
		vec[j] = temp;
	}
}

void sort_a_bitonique_suite(vector_t& vec, int begin, int n, int dir)
{
	int i;
	if(n>1){
		for(i=begin;i<(begin+n/2);i++)
		{
			compare_and_swap(vec, i, i+n/2, dir);
		}

		sort_a_bitonique_suite(vec, begin, n/2, dir);

		sort_a_bitonique_suite(vec, begin+n/2, n/2, dir);
	}
}

void sort_arbitrary_suite(vector_t& vec, int begin, int n, int dir)
{
	if(n>1){
		sort_arbitrary_suite(vec, begin, n/2, ASC_ORDER);

		sort_arbitrary_suite(vec, begin+n/2, n/2, DES_ORDER);

		sort_a_bitonique_suite(vec, begin, n, dir);
	}

}

void tri_fusion(vector_t& arr, int l, int r)
{
    if (l < r) {
        int m = l + (r - l) / 2;
  
        tri_fusion(arr, l, m);
        tri_fusion(arr, m + 1, r);
        fusion(arr, l, m, r);
    }
}

void fusion(vector_t& arr, int l, int m, int r)
{
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    //  create temp arrays 
    vector_t L(n1);
    vector_t R(n2);

    //  Copy data to temp arrays L[] and R[] 
    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    //  Merge the temp arrays back into arr[l..r]
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    //  Copy the remaining elements of L[], if there     are any
    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

  //  Copy the remaining elements of R[], if there     are any 
    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

int plus_grande_puissance_de_deux(int n)
{
	int k, k_precedent, deux_puissance_k;

	k = 0;
	k_precedent = 0;
	deux_puissance_k = 1;

	while(deux_puissance_k <= n)
	{
		k_precedent = k;
		k = k+1;
		deux_puissance_k = deux_puissance_k * 2;
	}
	return k_precedent;
}


void tri_bitonique_fusion(vector_t& tab_init, int taille_tab_init )
{
	//float k_f = log(taille_tab_init);
	//printf("log(taille) = %f\n",k_f);
	int k = plus_grande_puissance_de_deux(taille_tab_init);

	//int i, j;
	//printf("le ln de n vaut %d \n", k );
	int taille_bito = pow(2, k);
	//printf("taille_bito = %d \n", taille_bito );

	sort_arbitrary_suite(tab_init, 0, taille_bito, ASC_ORDER);
	tri_fusion(tab_init, taille_bito, taille_tab_init-1);
	fusion(tab_init, 0, taille_bito-1, taille_tab_init-1);
}



/*
int main(int argc, char *argv[])
{
	long i;
	int par;
	double elapsed;
	struct timeval t0, t1;
	int n = 16;

	printf("Enter vector size :  ");
	scanf("%d",&n);

	vector_t tab = allocate_vector(n);
	generate_vector_values(tab, n);
	print_vec(tab, n);


	gettimeofday(&t0, 0);

//	sort_arbitrary_suite(tab, 0, n, ASC_ORDER);
	tri_bitonique_fusion(tab, n);


	gettimeofday(&t1, 0);
	print_vec(tab, n);
	elapsed = (t1.tv_sec-t0.tv_sec) * 1.0f + (t1.tv_usec - t0.tv_usec) / 1000000.0f;
	printf(" in time %f s\n", elapsed);


	pthread_exit(NULL);
}*/
