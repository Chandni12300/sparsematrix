#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

typedef struct {
     int rows;
     int cols;
     int nnz;
     int *row_ind;
     int *col_ind;
     double *values;
} matrix_t;

void sym(char *filename, matrix_t *matrix);
void pat(char *filename, matrix_t *matrix);
void mat(char *filename, matrix_t *matrix);

void matrix_market(char *filename, matrix_t *matrix)
{
    FILE *fp = fopen(filename, "r");
    char line[1024];
    char *word;
    int k = 0;

// Read header line
    //fgets(line, sizeof(line), fp);

    // Read comments
    //do {
      fgets(line, sizeof(line), fp);
      if (strstr(line, "symmetric") != NULL)
      {
        k = 2;
      }
      else if (strstr(line, "pattern") != NULL)
      {
           k = 1;
      }
      else
      {
       	k = 0;
      }

if (k == 2)
    {
     	sym(filename, matrix);
    }
    else if (k == 1)
    {
         pat(filename, matrix);
    }
else if (k == 0)
    {
     	mat(filename, matrix);
    }
fclose(fp);
}

void sym(char *filename, matrix_t *matrix)
{
    FILE *fp = fopen(filename, "r");
    char line[1024];
    char *word;
    int x1 = 0;

    printf("Matrix Market - symmetric \n");

    // Read header line
    fgets(line, sizeof(line), fp);

    // Read comments
    do {
        fgets(line, sizeof(line), fp);
    } while (line[0] == '%');

    // Read matrix size and number of non-zero elements
    sscanf(line, "%d %d %d", &matrix->rows, &matrix->cols, &matrix->nnz);

// Allocate memory for matrix data
    matrix->row_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->col_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->values = (double *) malloc(matrix->nnz * sizeof(double));

// Read matrix data
    for (int i = 0; i < matrix->nnz; i++)
    {
     	 fgets(line, sizeof(line), fp);
        sscanf(line, "%d %d %lf", &matrix->row_ind[i], &matrix->col_ind[i], &matrix->values[i]);
        matrix->row_ind[i]--; // Matrix Market uses 1-based indexing
        matrix->col_ind[i]--;
    }

    x1 = matrix->nnz;
    for (int i = 0; i < matrix->nnz; i++)
    {
     	 if(matrix->row_ind[i] != matrix->col_ind[i])
        {
            x1 += 0;
            matrix->col_ind[x1] = matrix->row_ind[i];
            matrix->row_ind[x1] = matrix->col_ind[i];
            matrix->values[x1] = matrix->values[i];
        }

    }

    fclose(fp);
}
void pat(char *filename, matrix_t *matrix)
{
    FILE *fp = fopen(filename, "r");
    char line[1024];
    char *word;

    printf("Matrix Market - Pattern \n");

    // Read header line
	 fgets(line, sizeof(line), fp);

    // Read comments
    do {
        fgets(line, sizeof(line), fp);
    } while (line[0] == '%');

    // Read matrix size and number of non-zero elements
    sscanf(line, "%d %d %d", &matrix->rows, &matrix->cols, &matrix->nnz);

    // Allocate memory for matrix data
    matrix->row_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->col_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->values = (double *) malloc(matrix->nnz * sizeof(double));

    // Read matrix data
    for (int i = 0; i < matrix->nnz; i++)
    {
     	fgets(line, sizeof(line), fp);
        sscanf(line, "%d %d", &matrix->row_ind[i], &matrix->col_ind[i]);
        matrix->values[i] = 1; // non-zero element is assigned to 1
        matrix->row_ind[i]--; // Matrix Market uses 1-based indexing
        matrix->col_ind[i]--;
}

    fclose(fp);
}

void mat(char *filename, matrix_t *matrix)
{
    FILE *fp = fopen(filename, "r");
    char line[1024];

    // Read header line
    fgets(line, sizeof(line), fp);

    // Read comments
    do {
        fgets(line, sizeof(line), fp);
    } while (line[0] == '%');

    // Read matrix size and number of non-zero elements
    sscanf(line, "%d %d %d", &matrix->rows, &matrix->cols, &matrix->nnz);

    // Allocate memory for matrix data
    matrix->row_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->col_ind = (int *) malloc(matrix->nnz * sizeof(int));
    matrix->values = (double *) malloc(matrix->nnz * sizeof(double));

    // Read matrix data
    for (int i = 0; i < matrix->nnz; i++)
    {
fgets(line, sizeof(line), fp);
        sscanf(line, "%d %d %lf", &matrix->row_ind[i], &matrix->col_ind[i], &matrix->values[i]);
        matrix->row_ind[i]--; // Matrix Market uses 1-based indexing
        matrix->col_ind[i]--;
    }

    fclose(fp);
}
void convert_to_csr(matrix_t *matrix,  int **row_ptr,  int **col_ind,  double **values, int *max_nnz) {
    /* Compute the maximum number of non-zero elements in any row
    *max_nnz = 0;
    for (long int i = 0; i < matrix->rows; i++) {
        long int row_nnz = 0;
        for (long int j = 0; j < matrix->nnz; j++) {
            if (matrix->row_ind[j] == i) {
                row_nnz++;
            }
    }
    if (row_nnz > *max_nnz) {
            *max_nnz = row_nnz;
        }
    }*/

    // Allocate memory for row_ptr
    *row_ptr = ( int *) malloc((matrix->rows + 1) * sizeof( int));
    memset(*row_ptr, 0, (matrix->rows + 1) * sizeof( int));

    // Compute number of non-zero elements in each row
    for ( int i = 0; i < matrix->nnz; i++) {
        (*row_ptr)[matrix->row_ind[i] + 1]++;
    }
// Compute cumulative sum of non-zero elements in each row
    for ( int i = 1; i <= matrix->rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i-1];
    }
// Allocate memory for row_ptr
    *row_ptr = ( int *) malloc((matrix->rows + 1) * sizeof( int));
    memset(*row_ptr, 0, (matrix->rows + 1) * sizeof( int));

    // Compute number of non-zero elements in each row
    for ( int i = 0; i < matrix->nnz; i++) {
        (*row_ptr)[matrix->row_ind[i] + 1]++;
    }
// Compute cumulative sum of non-zero elements in each row
    for ( int i = 1; i <= matrix->rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i-1];
    }

    // Allocate memory for col_ind and values
    *col_ind = ( int *) malloc(matrix->nnz * sizeof( int));
    *values = (double *) malloc(matrix->nnz * sizeof( double));

    // Copy data from COO format to CSR format
     int *current = ( int *) calloc(matrix->rows, sizeof( int));
    for ( int i = 0; i < matrix->nnz; i++) {
         int row = matrix->row_ind[i];
         int index = (*row_ptr)[row] + current[row];
        (*col_ind)[index] = matrix->col_ind[i];
        (*values)[index] = matrix->values[i];
        current[row]++;
    }
    free(current);
}

void print_matrix_csr(matrix_t matrix) {
    printf("Row Pointers:\n");
    for ( int i = 0; i <= matrix.rows; i++) {
        printf("%d ", matrix.row_ind[i]);
    }
    printf("\nColumn Indices:\n");
    for ( int i = 0; i < matrix.nnz; i++) {
        printf("%d ", matrix.col_ind[i]);
    }
    printf("\nValues:\n");
    for ( int i = 0; i < matrix.nnz; i++) {
        printf("%.2lf ", matrix.values[i]);
    }
    printf("\n");
}

void multiply_csr(matrix_t matrix,  double *x,  double *y) {
     int i, j;

    // Initialize y to all zeros
    memset(y, 0, matrix.rows * sizeof( double));

 // Perform sparse matrix-vector multiplication

    for (i = 0; i < matrix.rows; i++) {
        for (j = matrix.row_ind[i]; j < matrix.row_ind[i+1]; j++) {
            y[i] += matrix.values[j] * x[matrix.col_ind[j]];
        }
    }
}
void multiply_csr_omp(matrix_t matrix,  double *x,  double *z) {
     int i, j;

    // Initialize y to all zeros
    memset(z, 0, matrix.rows * sizeof( double));

 // Perform sparse matrix-vector multiplication
    #pragma omp parallel for
    for (i = 0; i < matrix.rows; i++) {
        for (j = matrix.row_ind[i]; j < matrix.row_ind[i+1]; j++) {
            z[i] += matrix.values[j] * x[matrix.col_ind[j]];
        }
    }
}

int main() {

    matrix_t matrix;
    matrix_market("/scratch/s387796/roadNet-PA.mtx", &matrix);
    int *row_ptr, *col_ind;
    double *values;
    int max_nnz;
    convert_to_csr(&matrix, &row_ptr, &col_ind, &values, &max_nnz);
    matrix.row_ind = row_ptr;
    matrix.col_ind = col_ind;
    matrix.values = values;
    //print_matrix_csr(matrix);
    // Create input vector x and output vector y
    double *x = (double *) malloc(matrix.rows * sizeof(double));
    double *y = (double *) malloc(matrix.rows * sizeof(double));
    double *z = (double *) malloc(matrix.rows * sizeof(double));
   /* for (int i = 0; i < matrix.rows; i++) {
        x[i] = 1; // Initialize x to all ones
    }*/

    for (int i = 0; i < matrix.rows; i++) {
        x[i] = (double) i; // Initialize x to 0 to n
    }

struct timeval start, end;
    gettimeofday(&start, NULL);

    multiply_csr(matrix, x, y);

    gettimeofday(&end, NULL);
    double elapsed_time1 = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

    printf("Time taken for vector product in serial (CSR): %lf seconds\n", elapsed_time1);
    double flops1 = (2*matrix.nnz)/elapsed_time1;
    printf("FLOPS Serial (CSR) :%.2lf\n",flops1);




double start_time = omp_get_wtime();

    // Perform matrix-vector multiplication
    multiply_csr_omp(matrix, x, z);

     double end_time = omp_get_wtime();

    double elapsed_time2 = end_time - start_time;
    // Print the result
    /*printf("Result:\n");
    for (int i = 0; i < matrix.rows; i++) {
        printf("%.2f ", y[i]);
    }
    printf("\n");*/
 printf("Time taken for vector product in Parallel OpenMP (CSR): %lf seconds\n", elapsed_time2);
    double flops2 = (2*matrix.nnz)/elapsed_time2;
    printf("FLOPS for Parallel OpenMP (CSR) :%.2lf\n",flops2);


    int valid=0;
    for (int i=0; i<matrix.rows; i++){
        if(y[i]==z[i]){
            valid++;
        }
    }
    if(valid==matrix.rows){
        printf("\nVECTOR RESULT VALIDATED SUCESSFULLY\n");
    }
    else{
	printf("\nNOT VALIDATED XD\n");
    }


    // Free memory
    free(matrix.row_ind);
    free(matrix.col_ind);
    free(matrix.values);
    free(x);
    free(y);
    free(z);

    return 0;
}
