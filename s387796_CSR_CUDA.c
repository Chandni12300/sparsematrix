#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

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
__global__ void matrix_multiplication_kernel(int *row_ptr, int *col_idx, double *values, double *x, double *y, int num_rows);


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
void convert_to_csr(matrix_t *matrix, int **row_ptr, int **col_ind, double **values, int *max_nnz) {

    // Allocate memory for row_ptr
    *row_ptr = (int *) malloc((matrix->rows + 1) * sizeof(int));
    memset(*row_ptr, 0, (matrix->rows + 1) * sizeof(int));

    // Compute number of non-zero elements in each row
    for (int i = 0; i < matrix->nnz; i++) {
        (*row_ptr)[matrix->row_ind[i] + 1]++;
    }

    // Compute cumulative sum of non-zero elements in each row
    for (int i = 1; i <= matrix->rows; i++) {
        (*row_ptr)[i] += (*row_ptr)[i-1];
    }

    // Allocate memory for col_ind and values
    *col_ind = (int *) malloc(matrix->nnz * sizeof(int));
    *values = (double *) malloc(matrix->nnz * sizeof(double));

    // Copy data from COO format to CSR format
    int *current = (int *) calloc(matrix->rows, sizeof(int));
    for (int i = 0; i < matrix->nnz; i++) {
        int row = matrix->row_ind[i];
        int index = (*row_ptr)[row] + current[row];
        (*col_ind)[index] = matrix->col_ind[i];
        (*values)[index] = matrix->values[i];
        current[row]++;
    }
    free(current);
}
__global__ void matrix_multiplication_kernel(int *row_ptr, int *col_idx, double *values, double *x, double *y, int num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_rows)
    {
     	double sum = 0.0;
        int start = row_ptr[row];
        int end = row_ptr[row+1];

        for (int i = start; i < end; i++)
        {
            int col = col_idx[i];
            double val = values[i];
            sum += val * x[col];
        }

	y[row] = sum;
    }
}

void matrix_multiplication(matrix_t *matrix, double *x, double *y)
{
    int num_rows = matrix->rows;

    // Allocate memory on device
    int *d_row_ind, *d_col_ind;
    double *d_values, *d_x, *d_y;

    cudaMalloc(&d_row_ind, matrix->nnz * sizeof(int));
    cudaMalloc(&d_col_ind, matrix->nnz * sizeof(int));
    cudaMalloc(&d_values, matrix->nnz * sizeof(double));
    cudaMalloc(&d_x, matrix->cols * sizeof(double));
    cudaMalloc(&d_y, matrix->rows * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(d_row_ind, matrix->row_ind, matrix->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, matrix->col_ind, matrix->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, matrix->values, matrix->nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, matrix->cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, matrix->rows * sizeof(double), cudaMemcpyHostToDevice);

    // Set the number of blocks and threads
    int threads_per_block = 256;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    // Call the kernel

//struct timeval start_time, end_time;
//gettimeofday(&start_time, NULL);

    matrix_multiplication_kernel<<<blocks_per_grid, threads_per_block>>>(d_row_ind, d_col_ind, d_values, d_x, d_y, num_rows);

//gettimeofday(&end_time, NULL);

//double kernel_time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;


    // Copy the result from device to host
    cudaMemcpy(y, d_y, matrix->rows * sizeof(double), cudaMemcpyDeviceToHost);

//printf("Time taken for CUDA KERNEL:: %lf  \n", kernel_time_taken);
//double flops = (2*matrix->nnz)/kernel_time_taken;
//printf("FLOPS:%.2lf\n",flops);




    // Free memory on device and host
    cudaFree(d_row_ind);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}
void multiply_csr(matrix_t matrix,  double *x,  double *z) {
     int i, j;

    // Initialize y to all zeros
    memset(z, 0, matrix.rows * sizeof( double));

 // Perform sparse matrix-vector multiplication

    for (i = 0; i < matrix.rows; i++) {
        for (j = matrix.row_ind[i]; j < matrix.row_ind[i+1]; j++) {
            z[i] += matrix.values[j] * x[matrix.col_ind[j]];
        }
    }
}
int main() {

matrix_t matrix;
matrix_market("/scratch/s387796/Cube_Coup_dt0.mtx", &matrix);
int *row_ptr, *col_ind;
double *values;
int max_nnz;
convert_to_csr(&matrix, &row_ptr, &col_ind, &values, &max_nnz);
matrix.row_ind = row_ptr;
matrix.col_ind = col_ind;
matrix.values = values;

// Create input vector x and output vector y
double *x = (double *) malloc(matrix.rows * sizeof(double));
double *y = (double *) malloc(matrix.rows * sizeof(double));
double *z = (double *) malloc(matrix.rows * sizeof(double));

    for (int i = 0; i < matrix.rows; i++) {
        x[i] = (double) i; // Initialize x to 0 to n
    }

    //clock_t start, stop;

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    //start = clock();

    matrix_multiplication(&matrix, x, y);

    gettimeofday(&end_time, NULL);

    double kernel_time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;


struct timeval start, end;
gettimeofday(&start, NULL);

multiply_csr(matrix, x, z);

gettimeofday(&end, NULL);
double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

printf("Elapsed time for Serial (CSR): %lf seconds\n", elapsed_time);
double flops1 = (2*matrix.nnz)/elapsed_time;
printf("FLOPS Serial (CSR) :%.2lf\n",flops1);

printf("Time taken for CUDA KERNEL (CSR) : %lf  \n", kernel_time_taken);
double flops = (2*matrix.nnz)/kernel_time_taken;
printf("FLOPS for CUDA (CSR):%.2lf\n",flops);

/*
// Print the result
printf("Result:\n");
for (int i = 0; i < matrix.rows; i++) {
    printf("%.6f ", y[i]);
}
printf("\n");
*/

    int valid = 0;
    for (int i=0; i < matrix.rows; i++){
        if(fabs(z[i] - y[i]) < 5000  ){
            valid ++ ;

        }
    }
    if(valid == matrix.rows){
        printf("\nVECTOR RESULT VALIDATED SUCESSFULLY\n");
    }
    else{
	printf("\nNOT VALIDATED\n");
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