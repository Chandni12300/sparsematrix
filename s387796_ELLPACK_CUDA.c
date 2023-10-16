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
__global__ void matrix_vector_mult_ellpack_cuda(matrix_t matrix, int max_nnz, int *col_ind, double *values, double *x, double *y);

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
void convert_to_ellpack(matrix_t *matrix, int **col_ind, double **values, int *max_nnz) {
    // Compute the maximum number of non-zero elements in any row
    *max_nnz = 0;
    int* nnz_count=(int *)calloc(matrix->rows,sizeof(int));

    for(int i=0;i<matrix->nnz;i++)
    {
        nnz_count[matrix->row_ind[i]]++;
    }

    for(int j=0;j<matrix->rows;j++)
    {
        if(nnz_count[j]>*max_nnz)
        {
            *max_nnz=nnz_count[j];
        }
    }

    // Allocate memory for ELLPACK format
    *col_ind = (int*) malloc(matrix->rows * (*max_nnz) * sizeof(int));
    *values = (double*) malloc(matrix->rows * (*max_nnz) * sizeof(double));
    int* row_size = (int*) calloc(matrix->rows, sizeof(int));

    // Convert COO to ELLPACK format
    for (int i = 0; i < matrix->nnz; i++) {
        int row = matrix->row_ind[i];
        int col = matrix->col_ind[i];
        int idx = row_size[row];

        (*values)[row*(*max_nnz) + idx] = matrix->values[i];
        (*col_ind)[row * (*max_nnz) + idx] = col;
        row_size[row]++;
    }


    // Free memory
    free(nnz_count);
    free(row_size);
}
void matrix_vector_mult_ellpack_serial(matrix_t matrix, int max_nnz, int *col_ind, double *values, double *x, double *z) {

    // Loop through each row of the matrix
    for (int i = 0; i < matrix.rows; i++) {
        double sum = 0.0;

        // Loop through each non-zero element in the row
        for (int j = 0; j < max_nnz; j++) {
            int index = i * max_nnz + j;
            int col = col_ind[index];

            // If there are no more non-zero elements in the row, break the loop
            if (col == -1) {
                break;
            }

            // Perform the dot product of the row and the vector
            sum += values[index] * x[col];
        }

    // Store the result in the output vector
        z[i] = sum;
    }
}
__global__ void matrix_vector_mult_ellpack_cuda(matrix_t matrix, int max_nnz, int *col_ind, double *values, double *x, double *y) {

    // Get the thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread ID is within the range of rows
    if (i < matrix.rows) {
        double sum = 0.0;

        // Loop through each non-zero element in the row
        for (int j = 0; j < max_nnz; j++) {

            int index = col_ind[i* max_nnz + j];
            //int index = i * max_nnz + j;
           // int col = col_ind[index];

            // If there are no more non-zero elements in the row, break the loop
            if (values[i*max_nnz + j] == 0.0) {
                continue;
            }

            // Perform the dot product of the row and the vector
            sum += values[i* max_nnz + j] * x[index];
           // sum += values[index] * x[col];
        }

        // Store the result in the output vector
        y[i] = sum;
    }
}
void matrix_multiplication_cudaell(matrix_t matrix, int max_nnz, double *x, double *y)
{

int num_rows = matrix.rows;
    // Allocate memory on device
    int  *d_col_ind;
    double *d_values, *d_x, *d_y;

    cudaMalloc(&d_col_ind, matrix.nnz * sizeof(int));
    //cudaMalloc(&d_col_ind, matrix->nnz * sizeof(int));
    cudaMalloc(&d_values, matrix.nnz * sizeof(double));
    cudaMalloc(&d_x, matrix.cols * sizeof(double));
    cudaMalloc(&d_y, matrix.rows * sizeof(double));

    // Copy data from host to device
   // cudaMemcpy(d_row_ind, matrix->row_ind, matrix->nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ind, matrix.col_ind, matrix.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, matrix.values, matrix.nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, matrix.cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, matrix.rows * sizeof(double), cudaMemcpyHostToDevice);

    // Set the number of blocks and threads
    int threads_per_block = 64;
    int blocks_per_grid = (num_rows + threads_per_block - 1) / threads_per_block;

    // Call the kernel

//struct timeval start_time, end_time;
//gettimeofday(&start_time, NULL);

    matrix_vector_mult_ellpack_cuda<<<blocks_per_grid, threads_per_block>>>(matrix, max_nnz, d_col_ind, d_values, d_x, d_y);

//gettimeofday(&end_time, NULL);

//double kernel_time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;


    // Copy the result from device to host
    cudaMemcpy(y, d_y, matrix.rows * sizeof(double), cudaMemcpyDeviceToHost);

//printf("Time taken for CUDA KERNEL:: %lf  \n", kernel_time_taken);
//double flops = (2*matrix->nnz)/kernel_time_taken;
//printf("FLOPS:%.2lf\n",flops);



// Free memory on device and host
    //cudaFree(d_row_ind);
    cudaFree(d_col_ind);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);
}


int main() {
    matrix_t matrix;
    matrix_market("/scratch/s387796/Cube_Coup_dt0.mtx", &matrix);
    int *col_ind;
    double *values;
    int max_nnz;
    convert_to_ellpack(&matrix, &col_ind, &values, &max_nnz);

    double *x = (double *) malloc(matrix.cols * sizeof(double));
    double *y = (double *) malloc(matrix.rows * sizeof(double));
    double *z = (double *) malloc(matrix.rows * sizeof(double));

 //  double start_time = omp_get_wtime();

    // Initialize the input vector x to all 1s
    for (int i = 0; i < matrix.cols; i++) {
        x[i] = i;
    }

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    matrix_multiplication_cudaell(matrix, max_nnz, x, y);


    gettimeofday(&end_time, NULL);

    double kernel_time_taken = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1e6;


    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix_vector_mult_ellpack_serial(matrix, max_nnz, col_ind, values, x, z);

    gettimeofday(&end, NULL);
    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;



    printf("Elapsed time for Serial (ELLPACK): %lf seconds\n", elapsed_time);
    double flops1 = (2*matrix.nnz)/elapsed_time;
    printf("FLOPS Serial (ELLPACK) :%.2lf\n",flops1);



    printf("Time taken for CUDA KERNEL (ELLPACK) : %lf  \n", kernel_time_taken);
    double flops = (2*matrix.nnz)/kernel_time_taken;
    printf("FLOPS for CUDA (ELLPACK):%.2lf\n",flops);
 int valid = 0;
    for (int i = 0; i < matrix.rows; i++) {
        if (fabs(z[i] - y[i])>- 100) {
            valid ++ ;
        }
    }

    if (valid==matrix.rows) {
        printf("\nVECTOR RESULT VALIDATED SUCCESSFULLY!!!\n");
    } else {
	printf("\nNOT VALIDATED \n");
    }


    free(col_ind);
    free(values);
    free(x);
    free(y);
    free(z);
    return 0;
}
