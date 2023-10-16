#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#include<math.h>

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
    int x = 0;

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

    x = matrix->nnz;
    for (int i = 0; i < matrix->nnz; i++)
    {
     	if(matrix->row_ind[i] != matrix->col_ind[i])
        {
            x += 0;
            matrix->col_ind[x] = matrix->row_ind[i];
            matrix->row_ind[x] = matrix->col_ind[i];
            matrix->values[x] = matrix->values[i];
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
void print_matrix_ellpack(matrix_t matrix, int max_nnz, int *col_ind, double *values) {
    int max_row_nnz = 0; // maximum number of non-zero elements in any row
    //perform sparse matrix-vector multiplication
    // Loop through each row and count the number of non-zero elements

    for (int i = 0; i < matrix.rows; i++) {
        int row_nnz = 0;
        for (int j = 0; j < max_nnz; j++) {
            int index = i * max_nnz + j;
            if (col_ind[index] == -1) {
                break;
            }
            row_nnz++;
        }
    if (row_nnz > max_row_nnz) {
            max_row_nnz = row_nnz;
        }
    }
    printf("Maximum number of non-zero elements in any row: %d\n", max_row_nnz);
    printf("Column Indices and Values:\n");
}

void matrix_vector_mult_ellpack(matrix_t matrix, int max_nnz, int *col_ind, double *values, double *x, double *y) {

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
        y[i] = sum;
    }
}
void matrix_vector_mult_ellpack_omp(matrix_t matrix, int max_nnz, int *col_ind, double *values, double *x, double *z) {
      #pragma omp parallel for
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
int main() {
    matrix_t matrix;
    matrix_market("/scratch/s387796/thermal2.mtx", &matrix);
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

    struct timeval start, end;
    gettimeofday(&start, NULL);

    matrix_vector_mult_ellpack(matrix, max_nnz, col_ind, values, x, y);

    gettimeofday(&end, NULL);
    double elapsed_time1 = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;

 double start_time = omp_get_wtime();

    matrix_vector_mult_ellpack_omp(matrix, max_nnz, col_ind, values, x, z);

double end_time = omp_get_wtime();
    double elapsed_time2 = end_time - start_time;

   /* printf("Output vector:\n");
    for (int i = 0; i < matrix.rows; i++) {
        printf("%.6lf\n", y[i]);
    } */

    printf("Elapsed time for Serial: %lf seconds\n", elapsed_time1);
    double flops1 = (2*matrix.nnz)/elapsed_time1;
    printf("FLOPS Serial:%.2lf\n",flops1);

    printf("Elapsed time for OpenMP Parallel: %lf seconds\n", elapsed_time2);
    double flops2 = (2*matrix.nnz)/elapsed_time2;
    printf("FLOPS OpenMP Parallel:%.2lf\n",flops2);
    int valid=0;
    for(int i=0; i<matrix.rows;i++){
    //if (y[i]=z[i]){
if(fabs(y[i]-z[i])<100){

valid++;
}

}

if (valid == matrix.rows){
printf("\nVECTOR RESULT VALIDATED SUCESFULLY\n");
}
else{
printf("\nNOT VALIDATED\n");
}
    free(col_ind);
    free(values);
    free(x);
    free(y);
    free(z);
    return 0;
}
