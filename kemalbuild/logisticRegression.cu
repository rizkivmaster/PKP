#include <assert.h>
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SAMPLE 10000
#define MAX_FEATURE 100
#define STEP_SIZE 0.005
#define NUM_ITER 10000
#define TILE_WIDTH 32
#define BLOCK_SIZE 1024

/**
  * Check error when calling CUDA API.
  */
void checkCudaError(cudaError_t errorCode)
{
    if (errorCode != cudaSuccess)
    {
        fprintf(stderr, "Error %d\n", errorCode);
        exit(1);
    }
}

/**
  * Allocate memory on host for an mxn float matrix and return its pointer.
  */
float** create_matrix_on_host(int m, int n)
{
    float **mat;

    mat = (float **) malloc(m*sizeof(float *));
    if (!mat)
    {
        fprintf(stderr, "Error allocating row memory");
        exit(1);
    }

    mat[0] = (float *) malloc(m*n*sizeof(float));
    if (!mat[0])
    {
        fprintf(stderr, "error allocating col memory");
        exit(1);
    }

    for (int i = 1; i < m; i++)
        mat[i] = mat[i-1] + n;

    return mat;
}

/**
  * Free matrix memory on host.
  */
void free_matrix_on_host(float **mat)
{
    free(mat[0]);
    free(mat);
}

/**
  * Read a CSV file and return it as a matrix.
  */
void read_csv(char *filename, float **mat, int &m, int &n)
{
    const int BUFFER_SIZE = 100000;
    const char *DELIM = ",";

    int i, j;
    char *buffer, *pch;
    FILE *f;

    buffer = (char *) malloc(BUFFER_SIZE*sizeof(char));
    f = fopen(filename, "r");
    // Read until EOF
    i = 0;
    while (feof(f) == 0)
    {
        fscanf(f, "%s\n", buffer);
        pch = strtok(buffer, DELIM);
        j = 0;
        while (pch != NULL)
        {
            mat[i][j] = atof(pch);
            pch = strtok(NULL, DELIM);
            j++;
        }
        i++;
    }
    m = i;
    n = j;

    free(buffer);
    fclose(f);
}

/**
  * Print a matrix to stdout.
  */
void print_matrix(float **mat, int m, int n)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            printf(" %.2f", mat[i][j]);
        printf("\n");
    }
}

/**
  * Sigmoid function for a vector of length n.
  */
void vectorized_sigmoid_on_host(float *y, float *x, int n)
{
    int i;
    for (i = 0; i < n; i++)
        y[i] = 1.0f / (1.0f+exp(-x[i]));
}

__global__ void vectorized_sigmoid_on_device(float *y, float *x, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = 1.0f / (1.0f+exp(-x[idx]));
}

void test_vectorized_sigmoid_on_device()
{
    float *x_h, *y_h, *yt_h;
    float *x_d, *y_d;
    int i, grid_size, n = 10, nbytes = n*sizeof(float);

    x_h = (float *) malloc(nbytes);
    y_h = (float *) malloc(nbytes);
    yt_h = (float *) malloc(nbytes);

    checkCudaError(cudaMalloc((void **) &x_d, nbytes));
    checkCudaError(cudaMalloc((void **) &y_d, nbytes));

    for (i = 0; i < n; i++)
        x_h[i] = rand() % 10;
    checkCudaError(cudaMemcpy(x_d, x_h, nbytes, cudaMemcpyHostToDevice));

    vectorized_sigmoid_on_host(y_h, x_h, n);

    grid_size = (n/BLOCK_SIZE) + (n%BLOCK_SIZE==0?0:1);
    dim3 grid(grid_size), block(BLOCK_SIZE);
    vectorized_sigmoid_on_device<<<grid, block>>>(y_d, x_d, n);
    checkCudaError(cudaMemcpy(yt_h, y_d, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        assert(abs(y_h[i] - yt_h[i]) < 1e-5);

    free(x_h); free(y_h); free(yt_h);
    cudaFree(x_d); cudaFree(y_d);
}

/**
  * Subtract vector a and b (a - b) and store the result in c.
  *
  * All vectors should have length n.
  */
void subtract_vector_and_vector_on_host(float *c, float *a, float *b, int n)
{
    int i;
    for (i = 0; i < n; i++)
        c[i] = a[i] - b[i];
}

__global__ void subtract_vector_and_vector_on_device(float *c, float *a, float *b, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] - b[idx];
}

void test_subtract_vector_and_vector_on_device()
{
    float *c_h, *a_h, *b_h, *ct_h;
    float *c_d, *a_d, *b_d;
    int i, grid_size, n = 10, nbytes = n*sizeof(float);

    c_h = (float *) malloc(nbytes);
    a_h = (float *) malloc(nbytes);
    b_h = (float *) malloc(nbytes);
    ct_h = (float *) malloc(nbytes);

    checkCudaError(cudaMalloc((void **) &c_d, nbytes));
    checkCudaError(cudaMalloc((void **) &a_d, nbytes));
    checkCudaError(cudaMalloc((void **) &b_d, nbytes));

    for (i = 0; i < n; i++)
        a_h[i] = rand() % 10;
    for (i = 0; i < n; i++)
        b_h[i] = rand() % 10;
    checkCudaError(cudaMemcpy(a_d, a_h, nbytes, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(b_d, b_h, nbytes, cudaMemcpyHostToDevice));

    subtract_vector_and_vector_on_host(c_h, a_h, b_h, n);

    grid_size = (n/BLOCK_SIZE) + (n%BLOCK_SIZE==0?0:1);
    dim3 grid(grid_size), block(BLOCK_SIZE);
    subtract_vector_and_vector_on_device<<<grid, block>>>(c_d, a_d, b_d, n);
    checkCudaError(cudaMemcpy(ct_h, c_d, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        assert(abs(c_h[i] - ct_h[i]) < 1e-5);

    free(c_h); free(a_h); free(b_h); free(ct_h);
    cudaFree(c_d); cudaFree(a_d); cudaFree(b_d);
}

/**
  * Multiply vector x and scalar c and store the result in y.
  *
  * All vectors should have length n.
  */
void multiply_vector_and_scalar_on_host(float *y, float *x, float c, int n)
{
    int i;
    for (i = 0; i < n; i++)
        y[i] = x[i] * c;
}

__global__ void multiply_vector_and_scalar_on_device(float *y, float *x, float c, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n)
        y[idx] = x[idx] * c;
}

void test_multiply_vector_and_scalar_on_device()
{
    float *x_h, *y_h, *yt_h;
    float *x_d, *y_d;
    int i, grid_size, n = 10, nbytes = n*sizeof(float);

    x_h = (float *) malloc(nbytes);
    y_h = (float *) malloc(nbytes);
    yt_h = (float *) malloc(nbytes);

    checkCudaError(cudaMalloc((void **) &x_d, nbytes));
    checkCudaError(cudaMalloc((void **) &y_d, nbytes));

    for (i = 0; i < n; i++)
        x_h[i] = rand() % 10;
    checkCudaError(cudaMemcpy(x_d, x_h, nbytes, cudaMemcpyHostToDevice));

    multiply_vector_and_scalar_on_host(y_h, x_h, 1.5f, n);

    grid_size = (n/BLOCK_SIZE) + (n%BLOCK_SIZE==0?0:1);
    dim3 grid(grid_size), block(BLOCK_SIZE);
    multiply_vector_and_scalar_on_device<<<grid, block>>>(y_d, x_d, 1.5f, n);
    checkCudaError(cudaMemcpy(yt_h, y_d, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        assert(abs(y_h[i] - yt_h[i]) < 1e-5);

    free(x_h); free(y_h); free(yt_h);
    cudaFree(x_d); cudaFree(y_d);
}

/**
  * Compute the dot product of vector u and v.
  *
  * All vectors should have length n.
  */
float dot_product(float *u, float *v, int n)
{
    int i;
    float sum = 0.0;
    for (i = 0; i < n; i++)
        sum += u[i] * v[i];
    return sum;
}

/**
  * Multiply a matrix A and a vector x and store the result in vector y.
  *
  * Matrix A is of size mxn and hence vector x should have size n. Vector y
  * will have size m.
  */
void multiply_matrix_and_vector_on_host(float *y, float **A, float *x, int m, int n)
{
    int i;
    for (i = 0; i < m; i++)
        y[i] = dot_product(A[i], x, n);
}

__global__ void multiply_matrix_and_vector_on_device(float *y, float *A, float *x, int m, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < m)
    {
        int i;
        float sum = 0.0;
        for (i = 0; i < n; i++)
            sum += A[idx*n+i] * x[i];
        y[idx] = sum;
    }
}

void test_multiply_matrix_and_vector_on_device()
{
    float **A_h, *x_h, *y_h, *yt_h;
    float *A_d, *x_d, *y_d;
    int i, j, grid_size, m = 4, n = 3;

    A_h = create_matrix_on_host(m, n);
    x_h = (float *) malloc(n*sizeof(float));
    y_h = (float *) malloc(m*sizeof(float));
    yt_h = (float *) malloc(m*sizeof(float));

    checkCudaError(cudaMalloc((void **) &A_d, m*n*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &x_d, n*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &y_d, m*sizeof(float)));

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            A_h[i][j] = rand() % 10;
    for (i = 0; i < n; i++)
        x_h[i] = rand() % 10;
    checkCudaError(cudaMemcpy(A_d, A_h[0], m*n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(x_d, x_h, n*sizeof(float), cudaMemcpyHostToDevice));

    multiply_matrix_and_vector_on_host(y_h, A_h, x_h, m, n);

    grid_size = (m/BLOCK_SIZE) + (m%BLOCK_SIZE==0?0:1);
    dim3 grid(grid_size), block(BLOCK_SIZE);
    multiply_matrix_and_vector_on_device<<<grid, block>>>(y_d, A_d, x_d, m, n);
    checkCudaError(cudaMemcpy(yt_h, y_d, m*sizeof(float), cudaMemcpyDeviceToHost));

    for (i = 0; i < m; i++)
        assert(abs(y_h[i] - yt_h[i]) < 1e-5);

    free_matrix_on_host(A_h); free(x_h); free(y_h); free(yt_h);
    cudaFree(A_d); cudaFree(x_d); cudaFree(y_d);
}

/**
  * Transpose a matrix A of size mxn and store the result in T.
  */
void transpose_matrix_on_host(float **T, float **A, int m, int n)
{
    int i, j;
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            T[i][j] = A[j][i];
}

__global__ void transpose_matrix_on_device(float *T, float *A, int m, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if (i < n && j < m)
        T[i*m+j] = A[j*n+i];
}

void test_transpose_matrix_on_device()
{
    float **X_h, **T_h, **Tt_h;
    float *X_d, *T_d;
    int i, j, gridx, gridy, m = 3, n = 4, nbytes = m*n*sizeof(float);

    X_h = create_matrix_on_host(m, n);
    T_h = create_matrix_on_host(n, m);
    Tt_h = create_matrix_on_host(n, m);

    checkCudaError(cudaMalloc((void **) &X_d, nbytes));
    checkCudaError(cudaMalloc((void **) &T_d, nbytes));

    for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            X_h[i][j] = rand() % 10;
    checkCudaError(cudaMemcpy(X_d, X_h[0], nbytes, cudaMemcpyHostToDevice));

    transpose_matrix_on_host(T_h, X_h, m, n);

    gridx = (n/TILE_WIDTH) + (n%TILE_WIDTH==0?0:1);
    gridy = (m/TILE_WIDTH) + (m%TILE_WIDTH==0?0:1);
    dim3 grid(gridx, gridy), block(TILE_WIDTH, TILE_WIDTH);
    transpose_matrix_on_device<<<grid, block>>>(T_d, X_d, m, n);
    checkCudaError(cudaMemcpy(Tt_h[0], T_d, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            assert(abs(T_h[i][j] - Tt_h[i][j]) < 1e-5);

    free_matrix_on_host(X_h); free_matrix_on_host(T_h); free_matrix_on_host(Tt_h);
    cudaFree(X_d); cudaFree(T_d);
}

/**
  * Find the best parameter for logistic regression using gradient descent.
  *
  * The result will be stored in theta. Arguments init_theta, step_size, X, and y
  * are the initial value for the parameter, step size of gradient descent, design
  * matrix, and target vector respectively. Design matrix X should have size mxn.
  * The number of gradient descent iterations is controlled by niter.
  */
void logistic_regression_gradient_descent_on_host(float *theta, float *init_theta, float step_size, float **X, float *y, int m, int n, int niter)
{
    int t;
    float **T, *mtemp, *ntemp;

    T = create_matrix_on_host(n, m);
    mtemp = (float *) malloc(m*sizeof(float));
    ntemp = (float *) malloc(n*sizeof(float));
    memcpy(theta, init_theta, n*sizeof(float));

    for (t = 0; t < niter; t++)
    {
        // Compute X*theta
        multiply_matrix_and_vector_on_host(mtemp, X, theta, m, n);
        // Compute h(X)
        vectorized_sigmoid_on_host(mtemp, mtemp, m);
        // Compute error
        subtract_vector_and_vector_on_host(mtemp, mtemp, y, m);
        // Compute transpose of X
        transpose_matrix_on_host(T, X, m, n);
        // Compute derivative of cost function J
        multiply_matrix_and_vector_on_host(ntemp, T, mtemp, n, m);
        multiply_vector_and_scalar_on_host(ntemp, ntemp, (float) step_size / m, n);
        // Update theta
        subtract_vector_and_vector_on_host(theta, theta, ntemp, n);
    }
    free_matrix_on_host(T);
    free(mtemp);
    free(ntemp);
}

void parallel_logistic_regression_gradient_descent(float *theta, float *init_theta, float step_size, float **X, float *y, int m, int n, int niter)
{
    float *mtemp_d, *X_d, *theta_d, *y_d, *T_d;
    int t, grid_size, gridx, gridy;

    memcpy(theta, init_theta, n*sizeof(float));

    checkCudaError(cudaMalloc((void **) &mtemp_d, m*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &X_d, m*n*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &theta_d, n*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &y_d, m*sizeof(float)));
    checkCudaError(cudaMalloc((void **) &T_d, m*sizeof(float)));

    checkCudaError(cudaMemcpy(X_d, X[0], m*n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(theta_d, theta, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(y_d, y, m*sizeof(float), cudaMemcpyHostToDevice));

    for (t = 0; t < niter; t++)
    {
        // Compute X*theta
        grid_size = (m/BLOCK_SIZE) + (m%BLOCK_SIZE==0?0:1);
        dim3 grid(grid_size), block(BLOCK_SIZE);
        multiply_matrix_and_vector_on_device<<<grid, block>>>(mtemp_d, X_d, theta_d, m, n);
        // Compute h(X)
        vectorized_sigmoid_on_device<<<grid, block>>>(mtemp_d, mtemp_d, m);
        // Compute error
        subtract_vector_and_vector_on_device<<<grid, block>>>(mtemp_d, mtemp_d, y_d, m);
        // Compute transpose of X
        gridx = (m/TILE_WIDTH) + (m%TILE_WIDTH==0?0:1);
        gridy = (n/TILE_WIDTH) + (n%TILE_WIDTH==0?0:1);
        dim3 grid2d(gridx, gridy), block2d(TILE_WIDTH, TILE_WIDTH);
        transpose_matrix_on_device<<<grid2d, block2d>>>(T_d, X_d, m, n);
    }
}

/**
  * Predict based on estimate.
  */
void predict_on_host(float *prediction, float *estimate, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        if (estimate[i] >= 0.5f)
            prediction[i] = 1.0f;
        else
            prediction[i] = 0.0f;
    }
}

__global__ void predict_on_device(float *prediction, float *estimate, int n)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx < n)
    {
        if (estimate[idx] >= 0.5f)
            prediction[idx] = 1.0f;
        else
            prediction[idx] = 0.0f;
    }
}

void test_predict_on_device()
{
    float *est_h, *pred_h, *predt_h;
    float *est_d, *pred_d;
    int i, grid_size, n = 10, nbytes = n*sizeof(float);

    est_h = (float *) malloc(nbytes);
    pred_h = (float *) malloc(nbytes);
    predt_h = (float *) malloc(nbytes);

    checkCudaError(cudaMalloc((void **) &est_d, nbytes));
    checkCudaError(cudaMalloc((void **) &pred_d, nbytes));

    for (i = 0; i < n; i++)
        est_h[i] = (float) rand() / RAND_MAX;
    checkCudaError(cudaMemcpy(est_d, est_h, nbytes, cudaMemcpyHostToDevice));

    predict_on_host(pred_h, est_h, n);

    grid_size = (n/BLOCK_SIZE) + (n%BLOCK_SIZE==0?0:1);
    dim3 grid(grid_size), block(BLOCK_SIZE);
    predict_on_device<<<grid, block>>>(pred_d, est_d, n);
    checkCudaError(cudaMemcpy(predt_h, pred_d, nbytes, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++)
        assert(abs(pred_h[i] - predt_h[i]) < 1e-5);

    free(est_h); free(pred_h); free(predt_h);
    cudaFree(est_d); cudaFree(pred_d);
}

/**
  * Compute zero-one loss function, i.e. number of misclassification.
  */
int zero_one_loss_function(float *prediction, float *y, int n)
{
    int i, res = 0;
    for (i = 0; i < n; i++)
        if (prediction[i] != y[i])
            res++;
    return res;
}

void run_tests()
{
    srand(time(NULL));
    test_vectorized_sigmoid_on_device();
    test_subtract_vector_and_vector_on_device();
    test_multiply_vector_and_scalar_on_device();
    test_multiply_matrix_and_vector_on_device();
    test_transpose_matrix_on_device();
    test_predict_on_device();
}

void logistic_regression_on_host(char *train_filename, char *test_filename)
{
    int i, j, m, mt, n, loss;
    float **X, **Xt, **mat, *y, *yt, *theta, *init_theta, *ypred;

    // Get design matrix and target vector
    mat = create_matrix_on_host(MAX_SAMPLE, MAX_FEATURE);
    read_csv(train_filename, mat, m, n);
    X = create_matrix_on_host(m, n-1);
    y = (float *) malloc(m*sizeof(float));
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n-1; j++)
            X[i][j] = mat[i][j];
        y[i] = mat[i][n-1];
    }
    n--;

    // Perform gradient descent to find best parameter value
    theta = (float *) malloc(n*sizeof(float));
    init_theta = (float *) malloc(n*sizeof(float));
    memset(init_theta, 0, n*sizeof(float));
    logistic_regression_gradient_descent_on_host(theta, init_theta, STEP_SIZE, X, y, m, n, NUM_ITER);

    // Get testing data
    read_csv(test_filename, mat, mt, n);
    Xt = create_matrix_on_host(mt, n-1);
    yt = (float *) malloc(mt*sizeof(float));
    for (i = 0; i < mt; i++)
    {
        for (j = 0; j < n-1; j++)
            Xt[i][j] = mat[i][j];
        yt[i] = mat[i][n-1];
    }
    n--;
    free_matrix_on_host(mat);

    // Compute zero-one loss for training data
    ypred = (float *) malloc(m*sizeof(float));
    multiply_matrix_and_vector_on_host(ypred, X, theta, m, n);
    vectorized_sigmoid_on_host(ypred, ypred, m);
    predict_on_host(ypred, ypred, m);
    loss = zero_one_loss_function(ypred, y, m);
    printf("Loss on train data: %d/%d\n", loss, m);
    free(ypred);

    // Compute zero-one loss for testing data
    ypred = (float *) malloc(mt*sizeof(float));
    multiply_matrix_and_vector_on_host(ypred, Xt, theta, mt, n);
    vectorized_sigmoid_on_host(ypred, ypred, mt);
    predict_on_host(ypred, ypred, mt);
    loss = zero_one_loss_function(ypred, yt, mt);
    printf("Loss on testing data: %d/%d\n", loss, mt);

    free_matrix_on_host(X); free_matrix_on_host(Xt);
    free(y); free(yt); free(theta); free(init_theta); free(ypred);
}

void logistic_regression_on_device(char *train_filename, char *test_filename)
{
}

int main(int argc, char **argv)
{
    run_tests();
    logistic_regression_on_host(argv[1], argv[2]);
    return 0;
}
