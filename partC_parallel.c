#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <omp.h>

void multiply_ikj_seq(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n * n; i++) C[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

void multiply_ikj_omp(double *A, double *B, double *C, int n, int threads) {
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0.0;
        }
        for (int k = 0; k < n; k++) {
            double r = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C[i * n + j] += r * B[k * n + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <size_n> <threads> <mode: seq|omp>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int threads = atoi(argv[2]);
    char *mode = argv[3];

    double *A = (double *)malloc(n * n * sizeof(double));
    double *B = (double *)malloc(n * n * sizeof(double));
    double *C = (double *)malloc(n * n * sizeof(double));

    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    LARGE_INTEGER frequency, start, end;
    QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&start);
    if (strcmp(mode, "seq") == 0) {
        multiply_ikj_seq(A, B, C, n);
    } else {
        multiply_ikj_omp(A, B, C, n, threads);
    }
    QueryPerformanceCounter(&end);

    double time_ms = (double)(end.QuadPart - start.QuadPart) * 1000.0 / frequency.QuadPart;
    double checksum = C[0] + C[n * n - 1];

    printf("%d,%d,%s,%.4f,%.6f\n", n, threads, mode, time_ms, checksum);

    free(A); free(B); free(C);
    return 0;
}