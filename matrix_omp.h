#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

int nt;

int min(int a, int b);

void initmatzeros(double **m, int r, int c);

void initmatrand(double **m, int r, int c);

void matmul(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void t_matmul(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void matmul_t(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void matdot(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void matadd(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void matsub(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3);

void matprint(double **m, int r, int c);

double **createmat(int r, int c);

void matsclmul(double **m, int r, int c, double scalar, double **m2);

int min(int a, int b)
{
    return a < b ? a : b;
}

void initmatzeros(double **m, int r, int c)
{
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m[i][j] = 0.0;
}

void initmatrand(double **m, int r, int c)
{
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m[i][j] = 2.0 * (double)rand() / (double)RAND_MAX - 1.0;
}

void matmul(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert(c1 == r2);
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c2; j++)
        {
            m3[i][j] = 0.0;
            for (int k = 0; k < r2; k++)
                m3[i][j] += m1[i][k] * m2[k][j];
        }
}

void t_matmul(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert(r1 == r2);
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < c1; i++)
        for (int j = 0; j < c2; j++)
        {
            m3[i][j] = 0.0;
            for (int k = 0; k < r1; k++)
                m3[i][j] += m1[k][i] * m2[k][j];
        }
}

void matmul_t(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert(c1 == c2);
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < r2; j++)
            for (int k = 0; k < c1; k++)
                m3[i][j] += m1[i][k] * m2[j][k];
}

void matdot(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert((r1 == r2) && (c1 == c2));
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c1; j++)
            m3[i][j] = m1[i][j] * m2[i][j];
}

void matadd(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert((r1 == r2) && (c1 == c2));
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c1; j++)
            m3[i][j] = m1[i][j] + m2[i][j];
}

void matsub(double **m1, int r1, int c1, double **m2, int r2, int c2, double **m3)
{
    assert((r1 == r2) && (c1 == c2));
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c1; j++)
            m3[i][j] = m1[i][j] - m2[i][j];
}

void matsclmul(double **m, int r, int c, double scalar, double **m2)
{
// #pragma omp parallel for num_threads(128 / min(64, nt))
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m2[i][j] = m[i][j] * scalar;
}

void matprint(double **m, int r, int c)
{
    for (int i = 0; i < r; i++)
    {
        for (int j = 0; j < c; j++)
            printf("%.10f ", m[i][j]);
        printf("\n");
    }
}

double **createmat(int r, int c)
{
    double **m = (double **)malloc(sizeof(double *) * r);
    for (int i = 0; i < r; i++)
        m[i] = (double *)malloc(sizeof(double) * c);
    return m;
}

void free_matrix(double **m, int r)
{
#pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        free(m[i]);
    free(m);
}