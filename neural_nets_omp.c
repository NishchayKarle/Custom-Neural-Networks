#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <omp.h>

#include "matrix_omp.h"
#include "mnist.h"

struct neural_network
{
    double alpha;
    int nl, nh, ne, nb,
        in, on, tl;

    double ***w,
        ***b,
        ****z,
        ****a,
        ****e;
    int *nnl;
};
typedef struct neural_network neural_network;

double sigmoid(double val)
{
    return 1 / (1 + exp(-val));
}

double dsigmoid(double val)
{
    return sigmoid(val) * (1 - sigmoid(val));
}

void apply_sigmoid(double **m, int r, int c, double **m2)
{
    // #pragma omp parallel for num_threads(128 / min(64, nt))
    // #pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m2[i][j] = sigmoid(m[i][j]);
}

void apply_dsigmoid(double **m, int r, int c, double **m2)
{
    // #pragma omp parallel for num_threads(128 / min(64, nt))
    // #pragma omp parallel for num_threads(10)
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            m2[i][j] = dsigmoid(m[i][j]);
}

void neural_network_init(neural_network *network, double alpha,
                         int nl, int nh, int ne, int nb, int in, int on)
{
    network->alpha = alpha;
    network->nl = nl;
    network->nh = nh;
    network->ne = ne;
    network->nb = nb;
    network->in = in;
    network->on = on;
    network->tl = 2 + nl;

    network->w = (double ***)malloc(sizeof(double **) * (network->tl - 1));
    network->b = (double ***)malloc(sizeof(double **) * network->tl);
    network->z = (double ****)malloc(sizeof(double ***) * network->nb);
    network->a = (double ****)malloc(sizeof(double ***) * network->nb);
    network->e = (double ****)malloc(sizeof(double ***) * network->nb);

    network->nnl = (int *)malloc(sizeof(int) * network->tl);
    network->nnl[0] = network->in;
    for (int l = 1; l < network->tl - 1; l++)
        network->nnl[l] = network->nh;
    network->nnl[network->tl - 1] = network->on;

    // SET RANDOM INITIAL WEIGHTS
    {
        for (int l = 0; l < network->tl - 1; l++)
        {
            network->w[l] = createmat(network->nnl[l + 1], network->nnl[l]);
            initmatrand(network->w[l], network->nnl[l + 1], network->nnl[l]);
        }
    }

    // SET RANDOM INITIAL BIAS
    {
        for (int l = 0; l < network->tl; l++)
        {
            network->b[l] = createmat(network->nnl[l], 1);
            initmatrand(network->b[l], network->nnl[l], 1);
        }
    }

    // INIT OTHER PARAMS
    {
        for (int i = 0; i < network->nb; i++)
        {
            network->z[i] = (double ***)malloc(sizeof(double **) * network->tl);
            network->a[i] = (double ***)malloc(sizeof(double **) * network->tl);
            network->e[i] = (double ***)malloc(sizeof(double **) * network->tl);
        }

        for (int i = 0; i < network->nb; i++)
            for (int l = 0; l < network->tl; l++)
            {
                network->z[i][l] = createmat(network->nnl[l], 1);
                network->a[i][l] = createmat(network->nnl[l], 1);
                network->e[i][l] = createmat(network->nnl[l], 1);
            }
    }
}

void feed_forward(neural_network *network, double *input_x, int length, int index)
{
    assert(length == network->in);
    for (int i = 0; i < length; i++)
        network->a[index][0][i][0] = input_x[i];

    for (int l = 1; l < network->tl; l++)
    {
        matmul(network->w[l - 1], network->nnl[l], network->nnl[l - 1],
               network->a[index][l - 1], network->nnl[l - 1], 1, network->z[index][l]);
        matadd(network->z[index][l], network->nnl[l], 1, network->b[l], network->nnl[l], 1, network->z[index][l]);
        apply_sigmoid(network->z[index][l], network->nnl[l], 1, network->a[index][l]);
    }
}

void output_error(neural_network *network, double expected_val, int index)
{
    double **tempmat = createmat(network->on, 1);
    double **tempmat2 = createmat(network->on, 1);

    for (int i = 0; i < network->on; i++)
        tempmat[i][0] = (i == expected_val) ? 1 : 0;

    matsub(network->a[index][network->tl - 1], network->on, 1, tempmat, network->on, 1, tempmat);
    apply_dsigmoid(network->z[index][network->tl - 1], network->on, 1, tempmat2);
    matdot(tempmat, network->on, 1, tempmat2, network->on, 1, network->e[index][network->tl - 1]);

    free_matrix(tempmat, network->on);
    free_matrix(tempmat2, network->on);
}

void backpropogation(neural_network *network, int index)
{
    for (int l = network->tl - 2; l > 0; l--)
    {
        t_matmul(network->w[l], network->nnl[l + 1], network->nnl[l], network->e[index][l + 1],
                 network->nnl[l + 1], 1, network->e[index][l]);

        double **tempmat = createmat(network->nnl[l], 1);
        apply_dsigmoid(network->z[index][l], network->nnl[l], 1, tempmat);
        matdot(network->e[index][l], network->nnl[l], 1, tempmat, network->nnl[l], 1, network->e[index][l]);
        free_matrix(tempmat, network->nnl[l]);
    }
}

void gradient_descent(neural_network *network, double **images,
                      double *label, int *index_arr, double ***dw, double ***db)
{

    {
        for (int l = 0; l < network->tl; l++)
        {
            if (l != network->tl - 1)
                initmatzeros(dw[l], network->nnl[l + 1], network->nnl[l]);
            initmatzeros(db[l], network->nnl[l], 1);
        }
    }

#pragma omp parallel for num_threads(10)
    for (int i = 0; i < network->nb; i++)
    {

        feed_forward(network, images[index_arr[i]], SIZE, i);
        output_error(network, label[index_arr[i]], i);
        backpropogation(network, i);
    }

    for (int l = network->tl - 1; l > 0; l--)
        for (int i = 0; i < network->nb; i++)
        {
            matmul_t(network->e[i][l], network->nnl[l], 1, network->a[i][l - 1], network->nnl[l - 1], 1, dw[l - 1]);
            matadd(network->e[i][l], network->nnl[l], 1, db[l], network->nnl[l], 1, db[l]);
        }

    for (int l = network->tl - 1; l > 0; l--)
    {
        matsclmul(dw[l - 1], network->nnl[l], network->nnl[l - 1],
                  ((double)network->alpha / (double)network->nb), dw[l - 1]);
        matsclmul(db[l], network->nnl[l], 1,
                  ((double)network->alpha / (double)network->nb), db[l]);

        matsub(network->w[l - 1], network->nnl[l], network->nnl[l - 1], dw[l - 1],
               network->nnl[l], network->nnl[l - 1], network->w[l - 1]);
        matsub(network->b[l], network->nnl[l], 1, db[l], network->nnl[l], 1, network->b[l]);
    }
}

void shuffle_indexes(int *arr, int length)
{
    for (int i = length - 1; i >= 1; i--)
    {
        int j = rand() % (i + 1);
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void train_neural_network(neural_network *network, double **train_image,
                          int num_of_images, double *train_label)
{
    double ***dw = (double ***)malloc(sizeof(double **) * (network->tl - 1));
    double ***db = (double ***)malloc(sizeof(double **) * network->tl);
    for (int l = 0; l < network->tl; l++)
    {
        if (l != network->tl - 1)
            dw[l] = createmat(network->nnl[l + 1], network->nnl[l]);
        db[l] = createmat(network->nnl[l], 1);
    }

    int arr[num_of_images];
    for (int i = 0; i < num_of_images; i++)
        arr[i] = i;

    for (int i = 0; i < network->ne; i++)
    {
        shuffle_indexes(arr, num_of_images);
        for (int j = 0; j < num_of_images / network->nb; j++)
            gradient_descent(network, train_image, train_label, arr + (j * network->nb),
                             dw, db);
    }

    for (int l = 0; l < network->tl; l++)
    {
        if (l != network->tl - 1)
        {
            free_matrix(dw[l], network->nnl[l + 1]);
        }

        free_matrix(db[l], network->nnl[l]);
    }
    free(dw);
    free(db);
}

void arg_max(double **arr, int length, int *res)
{
    double max_val = arr[0][0];
    *res = 0;
    for (int i = 1; i < length; i++)
        if (arr[i][0] > max_val)
        {
            max_val = arr[i][0];
            *res = i;
        }
}

void test_neural_network(neural_network *network, double **test_image,
                         int num_of_images, double *test_label)
{
    int correct = 0;
    for (int i = 0; i < num_of_images; i++)
    {
        feed_forward(network, test_image[i], SIZE, 0);
        int predicted;
        arg_max(network->a[0][network->tl - 1], 10, &predicted);
        // printf("PRED = %d ACTUAL = %d\n", predicted, (int)test_label[i]);
        if (predicted == (int)test_label[i])
            correct++;
    }
    printf("ACC: %f\n", 100.0 * (double)correct / (double)num_of_images);
}

int main(int argc, char **argv)
{
    srand(1);
    load_mnist();

    int nl = atoi(argv[1]), // LAYERS
        nh = atoi(argv[2]), // NEURONS
        ne = atoi(argv[3]), // EPOCS
        nb = atoi(argv[4]); // BATCH SIZE

    // nt = nb;
    neural_network nn;
    neural_network_init(&nn,
                        0.1,
                        nl,  // LAYERS
                        nh,  // NEURONS
                        ne,  // EPOCS
                        nb,  // BATCH SIZE
                        784, // INPUT
                        10); // OUTPUT

    double start, end;
    start = omp_get_wtime();
    train_neural_network(&nn, train_image, 6000, train_label);
    end = omp_get_wtime();

    test_neural_network(&nn, test_image, 10000, test_label);

    for (int l = 0; l < nn.tl; l++)
    {
        if (l != nn.tl - 1)
            free_matrix(nn.w[l], nn.nnl[l + 1]);
        free_matrix(nn.b[l], nn.nnl[l]);
    }

    for (int i = 0; i < nn.nb; i++)
    {
        for (int l = 0; l < nn.tl; l++)
        {
            free_matrix(nn.z[i][l], nn.nnl[l]);
            free_matrix(nn.a[i][l], nn.nnl[l]);
            free_matrix(nn.e[i][l], nn.nnl[l]);
        }
        free(nn.z[i]);
        free(nn.a[i]);
        free(nn.e[i]);
    }

    free_mnist();
    free(nn.w);
    free(nn.b);
    free(nn.z);
    free(nn.a);
    free(nn.e);
    free(nn.nnl);

    printf("TIME TAKEN: %f (s)\n", (end - start));

    return EXIT_SUCCESS;
}
