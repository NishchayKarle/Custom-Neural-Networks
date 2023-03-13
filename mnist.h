#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define TRAIN_IMAGE "data/train-images-idx3-ubyte"
#define TRAIN_LABEL "data/train-labels-idx1-ubyte"
#define TEST_IMAGE "data/t10k-images-idx3-ubyte"
#define TEST_LABEL "data/t10k-labels-idx1-ubyte"

#define SIZE 784 // 28*28
#define NUM_TRAIN 60000
#define NUM_TEST 10000
#define LEN_INFO_IMAGE 4
#define LEN_INFO_LABEL 2

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1

unsigned char image[MAX_NUM_OF_IMAGES][MAX_IMAGESIZE][MAX_IMAGESIZE];
int width[MAX_NUM_OF_IMAGES], height[MAX_NUM_OF_IMAGES];

int info_image[LEN_INFO_IMAGE];
int info_label[LEN_INFO_LABEL];

unsigned char train_image_char[NUM_TRAIN][SIZE];
unsigned char test_image_char[NUM_TEST][SIZE];
unsigned char train_label_char[NUM_TRAIN][1];
unsigned char test_label_char[NUM_TEST][1];

double **train_image;
double **test_image;
double *train_label;
double *test_label;

void FlipLong(unsigned char *ptr)
{
    register unsigned char val;

    // Swap 1st and 4th bytes
    val = *(ptr);
    *(ptr) = *(ptr + 3);
    *(ptr + 3) = val;

    // Swap 2nd and 3rd bytes
    ptr += 1;
    val = *(ptr);
    *(ptr) = *(ptr + 1);
    *(ptr + 1) = val;
}

void read_mnist_char(char *file_path, int num_data, int len_info, int arr_n,
                     unsigned char data_char[][arr_n], int info_arr[])
{
    int i, fd;
    unsigned char *ptr;

    if ((fd = open(file_path, O_RDONLY)) == -1)
    {
        fprintf(stderr, "couldn't open image file");
        exit(-1);
    }

    read(fd, info_arr, len_info * sizeof(int));

    // read-in information about size of data
    for (i = 0; i < len_info; i++)
    {
        ptr = (unsigned char *)(info_arr + i);
        FlipLong(ptr);
        ptr = ptr + sizeof(int);
    }

    // read-in mnist numbers (pixels|labels)
    for (i = 0; i < num_data; i++)
    {
        read(fd, data_char[i], arr_n * sizeof(unsigned char));
    }

    close(fd);
}

void image_char2double(int num_data, unsigned char data_image_char[][SIZE], double **data_image)
{
    int i, j;
    for (i = 0; i < num_data; i++)
        for (j = 0; j < SIZE; j++)
            data_image[i][j] = (double)data_image_char[i][j] / 255.0;
}

void image_char2double2(int num_data, unsigned char data_label_char[][1], double *data_label)
{
    int i;
    for (i = 0; i < num_data; i++)
        data_label[i] = (int)data_label_char[i][0];
}

void load_mnist()
{
    train_image = (double **)malloc(NUM_TRAIN * sizeof(double *));
    test_image = (double **)malloc(NUM_TEST * sizeof(double *));
    train_label = (double *)malloc(NUM_TRAIN * sizeof(double));
    test_label = (double *)malloc(NUM_TRAIN * sizeof(double));

    for (int i = 0; i < NUM_TRAIN; i++)
        train_image[i] = (double *)malloc(SIZE * sizeof(double));

    for (int i = 0; i < NUM_TEST; i++)
        test_image[i] = (double *)malloc(SIZE * sizeof(double));

    read_mnist_char(TRAIN_IMAGE, NUM_TRAIN, LEN_INFO_IMAGE, SIZE, train_image_char, info_image);
    image_char2double(NUM_TRAIN, train_image_char, train_image);

    read_mnist_char(TEST_IMAGE, NUM_TEST, LEN_INFO_IMAGE, SIZE, test_image_char, info_image);
    image_char2double(NUM_TEST, test_image_char, test_image);

    read_mnist_char(TRAIN_LABEL, NUM_TRAIN, LEN_INFO_LABEL, 1, train_label_char, info_label);
    image_char2double2(NUM_TRAIN, train_label_char, train_label);

    read_mnist_char(TEST_LABEL, NUM_TEST, LEN_INFO_LABEL, 1, test_label_char, info_label);
    image_char2double2(NUM_TEST, test_label_char, test_label);
}

void free_mnist()
{
    for (int i = 0; i < NUM_TRAIN; i++)
        free(train_image[i]);

    for (int i = 0; i < NUM_TEST; i++)
        free(test_image[i]);

    free(train_image);
    free(train_label);
    free(test_image);
    free(test_label);
}