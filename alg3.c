#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ARRAY_SIZE 10000000
#define BUCKETS_NUM 100
#define MAX_ELEM 1000000000

// logging modes:
#define DBG 0
#define RES 1
#define NONE 2
#define LOG_MODE NONE  // can be changed

#define QUICK 0
#define INSERT 1
#define SORT_MODE QUICK // can be changed

struct Bucket{
  int* elems;
  int len;
};


void fillArray(int* array);

// BUCKET SORT METHODS:
void clearBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets, int threadId);
void splitArrayToSmallBuckets(struct Bucket** smallBuckets, int* array, int threadId);
void mergeBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets);
void sortBigBuckets(struct Bucket* bigBuckets);
void calculateBigBucketsOffsets(struct Bucket* bigBuckets, int* bigBucketsOffsets);
void writeBigBucketsToArray(struct Bucket* bigBuckets, int* bigBucketsOffsets, int* array);

// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Bucket*** smallBuckets, struct Bucket** bigBuckets, int** bigBucketsOffsets);
void freeBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets, int* bigBucketsOffsets);

// QUICK SORT METHODS:
void swap(int* a, int* b);
int partition (int arr[], int low, int high);
void quickSort(int arr[], int low, int high);

// PRINTING METHODS:
void printConfig();
void printArray(int* array);
void printInitialArray(int* array);
void printSortedArray(int* array);
void printSmallBuckets(struct Bucket** smallBuckets);
void printInitialBigBuckets(struct Bucket* bigBuckets);
void printBigBuckets(struct Bucket* bigBuckets);
void printSortedBigBuckets(struct Bucket* bigBuckets);
void printBigBucketsOffsets(int* bigBucketsOffsets);
void printAndSaveResults(double tfillArray, double tSplitArray, double tMergeBuckets, double tSort, double tWriteBucketsToArray, double tWhole);

int main(int argc, char* argv[])
{
  double startTime, tmp, tfillArray, tSplitArray, tMergeBuckets, tSort, tWriteBucketsToArray, tWhole;
  time_t t;
  int* array = (int*)malloc(sizeof(int) * ARRAY_SIZE);

  int num_threads = atoi(argv[1]);

  srand((unsigned) time(&t));

  struct Bucket** smallBuckets;
  struct Bucket* bigBuckets;
  int* bigBucketsOffsets;
  int threadIdx = 0;

  startTime = omp_get_wtime();
  #pragma omp parallel num_threads(num_threads)
  {
    int threadId;
    #pragma omp critical
    threadId = threadIdx++;

    allocateBuckets(&smallBuckets, &bigBuckets, &bigBucketsOffsets);
    clearBuckets(smallBuckets, bigBuckets, threadId);

    #pragma omp single
    tmp = omp_get_wtime();
    fillArray(array);
    #pragma omp single
    tfillArray = omp_get_wtime() - tmp;
    printInitialArray(array);


    #pragma omp single
    tmp = omp_get_wtime();
    splitArrayToSmallBuckets(smallBuckets, array, threadId);
    #pragma omp single
    tSplitArray = omp_get_wtime() - tmp;
    printSmallBuckets(smallBuckets);

    #pragma omp single
    tmp = omp_get_wtime();
    mergeBuckets(smallBuckets, bigBuckets);
    #pragma omp single
    tMergeBuckets = omp_get_wtime() - tmp;
    printInitialBigBuckets(bigBuckets);

    #pragma omp barrier
    tmp = omp_get_wtime();
    sortBigBuckets(bigBuckets);
    #pragma omp single
    tSort = omp_get_wtime() - tmp;
    printSortedBigBuckets(bigBuckets);

    #pragma omp single
    tmp = omp_get_wtime();
    calculateBigBucketsOffsets(bigBuckets, bigBucketsOffsets);
    printBigBucketsOffsets(bigBucketsOffsets);
    writeBigBucketsToArray(bigBuckets, bigBucketsOffsets, array);
    #pragma omp single
    tWriteBucketsToArray = omp_get_wtime() - tmp;
    printSortedArray(array);

    freeBuckets(smallBuckets, bigBuckets, bigBucketsOffsets);
  }
  tWhole = omp_get_wtime() - startTime;

  printConfig();
  printAndSaveResults(tfillArray, tSplitArray, tMergeBuckets, tSort, tWriteBucketsToArray, tWhole);

  free(array);

  return 0;
}


void fillArray(int* array){
  int i;
  unsigned int threadNum = omp_get_thread_num();
  #pragma omp for private(i)
  for(i = 0; i < ARRAY_SIZE; i++){
    array[i] = rand_r(&threadNum) % MAX_ELEM;
  }
}


// BUCKET SORT METHODS:
void clearBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets, int threadId){
  int i;
  for(i = 0; i < BUCKETS_NUM; i++){
    smallBuckets[threadId][i].elems = NULL;
    smallBuckets[threadId][i].len = 0;
  }
  #pragma omp for private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    bigBuckets[i].elems = NULL;
    bigBuckets[i].len = 0;
  }
}

void splitArrayToSmallBuckets(struct Bucket** smallBuckets, int* array, int threadId){
  int i, j, bucketId;
  #pragma omp for private(i)
  for(i = 0; i < ARRAY_SIZE; i++){
    bucketId =((double)array[i] / MAX_ELEM) * BUCKETS_NUM;
    smallBuckets[threadId][bucketId].len++;
  }
  for(j = 0; j < BUCKETS_NUM; j++){
    struct Bucket* bucket = &smallBuckets[threadId][j];
    (*bucket).elems = (int*)malloc(sizeof(int) * (*bucket).len);
    (*bucket).len = 0;
  }
  #pragma omp for private(i)
  for(i = 0; i < ARRAY_SIZE; i++){
    bucketId =((double)array[i] / MAX_ELEM) * BUCKETS_NUM;
    struct Bucket* bucket = &smallBuckets[threadId][bucketId];
    (*bucket).elems[(*bucket).len] = array[i];
    (*bucket).len++;
  }
}

void mergeBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets){
  int i, j, k, numThreads = omp_get_num_threads();;
  #pragma omp for schedule(static) private(i, j)
  for(i = 0; i < BUCKETS_NUM; i++){
    for(j = 0; j < numThreads; j++){
      bigBuckets[i].len += smallBuckets[j][i].len;
    }
    bigBuckets[i].elems = (int*)malloc(sizeof(int) * bigBuckets[i].len);

    int offset = 0;
    for(j = 0; j < numThreads; j++){
      for(k = 0; k < smallBuckets[j][i].len; k++){
        bigBuckets[i].elems[offset] = smallBuckets[j][i].elems[k];
        offset++;
      }
    }
  }
}

void sortBigBuckets(struct Bucket* bigBuckets){
  int i;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    quickSort(bigBuckets[i].elems, 0, bigBuckets[i].len - 1);
  }
}

void calculateBigBucketsOffsets(struct Bucket* bigBuckets, int* bigBucketsOffsets){
  #pragma omp single
  {
    int i, totalOffset = 0;
    for(i = 0; i < BUCKETS_NUM; i++){
      bigBucketsOffsets[i] = totalOffset;
      totalOffset += bigBuckets[i].len;
    }
  }
}

void writeBigBucketsToArray(struct Bucket* bigBuckets, int* bigBucketsOffsets, int* array){
  int i, j;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    int idx = bigBucketsOffsets[i];
    for(j = 0; j < bigBuckets[i].len; j++){
      array[idx] = bigBuckets[i].elems[j];
      idx++;
    }
  }
}


// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Bucket*** smallBuckets, struct Bucket** bigBuckets, int** bigBucketsOffsets){
  #pragma omp single
  {
    int i, numThreads = omp_get_num_threads();
    *smallBuckets = (struct Bucket**)malloc(sizeof(struct Bucket*) * numThreads);
    for(i = 0; i < numThreads; i++){
      (*smallBuckets)[i] = (struct Bucket*)malloc(sizeof(struct Bucket) * BUCKETS_NUM);
    }
    *bigBuckets = (struct Bucket*)malloc(sizeof(struct Bucket) * BUCKETS_NUM);
    *bigBucketsOffsets = (int*)malloc(sizeof(int) * BUCKETS_NUM);
  }
}

void freeBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets, int* bigBucketsOffsets){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    int i, j;
    for(i = 0; i < numThreads; i++){
      for(j = 0; j < BUCKETS_NUM; j++){
        free(smallBuckets[i][j].elems);
      }
      free(smallBuckets[i]);
    }
    free(smallBuckets);

    for(i = 0; i < BUCKETS_NUM; i++){
      free(bigBuckets[i].elems);
    }
    free(bigBuckets);
    free(bigBucketsOffsets);
  }
}


// QUICK SORT METHODS
void swap(int* a, int* b)
{
  int t = *a;
  *a = *b;
  *b = t;
}

int partition (int arr[], int low, int high)
{
  int pivot = arr[high];
  int i = (low - 1);
  int j;
  for (j = low; j <= high- 1; j++)
  {
    if (arr[j] <= pivot)
    {
      i++;
      swap(&arr[i], &arr[j]);
    }
  }
  swap(&arr[i + 1], &arr[high]);
  return (i + 1);
}

void quickSort(int arr[], int low, int high)
{
  if (low < high)
  {
    int pi = partition(arr, low, high);
    quickSort(arr, low, pi - 1);
    quickSort(arr, pi + 1, high);
  }
}

// PRINTING METHODS:
void printConfig(){
  printf("array size = \t\t%d\n", (int)ARRAY_SIZE);
  printf("buckets num = \t\t%d\n", (int)BUCKETS_NUM);
  printf("max array elem = \t%d\n", (int)MAX_ELEM);
  printf("logging mode = \t\t");
  if(DBG == LOG_MODE){
    printf("DBG\n");
  }
  else if(RES == LOG_MODE){
    printf("RES\n");
  }else{
    printf("NONE\n");
  }
  printf("sorting method = \t");
  if(QUICK == SORT_MODE){
    printf("QUICK SORT\n");
  }else{
    printf("INSERTION SORT\n");
  }
}

void printArray(int* array){
  int i;
  for(i = 0; i < ARRAY_SIZE; i++){
    printf("%d\t", array[i]);
  }
  printf("\n\n");
}

void printInitialArray(int* array){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      printf("Initial array:\n");
      printArray(array);
    }
  }
}

void printSortedArray(int* array){
  #pragma omp single
  {
    if(DBG == LOG_MODE || RES == LOG_MODE){
      printf("Sorted array:\n");
      printArray(array);
    }
  }
}

void printSmallBuckets(struct Bucket** smallBuckets){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      int i, j, k;
      printf("SmallBuckets:\n");
      for(i = 0; i < numThreads; i++){
        for(j = 0; j < BUCKETS_NUM; j++){
          printf("SmallBuckets for thread id=%d\tsmall bucket id: %u\telems: ", i, j);
          for(k = 0; k < smallBuckets[i][j].len; k++){
            printf("%d ", smallBuckets[i][j].elems[k]);
          }
          printf("\n");
        }
      }
      printf("\n");
    }
  }
}

void printInitialBigBuckets(struct Bucket* bigBuckets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      printf("Initial BigBuckets (right after merging SmallBuckets):\n");
      printBigBuckets(bigBuckets);
    }
  }
}

void printBigBuckets(struct Bucket* bigBuckets){
  printf("printBigBuckets called\n");
  int i, j;
  for(i = 0; i < BUCKETS_NUM; i++){
    printf("Big bucket id: %d\t elems: ", i);
    for(j = 0; j < bigBuckets[i].len; j++){
      printf("%d ", bigBuckets[i].elems[j]);
    }
    printf("\n");
  }
  printf("\n");
}

void printSortedBigBuckets(struct Bucket* bigBuckets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      printf("Sorted BigBuckets:\n");
      printBigBuckets(bigBuckets);
    }
  }
}

void printBigBucketsOffsets(int* bigBucketsOffsets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      int i;
      printf("BigBucketsOffsets:\n");
      for(i = 0; i < BUCKETS_NUM; i++){
        printf("Offset of BigBuckets %d = %d\n", i, bigBucketsOffsets[i]);
      }
      printf("\n");
    }
  }
}

void printAndSaveResults(double tfillArray, double tSplitArray, double tMergeBuckets, double tSort, double tWriteBucketsToArray, double tWhole){
  printf("Filling initial array time:\t\t%f seconds\n", tfillArray);
  printf("Splitting numbers to buckets time:\t%f seconds\n", tSplitArray);
  printf("Merging buckets time:\t\t\t%f seconds\n", tMergeBuckets);
  printf("Sorting buckets time:\t\t\t%f seconds\n", tSort);
  printf("Writing buckets to array time:\t\t%f seconds\n", tWriteBucketsToArray);
  printf("Total time:\t\t\t\t%f seconds\n", tWhole);

  FILE* results = fopen("resultsAlg3.txt", "w");
  fprintf(results, "Filling initial array time:\t\t%f seconds\n", tfillArray);
  fprintf(results, "Splitting numbers to buckets time:\t%f seconds\n", tSplitArray);
  fprintf(results, "Merging buckets time:\t\t\t%f seconds\n", tMergeBuckets);
  fprintf(results, "Sorting buckets time:\t\t\t%f seconds\n", tSort);
  fprintf(results, "Writing buckets to array time:\t\t%f seconds\n", tWriteBucketsToArray);
  fprintf(results, "Total time:\t\t\t\t%f seconds\n", tWhole);
  fclose(results);
}
