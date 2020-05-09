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
void clearBuckets(struct Bucket* buckets);
void splitArrayToBuckets(struct Bucket* buckets, int* array, omp_lock_t* locks, int threadId);
void mergeBuckets(struct Bucket** smallBuckets, struct Bucket* bigBuckets);
void sortBuckets(struct Bucket* buckets);
void calculateBucketsOffsets(struct Bucket* buckets, int* bucketsOffsets);
void writeBucketsToArray(struct Bucket* buckets, int* bucketsOffsets, int* array);

// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Bucket** buckets, int** bucketsOffsets, omp_lock_t** locks);
void freeBuckets(struct Bucket* buckets, int* bucketsOffsets, omp_lock_t* locks);

// QUICK SORT METHODS:
void swap(int* a, int* b);
int partition (int arr[], int low, int high);
void quickSort(int arr[], int low, int high);

// PRINTING METHODS:
void printConfig();
void printArray(int* array);
void printInitialArray(int* array);
void printSortedArray(int* array);
void printBuckets(struct Bucket* buckets);
void printInitialBigBuckets(struct Bucket* bigBuckets);
void printBigBuckets(struct Bucket* bigBuckets);
void printSortedBuckets(struct Bucket* buckets);
void printBucketsOffsets(int* bucketsOffsets);
void printAndSaveResults(double tfillArray, double tSplitArray, double tMergeBuckets, double tSort, double tWriteBucketsToArray, double tWhole);

int main(int argc, char* argv[])
{
  double startTime, tmp, tfillArray, tSplitArray, tMergeBuckets, tSort, tWriteBucketsToArray, tWhole;
  time_t t;
  int* array = (int*)malloc(sizeof(int) * ARRAY_SIZE);

  int num_threads = atoi(argv[1]);

  srand((unsigned) time(&t));

  struct Bucket* buckets;
  int* bucketsOffsets;
  omp_lock_t* locks;
  int threadIdx = 0;

  startTime = omp_get_wtime();
  #pragma omp parallel num_threads(num_threads)
  {
    int threadId;
    #pragma omp critical
    threadId = threadIdx++;

    allocateBuckets(&buckets, &bucketsOffsets, &locks);
    clearBuckets(buckets);

    #pragma omp single
    tmp = omp_get_wtime();
    fillArray(array);
    #pragma omp single
    tfillArray = omp_get_wtime() - tmp;
    printInitialArray(array);


    #pragma omp single
    tmp = omp_get_wtime();
    splitArrayToBuckets(buckets, array, locks, threadId);
    #pragma omp single
    tSplitArray = omp_get_wtime() - tmp;
    printBuckets(buckets);

    #pragma omp barrier
    tmp = omp_get_wtime();
    sortBuckets(buckets);
    #pragma omp single
    tSort = omp_get_wtime() - tmp;
    //printSortedBuckets(buckets);

    #pragma omp single
    tmp = omp_get_wtime();
    calculateBucketsOffsets(buckets, bucketsOffsets);
    printBucketsOffsets(bucketsOffsets);
    writeBucketsToArray(buckets, bucketsOffsets, array);
    #pragma omp single
    tWriteBucketsToArray = omp_get_wtime() - tmp;
    printSortedArray(array);

    freeBuckets(buckets, bucketsOffsets, locks);
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
void clearBuckets(struct Bucket* buckets){
  int i;
  #pragma omp for private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    buckets[i].elems = NULL;
    buckets[i].len = 0;
  }
}

void splitArrayToBuckets(struct Bucket* buckets, int* array, omp_lock_t* locks, int threadId){
  //int i, j, bucketId, num_threads = omp_get_num_threads();
  int i, j, bucketId;
  #pragma omp for private(i)
  for(i = 0; i < ARRAY_SIZE; i++){
    bucketId = ((double)array[i] / MAX_ELEM) * BUCKETS_NUM;
    omp_set_lock(&locks[bucketId]);
    buckets[bucketId].len++;
    omp_unset_lock(&locks[bucketId]);
  }
  #pragma omp for private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    buckets[i].elems = (int*)malloc(sizeof(int) * buckets[i].len);
    buckets[i].len = 0;
  }
  #pragma omp for private(i)
  for(i = 0; i < ARRAY_SIZE; i++){
    bucketId = ((double)array[i] / MAX_ELEM) * BUCKETS_NUM;
    omp_set_lock(&locks[bucketId]);
    buckets[bucketId].elems[buckets[bucketId].len] = array[i];
    buckets[bucketId].len++;
    omp_unset_lock(&locks[bucketId]);
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

void sortBuckets(struct Bucket* buckets){
  int i;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    quickSort(buckets[i].elems, 0, buckets[i].len - 1);
  }
}

void calculateBucketsOffsets(struct Bucket* buckets, int* bucketsOffsets){
  #pragma omp single
  {
    int i, totalOffset = 0;
    for(i = 0; i < BUCKETS_NUM; i++){
      bucketsOffsets[i] = totalOffset;
      totalOffset += buckets[i].len;
    }
  }
}

void writeBucketsToArray(struct Bucket* buckets, int* bucketsOffsets, int* array){
  int i, j;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    int idx = bucketsOffsets[i];
    for(j = 0; j < buckets[i].len; j++){
      array[idx] = buckets[i].elems[j];
      idx++;
    }
  }
}


// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Bucket** buckets, int** bucketsOffsets, omp_lock_t** locks){
  #pragma omp single
  {
    int i, numThreads = omp_get_num_threads();
    *buckets = (struct Bucket*)malloc(sizeof(struct Bucket) * BUCKETS_NUM);
    *bucketsOffsets = (int*)malloc(sizeof(int) * BUCKETS_NUM);
    *locks = (omp_lock_t*)malloc(sizeof(omp_lock_t) * BUCKETS_NUM);
  }
}

void freeBuckets(struct Bucket* buckets, int* bucketsOffsets, omp_lock_t* locks){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    int i;
//    for(i = 0; i < BUCKETS_NUM; i++){
//      free(buckets[i].elems);
//    }
    free(buckets);
    free(bucketsOffsets);
    free(locks);
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

void printBuckets(struct Bucket* buckets){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      int i, j, k;
      printf("Buckets:\n");
      for(j = 0; j < BUCKETS_NUM; j++){
        printf("Buckett id: %u\telems: ", j);
        for(k = 0; k < buckets[j].len; k++){
          printf("%d ", buckets[j].elems[k]);
        }
        printf("\n");
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

void printSortedBuckets(struct Bucket* buckets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      printf("Sorted Buckets:\n");
      printBuckets(buckets);
      printf("after printing");
    }
  }
}

void printBucketsOffsets(int* bucketsOffsets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      int i;
      printf("BucketsOffsets:\n");
      for(i = 0; i < BUCKETS_NUM; i++){
        printf("Offset of Buckets %d = %d\n", i, bucketsOffsets[i]);
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

  FILE* results = fopen("resultsAlg2.txt", "w");
  fprintf(results, "Filling initial array time:\t\t%f seconds\n", tfillArray);
  fprintf(results, "Splitting numbers to buckets time:\t%f seconds\n", tSplitArray);
  fprintf(results, "Merging buckets time:\t\t\t%f seconds\n", tMergeBuckets);
  fprintf(results, "Sorting buckets time:\t\t\t%f seconds\n", tSort);
  fprintf(results, "Writing buckets to array time:\t\t%f seconds\n", tWriteBucketsToArray);
  fprintf(results, "Total time:\t\t\t\t%f seconds\n", tWhole);
  fclose(results);
}
