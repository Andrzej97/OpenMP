#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define ARRAY_SIZE 1000000
#define BUCKETS_NUM 100
#define MAX_ELEM 1000

// logging modes:
#define DBG 0
#define RES 1
#define NONE 2
#define LOG_MODE NONE  // can be changed

#define QUICK 0
#define INSERT 1
#define SORT_MODE QUICK // can be changed

#define min(a,b) \
  ({ __typeof__ (a) _a = (a); \
      __typeof__ (b) _b = (b); \
    _a < _b ? _a : _b; })

struct Node{
  int val;
  struct Node* next;
};

void fillArray(int* array);

// BUCKET SORT METHODS:
void clearBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes, int* bigBucketsOffsets, int threadId);
void splitArrayToSmallBuckets(struct Node*** smallBuckets, int* array, int threadId);
void mergeBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes);
void sortBigBuckets(struct Node** bigBuckets);
void calculateBigBucketsOffsets(int* bigBucketsOffsets, int* bigBucketsSizes);
void writeBigBucketsToArray(struct Node** bigBuckets, int* bigBucketsOffsets, int* array);

// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Node**** smallBuckets, struct Node*** bigBuckets, int** bigBucketsSizes, int** bigBucketsOffsets);
void addNode(struct Node** head, int value);
void deleteAllNodes(struct Node* head);
void freeBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes, int* bigBucketsOffsets);

// QUICK SORT METHODS:
void quickSort(struct Node** head);
struct Node* partition(struct Node** head, struct Node** tail);
struct Node* quickSortRec(struct Node* head, struct Node* tail);
struct Node* getTail(struct Node* head);

// INSERTION SORT METHODS:
void insertionSort(struct Node** head);
void addNodeSorted(struct Node** head, struct Node* elem);

// PRINTING METHODS:
void printConfig();
void printArray(int* array);
void printInitialArray(int* array);
void printSortedArray(int* array);
void printSmallBuckets(struct Node*** smallBuckets);
void printInitialBigBuckets(struct Node** bigBuckets);
void printBigBuckets(struct Node** bigBuckets);
void printSortedBigBuckets(struct Node** bigBuckets);
void printBigBucketsOffsets(int* bigBucketsOffsets);
void printAndSaveResults(double tfillArray, double tSplitArray, double tMergeBuckets, double tSort, double tWriteBucketsToArray, double tWhole);

int main(int argc, char* argv[])
{
  double startTime, tmp, tfillArray, tSplitArray, tMergeBuckets, tSort, tWriteBucketsToArray, tWhole;
  time_t t;
  int* array = (int*)malloc(sizeof(int) * ARRAY_SIZE);

  srand((unsigned) time(&t));

  struct Node*** smallBuckets;
  struct Node** bigBuckets;
  int* bigBucketsSizes;
  int* bigBucketsOffsets;
  int threadIdx = 0;

  startTime = omp_get_wtime();
  #pragma omp parallel
  {
    int threadId;
    #pragma omp critical
    threadId = threadIdx++;

    allocateBuckets(&smallBuckets, &bigBuckets, &bigBucketsSizes, &bigBucketsOffsets);
    clearBuckets(smallBuckets, bigBuckets, bigBucketsSizes, bigBucketsOffsets, threadId);

    tmp = omp_get_wtime();
    fillArray(array);
    tfillArray = omp_get_wtime() - tmp;
    printInitialArray(array);

    tmp = omp_get_wtime();
    splitArrayToSmallBuckets(smallBuckets, array, threadId);
    tSplitArray = omp_get_wtime() - tmp;
    printSmallBuckets(smallBuckets);

    tmp = omp_get_wtime();
    mergeBuckets(smallBuckets, bigBuckets, bigBucketsSizes);
    tMergeBuckets = omp_get_wtime() - tmp;
    printInitialBigBuckets(bigBuckets);

    tmp = omp_get_wtime();
    sortBigBuckets(bigBuckets);
    tSort = omp_get_wtime() - tmp;
    printSortedBigBuckets(bigBuckets);

    tmp = omp_get_wtime();
    calculateBigBucketsOffsets(bigBucketsOffsets, bigBucketsSizes);
    printBigBucketsOffsets(bigBucketsOffsets);
    writeBigBucketsToArray(bigBuckets, bigBucketsOffsets, array);
    tWriteBucketsToArray = omp_get_wtime() - tmp;
    printSortedArray(array);

    freeBuckets(smallBuckets, bigBuckets, bigBucketsSizes, bigBucketsOffsets);
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
  #pragma omp for schedule(static) private(i)
  for(i = 0; i <ARRAY_SIZE; i++){
    array[i] = rand_r(&threadNum) % MAX_ELEM;
  }
}


// BUCKET SORT METHODS:
void clearBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes, int* bigBucketsOffsets, int threadId){
  int i;
  #pragma omp for private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    smallBuckets[threadId][i] = NULL;
    bigBuckets[i] = NULL;
    bigBucketsSizes[i] = 0;
    bigBucketsOffsets[i] = 0;
  }
}

void splitArrayToSmallBuckets(struct Node*** smallBuckets, int* array, int threadId){
  int i, bucketId;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i <ARRAY_SIZE; i++){
    bucketId =((double)array[i] / MAX_ELEM) * BUCKETS_NUM;
    addNode(&smallBuckets[threadId][bucketId], array[i]);
  }
}

void mergeBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes){
  int i, j, numThreads = omp_get_num_threads();;
  #pragma omp for schedule(static) private(i, j)
  for(i = 0; i < BUCKETS_NUM; i++){
    for(j = 0; j < numThreads; j++){
      struct Node* tmp = smallBuckets[j][i];
      while(tmp != NULL){
        addNode(&bigBuckets[i], tmp->val);
        bigBucketsSizes[i]++;
        tmp = tmp->next;
      }
    }
  }
}

void sortBigBuckets(struct Node** bigBuckets){
  int i;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    if(INSERT == SORT_MODE){
      insertionSort(&bigBuckets[i]);
    }
    else if(QUICK == SORT_MODE){
      quickSort(&bigBuckets[i]);
    }
  }
}

void calculateBigBucketsOffsets(int* bigBucketsOffsets, int* bigBucketsSizes){
  #pragma omp single
  {
    int i, totalOffset = 0;
    for(i = 0; i < BUCKETS_NUM; i++){
      bigBucketsOffsets[i] = totalOffset;
      totalOffset += bigBucketsSizes[i];
    }
  }
}

void writeBigBucketsToArray(struct Node** bigBuckets, int* bigBucketsOffsets, int* array){
  int i;
  #pragma omp for schedule(static) private(i)
  for(i = 0; i < BUCKETS_NUM; i++){
    int idx = bigBucketsOffsets[i];
    struct Node* tmp = bigBuckets[i];
    while(NULL != tmp){
      array[idx] = tmp->val;
      idx++;
      tmp = tmp->next;
    }
  }
}


// MEMORY MANAGEMENT FUNCTIONS:
void allocateBuckets(struct Node**** smallBuckets, struct Node*** bigBuckets, int** bigBucketsSizes, int** bigBucketsOffsets){
  #pragma omp single
  {
    int i, numThreads = omp_get_num_threads();
    *smallBuckets = (struct Node***)malloc(sizeof(struct Node**) * numThreads);
    for(i = 0; i < numThreads; i++){
      (*smallBuckets)[i] = (struct Node**)malloc(sizeof(struct Node*) * BUCKETS_NUM);
    }
    *bigBuckets = (struct Node**)malloc(sizeof(struct Node*) * BUCKETS_NUM);
    *bigBucketsSizes = (int*)malloc(sizeof(int) * BUCKETS_NUM);
    *bigBucketsOffsets = (int*)malloc(sizeof(int) * BUCKETS_NUM);
  }
}

void addNode(struct Node** head, int value){
  struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
  newNode->val = value;
  newNode->next = *head;
  *head = newNode;
}

void deleteAllNodes(struct Node* head){
  struct Node* tmp;
  while(head != NULL){
    tmp = head;
    head = head->next;
    free(tmp);
  }
}

void freeBuckets(struct Node*** smallBuckets, struct Node** bigBuckets, int* bigBucketsSizes, int* bigBucketsOffsets){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    int i, j;
    for(i = 0; i < numThreads; i++){
      for(j = 0; j < BUCKETS_NUM; j++){
        deleteAllNodes(smallBuckets[i][j]);
      }
      free(smallBuckets[i]);
    }
    free(smallBuckets);

    for(i = 0; i < BUCKETS_NUM; i++){
      deleteAllNodes(bigBuckets[i]);
    }
    free(bigBuckets);

    free(bigBucketsSizes);
    free(bigBucketsOffsets);
  }
}


// QUICK SORT METHODS:
void quickSort(struct Node** head){
  *head = quickSortRec(*head, getTail(*head));
}

struct Node* partition(struct Node** head, struct Node** tail)
{
    struct Node* cur = *head;
    struct Node* pivot = *tail;
    struct Node* tailFirstPart = NULL;
    struct Node* tailSecondPart = pivot;

    struct Node* newHead = NULL;

    while (cur != *tail)
    {
        if (cur->val < pivot->val)
        {
            if(newHead == NULL)
              newHead = cur;

            tailFirstPart = cur;
            cur = cur->next;
        }
        else
        {
            if (tailFirstPart)
                tailFirstPart->next = cur->next;
            struct Node *tmp = cur->next;
            cur->next = NULL;
            tailSecondPart->next = cur;
            tailSecondPart = cur;
            cur = tmp;
        }
    }

    *head = newHead;
    if(*head == NULL)
      *head = pivot;
    *tail = tailSecondPart;

    return pivot;
}

struct Node* quickSortRec(struct Node* head, struct Node* tail)
{
    if (!head || head == tail)
        return head;

    struct Node *pivot = partition(&head, &tail);

    if (head != pivot)
    {
        struct Node *tmp = head;
        while (tmp->next != pivot)
            tmp = tmp->next;
        tmp->next = NULL;
        head = quickSortRec(head, tmp);
        tmp = getTail(head);
        tmp->next = pivot;
    }
    pivot->next = quickSortRec(pivot->next, tail);
    return head;
}

struct Node* getTail(struct Node* head){
  while(NULL != head && head->next != NULL){
    head = head->next;
  }
  return head;
}


// INSERTION SORT METHODS:
void insertionSort(struct Node** head){
  struct Node* sorted = NULL;
  struct Node* tmp;
  while(NULL != *head){
    tmp = *head;
    *head = (*head)->next;
    addNodeSorted(&sorted, tmp);
  }
  *head = sorted;
}

void addNodeSorted(struct Node** head, struct Node* elem){
  if(NULL == *head){
    *head = elem;
    (*head)->next = NULL;
    return;
  }
  if((*head)->val > elem->val){
    elem->next = *head;
    *head = elem;
    return;
  }
  struct Node* tmp = *head;
  while(tmp->next != NULL){
    if(tmp->next->val >= elem->val){
      struct Node* tmp2 = tmp->next;
      tmp->next = elem;
      elem->next = tmp2;
      return;
    }
    tmp = tmp->next;
  }
  tmp->next = elem;
  elem->next = NULL;
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

void printSmallBuckets(struct Node*** smallBuckets){
  int numThreads = omp_get_num_threads();
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      int i, j;
      struct Node* tmp;
      printf("SmallBuckets:\n");
      for(i = 0; i < numThreads; i++){
        for(j = 0; j < BUCKETS_NUM; j++){
          printf("SmallBuckets for thread id=%d\tsmall bucket id: %u\telems: ", i, j);
          tmp = smallBuckets[i][j];
          while(tmp != NULL){
            printf("%d ", tmp->val);
            tmp = tmp->next;
          }
          printf("\n");
        }
      }
      printf("\n");
    }
  }
}

void printInitialBigBuckets(struct Node** bigBuckets){
  #pragma omp single
  {
    if(DBG == LOG_MODE){
      printf("Initial BigBuckets (right after merging SmallBuckets):\n");
      printBigBuckets(bigBuckets);
    }
  }
}

void printBigBuckets(struct Node** bigBuckets){
  int i;
  for(i = 0; i < BUCKETS_NUM; i++){
    printf("Big bucket id: %d\t elems: ", i);
    struct Node* tmp = bigBuckets[i];
    while(NULL != tmp){
      printf("%d ", tmp->val);
      tmp = tmp->next;
    }
    printf("\n");
  }
  printf("\n");
}

void printSortedBigBuckets(struct Node** bigBuckets){
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

  FILE* results = fopen("results.txt", "w");
  fprintf(results, "Filling initial array time:\t\t%f seconds\n", tfillArray);
  fprintf(results, "Splitting numbers to buckets time:\t%f seconds\n", tSplitArray);
  fprintf(results, "Merging buckets time:\t\t\t%f seconds\n", tMergeBuckets);
  fprintf(results, "Sorting buckets time:\t\t\t%f seconds\n", tSort);
  fprintf(results, "Writing buckets to array time:\t\t%f seconds\n", tWriteBucketsToArray);
  fprintf(results, "Total time:\t\t\t\t%f seconds\n", tWhole);
  fclose(results);
}