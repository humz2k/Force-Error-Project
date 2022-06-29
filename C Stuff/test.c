#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int countLines(char* filename){

  FILE* filePointer;
  filePointer = fopen(filename, "r");
  int length = 0;
  if (!(filePointer == NULL)) {
    char buffer[256];
    while (fgets(buffer,256,filePointer)){
      length++;
    }
  }
  fclose(filePointer);
  return length;

}

void readFile(char* filename,float** pos,float** vel,float* mass, const char* delim){

  FILE* filePointer;
  filePointer = fopen(filename, "r");

  if (!(filePointer == NULL)) {

    char buffer[256];
    int count = -1;
    int index;

    while (fgets(buffer,256,filePointer)){
      if (count != -1){
        pos[count] = malloc(3 * sizeof(float));
        vel[count] = malloc(3 * sizeof(float));

        index = 0;

        char *line = strtok(buffer, delim);
        while (line != NULL){
          
          if (index < 3){
            pos[count][index] = atof(line);
          }
          else if (index < 6){
            vel[count][index-3] = atof(line);
          }else{
            mass[count] = atof(line);
          }
          index++;
          line = strtok(NULL, delim);
        }


      }
      count++;
    }
  }

  fclose(filePointer);

}

void print2DArray(float** array, int length, int width){
  for (int i = 0; i < length; i++){
    for (int j = 0; j < width; j++){
      printf("%lf   ",array[i][j]);
    }
    printf("\n");
  }
}

void printArray(float* array, int length){

  for (int i = 0; i < length; i++){

    printf("%lf\n",array[i]);

  }

}

void freeArrays(float** pos, float** vel, float* mass, int nrows){

  for (int i = 0; i < nrows; i++){
    free(pos[i]);
    free(vel[i]);
  }
  free(pos);
  free(vel);
  free(mass);

}

int main(int argc, char* argv[]) {

  char* filename;

  int eps = 0;

  if (argc == 1){
    fprintf(stderr,"FILE NOT SPECIFIED\n");
    exit(1);
  }

  filename = argv[1];
  for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "-eps")) eps = atoi(argv[i + 1]);
  }

  printf("READING: %s\n",filename);
  printf("EPS: %d\n",eps);

  int nrows = countLines(filename)-1;

  printf("NROWS: %d\n",nrows);

  float** pos = malloc(nrows * sizeof(float*));
  float** vel = malloc(nrows * sizeof(float*));
  float* mass = malloc(nrows * sizeof(float));
  
  readFile(filename,pos,vel,mass,",");

  print2DArray(vel,nrows,3);
  printf("\n\n");
  printArray(mass,nrows);

  freeArrays(pos,vel,mass,nrows);

  return 0;
}