#include <stdlib.h>

int main() {
	int * temp = (int *) malloc(sizeof(int));
	free(temp);
	free(temp);
}