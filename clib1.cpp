#include <stdio.h>



int simple_function(void) {
    static int counter = 0;
    counter++;
    return counter;
}