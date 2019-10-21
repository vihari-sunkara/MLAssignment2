#include <stdio.h>
extern "C"{
int simple_func(void) {
    static int counter = 0;
    counter++;
    return counter;
}
}