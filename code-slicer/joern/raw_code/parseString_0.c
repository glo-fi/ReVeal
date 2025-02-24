#include <stdio.h>
#include <string.h>

/*
 * A single function that tokenizes a string.
 * It splits the input on spaces and prints each token.
 */
void parseString(char *input) {
    char *token = strtok(input, " ");
    while (token != NULL) {
        printf("Token: %s\n", token);
        token = strtok(NULL, " ");
    }
}
