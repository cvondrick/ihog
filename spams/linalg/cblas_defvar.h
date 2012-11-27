#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>

/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t  /* this may vary between platforms */

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};


char CBLAS_TRANSPOSE_CHAR[] = {'N', 'T', 'C'};
char *cblas_transpose(CBLAS_TRANSPOSE TransA)
{
	switch(TransA)
	{
		case 111:	return &CBLAS_TRANSPOSE_CHAR[0];
		case 112:	return &CBLAS_TRANSPOSE_CHAR[1];
		case 113:	return &CBLAS_TRANSPOSE_CHAR[2];
	}
	return NULL;
}

char CBLAS_UPLO_CHAR[] = {'U', 'L'};
char *cblas_uplo(CBLAS_UPLO Uplo)
{
	switch(Uplo)
	{
		case 121:	return &CBLAS_UPLO_CHAR[0];
		case 122:	return &CBLAS_UPLO_CHAR[1];
	}
	return NULL;
}

char CBLAS_DIAG_CHAR[] = {'N', 'U'};
char *cblas_diag(CBLAS_DIAG Diag)
{
	switch(Diag)
	{
		case 131:	return &CBLAS_DIAG_CHAR[0];
		case 132:	return &CBLAS_DIAG_CHAR[1];
	}
	return NULL;
}

char CBLAS_SIDE_CHAR[] = {'L', 'R'};
char *cblas_side(CBLAS_SIDE Side)
{
	switch(Side)
	{
		case 141:	return &CBLAS_SIDE_CHAR[0];
		case 142:	return &CBLAS_SIDE_CHAR[1];
	}
	return NULL;
}

#endif
