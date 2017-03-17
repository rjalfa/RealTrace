#include "vector3D.h"
#include "utilities.h"

double determinant(double a,double b,double c,double d)
{
	// a b
	// c d
	return a*d - b*c;
}

double abs(double a)
{
	if(a > 0) return a;
	return -a;
}

double determinant(Vector3D col1, Vector3D col2, Vector3D col3)
{
	//cout << col1 << " " << col2 << " " << col3 << endl;
	//getchar();
	return col1.X() * determinant(col2.Y(),col3.Y(),col2.Z(),col3.Z()) - col1.Y() * determinant(col2.X(),col3.X(),col2.Z(),col3.Z()) + col1.Z() * determinant(col2.X(),col3.X(),col2.Y(),col3.Y());
}