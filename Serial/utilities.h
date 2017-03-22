#ifndef _UTILITIES_H
#define _UTILITIES_H

double determinant(double a,double b,double c,double d);
double abs(double a);
double determinant(Vector3D col1, Vector3D col2, Vector3D col3);

class BBox {
public:
	double axis_min[3], axis_max[3];
	BBox() {
		// cout << std::numeric_limits < double >::max << endl;
		axis_min[0] = axis_min[1] = axis_min[2] = std::numeric_limits < double >::max();
		axis_max[0] = axis_max[1] = axis_max[2] = std::numeric_limits < double >::min();
	}
};

#endif