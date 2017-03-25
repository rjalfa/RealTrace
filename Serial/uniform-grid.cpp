#include "uniform-grid.h"

void Voxel::addPrimitive(Triangle * p, int i) {
	// cout << p << endl;
	idx.push_back(i);
	primitives.push_back(p);
}

bool Voxel::intersect(Ray& ray) {
	// cout << "in" << endl;
	bool hitSomething = false;
	for(int i = 0; i < primitives.size(); i++) {
		// cerr << "this ray: " << ray.getOrigin() << " " << ray.getDirection() << endl; 
		// cout << "---" << endl;
		// cout << primitives[i] << endl;
		// cout << "---" << endl;
		// cout << idx[i] << " " << ray.getParameter() << endl;
		if(primitives[i]->intersect(ray)) {
			// cerr << "hit" << endl;
			// cerr << ray.getOrigin() << " " << ray.getDirection() << " " << ray.didHit() << " " << ray.intersected() << endl;
			// cerr << "ok" << endl;
			// cerr << ray.getDirection() << "hit" << endl;
			// cout << ray.getDirection() << " " << idx[i] << endl;
			// cout << idx[i] << " " << ray.didHit() << endl;
			ray.setIdx(idx[i]);
			hitSomething = true;
		}
	}
	// cout << "out" << endl;
	return hitSomething;
}

double UniformGrid::findVoxelsPerUnitDist(vector < double > delta, int num) {
	double maxAxis = max(delta[0], max(delta[1], delta[2]));
	double invMaxWidth = 1.0 / maxAxis;
	double cubeRoot = 3.0 * pow(num, 1.0 / 3.0);
	double voxelsPerUnitDist = cubeRoot * invMaxWidth;
	return voxelsPerUnitDist;
}

int UniformGrid::posToVoxel(const Vector3D& pos, int axis) {
	int v = ((pos.e[axis] - bounds.axis_min[axis]) * invWidth[axis]);
	return min(max(v, 0), nVoxels[axis] - 1);
}

float UniformGrid::voxelToPos(int p, int axis) {
	return bounds.axis_min[axis] + p * width[axis];
}

inline int UniformGrid::offset(double x, double y, double z) {
	return z * nVoxels[0] * nVoxels[1] + y * nVoxels[0] + x;
}

UniformGrid::UniformGrid(vector < Triangle * > &p) {
	// for(auto val : p)
	// 	cout << val << endl;
	// cerr << "in" << endl;
	delta.resize(3);
	nVoxels.resize(3);
	width.resize(3), invWidth.resize(3);
	// assuming triangle objects do not require more refinements
	
	// cerr << p.size() << endl;
	for(int i = 0; i < p.size(); i++) {
		// cerr << i << ": ";
		for(int j = 0; j < 3; j++) {
			// cerr << j << " ";
			BBox temp = p[i]->getWorldBound();
			for(int axis = 0; axis < 3; axis++) {
				bounds.axis_min[axis] = min(bounds.axis_min[axis], temp.axis_min[axis]);
				bounds.axis_max[axis] = max(bounds.axis_max[axis], temp.axis_max[axis]);
			}
		}
		// cerr << endl;
	}
	// cerr << "bounds done" << endl;
		
	for(int axis = 0; axis < 3; axis++)
		delta[axis] = (bounds.axis_max[axis] - bounds.axis_min[axis]);

	// cerr << "delta done" << endl;
	// find voxelsPerUnitDist
	voxelsPerUnitDist = findVoxelsPerUnitDist(delta, p.size());

	// cerr << "voxels found: " <<  voxelsPerUnitDist << endl;
	for(int axis = 0; axis < 3; axis++) {
		nVoxels[axis] = ceil(delta[axis] * voxelsPerUnitDist);
		nVoxels[axis] = max(1, nVoxels[axis]);
		// 64 is magically determined number, lol
		nVoxels[axis] = min(nVoxels[axis], 64);
	}
	// cerr << "voxels per axis found: " << nVoxels[0] << " " << nVoxels[1] << " " << nVoxels[2] << endl;

	nv = 1;
	for(int axis = 0; axis < 3; axis++) {
		width[axis] = delta[axis] / nVoxels[axis];
		invWidth[axis] = (width[axis] == 0.0L) ? 0.0L : 1.0 / width[axis];
		nv *= nVoxels[axis];
	}

	// cerr << "total voxels: " << nv << endl;
	// voxels = new Voxel * [nv];
	voxels.resize(nv);

	// cerr << "allocated space" << endl;

	for(int i = 0; i < p.size(); i++) {
		BBox pb = p[i]->getWorldBound();
		int vmin[3], vmax[3];
		for(int axis = 0; axis < 3; axis++) {
			vmin[axis] = posToVoxel(Vector3D(pb.axis_min[0], pb.axis_min[1], pb.axis_min[2]), axis);
			vmax[axis] = posToVoxel(Vector3D(pb.axis_max[0], pb.axis_max[1], pb.axis_max[2]), axis);
		}

		// cerr << i << ": " << p[i].getVertex(0) << " " << p[i].getVertex(1) << " " << p[i].getVertex(2) << endl;
		// cerr << Vector3D(pb.axis_min[0], pb.axis_min[1], pb.axis_min[2]) << endl;
		// cerr << Vector3D(pb.axis_max[0], pb.axis_max[1], pb.axis_max[2]) << endl;

		for(int z = vmin[2]; z <= vmax[2]; z++) {
			for(int y = vmin[1]; y <= vmax[1]; y++) {
				for(int x = vmin[0]; x <= vmax[0]; x++) {
					int o = offset(x, y, z);
					// cerr << "offset: " << o << endl;
					// to-do
					// if(!voxels[o]) {
					// 	// cerr << "offset: " << o << endl;
					// 	voxels[o] = new Voxel;
					// }
					// voxels[o]->addPrimitive(p[i]);
					voxels[o].addPrimitive(p[i], i);
				}
			}
		}
		// for(int o = 0; o < nv; o++)
		// 	voxels[o].addPrimitive(p[i], i);
	}
	// for(int i = 0; i < nv; i++) {
	// 	cout << "voxel: " << i << endl;
	// 	for(auto val : voxels[i].primitives)
	// 		cout << val << endl;
	// 	cout << "-------" << endl;
	// }
}

bool UniformGrid::intersect(Ray& ray) {
	// check ray against overall grid bounds
	double rayT;
	// cout << ray.getDirection() << endl;
	// cerr << ray.getOrigin() << " " << ray.getDirection() << endl;
	bool flag = true;
	{
		double tmin = (bounds.axis_min[0] - ray.getOrigin().X()) / ray.getDirection().X();
		double tmax = (bounds.axis_max[0] - ray.getOrigin().X()) / ray.getDirection().X();
		if(tmin > tmax) swap(tmin, tmax);
		 
		float tymin = (bounds.axis_min[1] - ray.getOrigin().Y()) / ray.getDirection().Y(); 
		float tymax = (bounds.axis_max[1] - ray.getOrigin().Y()) / ray.getDirection().Y(); 

		if (tymin > tymax) swap(tymin, tymax); 

		if ((tmin > tymax) || (tymin > tmax)) 
		    flag = false;

		if (tymin > tmin) 
		    tmin = tymin; 

		if (tymax < tmax) 
		    tmax = tymax; 

		float tzmin = (bounds.axis_min[2] - ray.getOrigin().Z()) / ray.getDirection().Z(); 
		float tzmax = (bounds.axis_max[2] - ray.getOrigin().Z()) / ray.getDirection().Z(); 

		if (tzmin > tzmax) swap(tzmin, tzmax); 

		if ((tmin > tzmax) || (tzmin > tmax)) 
		    flag = false;

		if (tzmin > tmin) 
		    tmin = tzmin; 

		if (tzmax < tmax) 
		    tmax = tzmax; 

		rayT = tmin;
		if(flag) {
			ray.strictSetParameter(rayT);
		}
	}

	// cerr << rayT << endl;
	// return false;
	// cout << ray.getDirection() << " " << flag << endl; 
	if(!flag) return false;

	Vector3D gridIntersect = ray.getPosition();

	int pos[3], step[3], out[3];
	double nextCrossingT[3], deltaT[3];
	// set up 3D DDA for ray
	for(int axis = 0; axis < 3; axis++) {
		// compute current voxel for axis
		pos[axis] = posToVoxel(gridIntersect, axis);
		if(ray.getDirection().e[axis] >= 0) {
			// handle ray with positive direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect[axis]) / ray.getDirection().e[axis];
			deltaT[axis] = width[axis] / ray.getDirection().e[axis];
			step[axis] = 1;
			out[axis] = nVoxels[axis];
		} else {
			// handle ray with negative direction for voxel stepping
			nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect[axis]) / ray.getDirection().e[axis];
			deltaT[axis] = -width[axis] / ray.getDirection().e[axis];
			step[axis] = -1;
			out[axis] = -1;
		}
		// cerr << pos[axis] << " " << step[axis] << " " << out[axis] << endl;
	}
	// walk ray through voxel grid
	bool hitSomething = false;
	ray.strictSetParameter((FLT_MAX));
	for( ; ; ) {
		// check for intersection in current voxel and advance to next
		// Voxel * voxel = voxels[offset(pos[0], pos[1], pos[2])];
		Voxel& voxel = voxels[offset(pos[0], pos[1], pos[2])];
		// cout << "begin" << endl;
		// cout << ray.getDirection() << " " << hitSomething << " " << offset(pos[0], pos[1], pos[2]) << " " << ray.didHit() << endl; 
		if(voxel.primitives.size() != 0)
			hitSomething |= voxel.intersect(ray);
		// cout << ray.getDirection() << " " << hitSomething << " " << offset(pos[0], pos[1], pos[2]) << " " << ray.didHit() << endl; 
		// cout << "end" << endl;
		// if(voxel != NULL)
		// 	hitSomething |= voxel->intersect(ray);
		// advance to next voxel
		// find stepAxis for stepping to next voxel
		int bits =  ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
					((nextCrossingT[0] < nextCrossingT[2]) << 1) +
					((nextCrossingT[1] < nextCrossingT[2]));
		const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
		int stepAxis = cmpToAxis[bits];

		// if(ray.maxt < nextCrossingT[stepAxis])
		// 	break;
		pos[stepAxis] += step[stepAxis];
		if(pos[stepAxis] == out[stepAxis])
			break;
		if(hitSomething) break;
		nextCrossingT[stepAxis] += deltaT[stepAxis];
	}
	return hitSomething;
}