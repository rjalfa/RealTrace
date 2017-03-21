class Voxel {
	vector < Triangle& > primitives;
public:
	void addPrimitive(Triangle& p) {
		primitives.push_back(p);
	}

	bool intersect(Ray& ray) {
		bool hitSomething = false;
		for(int i = 0; i < primitives.size(); i++) {
			if(primitives[i].intersect(ray))
				hitSomething = true;
		}
		return hitSomething;
	}
};


class UniformGrid {

private:
	BBox bounds;
	vector < double > delta(3);
	vector < int > nVoxels(3);
	int voxelsPerUnitDist;
	vector < double > width(3), invWidth(3);
	vector < Voxel * > voxels(nv, 0);

	int findVoxelsPerUnitDist(vector < double > delta, int num) {
		double maxAxis = max({delta[0], delta[1], delta[2]});
		double invMaxWidth = 1.0 / maxAxis;
		double cubeRoot = 3.0 * pow(num, 1.0 / 3.0);
		double voxelsPerUnitDist = cubeRoot * invMaxWidth;
		return voxelsPerUnitDist;
	}

	int posToVoxel(const Vector3D& pos, int axis) {
		int v = ((p.e[axis] - bounds.axis_min[axis_min]) * invWidth[axis]);
		return min(max(v, 0), nVoxels[axis] - 1);
	}

	float voxelToPos(int p, int axis) const {
		return bounds.axis_min[axis] + p * width[axis];
	}

	inline int offset(double x, double y, double z) {
		return z * nVoxels[0] * nVoxels[1] + y * nVoxels[0] + x;
	}

public:
	UniformGrid(vector < Triangle& > &p) {
		// assuming triangle objects do not require more refinements
		
		for(int i = 0; i < p.size(); i++) {
			for(int j = 0; j < 3; j++) {
				BBox temp = p[i].getWorldBound();
				for(int axis = 0; axis < 3; axis++) {
					bounds.axis_min[axis] = temp.axis_min[axis];
					bounds.axis_max[axis] = temp.axis_max[axis];
				}
			}
		}

		
		for(int axis = 0; axis < 3; axis++)
			delta[axis] = (bounds.axis_max[axis] - bounds.axis_min[axis]);

		// find voxelsPerUnitDist
		voxelsPerUnitDist = findVoxelsPerUnitDist(delta, p.size());

		for(int axis = 0; axis < 3; axis++) {
			nVoxels[axis] = ceil(delta[axis] * voxelsPerUnitDist);
			nVoxels[axis] = max(1, nVoxels[axis]);
			// 64 is magically determined number, lol
			nVoxels[axis] = min(nVoxels[axis], 64);
		}

		int nv = 1;
		for(int axis = 0; axis < 3; axis++) {
			width[axis] = delta[axis] / nVoxels[axis];
			invWidth[axis] = (width[axis] == 0.0L) ? 0.0L : 1.0 / width[axis];
			nv *= nVoxels[axis];
		}

		for(int i = 0; i < p.size(); i++) {
			BBox pb = p[i].getWorldBound();
			int vmin[3], vmax[3];
			for(int axis = 0; axis < 3; axis++) {
				vmin[axis] = posToVoxel(pb.axis_min[axis], axis);
				vmax[axis] = posToVoxel(pb.axis_max[axis], axis);
			}

			for(int z = vmin[2]; z <= vmax[2]; z++) {
				for(int y = vmin[1]; y <= vmax[1]; y++) {
					for(int x = vmin[0]; x <= vmax[0]; x++) {
						int o = offset(x, y, z);
						// to-do
						if(!voxels[o]) voxels[o] = new Voxel;
						voxels[o].addPrimitive(p[i]);
					}
				}
			}
		}
	}

	bool intersect(const Ray& ray) const {
		// check ray against overall grid bounds
		float rayT;
		if()

		Vector3D gridIntersect = ray.getPosition(rayT);
	
		int pos[3], step[3], out[3];
		double nextCrossingT[3], deltaT[3];
		// set up 3D DDA for ray
		for(int axis = 0; axis < 3; axis++) {
			// compute current voxel for axis
			pos[axis] = posToVoxel(gridIntersect, axis);
			if(ray.getDirection.e[axis] >= 0) {
				// handle ray with positive direction for voxel stepping
				nextCrossingT[axis] = rayT + (voxelToPos(pos[axis] + 1, axis) - gridIntersect[axis]) / ray.getDirection.e[axis];
				deltaT[axis] = width[axis] / ray.getDirection.e[axis];
				step[axis] = 1;
				out[axis] = nVoxels[axis];
			} else {
				// handle ray with negative direction for voxel stepping
				nextCrossingT[axis] = rayT + (voxelToPos(pos[axis], axis) - gridIntersect[axis]) / ray.getDirection.e[axis];
				deltaT[axis] = -width[axis] / ray.getDirection.e[axis];
				step[axis] = -1;
				out[axis] = -1;
			}
		}
		// walk ray through voxel grid
		bool hitSomething = false;
		for( ; ; ) {
			// check for intersection in current voxel and advance to next
			Voxel * voxel = voxels[offset(pos[0], pos[1], pos[2])];
			if(voxel != NULL)
				hitSomething = |= voxel->intersect(ray);
			// advance to next voxel
			// find stepAxis for stepping to next voxel
			int bits =  ((nextCrossingT[0] < nextCrossingT[1]) << 2) +
						((nextCrossingT[0] < nextCrossingT[2]) << 1) +
						((nextCrossingT[1] < nextCrossingT[2]));
			const int cmpToAxis[8] = {2, 1, 2, 1, 2, 2, 0, 0};
			int stepAxis = cmpToAxis[bits];

			if(ray.maxt < nextCrossingT[stepAxis])
				break;
			pos[stepAxis] += step[stepAxis];
			if(pos[stepAxis] == out[stepAxis])
				break;
			nextCrossingT[stepAxis] += deltaT[stepAxis];
		}
	}
};