#include "kdtree.h"
#include <algorithm>
using namespace std;
void KdTreeNode::initInterior(int axis, int ac, float s)
{
	split = s;
	flags = axis;
	aboveChild |= (ac << 2);
}

void KdTreeNode::initLeaf(vector<Triangle*>& prims, vector<int>& primNumIdx)
{
	copy(prims.begin(), prims.end(),primitives.begin());
	copy(primNumIdx.begin(), primNumIdx.end(),idx.begin());
}

KdTree::KdTree(const vector<Triangle*> &primitives)
{
	//Assuming refinement not required
	//Build kd-tree for accelerator
	nextFreeNode = nAllocedNodes = 0;
	if (maxDepth <= 0) maxDepth = 8 + 1.3f * log(primitives.size())/log(2.0f);

	//Compute bounds for kd-tree construction
	vector<BBox> primBounds;
	for(int i = 0; i < primitives.size(); i++) {
		// cerr << i << ": ";
		for(int j = 0; j < 3; j++) {
			// cerr << j << " ";
			BBox temp = primitives[i]->getWorldBound();
			for(int axis = 0; axis < 3; axis++) {
				bounds.axis_min[axis] = min(bounds.axis_min[axis], temp.axis_min[axis]);
				bounds.axis_max[axis] = max(bounds.axis_max[axis], temp.axis_max[axis]);
			}
			primBounds.push_back(temp);
		}
	}
	//Allocate working memory for kd-tree construction
	//Initialize primNums for kd-tree construction
	vector<int> primNums(primitives.size());
	for(int i = 0; i < primitives.size(); i++) primNums[i] = i;
	//Start recursive construction of kd-tree
	buildTree(0, bounds, primBounds, primNums, primitives.size(), maxDepth, edges, prims0, prims1);
	//Free working memory for kd-tree construction
}

void KdTreeAccel::buildTree(int nodeNum, const BBox &nodeBounds, const vector<BBox> &allPrimBounds, uint32_t *primNums, int nPrimitives, int depth, BoundEdge *edges[3], uint32_t *prims0, uint32_t *prims1, int badRefines)
{
	//Get next free node from nodes array
	//Initialize leaf node if termination criteria met
	//Initialize interior node and continue recursion
}