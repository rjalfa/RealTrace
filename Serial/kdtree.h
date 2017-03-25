#ifndef _KDTREE_H_
#define _KDTREE_H_

#include <vector>
#include "triangle.h"
#include "ray.h"
#include "utilities.h"
#include "vector3D.h"

using namespace std;

//KdTree Local Declarations
class KdTreeNode {
public:
	//KdTreeNode Methods
	void initLeaf(vector<Triangle*>& prims, vector<int>& primNumIdx);
	void initInterior(int axis, int ac, float s);
	void addPrimitive(Triangle* prim, int index);

	float SplitPos() const { return split; }
	int nPrimitives() const { return nPrims >> 2; }
	int SplitAxis() const { return flags & 3; }
	bool IsLeaf() const { return (flags & 3) == 3; }
	int AboveChild() const { return aboveChild >> 2; }

	float split;
	vector < int > idx;
	vector < Triangle * > primitives;
private:
	int flags;
	int nPrims;
	int aboveChild;
};


//KdTree BSP - Full Binary
class KdTree {

private:
	//KdTree Private Data
	int isectcost; //Intersection Cost
	int traverselCost; // Traversal Cost
	int maxPrims; //maximum primitives per node
	int maxDepth; //maximum depth of tree
	float emptyBonus; //Don't know what this is 
	vector < Triangle * > primitives; // Collection of primitives

	KdTreeNode *nodes;
	int nAllocedNodes, nextFreeNode;

	BBox bounds;

public:
	KdTree() {}
	KdTree(const vector < Triangle * > &p);
	void buildTree(int nodeNum, const BBox &nodeBounds, const vector<BBox> &allPrimBounds, uint32_t *primNums, int nPrimitives, int depth, BoundEdge *edges[3], uint32_t *prims0, uint32_t *prims1, int badRefines);

	bool intersect(Ray& ray);
};

#endif