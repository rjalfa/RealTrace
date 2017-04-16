// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(Triangle * triangles, int idx, float extremes[3][2]) {
	x = (x - extremes[0][0]) / (extremes[0][1] - extremes[0][0]);
	y = (y - extremes[1][0]) / (extremes[1][1] - extremes[1][0]);
	z = (z - extremes[2][0]) / (extremes[2][1] - extremes[2][0]);
    x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}


const int MAXN;

int bvh_tree_low[MAXN], bvh_tree_high[MAXN];

BBox bvh_tree[MAXN];


void constructBVH(Triangle * triangles, int n) {
	float extremes[3][2];
	
}