
//
// Header-file for calculating splines using standard libraries
//
//			usage:
//
//			x bsplines class constructs a set of splines up to some given order,
//			  on some given knot sequence, the splines are stored in a vector,
//			  such that splines[a][b] accesses the spline of order a and index b
//			x get<some_member>() is an accessor that returns a pointer to some 
//			  data member of the spline
//			x calcsplines() calculates spline values as well as first and second 
//			  order derivatives on some predefined grid
//			x calcspline() returns the spline value as well as first and second
//			  derivatives in some point. This alborithm is slower than the grid
//			  one, due to unnecessary recalculations of intermediate results
//			x writesplines() writes the splines and their derivatives to a file
//			x for more details se the class declaration below
//
//			TODO:
//			x change to 0-indexation
//			x introduce the possibility of calculating higher order derivatives 
//			  iteratively
//
//			change log:
//
//			1.0		- initial release
//			1.1		- reworked grid such that the class now expects separate
//					grid and knot files.
//					- added the ability to calculate spline value in a point
//					rather than calculate values on a grid
//					- added a feature to change knots and grid
//			1.1.1	- reworked how returning single values works
//			1.1.2	- commented stuff
//			1.1.3	- enabled swapping grid
//
// Note:
// 
// This file uses 1-indexation rathar than 0-indexation, hence a qubic spline
// would be k = 4. Someone should eventually fix this as this is non-standard.
//
// Also, while only standard libraries are used here, you might want to check out
// some linear algebra package (e.g. Armadillo or Eigen) if you're going to use the
// splines in a context where you need linear algebraic operations.
//
// Originally developed by David Andersson
//


#include <iomanip>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>	
#include <fstream>
#include <functional>

using namespace std;
typedef unsigned int uint;

class bsplines								// class for bsplines
{
	// private section
	
	uint order;								// order of spline
	uint gridpts;							// number of grid points
	uint knotpts;							// number of knot points
	double tolerance;						// tolerance for float comparisons

	vector<double> knots;					// knot sequence
	vector<double> grid;					// grid points
	
	class spline							// a member spline in the set of splines
	{
		int index;							// the spline index, or number
		
		vector<double> vals;				// spline values
		vector<double> d1;					// spline first derivatives
		vector<double> d2;					// spline second derivatives
		double tval;						// same, but in one point
		double td1;
		double td2;
	
		friend bsplines;					// for ease of access

		public:
	};
	
	vector<vector <spline>> splines;		// the set of splines

	// puclic section
	
	public:

	void readknots(string);					// read knots from file
	void readknotsnorm(string);				// read knots from file and normalize
	void readgrid(string);					// read grid from file
	void swapgrid(string);					// reads and swaps new grid from file 
	void writesplines();					// write spline vals and derivs to file
	void buildsplines();					// build the set of splines
	void calcsplines();						// calculate spline vals and derivs
	void printknots();						// print knot sequence
	void printgrid();						// print grid
	void printgridsize();					// print gridsize
	void printvals(uint,uint);				// print values of a spline
	
	vector <double> calcspline(uint,uint,double); // calculate spline in point
	
	// accessors							// returns pointers to members
											// these are the droids you are looking for
	vector <double>* getknots(){return &knots;}
	vector <double>* getgrid(){return &grid;}
	uint* getknotpts(){return &knotpts;}
	uint* getgridpts(){return &gridpts;}
	uint getnosplines(uint m){return splines[m].size();}	
	vector <spline>* getsplines(uint m){return &splines[m];}
	vector <double>* getvals(uint m, uint n){return &splines[m][n].vals;}
	vector <double>* getd1(uint m, uint n){return &splines[m][n].d1;}
	vector <double>* getd2(uint m, uint n){return &splines[m][n].d2;}

	// constructor							// sets up the spline class
	bsplines (string iknots, string igrid, uint iorder, double itol)
		:order(iorder), tolerance(itol)
	{
		readknots(iknots);
		readgrid(igrid);
		buildsplines();
	}
};

void bsplines::buildsplines()
{
	{
	for (uint l = 1; l <= order; l++)
		{
		vector <spline> splinevec;
			for (uint k = 0; k < knotpts - l; k++)
			{
				spline tmp;
				tmp.index = k;
				tmp.vals.reserve(gridpts);
				tmp.d1.reserve(gridpts);
				tmp.d2.reserve(gridpts);
				splinevec.push_back(tmp);
			}
			splines.push_back(splinevec);
		}
	}
}

vector <double> bsplines::calcspline(uint m, uint n, double x)
{
	// first order splines					// exceptions handles infinities
	
	for (auto& sp : splines[0])
	{
		uint i = sp.index;
		
		if (x > knots[i+1])
			sp.tval = 0;
		else if ((x >= knots[i] && x < knots[i+1]) || x == knots.back())
			sp.tval = 1;
		else
			sp.tval = 0;
	}
	
	// higher order splines
	
	for (uint o = 1; o < order; o++)
	{
		uint oo = o+1;						// compensating for 1-indexation
		for (auto& sp : splines[o])
		{
			uint i = sp.index;
			
			double t1 = knots[i+oo-1] - knots[i];
			double t2 = knots[i+oo] - knots[i+1];
			
			double c = 0;
				
			if (abs(t1) > tolerance)
				c += (x - knots[i]) / t1 * splines[o-1][i].tval;
			if (abs(t2) > tolerance)
				c += (knots[i+oo] - x) / t2 * splines[o-1][i+1].tval;

			sp.tval = c;
		}
	}
	
	uint o = order - 1;	
	
	// first order derivatives
	
	for (auto& sp : splines[o])
	{
		uint i = sp.index;
		
		double t1 = knots[i+order-1] - knots[i];
		double t2 = knots[i+order] - knots[i+1];
		
		double c = 0;
			
		if (abs(t1) > tolerance)
			c += 1.0 / t1 * splines[o-1][i].tval;
		if (abs(t2) > tolerance)
			c -= 1.0 / t2 * splines[o-1][i+1].tval;
			
		c *= (order-1);

		sp.td1 = c;
	}
		
	// second order derivatives
	
	for (auto& sp : splines[o])
	{
		uint i = sp.index;
		
		double t1 = (knots[i+order-1] - knots[i+0]) * (knots[i+order-2] - knots[i+0]);
		double t2 = (knots[i+order-1] - knots[i+0]) * (knots[i+order-1] - knots[i+1]);
		double t3 = (knots[i+order-0] - knots[i+1]) * (knots[i+order-1] - knots[i+1]);
		double t4 = (knots[i+order-0] - knots[i+1]) * (knots[i+order-0] - knots[i+2]);
		
			double c = 0;
			
		if (abs(t1) > tolerance)
			c += 1.0 / t1 * splines[o-2][sp.index].tval;
		if (abs(t2) > tolerance)
			c -= 1.0 / t2 * splines[o-2][sp.index+1].tval;
		if (abs(t3) > tolerance)
			c -= 1.0 / t3 * splines[o-2][sp.index+1].tval;
		if (abs(t4) > tolerance)
			c += 1.0 / t4 * splines[o-2][sp.index+2].tval;
			
		c *= (order-1)*(order-2);

		sp.td2 = c;
	}

	vector <double> retvals = {splines[m][n].tval, splines[m][n].td1, splines[m][n].td2};

	return retvals;
}

void bsplines::calcsplines()
{
	// first order splines
	
	for (auto& sp : splines[0])
	{
		uint i = sp.index;
		for (auto& x : grid)
		{
			if (x > knots[i+1])
				sp.vals.push_back(0);
			else if ((x >= knots[i] && x < knots[i+1]) || x == knots.back())
				sp.vals.push_back(1);
			else
				sp.vals.push_back(0);
		}
	}
	
	// higher order splines
	
	for (uint o = 1; o < order; o++)
	{
		uint oo = o+1;		// compensating for 1-indexation
		for (auto& sp : splines[o])
		{
			uint i = sp.index;
			
			double t1 = knots[i+oo-1] - knots[i];
			double t2 = knots[i+oo] - knots[i+1];
			
			for (auto& x : grid)
			{
				uint k = &x - &grid[0];
				
				double c = 0;
				
				if (abs(t1) > tolerance)
					c += (x - knots[i]) / t1 * splines[o-1][i].vals[k];
				if (abs(t2) > tolerance)
					c += (knots[i+oo] - x) / t2 * splines[o-1][i+1].vals[k];

				sp.vals.push_back(c);
			}
		}
	}
	
	uint o = order - 1;		// use this one when accessing splines;
	
	// first order derivatives
	
	for (auto& sp : splines[o])
	{
		uint i = sp.index;
		
		double t1 = knots[i+order-1] - knots[i];
		double t2 = knots[i+order] - knots[i+1];
		
		for (auto& x : grid)
		{
			uint k = &x - &grid[0];
			double c = 0;
			
			if (abs(t1) > tolerance)
				c += 1.0 / t1 * splines[o-1][i].vals[k];
			if (abs(t2) > tolerance)
				c -= 1.0 / t2 * splines[o-1][i+1].vals[k];
			
			c *= (order-1);

			sp.d1.push_back(c);
		}
	}
		
	// second order derivatives
	
	for (auto& sp : splines[o])
	{
		uint i = sp.index;
		
		double t1 = (knots[i+order-1] - knots[i+0]) * (knots[i+order-2] - knots[i+0]);
		double t2 = (knots[i+order-1] - knots[i+0]) * (knots[i+order-1] - knots[i+1]);
		double t3 = (knots[i+order-0] - knots[i+1]) * (knots[i+order-1] - knots[i+1]);
		double t4 = (knots[i+order-0] - knots[i+1]) * (knots[i+order-0] - knots[i+2]);
		
		for (auto& x : grid)
		{
			uint k = &x - &grid[0];
			double c = 0;
			
			if (abs(t1) > tolerance)
				c += 1.0 / t1 * splines[o-2][sp.index].vals[k];
			if (abs(t2) > tolerance)
				c -= 1.0 / t2 * splines[o-2][sp.index+1].vals[k];
			if (abs(t3) > tolerance)
				c -= 1.0 / t3 * splines[o-2][sp.index+1].vals[k];
			if (abs(t4) > tolerance)
				c += 1.0 / t4 * splines[o-2][sp.index+2].vals[k];
			
			c *= (order-1)*(order-2);

			sp.d2.push_back(c);
		}
	}
}

void bsplines::readknots(string knotfile)
{
	double x;

	ifstream readknots(knotfile);
	while (readknots >> x)
		knots.push_back(x);
	
	for (uint k = 0; k < order - 1; k++)
	{   
		knots.insert(knots.begin(),knots.front());
		knots.insert(knots.end(),knots.back());
	}

	knotpts = knots.size();
}

void bsplines::readknotsnorm(string knotfile)
{
	double x;
	knots.reserve(knotpts + 2*(order - 1));

	ifstream readknots(knotfile);
	while (readknots >> x)
		knots.push_back(x);
	
	auto minmax = minmax_element(begin(knots), end(knots));
	double min = *(minmax.first);
	double max = *(minmax.second);

	for (auto& el : knots)
		el = (el - min) / (max-min);
}

void bsplines::readgrid(string gridfile)
{
	double x;

	ifstream readgrid(gridfile);
	while (readgrid >> x)
		grid.push_back(x);
	
	gridpts = grid.size();
}

void bsplines::swapgrid(string gridfile)
{
	grid = {};
	double x;

	ifstream readgrid(gridfile);
	while (readgrid >> x)
		grid.push_back(x);
	
	gridpts = grid.size();
}

void bsplines::printknots()
{
	cout << "content in knot vector: " << endl;
	for (auto& el : knots)
		cout << el << " ";
	cout << endl;
}

void bsplines::printgrid()
{
	cout << "content in grid vector: " << endl;
	for (auto& el : grid)
		cout << el << " ";
	cout << endl;
}

void bsplines::printgridsize()
{
	cout << "number of grid points: " << endl << grid.size() << endl; 
}

void bsplines::printvals(uint m, uint n)
{
	cout << "content in spline (B" << m  << "," <<  n <<  ") vals vector: " << endl;
	for (auto& el : splines[n][m].vals)
		cout << el << " ";
	cout << endl;
}

void bsplines::writesplines()
{
	for (uint o = 0; o < order; o++)
		for (auto& sp : splines[o])
		{
			uint i = sp.index;
			ostringstream namestream;

			namestream << "B(" << fixed << setprecision(1) << i << "," 
							   << fixed << setprecision(1) << o << ").csv";
			string filename = namestream.str();
	
			ofstream fs;
			fs.open(filename);

			if (o < order - 1)
			{
				for (uint k = 0; k < sp.vals.size(); k++)
					fs << sp.vals[k] << "," << 0 << "," << 0 << endl;
			
				fs.close();
			}
			else
			{
				for (uint k = 0; k < sp.vals.size(); k++)
					fs << sp.vals[k] << "," << sp.d1[k] << "," << sp.d2[k] << endl;
				fs.close();
			}

			cout << "write " << sp.vals.size() << " numbers to " << filename << endl;
		}
}



