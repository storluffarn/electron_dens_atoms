
//
// only implemented up to argon
// g means on gauss grid
// r means on radial grid
//

#include <cmath>
#include <iostream>
#include <fstream>
#include <armadillo>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include "bsplines.h"

using namespace std;
typedef unsigned int uint;
typedef vector<double> vecd;

const double pi = 4*atan(1);
const double alpha = 0.007297352566417;

int pingcount = 0;
void ping(const int line) {cout << "ping at line " << line << endl; pingcount++;}
void getpingcount() {cout << pingcount << endl;}

constexpr uint64_t factorial(uint64_t n)
{ 
	return n == 0 ? 1 : n*factorial(n-1); 
}

// -------------------------------------------------------------------------------
// ----------------------------- Hydrogen stuff ----------------------------------
// -------------------------------------------------------------------------------

void knotsngrid (double rmin, double rmax, uint steps, uint mode, vecd* absc, vecd* w, vecd* wgrid, vecd* pfgrid, string knotfile, string ggridfile, string rgridfile)
{
	vecd knots;
	knots.reserve(steps);
	vecd rgrid;
	knots.reserve(steps);
	vecd ggrid;
	ggrid.reserve(steps*absc->size());
	
	wgrid->reserve(steps*absc->size());
	pfgrid->reserve(steps*absc->size());

	if (mode == 0)
	{
		double step = (rmax - rmin) / steps;
		
		for (uint k = 0; k <= steps; k++)
		{
			double knot = rmin + k*step;

			knots.push_back(knot);
		}
	}
	else if(mode == 1)
	{
		double step = (rmax - rmin) / steps;
		
		for (uint k = 0; k <= steps; k++)
		{
			double knot = pow(1.5,k*step);

			knots.push_back(knot);
		}
	}

	rgrid = knots;
	//rgrid.erase(rgrid.end()-1);

	for (uint u = 0; u < knots.size()-1; u++)
	{
		double prefactor = 0.5*(knots.at(u+1)-knots.at(u));
		for (uint v = 0; v < absc->size(); v++)
		{
			double r = absc->at(v)*0.5*(knots.at(u+1)-knots.at(u)) + 0.5*(knots.at(u+1)+knots.at(u));
			
			if (knots.at(u) == knots.at(u+1))
				break;
		
			ggrid.push_back(r);
			wgrid->push_back(w->at(v));
			pfgrid->push_back(prefactor);
		}
	}
	
	ofstream knotstream, ggridstream, rgridstream;
	knotstream.open(knotfile); 
	ggridstream.open(ggridfile); 
	rgridstream.open(rgridfile);
	
	for (auto& el : knots)
		knotstream << el << endl;
	
	for (auto& el : ggrid)
		ggridstream << el << endl;

	for (auto& el : rgrid)
		rgridstream << el << endl;

	rgridstream.close(); 
	ggridstream.close(); 
	knotstream.close(); 
}

double gaussleft (uint i, uint j, int l, uint nz, vecd* wgrid, vecd* pfgrid, arma::vec* vex, uint o, bsplines* splines)
{
	double accval = 0;

	vecd* grid = splines->getgrid(); 

	for (auto& r : (*grid))
	{
		uint k = &r - &(*grid)[0];

		double bi = splines->getvals(o-1,i)->at(k);
		double dj = splines->getd2(o-1,j)->at(k);
		double bj = splines->getvals(o-1,j)->at(k);

		double el = pfgrid->at(k) * wgrid->at(k) * (bi * ((l*(l+1)/(2*pow(r,2)) - nz/r + vex->at(k))*bj - 0.5*dj));

		accval += el;
	}

	return accval;
}

double gaussright (uint i, uint j, vecd* wgrid, vecd* pfgrid, uint o, bsplines* splines)
{
	double accval = 0;

	vecd* grid = splines->getgrid(); 

	for (auto& r : (*grid))
	{
		uint k = &r - &(*grid)[0];

		double bi = splines->getvals(o-1,i)->at(k);
		double bj = splines->getvals(o-1,j)->at(k);

		double el = pfgrid->at(k) * wgrid->at(k) * (bi * bj);

		accval += el;
	}

	return accval;
}

void buildlhs (int l,uint nz, uint o, vecd* wgrid, vecd* pfgrid, arma::vec* vex, arma::mat* lhs, bsplines* splines)
{
	for (uint i = 0; i < lhs->n_rows; i++)
		for (uint j = 0; j < lhs->n_cols; j++)
			(*lhs)(i,j) = gaussleft (i,j,l,nz,wgrid,pfgrid,vex,o,splines);	// - for overall sign error...
}

void buildrhs (uint o, vecd* wgrid, vecd* pfgrid, arma::mat* rhs, bsplines* splines)
{
	for (uint i = 0; i < rhs->n_rows; i++)
		for (uint j = 0; j < rhs->n_cols; j++)
			(*rhs)(i,j) = gaussright (i,j,wgrid,pfgrid,o,splines);

}

void boundarylhs(arma::mat* lhs)
{
	arma::rowvec ltop(lhs->n_cols,arma::fill::zeros);
	arma::rowvec lbot(lhs->n_cols,arma::fill::zeros);

	(*lhs).row(0) = ltop;
	(*lhs).row(lhs->n_rows-1) = lbot;
}

void boundaryrhs(arma::mat* rhs)
{
	arma::rowvec rtop(rhs->n_cols,arma::fill::zeros);
	arma::rowvec rbot(rhs->n_cols,arma::fill::zeros);
	rtop(0) = 1;
	rbot(rbot.size()-1) = 1;

	(*rhs).row(0) = rtop;
	(*rhs).row(rhs->n_rows-1) = rbot;
}

void calcprob(uint o, arma::vec* prob, arma::vec* eigenvec, arma::mat* S, bsplines* splines)
{
	uint nsplines = splines->getnosplines(o-1);
	auto grid = (*splines->getgrid());

	cout << "nsplines " << nsplines << " gridsize " << grid.size() << " probsize " << prob->size() << endl;
	
	for (auto& r : grid)
	{
		uint k = &r - &grid[0];
		double x = 0;

		for (uint m = 1; m < nsplines-1; m++)		// exclude first and last?
		{
			x += (*eigenvec)(m)*splines->getvals(o-1,m)->at(k);
		}

		(*prob)(k) = x;						
	}
	
	//boundary conds
	(*prob)(0) = 0;
	(*prob)(prob->size()-1) = 0;	
	
	double norm = 0;
	for(uint k = 0; k < nsplines; k++)
		for(uint l = 0; l < nsplines; l++)
			norm += (*eigenvec)(k)*(*eigenvec)(l)*(*S)(k,l);
	norm = sqrt(norm);

	for(auto& el : *(prob))
		el /= norm;
}

void calcdens (vector <uint>* orbnums, vector <arma::vec>* prob, arma::vec* dens, bsplines* splines)
{	
	vecd* grid = splines->getgrid();
	
	cout << "gridsize " << grid->size() << " denssize " << dens->size() << " probsize " << prob->at(0).size() << endl;;

	for (auto& el : (*dens))
	{
		uint i = &el - &(*dens)[0];
		double accdens = 0;

		
		for (uint j = 0; j < orbnums->size(); j++)
		{
			if (i == 0)
				break;

			accdens += orbnums->at(j)*pow((*prob)[j](i)/grid->at(i),2);
		}

		el = accdens;
	}
}

vector <uint> getbs (vecd* enlvls, arma::vec* eigvals)
{
	vector <uint> bspos;
	for (auto& en : (*enlvls))
	{
		vecd diff;
		
		for (auto& el : (*eigvals))
			diff.push_back(abs(el-en));
		
		auto minel = min_element(begin(diff),end(diff));
		uint pos = distance(begin(diff),minel);
		
		auto k = &en - &(*enlvls)[0];
		while ((k > 0 && pos == bspos[k-1]) || (*eigvals)(pos) == 0)
		{
			diff[pos] = max((*eigvals));			// somewhat ugly, but works

			minel = min_element(begin(diff),end(diff));
			pos = distance(begin(diff),minel);
		}

		bspos.push_back(pos);
	}

	return bspos;
}

vecd calcenlvls (uint n, uint nz)
{
	vector <double> en;

	for (uint k = 1; k <= n; k++)
	{
		//en.push_back(-double(nz)/pow(2,k));
		en.push_back(- pow(nz,2)/(2.0*pow(k,2)));
	}

	return en;
}

double anpsi (uint n, int l, uint nz,	double r)
{
	double a = (1836+1)/1836;	// mz + me / mz;

	double t111 = factorial(n-l-1);
	double t112 = 2*n*factorial(n+l);
	double t11 = t111/t112;
	double t1 = sqrt(pow(2*nz/(n*a),3)*t11);
	double t2 = exp(-nz*r/(n*a))*pow(2*nz*r/(n*a),l);
	double t3 = pow(boost::math::laguerre(n-l-1,2*nz*r/(n*a)),2*l+1);

	double psi = t1*t2*t3;

	return psi;
}


// --------------------------------------------------------------------------------
// ------------------------------ Vee stuff ---------------------------------------
// ---------------------------------------------------------------------------.----

void vknotsngrid (bsplines* rsplines, string rknotfile, string rgridfile)
{
	// radial part
	auto rgrid = (*rsplines->getgrid());
	
	//for (auto& el : rgrid)
	//	cout << el << " ";
	//cout << endl;

	//rgrid.insert(rgrid.begin(),rmin);
	//rgrid.push_back(rmax);
	
	vecd rknots;

	rknots = rgrid;


	// below should compensate for the higher order, this is sloppy coding though! Will only work for o = 5. Easily fixed by adding n = abs(4-o) points instead.
	rknots.erase(rknots.begin());		
	rknots.erase(rknots.end()-1);
	rknots.erase(rknots.begin()+round(rknots.size()/2));
	
	cout << "knots in vknotsngrid " << rknots.size() << endl;
	cout << "grid in vknotsngrid " << rgrid.size() << endl;

	//for (auto& el : rknots)
	//	cout << el << " ";
	//cout << endl;
	//for (auto& el : rgrid)
	//	cout << el << " ";
	//cout << endl;

	ofstream rknotstream, rgridstream;
	rknotstream.open(rknotfile);
	rgridstream.open(rgridfile);

	for (auto& el : rknots)
		rknotstream << el << endl;

	for (auto& el : rgrid)
		rgridstream << el << endl;

	rgridstream.close();
	rknotstream.close();
}

vector <uint> orbdist (uint ne)
{
	uint occorbs = 0;
	vector <uint> occnums;
	vector <uint> orb = {2,2,3,2,3,2,5,3,2,5,3};
	
	uint k = 0;

	while (ne > 0)
	{
		uint occnum = 0;
		
		if (!(orb.at(k)%2))
		{
			uint it = 2;
			while(ne > 0 && it > 0)
			{
				occnum++;
				ne--;
				it--;
			}
		}
		else if (!(orb.at(k)%3))
		{
			uint it = 6;
			while(ne > 0 && it > 0)
			{
				occnum++;
				ne--;
				it--;
			}
		}
		else if (!(orb.at(k)%5))
		{
			uint it = 10;
			while(ne > 0 && it > 0)
			{
				occnum++;
				ne--;
				it--;
			}
		}

		occnums.push_back(occnum);
		occorbs++;

		k++;
	}

	return occnums;
}

int orbkey(int n)
{
	if (n == 2)
		return 0;
	else if (n == 3)
		return 1;
	else if (n == 5)
		return 2;
	else 
	{
		cout << "error orbkey not in range" << endl;
		return -1;
	}
}


void vbuildmatrix (uint order, uint gridpts, arma::mat* lhs, arma::vec* rhs, arma::vec* dens, bsplines* splines)
{
	uint nsplines = splines->getnosplines(order - 1);

	auto tmp = splines->getd2(order-1,0);

	cout << "lhssize " << lhs->n_rows << "x" << lhs->n_cols << " gridpts " << gridpts << " nsplines " << nsplines << " d2size " << tmp->size() << " rhssize " << rhs->size() << " denssize " << dens->size() << endl;

	for (uint k = 0; k < gridpts; k++)
		for (uint l = 0; l < nsplines; l++)
			(*lhs)(k,l) = splines->getd2(order-1,l)->at(k);

	for (uint i = 1; i < rhs->n_rows-2; i++)
		(*rhs)(i) = -(*dens)(i);
}

void vboundcond (uint order, arma::mat* lhs, arma::vec* rhs, bsplines* splines) 
{
	//cout << lhs->size() << " " <<  rhs->size() << endl;
	uint nosplines = splines->getnosplines(order-1);
	arma::rowvec bot(nosplines,arma::fill::zeros);
	arma::rowvec top = bot;
	bot(bot.n_cols-1) = 1;
	top(0) = 1;
	lhs->row(0) = top;
	lhs->row(lhs->n_rows-1) = bot;

	(*rhs)(0) = 0;
	(*rhs)((*rhs).n_rows-1) = 0;
	
}

void vsolving (arma::mat* splinemat, arma::colvec* dens, arma::colvec* sols)
{   
	// LU-decomposition for increased performance   
	arma::mat U,L;

	lu(U,L,*splinemat);
	arma::mat LUsplinemat = U*L;

	(*sols) = arma::solve(LUsplinemat,(*dens));
	//(*sols) = arma::solve((*splinemat),(*dens));
}

void vestval (uint o, arma::vec* estpots, arma::vec* sols, bsplines* splines)
{
	uint gridpts = (*splines->getgridpts());
	uint nosplines = splines->getnosplines(o-1);
	vector<double>* grid = splines->getgrid();

	estpots->resize(gridpts);
	
	for (auto& el : (*grid))
	{
		uint k = &el - &(*grid)[0];
		double accpot = 0;
	
		for (uint l = 1; l < nosplines - 1; l++)
		{
			double x = (*sols)(l)*(*splines->getvals(o-1,l))[k];
			accpot += x;		
		}

		(*estpots)(k) = (accpot/el);
	}

	//estpots->insert(estpots->begin(),0);
}

void vex (arma::vec* ex, arma::vec* dens)
{	
	for (auto& el : (*dens))
	{
		uint k = &el - &(*dens)[0];

		double x = -3*pow(3.0*el/(8*pi),1.0/3.0);
		(*ex)[k] = x;
	}
}

double statedist(arma::vec* v1, arma::vec* v2)
{
	if (v1->size() != v2->size())
	{
		cout << "vector sizes incompatible" << endl;
		
		return 0;
	}

	double accval = 0;
	
	for(uint k = 0; k < v1->size(); k++)
	{
		double x = pow(v1->at(k) - v2->at(k),2);

		accval += x;
	}

	return sqrt(accval);
}

void calcpsi (arma::vec* psi,arma::vec* probs, bsplines* splines)
{
	vecd* grid = splines->getgrid();
	
	for (auto& el : (*psi))
	{
		auto k = &el - &(*psi)[0];

		el = (*probs)[k]/(*grid)[k];

		el = pow(el,2);

		el *= pow((*grid)[k],2);
	}
}

void writepsi (bsplines* splines, arma::vec* psi, string filename)
{
	auto grid = splines->getgrid();

	ofstream fs;
	fs.open(filename);

	for (uint k = 0; k < psi->size(); k++)
		fs << (*grid)[k]  << "," << (*psi)(k) << endl;

	fs.close();
}

void writedens (bsplines* splines, arma::vec* dens, string filename)
{
	auto grid = splines->getgrid();

	ofstream fs;
	fs.open(filename);

	for (uint k = 0; k < dens->size(); k++)
		fs << (*grid)[k]  << "," << (*dens)(k)*pow((*grid)[k],2) << endl;

	fs.close();
}

// points for gaussian quadrature
vecd getabsc()
{
	vecd absc = {
		-0.973906528517172,
		-0.865063366688985,
		-0.679409568299024,
		-0.433395394129247,
		-0.148874338981631,
		 0.148874338981631,
		 0.433395394129247,
		 0.679409568299024,
		 0.865063366688985,
		 0.973906528517172};

	return absc;
}

vecd getw()
{
	vecd w = {
		0.295524224714753,
		0.269266719309996,
		0.219086362515982,
		0.149451349150581,
		0.066671344308688,
		0.066671344308688,
		0.149451349150581,
		0.219086362515982,
		0.269266719309996,
		0.295524224714753};

	return w;
}

int main()
{
	// files
	string knotfile = "knots.dat";
	string rgridfile = "rgrid.dat";
	string ggridfile = "ggrid.dat";
	string vknotfile = "vknots.dat";
	string rvgridfile = "rvgrid.dat";
	string gvgridfile = "gvgrid.dat";
	string outfile = "output.csv";
	
	// numbers
	uint nabsc = 10;					// number of abscissas
	uint knotpts = 100;					// without ghost pts
	uint ggridpts = knotpts*nabsc;
	//uint difford = 2;					// why did I introdce this?
	uint order = 5;
	//uint adjust = order - difford - 3;	// compensating for order being larger than 
	uint gvgridpts = ggridpts + 2;
	//uint vknotpts = vgridpts - adjust;
	//uint gridpts = knotpts + adjust;
	//vgridpts = knotpts;
	
	double tol = 1e-12;
	//double Q0 = 1;
	//uint n = 1;
	uint nz = 1;
	uint ne = 1;
	uint l = 0;
	
	vector <uint> orbnums = orbdist(ne);
	if (orbnums.size() > 6)
		l = 2;
	else if (orbnums.size() > 2)
		l = 1;
	
	vector <uint> orb = {2,2,3,2,3,2};
	vector <uint> orbsused( orb.begin(), orb.begin() + orbnums.size() );
	uint sorbs = count(orbsused.begin(),orbsused.end(),2);
	uint porbs = count(orbsused.begin(),orbsused.end(),3);

	//cout << sorbs << " " << porbs << endl;
	//for (auto& el : orbnums)
	//	cout << el << " ";
	//cout << endl;

	double rmin = 0.0;
	double rmax = 7.0;
	uint mode = 0;

	vecd absc = getabsc();
	vecd w = getw();
	vecd wgrid;
	vecd pfgrid;
	vecd vdir (0,knotpts);
	vecd vexch (0,knotpts);

	knotsngrid (rmin,rmax,knotpts,mode,&absc,&w,&wgrid,&pfgrid,knotfile,ggridfile,rgridfile);
	bsplines rsplines(knotfile,rgridfile,order,tol);
	bsplines gsplines(knotfile,ggridfile,order,tol);	// for the gauss grid

	//splines.printgrid();
	//splines.printknots();
	//splinesg.printgrid();
	//splinesg.printknots();

	//vector <double> rgrid = (*splines.getknots());
	//rgrid.erase(rgrid.begin(),rgrid.begin() + 4);
	//rgrid.erase(rgrid.end() - 4,rgrid.end());

	vknotsngrid (&rsplines,rvgridfile,vknotfile);

	//vknotsngrid (rmin, rmax, order, vknotfile, vgridfile, &splines);

	bsplines vsplines(vknotfile,rvgridfile,order,tol);
	
	uint nrsplines = rsplines.getnosplines(order-1);
	uint nrvsplines = vsplines.getnosplines(order-1);

	//cout << "h gridpts " << (*rsplines.getgridpts()) << " v gridpts " << (*vsplines.getgridpts()) << " h nsplines " << nrsplines << " v nsplines " << nrvsplines << endl;
	
	cout << "vknots " << (*vsplines.getknotpts()) << " vgridpts " << (*vsplines.getgridpts()) << " vnsplines " << nrvsplines << endl;


	rsplines.calcsplines();
	gsplines.calcsplines();
	vsplines.calcsplines();

	float scale = 0.7;
	double c = 1;
	double acc = 1e-9;
	uint loops = 0;
	uint limit = 0;

	arma::mat rhs(nrsplines,nrsplines,arma::fill::zeros);
	buildrhs (order,&wgrid,&pfgrid,&rhs,&gsplines);
	boundaryrhs (&rhs);
		
	vecd enlvlsl0old = calcenlvls (sorbs, nz);
	vecd enlvlsl1old;
	for (uint k = 1; k <= porbs; k++)
		enlvlsl1old.push_back(enlvlsl0old[k]);
	
	vecd enlvlsl0orig = enlvlsl0old;
	vecd enlvlsl1orig = enlvlsl1old;

	arma::vec rdens(knotpts+1,arma::fill::zeros);
	arma::vec rdensold = rdens;
	arma::vec debugpsi(gvgridpts,arma::fill::zeros);
	
	arma::vec rvestdir;
	arma::vec rvestex(nrvsplines,arma::fill::zeros);	
	arma::vec rvest (gvgridpts,arma::fill::zeros);
	arma::vec gvest (gvgridpts,arma::fill::zeros);
	arma::vec rvold (gvgridpts,arma::fill::zeros);
	arma::vec gvold (gvgridpts,arma::fill::zeros);

	while (c > acc)		// main loop
	{
		///////////////////////////
		// build all Pnl vectors //
		///////////////////////////
		
		// parts of this probably could -- and should -- go into a sub rutine

		vector <arma::mat> lhs (l+1);
		for (auto& el : lhs)
			el = arma::zeros<arma::mat>(nrsplines,nrsplines);

		for (uint i = 0; i <= l; i++)
		{
			buildlhs (i,nz,order,&wgrid,&pfgrid,&gvold,&lhs[i],&gsplines);
			boundarylhs (&lhs[i]);
		}
		
		// this should be done in classes to be c++ style, but armadillo and classes seems weird
		vector <arma::vec> rl0, rl1;
		vector <arma::vec> gl0, gl1;
		arma::vec rpnl (knotpts+1,arma::fill::zeros);
		arma::vec gpnl (ggridpts,arma::fill::zeros);

		for (uint k = 0; k < sorbs; k++)
		{
			rl0.push_back(rpnl);
			gl0.push_back(gpnl);
		}

		for (uint k = 0; k < porbs; k++)
		{
			rl1.push_back(rpnl);
			gl1.push_back(gpnl);
		}
		
		arma::cx_mat ceigenvecsl0;
		arma::cx_mat ceigenvecsl1;
		arma::cx_vec ceigenvalsl0;
		arma::cx_vec ceigenvalsl1;
	
		//cout << "lhs content:" << endl;
		//lhs.print();
		//cout << "rhs content:" << endl;
		//rhs.print();
		
		//rhs.print();
		
		//vold.print();
	
		eig_pair(ceigenvalsl0,ceigenvecsl0,lhs[0],rhs);
		
		if (l == 1)
			eig_pair(ceigenvalsl1,ceigenvecsl1,lhs[1],rhs);
		
		arma::vec eigenvalsl0 = real(ceigenvalsl0);
		arma::vec eigenvalsl1 = real(ceigenvalsl1);
		arma::mat eigenvecsl0 = real(ceigenvecsl0);
		arma::mat eigenvecsl1 = real(ceigenvecsl1);

		// this should be done exclusively using armadillo, you shouldn't do it like this, but I can't get armadillo to play along at the moment, so, there it is
	
		vector <uint> bsl0 = getbs(&enlvlsl0old, &eigenvalsl0);
		vector <uint> bsl1 = getbs(&enlvlsl1old, &eigenvalsl1);
		//vector <uint> bsl0 = getbs(&enlvlsl0orig, &eigenvalsl0);
		//vector <uint> bsl1 = getbs(&enlvlsl1orig, &eigenvalsl1);

		ping(__LINE__);
		for (uint k = 0; k < sorbs; k++)
		{
			cout << "bsl0: " << bsl0[k] << " eigval: " << eigenvalsl0(bsl0[k]) << endl;

			arma::vec eigvec = eigenvecsl0.col(bsl0[k]);
		
		ping(__LINE__);

			calcprob(order,&rl0[k],&eigvec,&rhs,&rsplines);
			calcprob(order,&gl0[k],&eigvec,&rhs,&gsplines);
		}

		ping(__LINE__);
		for (uint k = 0; k < porbs; k++)
		{
			cout << "bsl1: " << bsl1[k] << " eigval: " << eigenvalsl1(bsl1[k]) << endl;
			
			arma::vec eigvec = eigenvecsl1.col(bsl1[k]);
			calcprob(order,&rl1[k],&eigvec,&rhs,&rsplines);
			calcprob(order,&gl1[k],&eigvec,&rhs,&gsplines);
		}

		// for debigging 
		
		//uint dindex = bsl0[0];
		//cout << "debug vector index: " << dindex << " corresponding eigenvalue: " << eigenvalsl0(dindex) << endl;
		//arma::vec eigvec = eigenvecsl0.col(dindex);
		//calcprob(order,&debugpsi,&eigvec,&rhs,&splines);
		//
		//eigenvalsl0.print();
		//eigvec.print();
		
		// debugging end

		ping(__LINE__);

		vector <arma::vec> rpvector;
		vector <arma::vec> gpvector;

		rpvector.push_back(rl0[0]);
		gpvector.push_back(gl0[0]);
		
		auto sloops = sorbs - 1;
		auto ploops = porbs;
	
		while (sloops > 0 || ploops > 0)
		{
			if (sloops > 0)
			{
				//cout << "adding s: " << sorbs - sloops << endl;
				rpvector.push_back(rl0[sorbs-sloops]);
				gpvector.push_back(gl0[sorbs-sloops]);
				sloops--;
			}
		ping(__LINE__);

			if (ploops > 0)
			{
				//cout << "adding p: " << porbs - ploops << endl;
				rpvector.push_back(rl1[porbs-ploops]);
				gpvector.push_back(gl1[porbs-ploops]);
				ploops--;
			}
		}
		
		ping(__LINE__);
		arma::vec gdens(ggridpts,arma::fill::zeros);
		
		calcdens(&orbnums, &rpvector, &rdens, &rsplines);
		ping(__LINE__);
		calcdens(&orbnums, &gpvector, &gdens, &gsplines);
		
		//// more debugging

		ping(__LINE__);
		//arma::vec debugdens (gridpts+2,arma::fill::zeros);

		//debugdens = l0[0];

		//double csum = 0;
		//vecd* grid = rsplines.getgrid();
		//vecd diffgrid;
		//for (uint k = 0; k < grid->size()-1; k++)
		//{
		//	auto diff = abs(grid->at(k)	- grid->at(k+1));
		//	diffgrid.push_back(diff);
		//}
		//
		//for(uint k = 0; k < diffgrid.size(); k++)
		//{
		//	csum += 4*pi * pow((*grid)[k],2) * 0.5 * pow((debugdens[k]+ debugdens[k+1]),2) * diffgrid[k];
		//}
		//
		//cout << "this should be debug N electrons: " << 4*pi*csum << endl;
		//
		//// end more debugging

		// check normalization
		
		double csum = 0;
		vecd* grid = rsplines.getgrid();
		vecd diffgrid;
		for (uint k = 0; k < grid->size()-1; k++)
		{
			auto diff = abs(grid->at(k)	- grid->at(k+1));
			diffgrid.push_back(diff);
		}
		ping(__LINE__);
		
		for(uint k = 0; k < diffgrid.size(); k++)
		{
			csum += pow((*grid)[k],2)* 0.5 * (rdens[k]+ rdens[k+1]) * diffgrid[k];
			//csum += 0.5*(dens[k]+dens[k+1]) * diffgrid[k];
		}

		cout << "this should be Nocc: " << 4*pi*csum << endl;
		
		
		//arma::vec tmp(1,arma::fill::zeros);		// boundary conds
		//probnew.insert_rows(0,tmp);
		//probnew.insert_rows(probnew.size(),tmp);

		///////////////
		// Vee stuff //
		///////////////
		
		arma::mat vlhs(nrvsplines,nrvsplines,arma::fill::zeros); 
		ping(__LINE__);
		arma::vec vrhs(nrvsplines,arma::fill::zeros);


		////////////////////////////////
		// find directional potential //
		////////////////////////////////

		//cout << "density vector: " << endl;
		//dens.print();

		vbuildmatrix (order,knotpts,&vlhs,&vrhs,&rdens,&vsplines);
		ping(__LINE__);
		vboundcond (order,&vlhs,&vrhs,&vsplines); 
		
		//cout << "vlhs content:" << endl;
		//vlhs.print();
		//cout << "lrhs content:" << endl;
		//vrhs.print();

		arma::vec vsols;	
		ping(__LINE__);
		vsolving(&vlhs,&vrhs,&vsols);
	
		ping(__LINE__);
		vestval (order,&rvestdir,&vsols,&vsplines);
		ping(__LINE__);

		vsplines.swapgrid(gvgridfile);
		ping(__LINE__);
		vsplines.calcsplines();
		ping(__LINE__);

		arma::vec gvestdir(gvgridpts,arma::fill::zeros);	
		vestval (order,&gvestdir,&vsols,&vsplines);

		ping(__LINE__);
		gvestdir(0) = 0;		// bound conds
		gvestdir(gvestdir.size()-1) = 0;
		rvestdir(0) = 0;		// bound conds
		rvestdir(gvestdir.size()-1) = 0;

		ping(__LINE__);
		arma::vec gvestex(nrvsplines,arma::fill::zeros);	
		
		vex (&rvestex,&rdens);
		ping(__LINE__);
		vex (&gvestex,&gdens);
		
		gvest = (1-scale)*(gvestdir + gvestex) + scale*gvold;
		if (loops == 0)	
			gvest = gvestdir + gvestex;
		
		rvest = (1-scale)*(rvestdir + rvestex) + scale*rvold;
		if (loops == 0)	
			rvest = rvestdir + rvestex;
		//scale += 1;

		ping(__LINE__);
		if (loops == limit)
		{
			cout << "convergence not found within " << limit << " cycles, exiting " << endl;
			
			break;
		}
		else 
		{	
			c = statedist(&rdens,&rdensold);
	
			cout << "loop count: " << loops << ", c: " << c << endl;
			
			rvold = rvest;
			gvold = rvest;
			rdensold = rdens;
			
			for (auto& el : bsl0)
			{
				auto k = &el - &bsl0[0];
				enlvlsl0old[k] = eigenvalsl0(el);
			}
			
			for (auto& el : bsl1)
			{
				auto k = &el - &bsl1[0];
				enlvlsl1old[k] = eigenvalsl1(el);
			}
			
			loops++;
		}
	}

	ofstream fdir, fex, ftot;
	fdir.open("dir.dat"); fex.open("ex.dat"); ftot.open("tot.dat");
	
	for (auto& el : rvestdir)
		fdir << el << endl;
	for (auto& el : rvestex)
		fex << el << endl;
	for (auto& el : rvest)
		ftot << el << endl;
	
	fdir.close(); fex.close(); ftot.close();	
			
	
	//arma::vec psi(gridpts,arma::fill::zeros);
	//calcpsi(&psi,&probold,&splines);
	
	//writepsi(&vsplines,&debugpsi,"debugpsi.csv");
	writedens(&rsplines,&rdens,outfile);
	
	if (c < acc)
		cout << "sufficient convergence found in " << loops+0 << " iterations, with distance " << c << endl;

	return 0;
}



























