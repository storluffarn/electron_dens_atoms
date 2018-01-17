
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

void knotsngrid (double rmin, double rmax, uint steps, uint mode, vecd* absc, vecd* w, vecd* wgrid, vecd* pfgrid, string knotfile, string gridfile)
{
	vecd knots;
	knots.reserve(steps);
	
	vecd grid;
	grid.reserve(steps*absc->size());
	
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
		
		double emax = log(rmax+1);
		
		double step = emax / steps;

		for (uint k = 0; k <= steps; k++)
		{
			double knot = exp(step*k)-1;

			knots.push_back(knot);
		}
	}
	
	for (uint u = 0; u < knots.size()-1; u++)
	{
		double prefactor = 0.5*(knots.at(u+1)-knots.at(u));
		for (uint v = 0; v < absc->size(); v++)
		{
			double r = absc->at(v)*0.5*(knots.at(u+1)-knots.at(u)) + 0.5*(knots.at(u+1)+knots.at(u));
			
			if (knots.at(u) == knots.at(u+1))
				break;
		
			grid.push_back(r);
			wgrid->push_back(w->at(v));
			pfgrid->push_back(prefactor);
		}
		
	}

	ofstream knotstream, gridstream;
	knotstream.open(knotfile);
	gridstream.open(gridfile);
	
	for (auto& el : knots)
		knotstream << el << endl;
	
	for (auto& el : grid)
		gridstream << el << endl;

	gridstream.close();
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

		double el = pfgrid->at(k) * wgrid->at(k) * (bi * ((l*(l+1)/(2*pow(r,2)) - nz/r + vex->at(k+1))*bj - 0.5*dj));

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
	uint nosplines = splines->getnosplines(o-1);
	vecd* grid = splines->getgrid();

	for (auto& r : (*grid))
	{
		uint k = &r - &(*grid)[0];
		double x = 0;

		for (uint m = 1; m < nosplines-1; m++)		// exclude first and last?
		{
			x += (*eigenvec)(m)*splines->getvals(o-1,m)->at(k);
		}

		(*prob)(k+1) = x;							// + 1 due to boundary conds
	}
	
	double norm = 0;
	for(uint k = 0; k < nosplines; k++)
		for(uint l = 0; l < nosplines; l++)
			norm += (*eigenvec)(l)*(*S)(k,l)*(*eigenvec)(k);
	norm = sqrt(norm);

	for(auto& el : *(prob))
		el /= norm;
}

void calcdens (vector <uint>* orbnums, vector <arma::vec>* prob, arma::vec* dens, bsplines* splines)
{	
	vecd* grid = splines->getgrid();

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

		el = 1.0/(4.0*pi) * accdens;
	}
}

vector <uint> getbs (uint orbs, arma::vec* eigvals)
{
	vector <uint> bspos;

	vecd veigvals;
	veigvals.reserve(eigvals->size());

	for(auto& el : (*eigvals))
		veigvals.push_back(el);

	for (auto& el : veigvals)
		if (el == 0)
			el = 1e9;

	for (uint k = 0; k < orbs; k++)
	{

		auto minel = min_element(begin(veigvals),end(veigvals));
		uint pos = distance(begin(veigvals),minel);

		veigvals[pos] = 1e9;

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

// --------------------------------------------------------------------------------
// ------------------------------ Vee stuff ---------------------------------------
// ---------------------------------------------------------------------------.----

void vknotsngrid (double rmin, double rmax, string knotfile, string gridfile, bsplines* splines)
{
	vecd grid = (*splines->getgrid());

	grid.insert(grid.begin(),rmin);
	grid.push_back(rmax);

	vecd knots;
	knots = grid;

	knots.erase(knots.begin());
	knots.erase(knots.end()-1);
	knots.erase(knots.begin()+round(knots.size()/2));

	ofstream knotstream, gridstream;
	knotstream.open(knotfile);
	gridstream.open(gridfile);

	for (auto& el : knots)
		knotstream << el << endl;

	for (auto& el : grid)
		gridstream << el << endl;

	gridstream.close();
	knotstream.close();
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
	uint nosplines = splines->getnosplines(order - 1);
	auto grid = splines->getgrid();

	gridpts = grid->size();

	for (uint k = 0; k < gridpts; k++)
		for (uint l = 0; l < nosplines; l++)
			(*lhs)(k,l) = splines->getd2(order-1,l)->at(k);

	for (uint i = 1; i < rhs->n_rows-2; i++)
		(*rhs)(i) = -4*pi*grid->at(i)*(*dens)(i);
}

void vboundcond (uint order, uint ne, arma::mat* lhs, arma::vec* rhs, bsplines* splines) 
{
	uint nosplines = splines->getnosplines(order-1);
	arma::rowvec bot(nosplines,arma::fill::zeros);
	arma::rowvec top = bot;
	bot(bot.n_cols-1) = 1;
	top(0) = 1;
	lhs->row(0) = top;
	lhs->row(lhs->n_rows-1) = bot;

	(*rhs)(0) = 0;
	(*rhs)((*rhs).n_rows-1) = ne;
	
}

void vsolving (arma::mat* splinemat, arma::colvec* dens, arma::colvec* sols)
{   
	// LU-decomposition for increased performance   
	arma::mat L,U;

	lu(L,U,*splinemat);
	arma::mat LUsplinemat = L*U;

	(*sols) = arma::solve(LUsplinemat,(*dens));
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
}

void vex (arma::vec* ex, arma::vec* dens)
{	
	for (auto& el : (*dens))
	{
		uint k = &el - &(*dens)[0];

		double x = -3.0*pow(3.0*el/(8.0*pi),1.0/3.0);
		(*ex)(k) = x;
	}
}

// second way of calculating the total energy
//double intvee (vecd* wgrid, vecd* pfgrid, arma::vec* vest, vector <arma::vec>* Pvec, vecd* eigvals, vector <arma::vec>* eigvecs, vector <uint>* orbnums, uint o, bsplines* splines)
//{
//	//el = arma::zeros<arma::mat>(nosplines,nosplines);
//
//	auto nosplines = splines->getnosplines(o-1);
//	
//	arma::mat veemat (nosplines,nosplines);
//
//	// I should extract this result from previous calculations, but whatever
//	
//	for (uint i = 0; i < veemat.n_rows; i++)
//		for (uint j = 0; j < veemat.n_cols; j++)
//		{
//			double accval = 0;
//			vecd* grid = splines->getgrid(); 
//
//			for (uint k = 0; k < grid->size(); k++)
//			{
//				double bi = splines->getvals(o-1,i)->at(k);
//				double bj = splines->getvals(o-1,j)->at(k);
//
//				double el = pfgrid->at(k) * wgrid->at(k) * (bi * vest->at(k+1) * bj);
//
//				accval += el;
//			}
//
//		veemat(i,j) = accval;
//		}
//
//	double etot = 0;
//	for (uint k = 0; k < Pvec->size(); k++)
//	{
//		double accval = 0;
//		for (uint l = 0; l < veemat.n_rows; l++)
//			for (uint m = 0; m < veemat.n_cols; m++)
//			{
//				accval += eigvecs->at(k)(l)*veemat(l,m)*eigvecs->at(k)(m);
//			}
//
//		etot += orbnums->at(k) * (eigvals->at(k) - 0.5*accval);
//	}
//
//	return etot;
//}

double intvee (arma::vec* vest, vector <arma::vec>* Pvec, vecd* eigvals, vector <uint>* orbnums, bsplines* splines)
{
	vecd* grid = splines->getgrid();
	vecd rdiffgrid;
	rdiffgrid.reserve(grid->size());
	for (uint k = 0; k < grid->size()-1; k++)
	{
		auto diff = abs(grid->at(k)	- grid->at(k+1));
		rdiffgrid.push_back(diff);
	}
	
	double etot = 0;

	for (uint k = 0; k < Pvec->size(); k++)
	{
		vecd veediffgrid;
		veediffgrid.reserve(grid->size());

		for(uint l = 0; l < grid->size()-1; l++)
		{
			auto diff = 0.5 * (Pvec->at(k)(l)*vest->at(l)*Pvec->at(k)(l) + Pvec->at(k)(l+1)*vest->at(l+1)*Pvec->at(k)(l+1));
			
			veediffgrid.push_back(diff);
		}


		double accval = 0;
		for (uint i = 0; i < vest->n_rows; i++)
		{
			double el = rdiffgrid[i] * veediffgrid[i];

			accval += el;
		}
		
		etot += orbnums->at(k) * (eigvals->at(k) - 0.5*accval);
	}

	return etot;
}

double statedist(arma::vec* v1, arma::vec* v2)
{
	if (v1->size() != v2->size())
	{
		cout << "vector sizes incompatible" << endl;
		
		return -1;
	}

	double accval = 0;
	
	for(uint k = 0; k < v1->size(); k++)
	{
		double x = pow((*v1)(k) - (*v2)(k),2);

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
		fs << (*grid)[k]  << "," << 4*pi*(*dens)(k)*pow((*grid)[k],2) << endl;

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
	string gridfile = "grid.dat";
	string vknotfile = "vknots.dat";
	string vgridfile = "vgrid.dat";
	string outfile = "output.csv";
	
	// numbers
	uint nabsc = 10;					// number of abscissas
	uint knotpts = 125;					// without ghost pts
	uint gridpts = knotpts*nabsc;
	uint order = 5;
	uint vgridpts = gridpts + 2;
	
	double tol = 1e-12;
	uint nz = 10;
	uint ne = 10;
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
	double rmax = 4.0;
	uint mode = 1;

	vecd absc = getabsc();
	vecd w = getw();
	vecd wgrid;
	vecd pfgrid;
	vecd vdir (0,knotpts);
	vecd vexch (0,knotpts);

	knotsngrid (rmin,rmax,knotpts,mode,&absc,&w,&wgrid,&pfgrid,knotfile,gridfile);
	bsplines splines(knotfile,gridfile,order,tol);
	
	vknotsngrid (rmin, rmax, vknotfile, vgridfile, &splines);

	bsplines vsplines(vknotfile,vgridfile,order,tol);

	uint nosplines = splines.getnosplines(order-1);
	uint novsplines = vsplines.getnosplines(order-1);

	splines.calcsplines();
	vsplines.calcsplines();

	//splines.printgrid();
	//vsplines.printgrid();
	
	//cout << "h gridpts " << (*splines.getgridpts()) << " v gridpts " << (*vsplines.getgridpts()) << " h nsplines " << nosplines << " v nsplines " << novsplines << endl;
	
	float scale = 0.4;
	double c = 1;
	double acc = 1e-9;
	uint loops = 0;
	uint limit = 29;

	arma::mat rhs(nosplines,nosplines,arma::fill::zeros);
	buildrhs (order,&wgrid,&pfgrid,&rhs,&splines);
	boundaryrhs (&rhs);
		
	vecd enlvlsl0old = calcenlvls (sorbs, nz);
	vecd enlvlsl1old;
	for (uint k = 1; k <= porbs; k++)
		enlvlsl1old.push_back(enlvlsl0old[k]);
	
	vecd enlvlsl0orig = enlvlsl0old;
	vecd enlvlsl1orig = enlvlsl1old;

	arma::vec dens(vgridpts,arma::fill::zeros);
	arma::vec debugpsi(vgridpts,arma::fill::zeros);
	
	arma::vec vestdir;
	arma::vec vestex(novsplines,arma::fill::zeros);	
	arma::vec vest (gridpts+2,arma::fill::zeros);
	arma::vec vold (gridpts+2,arma::fill::zeros);
	vecd cs;
	cs.reserve(limit);

	vecd etots;
	arma::vec densold (gridpts+2,arma::fill::zeros);

	while (c > acc)		// main loop
	{
		///////////////////////////
		// build all Pnl vectors //
		///////////////////////////
		
		// parts of this probably could -- and should -- go into a sub rutine

		vector <arma::mat> lhs (l+1);
		for (auto& el : lhs)
			el = arma::zeros<arma::mat>(nosplines,nosplines);

		for (uint i = 0; i <= l; i++)
		{
			buildlhs (i,nz,order,&wgrid,&pfgrid,&vold,&lhs[i],&splines);
			boundarylhs (&lhs[i]);
		}

		// this should be done in classes to be c++ style, but armadillo and classes seems weird
		vector <arma::vec> l0, l1;
		arma::vec pnl (gridpts+2,arma::fill::zeros);

		for (uint k = 0; k < sorbs; k++)
			l0.push_back(pnl);
		
		for (uint k = 0; k < porbs; k++)
			l1.push_back(pnl);
		
		arma::cx_mat ceigenvecsl0;
		arma::cx_mat ceigenvecsl1;
		arma::cx_vec ceigenvalsl0;
		arma::cx_vec ceigenvalsl1;
	
		//cout << "lhs content:" << endl;
		//lhs.print();
		//cout << "rhs content:" << endl;
		//rhs.print();
		
		eig_pair(ceigenvalsl0,ceigenvecsl0,lhs[0],rhs);
		
		if (l == 1)
			eig_pair(ceigenvalsl1,ceigenvecsl1,lhs[1],rhs);
		
		arma::vec eigenvalsl0 = real(ceigenvalsl0);
		arma::vec eigenvalsl1 = real(ceigenvalsl1);
		arma::mat eigenvecsl0 = real(ceigenvecsl0);
		arma::mat eigenvecsl1 = real(ceigenvecsl1);

		// this should be done exclusively using armadillo, you shouldn't do it like this, but I can't get armadillo to play along at the moment, so, there it is
	
		//vector <uint> bsl0 = getbs(&enlvlsl0old, &eigenvalsl0);
		//vector <uint> bsl1 = getbs(&enlvlsl1old, &eigenvalsl1);
		//vector <uint> bsl0 = getbs(&enlvlsl0orig, &eigenvalsl0);
		//vector <uint> bsl1 = getbs(&enlvlsl1orig, &eigenvalsl1);
		vector <uint> bsl0 = getbs(sorbs,&eigenvalsl0);
		vector <uint> bsl1 = getbs(porbs,&eigenvalsl1);

		vecd eigvalsl0;
		vecd eigvalsl1;
		vector <arma::vec> eigvecsl0;
		vector <arma::vec> eigvecsl1;
		
		//eigenvalsl0.print();
		for (uint k = 0; k < sorbs; k++)
		{
			cout << "bsl0: " << bsl0[k] << " eigval: " << eigenvalsl0(bsl0[k]) << endl;
			arma::vec eigvec = eigenvecsl0.col(bsl0[k]);
			calcprob(order,&l0[k],&eigvec,&rhs,&splines);
			eigvalsl0.push_back(eigenvalsl0(bsl0[k]));
			eigvecsl0.push_back(eigvec);
		}

		//eigenvalsl1.print();
		for (uint k = 0; k < porbs; k++)
		{
			cout << "bsl1: " << bsl1[k] << " eigval: " << eigenvalsl1(bsl1[k]) << endl;
			arma::vec eigvec = eigenvecsl1.col(bsl1[k]);
			calcprob(order,&l1[k],&eigvec,&rhs,&splines);
			eigvalsl1.push_back(eigenvalsl1(bsl1[k]));
			eigvecsl1.push_back(eigvec);
		}

		vector <arma::vec> pvector;
		vecd eigvals;
		vector <arma::vec> eigvecs;

		pvector.push_back(l0[0]);
		eigvals.push_back(eigvalsl0[0]);
		eigvecs.push_back(eigvecsl0[0]);
		
		auto sloops = sorbs - 1;
		auto ploops = porbs;
	
		while (sloops > 0 || ploops > 0)
		{
			if (sloops > 0)
			{
				uint k = sorbs-sloops;
				pvector.push_back(l0[k]);
				eigvals.push_back(eigvalsl0[k]);
				eigvecs.push_back(eigvecsl0[k]);
				sloops--;
			}

			if (ploops > 0)
			{
				uint k = porbs-ploops;
				pvector.push_back(l1[k]);
				eigvals.push_back(eigvalsl1[k]);
				eigvecs.push_back(eigvecsl1[k]);
				ploops--;
			}
		}
	
		calcdens(&orbnums, &pvector, &dens, &vsplines);

		double csum = 0;
		vecd* vgrid = vsplines.getgrid();
		vecd diffgrid;
		for (uint k = 0; k < vgrid->size()-1; k++)
		{
			auto diff = abs(vgrid->at(k) - vgrid->at(k+1));
			diffgrid.push_back(diff);
		}
		
		for(uint k = 0; k < diffgrid.size(); k++)
			csum += pow( (*vgrid)[k] + diffgrid[k]/2 ,2)* 0.5 * (dens[k]+ dens[k+1]) * diffgrid[k];

		csum *= 4*pi;

		cout << "this should be Nocc: " << csum << endl;
	
		uint pick = 0;
		arma::vec dpnl = pvector[pick];
		for (auto& el : dpnl)
		{
			auto k = &el - &dpnl[0];

			if (k == 0)
				continue;

			el = pow(el/vgrid->at(k),2);
		}

		double dcsum = 0;
		for(uint k = 0; k < diffgrid.size(); k++)
			dcsum += pow( (*vgrid)[k] + diffgrid[k]/2 ,2)* 0.5 * (dpnl[k]+ dpnl[k+1]) * diffgrid[k];
		
		cout << "this should be 1: " << dcsum << endl;

		///////////////
		// Vee stuff //
		///////////////
		
		arma::mat vlhs(novsplines,novsplines,arma::fill::zeros); 
		arma::vec vrhs(novsplines,arma::fill::zeros);

		vbuildmatrix (order,vgridpts,&vlhs,&vrhs,&dens,&vsplines);
		vboundcond (order,ne,&vlhs,&vrhs,&vsplines); 
		
		//cout << "vlhs content:" << endl;
		//vlhs.print();
		//cout << "lrhs content:" << endl;
		//vrhs.print();

		arma::vec vsols;	
		vsolving(&vlhs,&vrhs,&vsols);
		
		vestval (order,&vestdir,&vsols,&vsplines);

		vestdir(0) = 0;		// bound conds
		vestdir(vestdir.size()-1) = 0;
		
		vex (&vestex,&dens);
		
		vest = (1-scale)*(vestdir + vestex) + scale*vold;
		if (loops == 0)	
			vest = vestdir + vestex;
		
		//double en = intvee (&wgrid,&pfgrid,&vold,&pvector,&eigvals,&eigvecs,&orbnums,order,&splines);
		double en = intvee (&vest,&pvector,&eigvals,&orbnums,&vsplines);

		etots.push_back(en);

		if (loops == limit)
		{
			c = statedist(&dens,&densold);
	
			cout << "loop count: " << loops << ", c: " << c << endl;
			
			vold = vest;
			densold = dens;
			cs.push_back(c);
			
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

			cout << "convergence not found within " << limit << " cycles, exiting " << endl;		
			break;
		}
		else 
		{	
			c = statedist(&dens,&densold);
	
			cout << "loop count: " << loops << ", c: " << c << endl;
			
			vold = vest;
			densold = dens;
			cs.push_back(c);
			
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

	ofstream fdir, fex, ftot, etot, fcs;
	fdir.open("dir.dat"); fex.open("ex.dat"); ftot.open("tot.dat"); etot.open("etot.dat"); fcs.open("converge.dat");
	
	for (auto& el : vestdir)
		fdir << el << endl;
	for (auto& el : vestex)
		fex << el << endl;
	for (auto& el : vest)
		ftot << el << endl;
	for (auto& el : etots)
		etot << el << endl;
	for (auto& el : cs)
		fcs << el << endl;
	
	fdir.close(); fex.close(); ftot.close(); etot.close(); fcs.close();
			
	//arma::vec psi(gridpts,arma::fill::zeros);
	//calcpsi(&psi,&probold,&splines);
	
	writedens(&splines,&dens,outfile);
	
	if (c < acc)
		cout << "sufficient convergence found in " << loops+0 << " iterations, with distance " << c << endl;

	cout << "This was a run for Z=" << nz << " e=" << ne << endl;
	
	return 0;
}



























