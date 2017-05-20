#include <vector>
#include <boost/array.hpp>
#include <boost/numeric/odeint.hpp>
#include <string>
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/arrayobject.h>
#include <algorithm>

using namespace std;
using namespace boost::numeric::odeint;
namespace py = boost::python;

typedef std::vector<double> vtype;

class neuron 
{
    protected:
        int ndim;
        double* iapp;
        double res;
        boost::function<void(const vtype&, vtype&, double)> ode;

    public:

    	void apply_current(PyObject* niapp, double nres) 
        {

            res = nres;

            PyArrayObject* niapp_arrayob= reinterpret_cast<PyArrayObject*>(niapp);
            iapp = reinterpret_cast<double*>(PyArray_DATA(niapp_arrayob));

        }
        py::object integrate(PyObject* niapp, double nres, double tspan, double dt, py::list init_list) 
        {
                res = nres;

                vtype x = {};
            	for(int i = 0; i < py::len(init_list); i++) x.push_back(py::extract<double>(init_list[i]));

            	PyArrayObject* niapp_arrayob= reinterpret_cast<PyArrayObject*>(niapp);
            	iapp = reinterpret_cast<double*>(PyArray_DATA(niapp_arrayob));

                int currpoint = 0;
                int gridlength = tspan/dt;

                npy_intp size[1];
                size[0] = gridlength;
                size[1] = ndim;

                double data[gridlength*ndim];

                auto write = [this, &data, dt](const vtype &x, const double t)
                {
                    int i = round(t/dt);
                    for (int j = 0; j < ndim; j++){
                        data[i*ndim+j] = x[j];
                    }
                };

                runge_kutta_dopri5<vtype> stepper;

                integrate_const( make_dense_output( 1.0e-4 , 1.0e-4, stepper), ode, x, 0.0, tspan, dt, write);

                double (*data2)[gridlength][ndim] = reinterpret_cast<double (*)[gridlength][ndim]>(data);

                PyObject* pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data2 );
                py::handle<> handle( pyObj );
                py::numeric::array arr( handle );
                return arr.copy();

        }
        py::object calc_modelerr(boost::function<void(const vtype&, vtype&, double)> ode, int ndim, py::numeric::array data, double dt) 
        {
                npy_intp size[1];
                size[0] = len(data)-1;

                double out[len(data)-1];
                runge_kutta4<vtype> stepper;

                vtype curr(ndim);
                vtype nex(ndim);
                vtype predict(ndim);

                for(int i = 0; i < len(data)-1; i++)
                {
                        for (int j = 0; j < ndim; j++)
                        {
                                curr[j] = py::extract<double>(data[i][j]);
                                nex[j] = py::extract<double>(data[i+1][j]);
                        }

                        stepper.do_step(ode,curr,i*dt,predict,dt);

                        out[i] = 0.0;
                        for (int j = 0; j < ndim; j++) {
                                out[i] += predict[j] - nex[j];
                        }

                }

                PyObject* pyObj = PyArray_SimpleNewFromData( 1, size, NPY_DOUBLE, out );
                py::handle<> handle( pyObj );
                py::numeric::array arr( handle );
                return arr.copy();
        }
};

class neuron_reset: public neuron
{
	protected:
		virtual bool reset_condition(const vtype &x){};
		virtual vtype prereset(const vtype &x){};
		virtual vtype reset(const vtype &x){};

	public:
		void write(const vtype &x, const double t, const double dt, double *data)
		{
			int i = round(t/dt);
			for (int j = 0; j < ndim; j++)
			{
				data[i*ndim+j] = x[j];
			}
		}
			
		py::object integrate(double tspan, double dt, vtype x) 
		{
			int currpoint = 0;
			int gridlength = tspan/dt;

			npy_intp size[1];
			size[0] = gridlength;
			size[1] = ndim;

			double data[gridlength*ndim];

			auto stepper =  euler<vtype>();				
			double t = 0.0;


			int i = 1;
			double target = 0.0;
			
			py::list spikes;
			while(t < tspan)
			{

				stepper.do_step(ode,x,t,dt);

					
				if(reset_condition(x))
				{
					spikes.append(t);
					write(prereset(x), t, dt, data);
					x = reset(x);					
					t += dt;
				}
			
				write(x,t, dt, data);
				t += dt;
			}


			double (*data2)[gridlength][ndim] = reinterpret_cast<double (*)[gridlength][ndim]>(data);

			PyObject* pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data2 );
			py::handle<> handle( pyObj );
			py::numeric::array arr( handle );
			return arr.copy();

		}
};

class iz: public neuron_reset
{
    public:
		double a,b,c,d,h;
		void set_equations()
   		{
            ndim = 2;
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                // Getting the applied current at time t
                int rt = round(t)/res;
                double I = iapp[rt];

                dxdt[0] = 0.04*x[0]*x[0] + 5*x[0] + 140 - x[1] + I;
                dxdt[1] = a*(b*x[0] - x[1]);
            };
    	}
		iz() 
		{
			a = 0.02;
			b = 0.2;
			c = -65.0;
			d = 8.0;
			h = 30.0;
			set_equations();
		};
		iz(double na, double nb, double nc, double nd, double nh)
		{
			a = na;
			b = nb;
			c = nc;
			d = nd;
			h = nh;
			set_equations();
		};
		
		bool reset_condition(const vtype &x)
		{
			return x[0] >= h;
		}
		
		vtype prereset(const vtype &x)
		{
			return {h,x[1]};
		}
		
		vtype reset(const vtype &x) 
		{
			return {c,x[1]+d};
		}

		py::object simulate(const double tspan, const double dt)
		{	
			vtype start = {c, -14.0};
            return integrate(tspan, dt, start);
		}
};

class adex: public neuron_reset
{
    public:
        double C,gl,el,delt,vt,tw,a,vr,b,h,R;
        void set_equations()
   		{
            ndim = 2;
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                // Getting the applied current at time t
                int rt = round(t)/res;
                double I = iapp[rt];

                dxdt[0] = 1/C*(-gl*(x[0]-el) + gl*delt*exp((x[0]-vt)/delt) - x[1] + R*I);
                dxdt[1] = 1/tw*(a*(x[0]-el) - x[1]);
            };
    	}
        adex() 
        {
            C    = 250.0; 
            gl   = 30.0;
            el   = -70.6;
            delt = 2.0;
            vt   = -55.0;
            tw   = 144.0;
            a    = 4.0;
            vr   = -70.6;
            b    = 80.5;
            h    = 30.0;
            R    = 1.0;
            set_equations();
        };
        adex(double nC,double ngl,double nel,double ndelt,double nvt,double ntw,double na,double nvr,double nb)
        {
            C    = nC; 
            gl   = ngl;
            el   = nel;
            delt = ndelt;
            vt   = nvt;
            tw   = ntw;
            a    = na;
            vr   = nvr;
            b    = nb;
            h    = 30.0;
            R    = 1.0;
            set_equations();
        };
		
		bool reset_condition(const vtype &x)
		{
			return x[0] >= h;
		}
		
		vtype prereset(const vtype &x)
		{	
			return {h,x[1]};
		}
		
		vtype reset(const vtype& x)
		{ 	
			return {vr,x[1]+b};
		}

        py::object simulate(double tspan, double dt)
        {
            vtype start = {el, 0.0};
            return integrate(tspan, dt, start);
        }
};

class mat: public neuron
{
    public:
            double tm,R,a,b,w,t1,t2,h,tref;
            mat() 
            {
                tm=10;
                R=50;
                a=15;
                b=3;
                w=5;
                t1=10;
                t2=200;
                tref=2;
                            };
            mat(double na, double nb, double nw)
            {
                a  = na;
                b  = nb;
                w  = nw;
                tm  = 10;
                R  = 50;
                t1 = 10;
                t2 = 200;
                tref = 2;
            };

            py::object simulate(const double duration, const double dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            double v = 0;
                            double h = 0;

                            double h1 = 0;
                            double h2 = 0;
                            double tf = 0;

                            double dv = 0;
                            double dh1 = 0;
                            double dh2 = 0;
                            
                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            double data[grid][2];

                            py::list spikes;

                            for(int i = 0; i < grid; i++)
                            {

                                rt = round(i*dt)/res;

                                dh1 = dt*(-h1/t1);
                                dh2 = dt*(-h2/t2);

                                dv  = dt*(-v + R*iapp[rt])/tm;

                                v  = v + dv;
                                h1 = h1 + dh1;
                                h2 = h2 + dh2;

                                h  = w + h1 + h2;

                                if(v > h && tf + tref/dt < i + 1)
                                {

                                    h1 += a;
                                    h2 += b;

                                    tf = i + 1;
                                    spikes.append(i);
                                }

                                data[i][0] = v;
                                data[i][1] = h;   
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return py::make_tuple(arr,spikes);
                    }
};

class augmat: public neuron
{
    public:
            double tm,R,a,b,w,t1,t2,h,tref,tv,c;
            augmat() 
            {
                tm=10;
                R=50;
                a=-0.5;
                b=0.35;
                w=5;
                t1=10;
                t2=200;
                tref=2;
                tv=5;
                c=0.3;
            };
            augmat(double na, double nb, double nc, double nw)
            {
                a  = na;
                b  = nb;
                w  = nw;
                c  = nc;
                tm  = 10;
                R  = 50;
                t1 = 10;
                t2 = 200;
                tref = 2;
                tv = 5;

            };

            py::object simulate(const double duration, const double dt)
                    {
                            int grid = duration/dt;
                            int rt = 0;

                            double v = 0;
                            double h = 0;

                            double h1 = 0;
                            double h2 = 0;
                            double hv = 0;
                            double tf = 0;

                            double dv = 0;
                            double dh1 = 0;
                            double dh2 = 0;
                            double dhv = 0;

                            double ddhv = 0;
                            
                            npy_intp size[2];
                            size[0] = grid;
                            size[1] = 2;

                            double data[grid][2];

                            py::list spikes;

                            for(int i = 0; i < grid; i++)
                            {

                                rt = round(i*dt)/res;

                                dh1 = dt*(-h1/t1);
                                dh2 = dt*(-h2/t2);
                                dhv = dt*(-hv/tv + ddhv);
                                ddhv = ddhv + dt*(-(hv/tv + dhv)/tv) + c*dv;

                                dv  = dt*(-v + R*iapp[rt])/tm;

                                v  = v + dv;
                                h1 = h1 + dh1;
                                h2 = h2 + dh2;
                                hv = hv + dhv;

                                h  = w + h1 + h2 + hv;

                                if(v > h && tf + tref/dt < i + 1)
                                {

                                    h1 += a;
                                    h2 += b;

                                    tf = i + 1;
                                    spikes.append(i);
                                }

                                data[i][0] = v;
                                data[i][1] = h;   
                            }

                            PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                            py::handle<> handle( pyObj );
                            py::numeric::array arr( handle );
                            return py::make_tuple(arr,spikes);
                    }
};

class augmatsto: public neuron
{
    public:
            double tm,R,a,b,w,t1,t2,h,tref,tv,c;

            augmatsto() 
            {
                tm=10;
                R=50;
                a=-0.5;
                b=0.35;
                w=5;
                t1=10;
                t2=200;
                tref=2;
                tv=5;
                c=0.3;
            };
            augmatsto(double na, double nb, double nc, double nw)
            {
                a  = na;
                b  = nb;
                w  = nw;
                c  = nc;
                tm  = 10;
                R  = 1;
                t1 = 10;
                t2 = 200;
                tref = 2;
                tv = 5;

            };

            py::object simulate(const double duration, const double dt)
            {
                default_random_engine rng(random_device{}());
                uniform_real_distribution<double> udist(0,1);


                int grid = duration/dt;
                int rt = 0;

                double v = 0;
                double h = 0;

                double h1 = 0;
                double h2 = 0;
                double hv = 0;
                double tf = 0;

                double dv = 0;
                double dh1 = 0;
                double dh2 = 0;
                double dhv = 0;

                double ddhv = 0;

                double ps = 0;

                
                npy_intp size[2];
                size[0] = grid;
                size[1] = 3;

                double data[grid][3];

                py::list spikes;

                for(int i = 0; i < grid; i++)
                {

                    rt = round(i*dt)/res;

                    dh1 = dt*(-h1/t1);
                    dh2 = dt*(-h2/t2);
                    dhv = dt*(-hv/tv + ddhv);
                    ddhv = ddhv + dt*(-(hv/tv + dhv)/tv) + c*dv;

                    dv  = dt*(-v + R*iapp[rt])/tm;

                    v  = v + dv;
                    h1 = h1 + dh1;
                    h2 = h2 + dh2;
                    hv = hv + dhv;

                    h  = w + h1 + h2 + hv;

                    ps = exp(v-h);

                    if(udist(rng) < ps && tf + tref/dt < i + 1)
                    {

                        h1 += a;
                        h2 += b;

                        tf = i + 1;
                        spikes.append(i);
                    }

                    data[i][0] = v;
                    data[i][1] = h;  
                    data[i][2] = ps; 
                }

                PyObject * pyObj = PyArray_SimpleNewFromData( 2, size, NPY_DOUBLE, data );
                py::handle<> handle( pyObj );
                py::numeric::array arr( handle );
                return py::make_tuple(arr,spikes);
            }
};

class hr: public neuron
{
    public:
        typedef boost::array< double , 3> state_type; // holds variable values form previous timestep
        double a,b,c,d,r,s,xn;

        void set_equations()
        {
            ndim = 3;
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                // Getting the applied current at time t
                int rt = round(t)/res;
                double I = iapp[rt];

                // Membrane potential
                dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                // Spiking variable/Recovery Current
                dxdt[1] = c - d*x[0]*x[0] - x[1];

                // Bursting variable
                dxdt[2] = r*(s*(x[0] - xn) - x[2]);
            };
        }

        hr() 
        {
            a = 1.0;
            b = 3.0;
            c = 1.0;
            d = 5.0;
            r = 0.005;
            s = 4.0;
            xn = -2.0;

            set_equations();
        };

        hr(double na, double nb, double nc, double nd, double nr, double ns, double nxn)
        {
            a = na;
            b = nb;
            c = nc;
            d = nd;
            r = nr;
            s = ns;
            xn = nxn;

            set_equations();
        };
};

class hr4: public neuron
{
    public:
        double a,b,c,d,r,s,xn,v,g,k,l;

        void set_equations()
        {
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                // Getting the applied current at time t
                int rt = round(t)/res;
                double I = iapp[rt];

                // Membrane potential
                dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                // Spiking variable/Recovery Current
                dxdt[1] = c - d*x[0]*x[0] - x[1] - g*x[3];

                // Bursting variable
                dxdt[2] = r*(s*(x[0] - xn) - x[2]);

                dxdt[3] = v*(k*(x[1] + l) - x[3]);
            };

            ndim = 4;
        }

        hr4() 
        {
            a = 1.0;
            b = 3.0;
            c = 1.0;
            d = 5.0;
            r = 0.005;
            s = 4.0;
            xn = -2.0;
            v = 0.001;
            g = 0.1;
            k = 3.0;
            l = 1.6;

            set_equations();
        };

        hr4(double na, double nb, double nc, double nd, double nr, double ns, double nxn, double nv, double nk, double ng, double nl)
        {
            a = na;
            b = nb;
            c = nc;
            d = nd;
            r = nr;
            s = ns;
            xn = nxn;
            v = nv;
            g = ng;
            k = nk;
            l = nl;

            set_equations();
        };
};

class hr2: public neuron
{

    public:
        double a,b,c,d,xn;

        void set_equations()
        {
            ndim = 2;
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                // Getting the applied current at time t
                int rt = round(t)/res;
                double I = iapp[rt];

                // Membrane potential
                dxdt[0] = x[1] - a*x[0]*x[0]*x[0] + b*x[0]*x[0] - x[2] + I;

                // Spiking variable/Recovery Current
                dxdt[1] = c - d*x[0]*x[0] - x[1];

            };
        }

        hr2() 
        {
            a = 1.0;
            b = 3.0;
            c = 1.0;
            d = 5.0;
            xn = -2.0;

            set_equations();
        };

        hr2(double na, double nb, double nc, double nd, double nr, double ns, double nxn)
        {
            a = na;
            b = nb;
            c = nc;
            d = nd;
            xn = nxn;

            set_equations();
        };
};

class hh: public neuron
{
    public:
        double C,gna,Ena,gk,Ek,gl,El,vm,dvm,tm0,tm1,vmt,dvmt,vh,dvh,
                th0,th1,vht,dvht,vn,dvn,tn0,tn1,vnt,dvnt;

        void set_equations()
        {
            ndim = 4;
            ode = [this]( const vtype &x , vtype &dxdt , double t )
            {
                int rt = round(round(t)/res);
                double I = iapp[rt];

                dxdt[0] = ((gna*x[1]*x[1]*x[1]*x[2]*(Ena-x[0]))+(gk*x[3]*x[3]*x[3]*x[3]*(Ek - x[0]))+
                           (gl*(El-x[0])) + I)/C;

                double taum = tm0 + tm1 * (1-pow(tanh((x[0] - vmt)/dvmt),2));
                double m0 = (1+tanh((x[0]-vm)/dvm))/2;
                dxdt[1] = (m0 - x[1])/taum;

                double tauh = th0 + th1 * (1-pow(tanh((x[0] - vht)/dvht),2));
                double h0 = (1+tanh((x[0]-vh)/dvh))/2;
                dxdt[2] = (h0 - x[2])/tauh;

                double taun = tn0 + tn1 * (1-pow(tanh((x[0] - vnt)/dvnt),2));
                double n0 = (1+tanh((x[0]-vn)/dvn))/2;
                dxdt[3] = (n0 - x[3])/taun;
            };
        }

        hh() 
        {
                C=1.0;
                gna=120.0;
                Ena=50.0;
                gk=20.0;
                Ek=-77.0;
                gl=0.3;
                El=-54.4;
                vm=-40.0;
                dvm=15.0;
                tm0=0.1;
                tm1=0.4;
                vmt=-40.0;
                dvmt=15.0;
                vh=-60.0;
                dvh=-15.0;
                th0=1.0;
                th1=7.0;
                vht=-60.0;
                dvht=-15.0;
                vn=-55.0;
                dvn=30.0;
                tn0=1.0;
                tn1=5.0;
                vnt=-55.0;
                dvnt=30.0;
                set_equations();
                
            }

            hh(py::list param)
            {
                gna=py::extract<double>(param[0]);
                Ena=py::extract<double>(param[1]);
                gk=py::extract<double>(param[2]);
                Ek=py::extract<double>(param[3]);
                gl=py::extract<double>(param[4]);
                El=py::extract<double>(param[5]);
                vm=py::extract<double>(param[6]);
                dvm=py::extract<double>(param[7]);
                tm0=py::extract<double>(param[8]);
                tm1=py::extract<double>(param[9]);
                vmt=py::extract<double>(param[10]);
                dvmt=py::extract<double>(param[11]);
                vh=py::extract<double>(param[12]);
                dvh=py::extract<double>(param[13]);
                th0=py::extract<double>(param[14]);
                th1=py::extract<double>(param[15]);
                vht=py::extract<double>(param[16]);
                dvht=py::extract<double>(param[17]);
                vn=py::extract<double>(param[18]);
                dvn=py::extract<double>(param[19]);
                tn0=py::extract<double>(param[20]);
                tn1=py::extract<double>(param[21]);
                vnt=py::extract<double>(param[22]);
                dvnt=py::extract<double>(param[23]);
                C=py::extract<double>(param[24]);
                
                set_equations();
            }
};

BOOST_PYTHON_MODULE(cneurons)
{
        using namespace py;
        import_array();
        py::numeric::array::set_module_and_type("numpy", "ndarray");

        class_<neuron, boost::noncopyable>("_neuron")
        	.def("apply_current",&neuron::apply_current)
        	.def("integrate",&neuron::integrate);

        class_<iz,bases<neuron>>("iz")
                .def(init<double, double, double, double, double>())
                .def("simulate", &iz::simulate)
                .def_readwrite("a", &iz::a)
                .def_readwrite("b", &iz::b)
                .def_readwrite("c", &iz::c)
                .def_readwrite("d", &iz::d)
                .def_readwrite("h", &iz::h);

        
        class_<adex,bases<neuron>>("adex")
                .def(init<double, double, double, double, double, double, double, double, double>())
                .def("simulate", &adex::simulate)
                .def_readwrite("C", &adex::C)
                .def_readwrite("gl", &adex::gl)
                .def_readwrite("el", &adex::el)
                .def_readwrite("delt", &adex::delt)
                .def_readwrite("vt", &adex::vt)
                .def_readwrite("tw", &adex::tw)
                .def_readwrite("a", &adex::a)
                .def_readwrite("vr", &adex::vr)
                .def_readwrite("b", &adex::b)
                .def_readwrite("h", &adex::h)
                .def_readwrite("R", &adex::R);

        class_<mat,bases<neuron>>("mat")
                .def(init<double,double,double>())
                .def("simulate", &mat::simulate)
                .def_readwrite("tm", &mat::tm)
                .def_readwrite("R", &mat::R)
                .def_readwrite("a", &mat::a)
                .def_readwrite("b", &mat::b)
                .def_readwrite("w", &mat::w)
                .def_readwrite("t1", &mat::t1)
                .def_readwrite("t2", &mat::t2)
                .def_readwrite("tref", &mat::tref);

        class_<augmat,bases<neuron>>("augmat")
                .def(init<double,double,double,double>())
                .def("simulate", &augmat::simulate)
                .def_readwrite("tm", &augmat::tm)
                .def_readwrite("R", &augmat::R)
                .def_readwrite("a", &augmat::a)
                .def_readwrite("b", &augmat::b)
                .def_readwrite("w", &augmat::w)
                .def_readwrite("t1", &augmat::t1)
                .def_readwrite("t2", &augmat::t2)
                .def_readwrite("tref", &augmat::tref)
                .def_readwrite("tv", &augmat::tv)
                .def_readwrite("c", &augmat::c);

        class_<augmatsto,bases<neuron>>("augmatsto")
                .def(init<double,double,double,double>())
                .def("simulate", &augmatsto::simulate)
                .def_readwrite("tm", &augmatsto::tm)
                .def_readwrite("R", &augmatsto::R)
                .def_readwrite("a", &augmatsto::a)
                .def_readwrite("b", &augmatsto::b)
                .def_readwrite("w", &augmatsto::w)
                .def_readwrite("t1", &augmatsto::t1)
                .def_readwrite("t2", &augmatsto::t2)
                .def_readwrite("tref", &augmatsto::tref)
                .def_readwrite("tv", &augmatsto::tv)
                .def_readwrite("c", &augmatsto::c);


        class_<hr,bases<neuron>>("hr")
                .def(init<double, double, double, double, double, double, double>())
                .def_readwrite("a", &hr::a)
                .def_readwrite("b", &hr::b)
                .def_readwrite("c", &hr::c)
                .def_readwrite("d", &hr::d)
                .def_readwrite("r", &hr::r)
                .def_readwrite("s", &hr::s)
                .def_readwrite("xn", &hr::xn);

        class_<hr2,bases<neuron>>("hr2")
                .def(init<double, double, double, double, double, double, double>())
                .def_readwrite("a", &hr2::a)
                .def_readwrite("b", &hr2::b)
                .def_readwrite("c", &hr2::c)
                .def_readwrite("d", &hr2::d)
                .def_readwrite("xn", &hr2::xn);

        class_<hr4,bases<neuron>>("hr4")
                .def(init<double, double, double, double, double, double, double, double, double, double, double>())
                .def_readwrite("a", &hr4::a)
                .def_readwrite("b", &hr4::b)
                .def_readwrite("c", &hr4::c)
                .def_readwrite("d", &hr4::d)
                .def_readwrite("r", &hr4::r)
                .def_readwrite("s", &hr4::s)
                .def_readwrite("xn",&hr4::xn)
                .def_readwrite("v", &hr4::v)
                .def_readwrite("g", &hr4::g)
                .def_readwrite("k", &hr4::k)
                .def_readwrite("l", &hr4::l);

        class_<hh,bases<neuron>>("hh")
                .def(init<py::list>())
                .def_readwrite("C", &hh::C)
                .def_readwrite("gna", &hh::gna)
                .def_readwrite("Ena", &hh::Ena)
                .def_readwrite("gk", &hh::gk)
                .def_readwrite("Ek", &hh::Ek)
                .def_readwrite("gl", &hh::gl)
                .def_readwrite("El", &hh::El)
                .def_readwrite("vm", &hh::vm)
                .def_readwrite("dvm", &hh::dvm)
                .def_readwrite("tm0", &hh::tm0)
                .def_readwrite("tm1", &hh::tm1)
                .def_readwrite("vmt", &hh::vmt)
                .def_readwrite("dvmt", &hh::dvmt)
                .def_readwrite("vh", &hh::vh)
                .def_readwrite("dvh", &hh::dvh)
                .def_readwrite("th0", &hh::th0)
                .def_readwrite("th1", &hh::th1)
                .def_readwrite("vht", &hh::vht)
                .def_readwrite("dvht", &hh::dvht)
                .def_readwrite("vn", &hh::vn)
                .def_readwrite("dvn", &hh::dvn)
                .def_readwrite("tn0", &hh::tn0)
                .def_readwrite("tn1", &hh::tn1)
                .def_readwrite("vnt", &hh::vnt)
                .def_readwrite("dvnt", &hh::dvnt);
}
