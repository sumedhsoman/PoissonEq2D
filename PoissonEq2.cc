#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

class PoissonEq2
{
public:
 PoissonEq2();
 void run();


private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;


  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;


  Vector<double> solution;
  Vector<double> system_rhs;
};
PoissonEq2::PoissonEq2()
 : fe(2)
 , dof_handler(triangulation)
{}

void PoissonEq2::make_grid()
{
    const Point<2> center(0,0);
    const double radius = 1;
    GridGenerator::hyper_shell(triangulation,center,radius,100);
    triangulation.refine_global(5);
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl;
}
void PoissonEq2::setup_system()
{
    dof_handler.distribute_dofs(fe);


    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
}
void PoissonEq2::assemble_system()
{
    QGauss<2> quadrature_formula(fe.degree+1);

    FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values);
    
    
 const unsigned int dofs_per_cell = fe.dofs_per_cell;
 FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
 Vector<double>     cell_rhs(dofs_per_cell);

 std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

 for(const auto &cell : dof_handler.active_cell_iterators())
 {
     fe_values.reinit(cell);

     cell_matrix = 0;
     cell_rhs = 0;

     for(const unsigned int q_index : fe_values.quadrature_point_indices())
     {
         for(const unsigned int i : fe_values.dof_indices())
             for(const unsigned int j : fe_values.dof_indices())
             cell_matrix(i, j) += 
                (fe_values.shape_grad(i, q_index)*
                 fe_values.shape_grad(j, q_index)*
                 fe_values.JxW(q_index));
         for (const unsigned int i : fe_values.dof_indices())
             cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            1 *                                 // f(x_q)
                            fe_values.JxW(q_index)); 
     }
     cell->get_dof_indices(local_dof_indices);
     for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));

      
     for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
 }
 std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);

  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);


}
void PoissonEq2::solve()
{


  SolverControl solver_control(1000, 1e-12);

  SolverCG<Vector<double>> solver(solver_control);

  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

}
void PoissonEq2::output_results() const
{
  // To write the output to a file, we need an object which knows about output
  // formats and the like. This is the DataOut class, and we need an object of
  // that type:
  DataOut<2> data_out;
  // Now we have to tell it where to take the values from which it shall
  // write. We tell it which DoFHandler object to use, and the solution vector
  // (and the name by which the solution variable shall appear in the output
  // file). If we had more than one vector which we would like to look at in
  // the output (for example right hand sides, errors per cell, etc) we would
  // add them as well:
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  // After the DataOut object knows which data it is to work on, we have to
  // tell it to process them into something the back ends can handle. The
  // reason is that we have separated the frontend (which knows about how to
  // treat DoFHandler objects and data vectors) from the back end (which knows
  // many different output formats) and use an intermediate data format to
  // transfer data from the front- to the backend. The data is transformed
  // into this intermediate format by the following function:
  data_out.build_patches();

  // Now we have everything in place for the actual output. Just open a file
  // and write the data into it, using VTK format (there are many other
  // functions in the DataOut class we are using here that can write the
  // data in postscript, AVS, GMV, Gnuplot, or some other file
  // formats):
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
}

void PoissonEq2::run()
{
  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}



int main()
{
  deallog.depth_console(2);

  PoissonEq2 laplace_problem;
  laplace_problem.run();

  return 0;
}

 


