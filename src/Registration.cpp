#include "Registration.h"


struct PointDistance
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////
  // This class should include an auto-differentiable cost function. 
  // To rotate a point given an axis-angle rotation, use
  // the Ceres function:
  // AngleAxisRotatePoint(...) (see ceres/rotation.h)
  // Similarly to the Bundle Adjustment case initialize the struct variables with the source and  the target point.
  // You have to optimize only the 6-dimensional array (rx, ry, rz, tx ,ty, tz).
  // WARNING: When dealing with the AutoDiffCostFunction template parameters,
  // pay attention to the order of the template parameters
  ////////////////////////////////////////////////////////////////////////////////////////////////////
};


Registration::Registration(std::string cloud_source_filename, std::string cloud_target_filename)
{
  open3d::io::ReadPointCloud(cloud_source_filename, source_ );
  open3d::io::ReadPointCloud(cloud_target_filename, target_ );
  Eigen::Vector3d gray_color;
  source_for_icp_ = source_;
}


Registration::Registration(open3d::geometry::PointCloud cloud_source, open3d::geometry::PointCloud cloud_target)
{
  source_ = cloud_source;
  target_ = cloud_target;
  source_for_icp_ = source_;
}


void Registration::draw_registration_result()
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  //different color
  Eigen::Vector3d color_s;
  Eigen::Vector3d color_t;
  color_s<<1, 0.706, 0;
  color_t<<0, 0.651, 0.929;

  target_clone.PaintUniformColor(color_t);
  source_clone.PaintUniformColor(color_s);
  source_clone.Transform(transformation_);

  auto src_pointer =  std::make_shared<open3d::geometry::PointCloud>(source_clone);
  auto target_pointer =  std::make_shared<open3d::geometry::PointCloud>(target_clone);
  open3d::visualization::DrawGeometries({src_pointer, target_pointer});
  return;
}



void Registration::execute_icp_registration(double threshold, int max_iteration, double relative_rmse, std::string mode)
{ 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //ICP main loop
  //Check convergence criteria and the current iteration.
  //If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  //Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    double previous_rmse = std::numeric_limits<double>::max();
    int iteration = 0;

    while (iteration < max_iteration) {
        // Find closest points
        auto [source_indices, target_indices, rmse] = find_closest_point(threshold);

        // Check for convergence based on RMSE change
        if ((previous_rmse - rmse) < relative_rmse) {
            std::cout << "Converged at iteration " << iteration << " with RMSE: " << rmse << std::endl;
            break;
        }

        // Get the transformation matrix
        if (mode == "svd") {
            transformation_ = get_svd_icp_transformation(source_indices, target_indices);
        } else if (mode == "lm") {
            transformation_ = get_lm_icp_registration(source_indices, target_indices);
        } else {
            std::cerr << "Use'svd' or 'lm'." << std::endl;
            return;
        }

        // Transform the source point cloud
        source_for_icp_.Transform(transformation_);

        // Update the RMSE
        previous_rmse = rmse;

        // Increment the iteration counter
        iteration++;
    }

    // Final transformation
    source_ = source_for_icp_;
    std::cout << "Final transformation after " << iteration << " iterations: " << std::endl << transformation_ << std::endl;
}


std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{ ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find source and target indices: for each source point find the closest one in the target and discard if their 
  //distance is bigger than threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<size_t> target_indices;
  std::vector<size_t> source_indices;
  Eigen::Vector3d source_point;
  double rmse;

  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  double total_squared_error = 0.0;
  size_t num_valid_pairs = 0;

  for (size_t i = 0; i < source_.points_.size(); ++i) {
      const auto& source_point = source_.points_[i];
     
      std::vector<int> indices(1);
      std::vector<double> distances(1);
      
      if (target_kd_tree.SearchKNN(source_point, 1, indices, distances) > 0) {
          if (distances[0] <= threshold * threshold) {
              source_indices.push_back(i);
              target_indices.push_back(indices[0]);
              total_squared_error += distances[0];
              num_valid_pairs++;
           }
      }
  }  



  if (num_valid_pairs > 0) {
    rmse = std::sqrt(total_squared_error / num_valid_pairs);
  }else {
    rmse = std::numeric_limits<double>::infinity();
  }  
  return {source_indices, target_indices, rmse};
}

Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> source_indices, std::vector<size_t> target_indices){
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Find point clouds centroids and subtract them. 
  //Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  //Use source_indices and target_indices to extract point to compute the 3x3 matrix to be decomposed.
  //Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  return transformation;
}

Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> source_indices, std::vector<size_t> target_indices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use source_indices and target_indices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  std::vector<double> transformation_arr(6, 0.0);
  int num_points = source_indices.size();
  // For each point....
  for( int i = 0; i < num_points; i++ )
  {
    
  }

  return transformation;
}


void Registration::set_transformation(Eigen::Matrix4d init_transformation)
{
  transformation_=init_transformation;
}


Eigen::Matrix4d  Registration::get_transformation()
{
  return transformation_;
}

double Registration::compute_rmse()
{
  open3d::geometry::KDTreeFlann target_kd_tree(target_);
  open3d::geometry::PointCloud source_clone = source_;
  source_clone.Transform(transformation_);
  int num_source_points  = source_clone.points_.size();
  Eigen::Vector3d source_point;
  std::vector<int> idx(1);
  std::vector<double> dist2(1);
  double mse;
  for(size_t i=0; i < num_source_points; ++i) {
    source_point = source_clone.points_[i];
    target_kd_tree.SearchKNN(source_point, 1, idx, dist2);
    mse = mse * i/(i+1) + dist2[0]/(i+1);
  }
  return sqrt(mse);
}

void Registration::write_tranformation_matrix(std::string filename)
{
  std::ofstream outfile (filename);
  if (outfile.is_open())
  {
    outfile << transformation_;
    outfile.close();
  }
}

void Registration::save_merged_cloud(std::string filename)
{
  //clone input
  open3d::geometry::PointCloud source_clone = source_;
  open3d::geometry::PointCloud target_clone = target_;

  source_clone.Transform(transformation_);
  open3d::geometry::PointCloud merged = target_clone+source_clone;
  open3d::io::WritePointCloud(filename, merged );
}


