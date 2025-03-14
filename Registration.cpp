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

  const Eigen::Vector3d source_point_;
  const Eigen::Vector3d target_point_;

  PointDistance(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point)
      : source_point_(source_point), target_point_(target_point) {}

  template<typename T>
  bool operator()(const T* const rotation, const T* const translation, T* residual) const {
    // Convert the rotation parameters to an Eigen type.
    Eigen::Matrix<T, 3, 1> rotation_vector(rotation[0], rotation[1], rotation[2]);

    // Convert source and target points to Eigen types.
    Eigen::Matrix<T, 3, 1> source_point_t = source_point_.template cast<T>();
    Eigen::Matrix<T, 3, 1> target_point_t = target_point_.template cast<T>();

    T transformed_point[3];

    // Use ceres provided AngleAxisRotatePoint to rotate the source point.
    ceres::AngleAxisRotatePoint(rotation_vector.data(), source_point_t.data(), transformed_point);

    // Apply the translation.
    for (int i = 0; i < 3; ++i) {
      transformed_point[i] += translation[i];
    }

    // Compute the residual as the difference between the transformed source point and the target point.
    for (int i = 0; i < 3; ++i) {
      residual[i] = transformed_point[i] - target_point_t[i];
    }

    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point) {
    return new ceres::AutoDiffCostFunction<PointDistance, 3, 3, 3>(
        new PointDistance(source_point, target_point));
  }
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
  // ICP main loop
  // Check convergence criteria and the current iteration.
  // If mode=="svd" use get_svd_icp_transformation if mode=="lm" use get_lm_icp_transformation.
  // Remember to update transformation_ class variable, you can use source_for_icp_ to store transformed 3d points.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  double pre_rmse = std::numeric_limits<double>::max();


  // Start timing
  auto start_time = std::chrono::high_resolution_clock::now();


  int iteration = 0;
  while (iteration < max_iteration) 
  {

    // Find closest points
    const auto closest_point = find_closest_point(threshold);

    // Check for convergence based on RMSE change
    if ((pre_rmse - std::get<2>(closest_point)) < relative_rmse) {
        std::cout << "Converged at iteration " << iteration << std::endl;
        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        std::cout << "Total time for ICP registration: " << duration/ 1000.0 << " Sec" << std::endl;
        break;
    }


    std::cout << "We are in iterarion: " <<iteration << " of " << max_iteration << std::endl;
    source_for_icp_ = source_;
    Eigen::Matrix4d transform;

    if (mode == "svd"){
      transform = get_svd_icp_transformation(std::get<0>(closest_point), std::get<1>(closest_point));
      source_.Transform(transform);
    }else if (mode == "lm") {
      transform = get_lm_icp_registration(std::get<0>(closest_point), std::get<1>(closest_point));
      source_.Transform(transform);
    }else{
      std::cout<<"Plese enter svd OR lm as last arguman"<<std::endl;
    }

    double new_rmse = compute_rmse();
    if (new_rmse < pre_rmse){
      transformation_ = transform;
    }


    pre_rmse = std::get<2>(closest_point);
    iteration++;
  }
  return;
}



std::tuple<std::vector<size_t>, std::vector<size_t>, double> Registration::find_closest_point(double threshold)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Find source and target indices: for each source point, find the closest one in the target and discard if their
  // distance is bigger than the threshold
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Vectors to store indices of matching points in source and target
  std::vector<size_t> targetIndices;
  std::vector<size_t> sourceIndices;

  // Variable to hold the Root Mean Squared Error (RMSE)
  double rmse = 0.0;

  // Create a KDTree for the target point cloud to efficiently find the nearest neighbors
  open3d::geometry::KDTreeFlann target_kd_tree(target_);

  // Variables to accumulate total squared error and count of valid pairs
  double total_squared_error = 0.0;
  size_t num_valid_pairs = 0;

  // Iterate over each point in the source point cloud
  for (size_t i = 0; i < source_.points_.size(); ++i) {
      const Eigen::Vector3d& source_point = source_.points_[i];
     
      std::vector<int> indices(1);
      std::vector<double> squared_distances(1);
      
      if (target_kd_tree.SearchKNN(source_point, 1, indices, squared_distances) > 0) {
          if (squared_distances[0] <= threshold * threshold) {
              sourceIndices.push_back(i);
              targetIndices.push_back(indices[0]);
              total_squared_error += squared_distances[0];
              num_valid_pairs++;
           }
      }
  }  

  // Calculate RMSE if there are valid pairs, otherwise set it to infinity
  if (num_valid_pairs > 0) {
    rmse = std::sqrt(total_squared_error / num_valid_pairs);
  } else {
    rmse = std::numeric_limits<double>::infinity();
  }  

  return {sourceIndices, targetIndices, rmse};
}


Eigen::Matrix4d Registration::get_svd_icp_transformation(std::vector<size_t> sourceIndices, std::vector<size_t> targetIndices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Find point clouds centroids and subtract them. 
  // Use SVD (Eigen::JacobiSVD<Eigen::MatrixXd>) to find best rotation and translation matrix.
  // Use sourceIndices and targetIndices to extract point to compute the 3x3 matrix to be decomposed.
  // Remember to manage the special reflection case.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Ensure the indices are valid and the same size
  assert(sourceIndices.size() == targetIndices.size());
  size_t num_points = sourceIndices.size();

  // Convert the point clouds to Eigen matrices using the indices
  Eigen::Matrix<double, 3, Eigen::Dynamic> sourcePoints(3, num_points);
  Eigen::Matrix<double, 3, Eigen::Dynamic> targetPoints(3, num_points);

  for (size_t i = 0; i < num_points; ++i) {
    sourcePoints.col(i) = source_.points_[sourceIndices[i]];
    targetPoints.col(i) = target_.points_[targetIndices[i]];
  }

  // Compute centroids of source and target point clouds
  Eigen::Vector3d center_source = sourcePoints.rowwise().mean();
  Eigen::Vector3d center_target = targetPoints.rowwise().mean();

  // Subtract centroids to get centered vectors
  Eigen::Matrix<double, 3, Eigen::Dynamic> centered_source = sourcePoints.colwise() - center_source;
  Eigen::Matrix<double, 3, Eigen::Dynamic> centered_target = targetPoints.colwise() - center_target;

  // Compute the covariance matrix
  Eigen::Matrix3d covariance_matrix = centered_source * centered_target.transpose();

  // Perform SVD
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance_matrix, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();

  // Compute the rotation matrix
  Eigen::Matrix3d R = V * U.transpose();

  // Handle special reflection case
  if (R.determinant() < 0) {
    V.col(2) *= -1;
    R = V * U.transpose();
  }

  // Compute the translation vector
  Eigen::Vector3d t = center_target - R * center_source;

  // Form the transformation matrix
  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
  transformation.block<3, 3>(0, 0) = R;
  transformation.block<3, 1>(0, 3) = t;

  return transformation;
}


Eigen::Matrix4d Registration::get_lm_icp_registration(std::vector<size_t> sourceIndices, std::vector<size_t> targetIndices)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //Use LM (Ceres) to find best rotation and translation matrix. 
  //Remember to convert the euler angles in a rotation matrix, store it coupled with the final translation on:
  //Eigen::Matrix4d transformation.
  //The first three elements of std::vector<double> transformation_arr represent the euler angles, the last ones
  //the translation.
  //use sourceIndices and targetIndices to extract point to compute the matrix to be decomposed.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity(4,4);

  // Set up the Ceres solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  options.num_threads = 4;
  options.max_num_iterations = 100;

  ceres::Problem problem;

  // Initialize transformation parameters: first 3 for rotation (Euler angles), last 3 for translation
  std::vector<double> transformation_arr(6, 0.0); 

  // Ensure the indices are valid and the same size
  assert(sourceIndices.size() == targetIndices.size());
  int num_points = sourceIndices.size();

  // Add residual blocks for each point pair
  for (int i = 0; i < num_points; ++i) {
    // Extract corresponding source and target points
    const Eigen::Vector3d& source_point = source_.points_[sourceIndices[i]];
    const Eigen::Vector3d& target_point = target_.points_[targetIndices[i]];

    // Create a cost function for the current point pair
    ceres::CostFunction* cost_function = PointDistance::Create(source_point, target_point);

    // Add the cost function to the problem
    problem.AddResidualBlock(cost_function, nullptr, transformation_arr.data(), transformation_arr.data() + 3);
  }

  // Set the options and solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Extract the optimized rotation (angle-axis) and translation
  double* rotation = transformation_arr.data();
  double* translation = transformation_arr.data() + 3;

  // Convert the optimized rotation (angle-axis) and translation to a transformation matrix
  Eigen::Matrix3d rotationMat;
  ceres::AngleAxisToRotationMatrix(rotation, rotationMat.data());

  transformation.block<3, 3>(0, 0) = rotationMat;
  transformation.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(translation);

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