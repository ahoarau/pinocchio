//
// Copyright (c) 2025 INRIA
//

#include <iostream>

#include "pinocchio/algorithm/aba.hpp"
#include "pinocchio/algorithm/aba-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/crba.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/kinematics-derivatives.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/frames-derivatives.hpp"
#include "pinocchio/algorithm/jacobian.hpp"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/center-of-mass-derivatives.hpp"
#include "pinocchio/algorithm/centroidal.hpp"
#include "pinocchio/algorithm/centroidal-derivatives.hpp"
#include "pinocchio/algorithm/compute-all-terms.hpp"
#include "pinocchio/algorithm/energy.hpp"
#include "pinocchio/algorithm/cholesky.hpp"
#include "pinocchio/algorithm/contact-info.hpp"
#include "pinocchio/algorithm/contact-dynamics.hpp"
#include "pinocchio/algorithm/contact-cholesky.hpp"
#include "pinocchio/algorithm/constrained-dynamics.hpp"
#include "pinocchio/algorithm/constrained-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/impulse-dynamics.hpp"
#include "pinocchio/algorithm/impulse-dynamics-derivatives.hpp"
#include "pinocchio/algorithm/regressor.hpp"
#include "pinocchio/algorithm/joint-configuration.hpp"
#include "pinocchio/multibody/sample-models.hpp"
#include "pinocchio/spatial/classic-acceleration.hpp"
#include "pinocchio/spatial/explog.hpp"
using namespace pinocchio;

#include <boost/test/unit_test.hpp>

#if !(defined(__has_feature) && __has_feature(realtime_sanitizer))
  #error "rtsan not enabled. Please enable rtsan with -fsanitize=realtime"
#endif

// Mimic what we have in the "standalone" version of rtsan
extern "C"
{
  void __rtsan_realtime_enter(void);
  void __rtsan_realtime_exit(void);
} // extern "C"

namespace
{
  // RAII class to enter/exit RTSan realtime mode
  struct ScopedSanitizeRealtime
  {
    ScopedSanitizeRealtime()
    {
      __rtsan_realtime_enter();
    }
    ~ScopedSanitizeRealtime()
    {
      __rtsan_realtime_exit();
    }
  };
} // namespace

// Custom error reporter to count number of errors
static int return_code = 0;
extern "C" void __sanitizer_report_error_summary(const char * error_summary)
{
  fprintf(stderr, "%s\n", error_summary);
  return_code++;
}

// Use halt_on_error=false to continue execution on errors and report all errors at once
// __attribute__((__visibility__("default"))) extern "C" const char * __rtsan_default_options()
// {
//   return "halt_on_error=false";
// }

bool hasMimicJoints(const Model & model)
{
  return !model.mimicked_joints.empty();
}

int run_kinematics_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Forward kinematics (position only)
    forwardKinematics(model, data, q);

    // Forward kinematics (position + velocity)
    forwardKinematics(model, data, q, v);

    // Forward kinematics (position + velocity + acceleration)
    forwardKinematics(model, data, q, v, a);

    // Update global placements
    updateGlobalPlacements(model, data);
  }
  return return_code;
}

int run_jacobians_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute joint Jacobians
    computeJointJacobians(model, data, q);
  }

  // Get specific joint Jacobian
  const Data::Matrix6x J = Data::Matrix6x::Zero(6, model.nv);
  const JointIndex joint_id = static_cast<JointIndex>(model.njoints - 1);
  {
    ScopedSanitizeRealtime sanitizer;
    getJointJacobian(model, data, joint_id, LOCAL, J);
    getJointJacobian(model, data, joint_id, WORLD, J);
    getJointJacobian(model, data, joint_id, LOCAL_WORLD_ALIGNED, J);

    // Compute Jacobian time variation
    computeJointJacobiansTimeVariation(model, data, q, v);
  }
  return return_code;
}

int run_non_linear_effects_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Non-linear effects
    nonLinearEffects(model, data, q, v);
  }
  return return_code;
}

int run_rnea_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // RNEA (Recursive Newton-Euler Algorithm)
    rnea(model, data, q, v, a);
  }
  return return_code;
}

int run_crba_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  {
    ScopedSanitizeRealtime sanitizer;
    // CRBA (Composite Rigid Body Algorithm)
    crba(model, data, q);
    crba(model, data, q, Convention::WORLD);
    crba(model, data, q, Convention::LOCAL);
  }
  return return_code;
}

int run_aba_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd tau = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // ABA (Articulated Body Algorithm)
    aba(model, data, q, v, tau);
  }
  return return_code;
}

int run_derivatives_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd tau = Eigen::VectorXd::Random(model.nv);

  {
    ScopedSanitizeRealtime sanitizer;
    // Compute forward kinematics derivatives
    computeForwardKinematicsDerivatives(model, data, q, v, a);
  }

  // RNEA derivatives
  const Data::MatrixXs rnea_partial_dq = Data::MatrixXs::Zero(model.nv, model.nv);
  const Data::MatrixXs rnea_partial_dv = Data::MatrixXs::Zero(model.nv, model.nv);
  const Data::MatrixXs rnea_partial_da = Data::MatrixXs::Zero(model.nv, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    computeRNEADerivatives(model, data, q, v, a, rnea_partial_dq, rnea_partial_dv, rnea_partial_da);
  }

  // ABA derivatives
  const Data::MatrixXs aba_partial_dq = Data::MatrixXs::Zero(model.nv, model.nv);
  const Data::MatrixXs aba_partial_dv = Data::MatrixXs::Zero(model.nv, model.nv);
  const Data::MatrixXs aba_partial_dtau = Data::MatrixXs::Zero(model.nv, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    computeABADerivatives(model, data, q, v, tau, aba_partial_dq, aba_partial_dv, aba_partial_dtau);
  }
  return return_code;
}

int run_center_of_mass_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute center of mass position
    centerOfMass(model, data, q);

    // Compute center of mass velocity
    centerOfMass(model, data, q, v);

    // Compute center of mass acceleration
    centerOfMass(model, data, q, v, a);

    // Jacobian of center of mass
    jacobianCenterOfMass(model, data, q);
  }
  return return_code;
}

int run_center_of_mass_derivatives_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Data::Matrix3x vcom_partial_dq = Data::Matrix3x::Zero(3, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Center of mass derivatives
    getCenterOfMassVelocityDerivatives(model, data, vcom_partial_dq);
  }
  return return_code;
}

int run_centroidal_dynamics_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  const Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  Data::Matrix6x dh_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_dv = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_da = Data::Matrix6x::Zero(6, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute centroidal momentum
    computeCentroidalMomentum(model, data, q, v);

    // Compute centroidal momentum time variation
    computeCentroidalMomentumTimeVariation(model, data, q, v, a);

    // Centroidal momentum Jacobian
    ccrba(model, data, q, v);

    // Centroidal derivatives
    computeCentroidalDynamicsDerivatives(model, data, q, v, a, dh_dq, dhdot_dq, dhdot_dv, dhdot_da);
  }
  return return_code;
}

int run_frame_algorithms_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  {
    ScopedSanitizeRealtime sanitizer;
    // Update frame placements
    updateFramePlacements(model, data);

    // Forward kinematics for frames
    framesForwardKinematics(model, data, q);
  }

  const Data::Matrix6x frame_J = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x v_partial_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x v_partial_dv = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x a_partial_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x a_partial_dv = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x a_partial_da = Data::Matrix6x::Zero(6, model.nv);

  for (FrameIndex frame_idx = 0; frame_idx < static_cast<FrameIndex>(model.nframes); ++frame_idx)
  {
    ScopedSanitizeRealtime sanitizer;
    getFrameJacobian(model, data, frame_idx, LOCAL, frame_J);
    getFrameJacobian(model, data, frame_idx, WORLD, frame_J);
    getFrameJacobian(model, data, frame_idx, LOCAL_WORLD_ALIGNED, frame_J);

    // Compute frame Jacobian
    computeFrameJacobian(model, data, q, frame_idx, LOCAL, frame_J);
    computeFrameJacobian(model, data, q, frame_idx, WORLD, frame_J);

    // Frame Jacobian time variation
    getFrameJacobianTimeVariation(model, data, frame_idx, LOCAL, frame_J);
    getFrameJacobianTimeVariation(model, data, frame_idx, WORLD, frame_J);

    // Frame velocity
    getFrameVelocity(model, data, frame_idx, LOCAL);
    getFrameVelocity(model, data, frame_idx, WORLD);
    getFrameVelocity(model, data, frame_idx, LOCAL_WORLD_ALIGNED);

    // Frame acceleration
    getFrameAcceleration(model, data, frame_idx, LOCAL);
    getFrameAcceleration(model, data, frame_idx, WORLD);
    getFrameAcceleration(model, data, frame_idx, LOCAL_WORLD_ALIGNED);

    // Frame classical acceleration
    getFrameClassicalAcceleration(model, data, frame_idx, LOCAL);
    getFrameClassicalAcceleration(model, data, frame_idx, WORLD);
    getFrameClassicalAcceleration(model, data, frame_idx, LOCAL_WORLD_ALIGNED);

    // Frame derivatives
    if (!hasMimicJoints(model))
    {
      ScopedSanitizeRealtime sanitizer;
      getFrameVelocityDerivatives(model, data, frame_idx, LOCAL, v_partial_dq, v_partial_dv);
      getFrameAccelerationDerivatives(
        model, data, frame_idx, LOCAL, v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);
    }
  }
  return return_code;
}

int run_compute_all_terms_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // pinocchio::forwardKinematics
    // pinocchio::crba
    // pinocchio::nonLinearEffects
    // pinocchio::computeJointJacobians
    // pinocchio::centerOfMass
    // pinocchio::jacobianCenterOfMass
    // pinocchio::ccrba
    // pinocchio::computeKineticEnergy
    // pinocchio::computePotentialEnergy
    // pinocchio::computeGeneralizedGravity
    computeAllTerms(model, data, q, v);
  }
  return return_code;
}

int run_energy_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  const Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute kinetic energy
    computeKineticEnergy(model, data, q, v);

    // Compute potential energy
    computePotentialEnergy(model, data, q);
  }
  return return_code;
}

int run_compute_generalized_gravity_test(const Model & model, Data & data)
{
  const Eigen::VectorXd q = randomConfiguration(model);
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute generalized gravity
    computeGeneralizedGravity(model, data, q);
  }
  return return_code;
}

int run_cholesky_test(const Model & model, Data & data)
{
  Eigen::VectorXd v_chol = Eigen::VectorXd::Random(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    cholesky::decompose(model, data);
    cholesky::solve(model, data, v_chol);
  }

  Data::MatrixXs M_inv = Data::MatrixXs::Zero(model.nv, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    cholesky::computeMinv(model, data, M_inv);
  }
  return return_code;
}

int run_joint_configuration_operations_test(const Model & model)
{
  Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXd dq = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd q_neutral = neutral(model);
  Eigen::VectorXd q_integrated(model.nq);
  Eigen::VectorXd q_interp(model.nq);

  {
    ScopedSanitizeRealtime sanitizer;
    normalize(model, q);
    difference(model, q, q_neutral, dq);
    integrate(model, q, v, q_integrated);
    interpolate(model, q, q_neutral, 0.5, q_interp);
    distance(model, q, q_neutral);
    isNormalized(model, q);
    isSameConfiguration(model, q, q, 1e-12);
  }
  return return_code;
}

BOOST_AUTO_TEST_CASE(dynamic_allocations_spatial_operations)
{
  {
    ScopedSanitizeRealtime sanitizer;
    // Classic acceleration
    SE3 M = SE3::Random();
    Motion v_spatial = Motion::Random();
    Motion a_spatial = Motion::Random();
    classicAcceleration(v_spatial, a_spatial);

    // Exponential/logarithm maps
    SE3::Vector3 w = SE3::Vector3::Random();
    exp3(w);
    log3(SE3::Random().rotation());

    Motion::Vector6 nu = Motion::Vector6::Random();
    exp6(nu);
    log6(M);
  }
}

void run_dynamic_allocations_test(const Model & model)
{
  // mimic only support the following algorithms:
  // RNEA
  // CRBA
  // Forward Kinematics
  // Jacobians and Frames
  // Centroidal Algorithm (ccrba)

  Data data(model);
  run_kinematics_test(model, data);
  run_jacobians_test(model, data);
  run_non_linear_effects_test(model, data);
  run_frame_algorithms_test(model, data);
  run_rnea_test(model, data);
  run_crba_test(model, data);

  run_aba_test(model, data);
  run_center_of_mass_derivatives_test(model, data);
  run_derivatives_test(model, data);
  run_center_of_mass_test(model, data);
  run_centroidal_dynamics_test(model, data);
  run_compute_all_terms_test(model, data);
  run_energy_test(model, data);
  run_compute_generalized_gravity_test(model, data);
  run_cholesky_test(model, data);
  run_joint_configuration_operations_test(model);

#if 0 // /!\ Those tests currently fails /!\
  // ============ REGRESSOR ============
  // Compute joint torque regressor
  {
    ScopedSanitizeRealtime sanitizer;
    const Data::MatrixXs & regressor = computeJointTorqueRegressor(model, data, q, v, a);
    boost::ignore_unused(regressor);
  }

  // ============ CONTACT DYNAMICS ============
  // Create contacts
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel) contact_models;
  PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData) contact_data;

  if (model.nframes > 1)
  {
    size_t frame_idx = static_cast<size_t>(model.nframes - 1);
    JointIndex contact_joint = model.frames[frame_idx].parentJoint;
    RigidConstraintModel contact_model_6d(CONTACT_6D, model, contact_joint, LOCAL);
    contact_models.push_back(contact_model_6d);
    contact_data.push_back(RigidConstraintData(contact_model_6d));

    // Initialize contact data
    initConstraintDynamics(model, data, contact_models);

    {
      // /!\ This test currently fails due to dynamic allocations in constraintDynamics /!\
      // ScopedSanitizeRealtime sanitizer;
      // Constrained forward dynamics
      constraintDynamics(model, data, q, v, tau, contact_models, contact_data);
    }

    // Contact Cholesky
    ContactCholeskyDecomposition contact_chol;
    contact_chol.allocate(model, contact_models);
    {
      ScopedSanitizeRealtime sanitizer;
      crba(model, data, q, Convention::WORLD);
      contact_chol.compute(model, data, contact_models, contact_data);
    }

    // Constrained dynamics derivatives
    Data::MatrixXs ddq_dq = Data::MatrixXs::Zero(model.nv, model.nv);
    Data::MatrixXs ddq_dv = Data::MatrixXs::Zero(model.nv, model.nv);
    Data::MatrixXs ddq_dtau = Data::MatrixXs::Zero(model.nv, model.nv);
    const int constraint_dim = contact_models[0].size();
    Data::MatrixXs lambda_dq = Data::MatrixXs::Zero(constraint_dim, model.nv);
    Data::MatrixXs lambda_dv = Data::MatrixXs::Zero(constraint_dim, model.nv);
    Data::MatrixXs lambda_dtau = Data::MatrixXs::Zero(constraint_dim, model.nv);

    {
      ScopedSanitizeRealtime sanitizer;
      computeConstraintDynamicsDerivatives(
        model, data, contact_models, contact_data, ddq_dq, ddq_dv, ddq_dtau, lambda_dq, lambda_dv,
        lambda_dtau);
    }

    // Impulse dynamics
    Eigen::VectorXd v_before = v;
    const double r_coeff = 0.0;
    ProximalSettings prox_settings(1e-12, 0., 1);
    {
      ScopedSanitizeRealtime sanitizer;
      impulseDynamics(
        model, data, q, v_before, contact_models, contact_data, r_coeff, prox_settings);
    }

    // Impulse dynamics derivatives
    Data::MatrixXs ddv_dq = Data::MatrixXs::Zero(model.nv, model.nv);
    Data::MatrixXs ddv_dvbefore = Data::MatrixXs::Zero(model.nv, model.nv);
    Data::MatrixXs impulse_dq = Data::MatrixXs::Zero(constraint_dim, model.nv);
    Data::MatrixXs impulse_dv = Data::MatrixXs::Zero(constraint_dim, model.nv);

    {
      ScopedSanitizeRealtime sanitizer;
      computeImpulseDynamicsDerivatives(
        model, data, contact_models, contact_data, r_coeff, prox_settings);
    }
  }
#endif
}

// BOOST_AUTO_TEST_CASE(dynamic_allocations_humanoid_random_free_floating)
// {
//   // Test with humanoid random model (Free floating)
//   Model model;
//   const bool using_free_floating = true;
//   const bool using_mimic = false;
//   buildModels::humanoidRandom(model, using_free_floating, using_mimic);
//   run_dynamic_allocations_test(model);
// }

// BOOST_AUTO_TEST_CASE(dynamic_allocations_humanoid_random_composite)
// {
//   // Test with humanoid random model (Composite)
//   Model model;
//   const bool using_free_floating = false;
//   const bool using_mimic = false;
//   buildModels::humanoidRandom(model, using_free_floating, using_mimic);
//   run_dynamic_allocations_test(model);
// }

// BOOST_AUTO_TEST_CASE(dynamic_allocations_manipulator)
// {
//   // Test with manipulator
//   Model model;
//   const bool using_mimic = false;
//   buildModels::manipulator(model, using_mimic);
//   run_dynamic_allocations_test(model);
// }

BOOST_AUTO_TEST_CASE(dynamic_allocations_manipulator_mimic)
{
  // Test with manipulator (Mimic)
  Model model;
  const bool using_mimic = true;
  buildModels::manipulator(model, using_mimic);
  run_dynamic_allocations_test(model);
}

// BOOST_AUTO_TEST_CASE(dynamic_allocations_humanoid_free_floating)
// {
//   // Test with humanoid (Free floating)
//   Model model;
//   const bool using_free_floating = true;
//   buildModels::humanoid(model, using_free_floating);
//   run_dynamic_allocations_test(model);
// }

// BOOST_AUTO_TEST_CASE(dynamic_allocations_humanoid_composite)
// {
//   // Test with humanoid (Composite)
//   Model model;
//   const bool using_free_floating = false;
//   buildModels::humanoid(model, using_free_floating);
//   run_dynamic_allocations_test(model);
// }
