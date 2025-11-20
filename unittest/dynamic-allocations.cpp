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

// Custom error reporter to count number of errors
static int return_code = 0;
extern "C" void __sanitizer_report_error_summary(const char * error_summary)
{
  fprintf(stderr, "%s\n", error_summary);
  return_code++;
}

// Use halt_on_error=false to continue execution on errors and report all errors at once
__attribute__((__visibility__("default"))) extern "C" const char * __rtsan_default_options()
{
  return "halt_on_error=false";
}

void run_dynamic_allocations_test(const Model & model)
{
  return_code = 0;
  BOOST_CHECK(model.njoints > 0);
  Data data(model);

  // Generate random configuration, velocity, and acceleration vectors
  Eigen::VectorXd q = randomConfiguration(model);
  Eigen::VectorXd v = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd a = Eigen::VectorXd::Random(model.nv);
  Eigen::VectorXd tau = Eigen::VectorXd::Random(model.nv);

  // ============ KINEMATICS ALGORITHMS ============
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

    // Compute forward kinematics derivatives
    computeForwardKinematicsDerivatives(model, data, q, v, a);
  }

  // ============ JACOBIAN ALGORITHMS ============
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute joint Jacobians
    computeJointJacobians(model, data, q);
  }

  // Get specific joint Jacobian
  Data::Matrix6x J = Data::Matrix6x::Zero(6, model.nv);
  JointIndex joint_id = static_cast<JointIndex>(model.njoints - 1);
  {
    ScopedSanitizeRealtime sanitizer;
    getJointJacobian(model, data, joint_id, LOCAL, J);
    getJointJacobian(model, data, joint_id, WORLD, J);
    getJointJacobian(model, data, joint_id, LOCAL_WORLD_ALIGNED, J);

    // Compute Jacobian time variation
    computeJointJacobiansTimeVariation(model, data, q, v);
  }

  // ============ DYNAMICS ALGORITHMS ============
  {
    ScopedSanitizeRealtime sanitizer;
    // RNEA (Recursive Newton-Euler Algorithm)
    rnea(model, data, q, v, a);

    // Non-linear effects
    nonLinearEffects(model, data, q, v);

    // Generalized gravity
    computeGeneralizedGravity(model, data, q);

    // CRBA (Composite Rigid Body Algorithm)
    crba(model, data, q);
    crba(model, data, q, Convention::WORLD);
    crba(model, data, q, Convention::LOCAL);

    // ABA (Articulated Body Algorithm)
    aba(model, data, q, v, tau);

    // Compute minimal inverse inertia matrix
    computeMinverse(model, data, q);
  }

  // ============ DERIVATIVES ALGORITHMS ============
  // RNEA derivatives
  Data::MatrixXs rnea_partial_dq = Data::MatrixXs::Zero(model.nv, model.nv);
  Data::MatrixXs rnea_partial_dv = Data::MatrixXs::Zero(model.nv, model.nv);
  Data::MatrixXs rnea_partial_da = Data::MatrixXs::Zero(model.nv, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    computeRNEADerivatives(model, data, q, v, a, rnea_partial_dq, rnea_partial_dv, rnea_partial_da);
  }

  // ABA derivatives
  Data::MatrixXs aba_partial_dq = Data::MatrixXs::Zero(model.nv, model.nv);
  Data::MatrixXs aba_partial_dv = Data::MatrixXs::Zero(model.nv, model.nv);
  Data::MatrixXs aba_partial_dtau = Data::MatrixXs::Zero(model.nv, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    computeABADerivatives(model, data, q, v, tau, aba_partial_dq, aba_partial_dv, aba_partial_dtau);
  }

  // ============ CENTER OF MASS ALGORITHMS ============
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

  // Center of mass derivatives
  Data::Matrix3x vcom_partial_dq = Data::Matrix3x::Zero(3, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    getCenterOfMassVelocityDerivatives(model, data, vcom_partial_dq);
  }

  // ============ CENTROIDAL DYNAMICS ============
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute centroidal momentum
    computeCentroidalMomentum(model, data, q, v);

    // Compute centroidal momentum time variation
    computeCentroidalMomentumTimeVariation(model, data, q, v, a);

    // Centroidal momentum Jacobian
    ccrba(model, data, q, v);
  }

  // Centroidal derivatives
  Data::Matrix6x dh_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_dq = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_dv = Data::Matrix6x::Zero(6, model.nv);
  Data::Matrix6x dhdot_da = Data::Matrix6x::Zero(6, model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    computeCentroidalDynamicsDerivatives(model, data, q, v, a, dh_dq, dhdot_dq, dhdot_dv, dhdot_da);
  }

  // ============ FRAMES ALGORITHMS ============
  {
    ScopedSanitizeRealtime sanitizer;
    // Update frame placements
    updateFramePlacements(model, data);

    // Forward kinematics for frames
    framesForwardKinematics(model, data, q);
  }

  // Get frame Jacobian
  Data::Matrix6x frame_J = Data::Matrix6x::Zero(6, model.nv);
  if (model.nframes > 0)
  {
    FrameIndex frame_id = static_cast<FrameIndex>(model.nframes - 1);
    {
      ScopedSanitizeRealtime sanitizer;
      getFrameJacobian(model, data, frame_id, LOCAL, frame_J);
      getFrameJacobian(model, data, frame_id, WORLD, frame_J);
      getFrameJacobian(model, data, frame_id, LOCAL_WORLD_ALIGNED, frame_J);

      // Compute frame Jacobian
      computeFrameJacobian(model, data, q, frame_id, LOCAL, frame_J);
      computeFrameJacobian(model, data, q, frame_id, WORLD, frame_J);

      // Frame Jacobian time variation
      getFrameJacobianTimeVariation(model, data, frame_id, LOCAL, frame_J);
      getFrameJacobianTimeVariation(model, data, frame_id, WORLD, frame_J);

      // Frame velocity
      getFrameVelocity(model, data, frame_id, LOCAL);
      getFrameVelocity(model, data, frame_id, WORLD);
      getFrameVelocity(model, data, frame_id, LOCAL_WORLD_ALIGNED);

      // Frame acceleration
      getFrameAcceleration(model, data, frame_id, LOCAL);
      getFrameAcceleration(model, data, frame_id, WORLD);
      getFrameAcceleration(model, data, frame_id, LOCAL_WORLD_ALIGNED);

      // Frame classical acceleration
      getFrameClassicalAcceleration(model, data, frame_id, LOCAL);
      getFrameClassicalAcceleration(model, data, frame_id, WORLD);
      getFrameClassicalAcceleration(model, data, frame_id, LOCAL_WORLD_ALIGNED);
    }

    // Frame derivatives
    Data::Matrix6x v_partial_dq = Data::Matrix6x::Zero(6, model.nv);
    Data::Matrix6x v_partial_dv = Data::Matrix6x::Zero(6, model.nv);
    Data::Matrix6x a_partial_dq = Data::Matrix6x::Zero(6, model.nv);
    Data::Matrix6x a_partial_dv = Data::Matrix6x::Zero(6, model.nv);
    Data::Matrix6x a_partial_da = Data::Matrix6x::Zero(6, model.nv);
    {
      ScopedSanitizeRealtime sanitizer;
      getFrameVelocityDerivatives(model, data, frame_id, LOCAL, v_partial_dq, v_partial_dv);
      getFrameAccelerationDerivatives(
        model, data, frame_id, LOCAL, v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da);
    }
  }

  // ============ COMPUTE ALL TERMS ============
  {
    ScopedSanitizeRealtime sanitizer;
    computeAllTerms(model, data, q, v);
  }

  // ============ ENERGY ============
  {
    ScopedSanitizeRealtime sanitizer;
    // Compute kinetic energy
    computeKineticEnergy(model, data, q, v);

    // Compute potential energy
    computePotentialEnergy(model, data, q);
  }

  // ============ CHOLESKY DECOMPOSITION ============
  {
    ScopedSanitizeRealtime sanitizer;
    crba(model, data, q);
  }

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
  // ============ SPATIAL OPERATIONS ============
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

  // ============ JOINT CONFIGURATION OPERATIONS ============

  // Neutral configuration
  const Eigen::VectorXd q_neutral = neutral(model);

  // Normalize configuration
  {
    ScopedSanitizeRealtime sanitizer;
    normalize(model, q);
  }

  // Difference between configurations
  Eigen::VectorXd dq(model.nv);
  {
    ScopedSanitizeRealtime sanitizer;
    difference(model, q, q_neutral, dq);
  }

  // Integrate configuration
  Eigen::VectorXd q_integrated(model.nq);
  {
    ScopedSanitizeRealtime sanitizer;
    integrate(model, q, v, q_integrated);
  }

  // Interpolate configurations
  Eigen::VectorXd q_interp(model.nq);
  {
    ScopedSanitizeRealtime sanitizer;
    interpolate(model, q, q_neutral, 0.5, q_interp);
  }

  // Distance between configurations
  {
    ScopedSanitizeRealtime sanitizer;
    distance(model, q, q_neutral);
  }

  // Check if configuration is normalized
  {
    ScopedSanitizeRealtime sanitizer;
    isNormalized(model, q);
  }

  // Check if configuration is within limits
  {
    ScopedSanitizeRealtime sanitizer;
    isSameConfiguration(model, q, q, 1e-12);
  }
  BOOST_REQUIRE_EQUAL(return_code, 0);
}

BOOST_AUTO_TEST_CASE(dynamic_allocations)
{
  // Test with humanoid random model (Free floating)
  {
    Model model;
    buildModels::humanoidRandom(model, true);
    run_dynamic_allocations_test(model);
  }

  // Test with humanoid random model (Composite)
  {
    Model model;
    buildModels::humanoidRandom(model, false);
    run_dynamic_allocations_test(model);
  }

  // Test with manipulator
  {
    Model model;
    buildModels::manipulator(model);
    run_dynamic_allocations_test(model);
  }

  // Test with manipulator (Mimic)
  {
    Model model;
    buildModels::manipulator(model, true);
    run_dynamic_allocations_test(model);
  }

  // Test with humanoid (Free floating)
  {
    Model model;
    buildModels::humanoid(model, true);
    run_dynamic_allocations_test(model);
  }

  // Test with humanoid (Composite)
  {
    Model model;
    buildModels::humanoid(model, false);
    run_dynamic_allocations_test(model);
  }
}
